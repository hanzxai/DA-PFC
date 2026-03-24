# models/kernels.py
"""
JIT 编译的仿真内核
三种模式：静态 / 分步给药 / D1 动力学

注意: @torch.jit.script 函数内部不能访问 Python 全局模块 (如 config)，
      因此 LIF 参数在每个函数顶部以字面量定义。
      所有数值与 config.py 保持一致，修改时需同步。
"""
import torch


# ======================================================================
# 工具函数: alpha_D1 一阶动力学单步更新 (可单独调试)
# ======================================================================
@torch.jit.script
def compute_alpha_d1_step(
    alpha_d1: torch.Tensor,
    da_t: torch.Tensor,
    current_time: float,
    da_onset: float,
    dt: float,
) -> torch.Tensor:
    """
    alpha_D1 一阶动力学单步更新。

    Args:
        alpha_d1 : (batch, 1) 当前 D1 受体激活状态
        da_t     : (batch, 1) 当前时刻各 batch 的 DA 浓度 (nM)
        current_time : 当前仿真时间 (ms)
        da_onset     : DA 给药起始时间 (ms)
        dt           : 时间步长 (ms)

    Returns:
        alpha_d1_new : (batch, 1) 更新后的 D1 受体激活状态
    """
    TAU_ON_D1  = 30876.1
    TAU_OFF_D1 = 164472.5
    EC50_D1    = 4.0
    BETA       = 1.0

    # 目标值 S(t) — Sigmoid
    s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))
    if current_time < da_onset:
        s_d1 = torch.zeros_like(s_d1)
    s_d1[0] = 0.0  # Control batch 始终为 0

    # 一阶动力学: d(alpha_d1)/dt = (s_d1 - alpha_d1) / tau
    diff = s_d1 - alpha_d1
    tau_dynamic = torch.where(
        diff > 0,
        torch.tensor(TAU_ON_D1,  device=alpha_d1.device),
        torch.tensor(TAU_OFF_D1, device=alpha_d1.device),
    )
    alpha_d1_new = alpha_d1 + (diff / tau_dynamic) * dt
    return alpha_d1_new


# ======================================================================
# 工具函数: alpha_D1 一阶动力学单步更新 —— k_on / k_off 参数化版本
#
# 方程形式:
#   dα/dt = k_on · (s_D1 - α)⁺  -  k_off · (α - s_D1)⁺
#
# 其中 (x)⁺ = max(x, 0)，即:
#   - 当 α < s_D1 (上升阶段): dα/dt = k_on  · (s_D1 - α)
#   - 当 α > s_D1 (下降阶段): dα/dt = -k_off · (α - s_D1)
#   - 当 α = s_D1 (稳态):     dα/dt = 0  → 稳态精确等于 s_D1
#
# 与 tau 版本的关系:
#   k_on  = 1 / TAU_ON_D1   (上升速率常数, 单位 ms⁻¹)
#   k_off = 1 / TAU_OFF_D1  (下降速率常数, 单位 ms⁻¹)
#
# 物理意义:
#   k_on  描述受体被 DA 激活的速率 (越大上升越快)
#   k_off 描述受体自发失活/解离的速率 (越大下降越快)
#   稳态激活水平由 s_D1 (Sigmoid of DA concentration) 决定
# ======================================================================
@torch.jit.script
def compute_alpha_d1_step_kon_koff(
    alpha_d1: torch.Tensor,
    da_t: torch.Tensor,
    current_time: float,
    da_onset: float,
    dt: float,
    k_on: float,
    k_off: float,
) -> torch.Tensor:
    """
    alpha_D1 一阶动力学单步更新 —— k_on / k_off 参数化版本。

    方程: dα/dt = k_on · (s_D1 - α)⁺ - k_off · (α - s_D1)⁺
    稳态: α_ss = s_D1  (峰值精确等于 Sigmoid 目标值)

    Args:
        alpha_d1     : (batch, 1) 当前 D1 受体激活状态
        da_t         : (batch, 1) 当前时刻各 batch 的 DA 浓度 (nM)
        current_time : 当前仿真时间 (ms)
        da_onset     : DA 给药起始时间 (ms)
        dt           : 时间步长 (ms)
        k_on         : 上升速率常数 (ms⁻¹), 默认 = 1/TAU_ON_D1  ≈ 3.24e-5
        k_off        : 下降速率常数 (ms⁻¹), 默认 = 1/TAU_OFF_D1 ≈ 6.08e-6

    Returns:
        alpha_d1_new : (batch, 1) 更新后的 D1 受体激活状态
    """
    EC50_D1 = 4.0
    BETA    = 1.0

    # 目标值 S(t) — Sigmoid
    s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))
    if current_time < da_onset:
        s_d1 = torch.zeros_like(s_d1)
    s_d1[0] = 0.0  # Control batch 始终为 0

    # 一阶动力学: 上升项 / 下降项分离
    rise_term = torch.clamp(s_d1 - alpha_d1, min=0.0)   # (s_D1 - α)⁺
    fall_term = torch.clamp(alpha_d1 - s_d1, min=0.0)   # (α - s_D1)⁺

    d_alpha = k_on * rise_term - k_off * fall_term
    alpha_d1_new = alpha_d1 + d_alpha * dt
    return alpha_d1_new


# ======================================================================
# 工具函数: alpha_D1 Langmuir 受体结合动力学单步更新
#
# 方程形式 (标准配体-受体结合模型):
#   dα/dt = k_on · s_D1 · (1 - α)  -  k_off · α
#
# 物理意义:
#   k_on · s_D1 · (1 - α) : DA 浓度驱动的受体结合速率 (正比于游离受体 1-α)
#   k_off · α              : 受体自发解离速率 (正比于已结合受体 α)
#
# 稳态解 (令 dα/dt = 0):
#   α_ss = (k_on · s_D1) / (k_on · s_D1 + k_off)
#        = s_D1 / (s_D1 + k_off/k_on)
#        = s_D1 / (s_D1 + Kd)    其中 Kd = k_off/k_on 为解离常数
#
# 注意: 稳态值 α_ss ≠ s_D1，而是由 Kd 决定的饱和曲线
# ======================================================================
@torch.jit.script
def compute_alpha_d1_step_langmuir(
    alpha_d1: torch.Tensor,
    da_t: torch.Tensor,
    current_time: float,
    da_onset: float,
    dt: float,
    k_on: float,
    k_off: float,
) -> torch.Tensor:
    """
    alpha_D1 Langmuir 受体结合动力学单步更新。

    方程: dα/dt = k_on · s_D1 · (1 - α) - k_off · α
    稳态: α_ss = k_on·s_D1 / (k_on·s_D1 + k_off)  (< s_D1)

    Args:
        alpha_d1     : (batch, 1) 当前 D1 受体激活状态
        da_t         : (batch, 1) 当前时刻各 batch 的 DA 浓度 (nM)
        current_time : 当前仿真时间 (ms)
        da_onset     : DA 给药起始时间 (ms)
        dt           : 时间步长 (ms)
        k_on         : 结合速率常数 (ms⁻¹)
        k_off        : 解离速率常数 (ms⁻¹)

    Returns:
        alpha_d1_new : (batch, 1) 更新后的 D1 受体激活状态
    """
    EC50_D1 = 4.0
    BETA    = 1.0

    # 目标值 S(t) — Sigmoid (DA 浓度对应的激活目标)
    s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))
    if current_time < da_onset:
        s_d1 = torch.zeros_like(s_d1)
    s_d1[0] = 0.0  # Control batch 始终为 0

    # Langmuir 动力学: 结合项 - 解离项
    bind_term   = k_on * s_d1 * (1.0 - alpha_d1)   # 结合: 正比于游离受体
    unbind_term = k_off * alpha_d1                   # 解离: 正比于已结合受体

    d_alpha = bind_term - unbind_term
    alpha_d1_new = alpha_d1 + d_alpha * dt
    # 限制在 [0, 1]
    alpha_d1_new = torch.clamp(alpha_d1_new, 0.0, 1.0)
    return alpha_d1_new


# ======================================================================
# 工具函数: alpha_D2 一阶动力学单步更新 (tau 版本)
#
# D2 受体与 D1 受体的关键差异:
#   1. 亲和力更高: EC50_D2 = 8.0 nM (当前代码设定; 生物学上 D2 亲和力
#      实际高于 D1, 但此处沿用已有参数体系)
#   2. 动力学更快: D2 受体偶联 Gi 蛋白, 信号转导链更短, 响应更迅速
#      τ_on_D2  ≈ 10000 ms  (约为 D1 的 1/3)
#      τ_off_D2 ≈ 50000 ms  (约为 D1 的 1/3)
#   3. 效应方向相反: D2 激活 → 抑制 cAMP → 降低神经元兴奋性
#      在网络中体现为: mod_R 减小 (R_eff 降低), I_mod 为负 (抑制电流)
#
# 方程形式 (与 D1 tau 版本结构相同, 仅参数不同):
#   dα_D2/dt = (s_D2(t) - α_D2) / τ_D2
#
# 其中 τ_D2 根据方向动态选择:
#   τ_D2 = τ_on_D2  当 s_D2 > α_D2 (上升)
#   τ_D2 = τ_off_D2 当 s_D2 < α_D2 (下降)
# ======================================================================
@torch.jit.script
def compute_alpha_d2_step(
    alpha_d2: torch.Tensor,
    da_t: torch.Tensor,
    current_time: float,
    da_onset: float,
    dt: float,
) -> torch.Tensor:
    """
    alpha_D2 一阶动力学单步更新 (tau 版本)。

    D2 受体动力学比 D1 更快 (τ 约为 D1 的 1/3):
      τ_on_D2  = 10000 ms
      τ_off_D2 = 50000 ms

    Args:
        alpha_d2     : (batch, 1) 当前 D2 受体激活状态
        da_t         : (batch, 1) 当前时刻各 batch 的 DA 浓度 (nM)
        current_time : 当前仿真时间 (ms)
        da_onset     : DA 给药起始时间 (ms)
        dt           : 时间步长 (ms)

    Returns:
        alpha_d2_new : (batch, 1) 更新后的 D2 受体激活状态
    """
    TAU_ON_D2  = 10000.0   # ms — D2 上升时间常数 (比 D1 快 ~3x)
    TAU_OFF_D2 = 50000.0   # ms — D2 下降时间常数 (比 D1 快 ~3x)
    EC50_D2    = 8.0       # nM — D2 半效浓度
    BETA       = 1.0

    # 目标值 S(t) — Sigmoid
    s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))
    if current_time < da_onset:
        s_d2 = torch.zeros_like(s_d2)
    s_d2[0] = 0.0  # Control batch 始终为 0

    # 一阶动力学: d(alpha_d2)/dt = (s_d2 - alpha_d2) / tau
    diff = s_d2 - alpha_d2
    tau_dynamic = torch.where(
        diff > 0,
        torch.tensor(TAU_ON_D2,  device=alpha_d2.device),
        torch.tensor(TAU_OFF_D2, device=alpha_d2.device),
    )
    alpha_d2_new = alpha_d2 + (diff / tau_dynamic) * dt
    return alpha_d2_new


# ======================================================================
# 工具函数: alpha_D2 一阶动力学单步更新 —— k_on / k_off 参数化版本
#
# 方程形式 (与 D1 k_on/k_off 版本结构相同):
#   dα_D2/dt = k_on_D2 · (s_D2 - α_D2)⁺  -  k_off_D2 · (α_D2 - s_D2)⁺
#
# 稳态: α_ss = s_D2  (精确等于 Sigmoid 目标值)
#
# 参数关系:
#   k_on_D2  = 1 / TAU_ON_D2  ≈ 1.0e-4  ms⁻¹  (比 D1 快 ~3x)
#   k_off_D2 = 1 / TAU_OFF_D2 ≈ 2.0e-5  ms⁻¹  (比 D1 快 ~3x)
# ======================================================================
@torch.jit.script
def compute_alpha_d2_step_kon_koff(
    alpha_d2: torch.Tensor,
    da_t: torch.Tensor,
    current_time: float,
    da_onset: float,
    dt: float,
    k_on: float,
    k_off: float,
) -> torch.Tensor:
    """
    alpha_D2 一阶动力学单步更新 —— k_on / k_off 参数化版本。

    方程: dα/dt = k_on · (s_D2 - α)⁺ - k_off · (α - s_D2)⁺
    稳态: α_ss = s_D2

    Args:
        alpha_d2     : (batch, 1) 当前 D2 受体激活状态
        da_t         : (batch, 1) 当前时刻各 batch 的 DA 浓度 (nM)
        current_time : 当前仿真时间 (ms)
        da_onset     : DA 给药起始时间 (ms)
        dt           : 时间步长 (ms)
        k_on         : 上升速率常数 (ms⁻¹)
        k_off        : 下降速率常数 (ms⁻¹)

    Returns:
        alpha_d2_new : (batch, 1) 更新后的 D2 受体激活状态
    """
    EC50_D2 = 8.0
    BETA    = 1.0

    # 目标值 S(t) — Sigmoid
    s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))
    if current_time < da_onset:
        s_d2 = torch.zeros_like(s_d2)
    s_d2[0] = 0.0  # Control batch 始终为 0

    # 一阶动力学: 上升项 / 下降项分离
    rise_term = torch.clamp(s_d2 - alpha_d2, min=0.0)   # (s_D2 - α)⁺
    fall_term = torch.clamp(alpha_d2 - s_d2, min=0.0)   # (α - s_D2)⁺

    d_alpha = k_on * rise_term - k_off * fall_term
    alpha_d2_new = alpha_d2 + d_alpha * dt
    return alpha_d2_new


# ======================================================================
# 工具函数: alpha_D2 Langmuir 受体结合动力学单步更新
#
# 方程形式 (标准配体-受体结合模型):
#   dα_D2/dt = k_on_D2 · s_D2 · (1 - α_D2)  -  k_off_D2 · α_D2
#
# 稳态解:
#   α_ss = (k_on · s_D2) / (k_on · s_D2 + k_off)
#        = s_D2 / (s_D2 + Kd_D2)    其中 Kd_D2 = k_off/k_on
#
# 注意: D2 的 Kd 与 D1 不同, 反映其不同的受体-配体亲和力
# ======================================================================
@torch.jit.script
def compute_alpha_d2_step_langmuir(
    alpha_d2: torch.Tensor,
    da_t: torch.Tensor,
    current_time: float,
    da_onset: float,
    dt: float,
    k_on: float,
    k_off: float,
) -> torch.Tensor:
    """
    alpha_D2 Langmuir 受体结合动力学单步更新。

    方程: dα/dt = k_on · s_D2 · (1 - α) - k_off · α
    稳态: α_ss = k_on·s_D2 / (k_on·s_D2 + k_off)  (< s_D2)

    Args:
        alpha_d2     : (batch, 1) 当前 D2 受体激活状态
        da_t         : (batch, 1) 当前时刻各 batch 的 DA 浓度 (nM)
        current_time : 当前仿真时间 (ms)
        da_onset     : DA 给药起始时间 (ms)
        dt           : 时间步长 (ms)
        k_on         : 结合速率常数 (ms⁻¹)
        k_off        : 解离速率常数 (ms⁻¹)

    Returns:
        alpha_d2_new : (batch, 1) 更新后的 D2 受体激活状态
    """
    EC50_D2 = 8.0
    BETA    = 1.0

    # 目标值 S(t) — Sigmoid
    s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))
    if current_time < da_onset:
        s_d2 = torch.zeros_like(s_d2)
    s_d2[0] = 0.0  # Control batch 始终为 0

    # Langmuir 动力学: 结合项 - 解离项
    bind_term   = k_on * s_d2 * (1.0 - alpha_d2)   # 结合: 正比于游离受体
    unbind_term = k_off * alpha_d2                   # 解离: 正比于已结合受体

    d_alpha = bind_term - unbind_term
    alpha_d2_new = alpha_d2 + d_alpha * dt
    # 限制在 [0, 1]
    alpha_d2_new = torch.clamp(alpha_d2_new, 0.0, 1.0)
    return alpha_d2_new


# ======================================================================
# 内核 1: 静态 DA (一次性跑完, DA 浓度不随时间变化)
# ======================================================================
@torch.jit.script
def run_batch_network(
    W_t: torch.Tensor,
    mod_R: torch.Tensor,
    I_mod: torch.Tensor,
    scale_syn: torch.Tensor,
    duration: float,
    dt: float,
    record_indices: torch.Tensor,
    n_exc: int,
):
    """
    静态并行仿真内核。
    W_t: (N, N), mod_R/I_mod/scale_syn: (Batch, N)
    record_indices: (K, 2) -> [batch_idx, neuron_idx]
    Returns: (spike_records, v_traces)
    """
    batch_size, N = mod_R.shape
    steps = int(duration / dt)

    # --- LIF 参数 (与 config.py 一致) ---
    V_rest, V_reset, V_th = -70.0, -75.0, -50.0
    R_base, t_ref, tau_syn = 0.1, 5.0, 5.0
    bg_mean, bg_std = 250.0, 50.0  # [Aligned with Gemini] V_ss=-45mV, super-threshold drive
    C_E, C_I = 250.0, 90.0  # membrane capacitance (pF): E=250, I=90

    # --- C_m 向量: (1, N), E 神经元=250, I 神经元=90 ---
    C_m = torch.full((1, N), C_I, device=W_t.device)
    C_m[0, :n_exc] = C_E

    # --- 状态变量 ---
    # Share identical initial V across batches so baseline is consistent
    V_init = torch.rand((1, N), device=W_t.device) * (V_th - V_reset) + V_reset
    V = V_init.expand(batch_size, -1).clone()
    t_last_spike = torch.full((batch_size, N), -1000.0, device=W_t.device)
    I_syn = torch.zeros((batch_size, N), device=W_t.device)

    # --- 记录容器 ---
    max_spikes = int(steps * N * batch_size * 0.15)
    spike_records = torch.zeros((max_spikes, 3), device=W_t.device, dtype=torch.long)
    spike_count = 0
    num_record = record_indices.shape[0]
    v_traces = torch.zeros((steps, num_record), device=W_t.device)

    # --- 预计算常量 ---
    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))

    for i in range(steps):
        current_time = i * dt

        # 1. 突触电流衰减
        I_syn = I_syn * decay_factor

        # 2. 总电流 — share identical noise across batches for consistent baseline
        I_bg = (torch.randn((1, N), device=W_t.device) * bg_std + bg_mean).expand(batch_size, -1)
        I_total = (I_syn * scale_syn) + I_bg + I_mod

        # 3. LIF exact integration: V_new = V_inf + (V - V_inf) * exp(-dt/tau_m)
        #    where V_inf = V_rest + R_eff * I_total, tau_m = R_eff * C_m
        R_eff = R_base * mod_R
        V_inf = V_rest + R_eff * I_total
        tau_m = R_eff * C_m
        decay_v = torch.exp(-dt / tau_m)
        V_new = V_inf + (V - V_inf) * decay_v
        is_refractory = (current_time - t_last_spike) <= t_ref
        V = torch.where(is_refractory, torch.tensor(V_reset, device=W_t.device), V_new)

        # 4. 记录电压
        for k in range(num_record):
            v_traces[i, k] = V[record_indices[k, 0], record_indices[k, 1]]

        # 5. 发放与传播
        spikes = V > V_th
        if spikes.any():
            indices = torch.nonzero(spikes)
            num_now = indices.shape[0]
            if spike_count + num_now < max_spikes:
                spike_records[spike_count : spike_count + num_now, 0] = i
                spike_records[spike_count : spike_count + num_now, 1] = indices[:, 0]
                spike_records[spike_count : spike_count + num_now, 2] = indices[:, 1]
                spike_count += num_now
            V[spikes] = V_reset
            t_last_spike[spikes] = torch.tensor(current_time, device=W_t.device)
            I_syn += torch.matmul(spikes.float(), W_t)

    return spike_records[:spike_count], v_traces


# ======================================================================
# 内核 2: 分步给药 (Baseline → DA, 参数瞬时切换)
# ======================================================================
@torch.jit.script
def run_batch_network_stepped(
    W_t: torch.Tensor,
    params_rest: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    params_active: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    duration: float,
    dt: float,
    da_onset: float,
    record_indices: torch.Tensor,
    n_exc: int,
):
    """分步给药内核: 在 da_onset 时刻瞬时切换调节参数。"""
    mod_R_rest, I_mod_rest, scale_rest = params_rest
    mod_R_act, I_mod_act, scale_act = params_active

    batch_size, N = mod_R_rest.shape
    steps = int(duration / dt)

    # --- LIF 参数 ---
    V_rest, V_reset, V_th = -70.0, -75.0, -50.0
    R_base, t_ref, tau_syn = 0.1, 5.0, 5.0  # R_base unit: GΩ (= mV/pA)
    bg_mean, bg_std = 250.0, 50.0  # [Aligned with Gemini] V_ss=-45mV, super-threshold drive
    C_E, C_I = 250.0, 90.0  # membrane capacitance (pF): E=250, I=90

    # --- C_m 向量: (1, N), E 神经元=250, I 神经元=90 ---
    C_m = torch.full((1, N), C_I, device=W_t.device)
    C_m[0, :n_exc] = C_E

    # --- 状态变量 ---
    # Share identical initial V across batches so baseline is consistent
    V_init = torch.rand((1, N), device=W_t.device) * (V_th - V_reset) + V_reset
    V = V_init.expand(batch_size, -1).clone()
    t_last_spike = torch.full((batch_size, N), -1000.0, device=W_t.device)
    I_syn = torch.zeros((batch_size, N), device=W_t.device)

    # --- 记录容器 ---
    max_spikes = int(steps * N * batch_size * 0.15)
    spike_records = torch.zeros((max_spikes, 3), device=W_t.device, dtype=torch.long)
    spike_count = 0
    num_record = record_indices.shape[0]
    v_traces = torch.zeros((steps, num_record), device=W_t.device)

    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))

    for i in range(steps):
        current_time = i * dt

        # 根据时间切换参数
        if current_time < da_onset:
            cur_mod_R = mod_R_rest
            cur_I_mod = I_mod_rest
            cur_scale = scale_rest
        else:
            cur_mod_R = mod_R_act
            cur_I_mod = I_mod_act
            cur_scale = scale_act

        I_syn = I_syn * decay_factor
        # Share identical noise across batches for consistent baseline
        I_bg = (torch.randn((1, N), device=W_t.device) * bg_std + bg_mean).expand(batch_size, -1)
        I_total = (I_syn * cur_scale) + I_bg + cur_I_mod

        # LIF exact integration: V_new = V_inf + (V - V_inf) * exp(-dt/tau_m)
        R_eff = R_base * cur_mod_R
        V_inf = V_rest + R_eff * I_total
        tau_m = R_eff * C_m
        decay_v = torch.exp(-dt / tau_m)
        V_new = V_inf + (V - V_inf) * decay_v
        is_refractory = (current_time - t_last_spike) <= t_ref
        V = torch.where(is_refractory, torch.tensor(V_reset, device=W_t.device), V_new)

        for k in range(num_record):
            v_traces[i, k] = V[record_indices[k, 0], record_indices[k, 1]]

        spikes = V > V_th
        if spikes.any():
            indices = torch.nonzero(spikes)
            num_now = indices.shape[0]
            if spike_count + num_now < max_spikes:
                spike_records[spike_count : spike_count + num_now, 0] = i
                spike_records[spike_count : spike_count + num_now, 1] = indices[:, 0]
                spike_records[spike_count : spike_count + num_now, 2] = indices[:, 1]
                spike_count += num_now
            V[spikes] = V_reset
            t_last_spike[spikes] = torch.tensor(current_time, device=W_t.device)
            I_syn += torch.matmul(spikes.float(), W_t)

    return spike_records[:spike_count], v_traces


# ======================================================================
# 内核 3: D1 受体动力学 (alpha_D1 遵循一阶 ODE, D2 瞬时)
# ======================================================================
@torch.jit.script
def run_dynamic_d1_kernel(
    W_t: torch.Tensor,
    mask_d1: torch.Tensor,
    mask_d2: torch.Tensor,
    da_level: float,
    da_onset: float,
    duration: float,
    dt: float,
    record_indices: torch.Tensor,
    n_exc: int,
):
    """
    D1 受体动力学仿真内核。
    - D1: 一阶动力学 (Tau_on / Tau_off)
    - D2: 瞬时响应
    Batch 0 = Control (0 nM), Batch 1 = Experiment (da_level nM)
    """
    # --- 药理学常数 (与 config.py 一致) ---
    BIAS_D1, BIAS_D2 = 3.0, -3.0
    LAM_D1, LAM_D2 = 0.3, 0.2
    # --- D1 Langmuir 动力学参数 ---
    TAU_ON_D1 = 30876.1
    TAU_OFF_D1 = 164472.5
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)   # D1 binding rate (ms⁻¹)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)  # D1 unbinding rate (ms⁻¹)
    # --- D2 Langmuir 动力学参数 ---
    TAU_ON_D2 = 10000.0
    TAU_OFF_D2 = 50000.0
    k_on_d2  = 1.0 / TAU_ON_D2   # D2 binding rate (ms⁻¹)
    k_off_d2 = 1.0 / TAU_OFF_D2  # D2 unbinding rate (ms⁻¹)

    EC50_D1 = 4.0
    EC50_D2 = 8.0
    BETA = 1.0
    EPS_D1, EPS_D2 = 0.15, 0.10   # D1: Gain 增强比例; D2: Gain 减弱比例
    BIAS_D1, BIAS_D2 = 3.0, -3.0
    # R_base: 基础膜电阻 (GΩ = mV/pA), V in mV, I in pA
    # τ_m = R_base * C_m: E 神经元 τ_m = 0.1 GΩ * 250 pF = 25 ms
    V_rest, V_reset, V_th = -70.0, -75.0, -50.0
    R_base, tau_syn, t_ref = 0.1, 5.0, 5.0  # R_base unit: GΩ (= mV/pA)
    bg_mean, bg_std = 250.0, 50.0  # [Aligned with Gemini] V_ss=-45mV, super-threshold drive
    C_E, C_I = 250.0, 90.0  # membrane capacitance (pF): E=250, I=90

    # --- 初始化 ---
    N = W_t.shape[0]
    batch_size = 2
    steps = int(duration / dt)

    # --- C_m 向量: (1, N), E 神经元=250, I 神经元=90 ---
    C_m = torch.full((1, N), C_I, device=W_t.device)
    C_m[0, :n_exc] = C_E

    # Share identical initial V across batches so baseline is consistent
    V_init = torch.rand((1, N), device=W_t.device) * (V_th - V_reset) + V_reset
    V = V_init.expand(batch_size, -1).clone()
    t_last_spike = torch.full((batch_size, N), -1000.0, device=W_t.device)
    I_syn = torch.zeros((batch_size, N), device=W_t.device)
    alpha_d1 = torch.zeros((batch_size, 1), device=W_t.device)
    alpha_d2 = torch.zeros((batch_size, 1), device=W_t.device)  # D2 动力学状态

    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))

    max_spikes = int(steps * N * batch_size * 0.15)
    spike_records = torch.zeros((max_spikes, 3), device=W_t.device, dtype=torch.long)
    spike_count = 0
    num_record = record_indices.shape[0]
    v_traces = torch.zeros((steps, num_record), device=W_t.device)

    # --- 时间步循环 ---
    for i in range(steps):
        current_time = i * dt

        # A. 当前 DA 浓度
        current_da_val = 0.0
        if current_time >= da_onset:
            current_da_val = float(da_level)
        da_t = torch.tensor([[0.0], [current_da_val]], device=W_t.device)

        # B. 目标值 S(t) — Sigmoid (D1 用于调试参考, D2 由动力学函数内部计算)
        s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))
        if current_time < da_onset:
            s_d1[:] = 0.0
        s_d1[0] = 0.0  # Control 始终为 0

        # C. 更新 alpha_D1 (Langmuir 受体结合动力学)
        alpha_d1 = compute_alpha_d1_step_langmuir(alpha_d1, da_t, current_time, da_onset, dt, k_on_d1, k_off_d1)

        # D. 更新 alpha_D2 (Langmuir 受体结合动力学)
        alpha_d2 = compute_alpha_d2_step_langmuir(alpha_d2, da_t, current_time, da_onset, dt, k_on_d2, k_off_d2)

        # E. 组装调节参数
        # mod_R: D1 区域 R_eff = R_base*(1+EPS_D1*alpha_d1), D2 区域 R_eff = R_base*(1-EPS_D2*alpha_d2)
        # 非受体区域保持 R_base 不变
        mod_R = R_base * (torch.ones((batch_size, N), device=W_t.device)
                          + (EPS_D1 * alpha_d1) * mask_d1
                          - (EPS_D2 * alpha_d2) * mask_d2)
        I_mod = torch.zeros((batch_size, N), device=W_t.device)
        scale_syn = torch.ones((batch_size, N), device=W_t.device)

        I_mod += (BIAS_D1 * alpha_d1) * mask_d1
        scale_syn += (LAM_D1 * alpha_d1) * mask_d1

        I_mod += (BIAS_D2 * alpha_d2) * mask_d2
        scale_syn -= (LAM_D2 * alpha_d2) * mask_d2

        # F. LIF exact integration: V_new = V_inf + (V - V_inf) * exp(-dt/tau_m)
        I_syn = I_syn * decay_factor
        # Share identical noise across batches for consistent baseline
        I_bg = (torch.randn((1, N), device=W_t.device) * bg_std + bg_mean).expand(batch_size, -1)
        I_total = (I_syn * scale_syn) + I_bg + I_mod

        R_eff = mod_R  # mod_R 已包含 R_base，直接作为有效膜电阻
        # Exact integration: V_new = V_inf + (V - V_inf) * exp(-dt/tau_m)
        V_inf = V_rest + R_eff * I_total
        tau_m = R_eff * C_m
        decay_v = torch.exp(-dt / tau_m)
        V_new = V_inf + (V - V_inf) * decay_v
        is_refractory = (current_time - t_last_spike) <= t_ref
        V = torch.where(is_refractory, torch.tensor(V_reset, device=W_t.device), V_new)

        for k in range(num_record):
            v_traces[i, k] = V[record_indices[k, 0], record_indices[k, 1]]

        spikes = V > V_th
        if spikes.any():
            indices = torch.nonzero(spikes)
            num_now = indices.shape[0]
            if spike_count + num_now < max_spikes:
                spike_records[spike_count : spike_count + num_now, 0] = i
                spike_records[spike_count : spike_count + num_now, 1] = indices[:, 0]
                spike_records[spike_count : spike_count + num_now, 2] = indices[:, 1]
                spike_count += num_now
            V[spikes] = V_reset
            t_last_spike[spikes] = torch.tensor(current_time, device=W_t.device)
            I_syn += torch.matmul(spikes.float(), W_t)

    return spike_records[:spike_count], v_traces


# ======================================================================
# 内核 4: D1 + D2 受体动力学 (alpha_D1 和 alpha_D2 均遵循一阶 ODE)
# ======================================================================
@torch.jit.script
def run_dynamic_d1_d2_kernel(
    W_t: torch.Tensor,
    mask_d1: torch.Tensor,
    mask_d2: torch.Tensor,
    da_level: float,
    da_onset: float,
    duration: float,
    dt: float,
    record_indices: torch.Tensor,
    n_exc: int,
):
    """
    D1 + D2 受体动力学仿真内核。
    - D1: 一阶动力学 (τ_on=30876ms / τ_off=164472ms)
    - D2: 一阶动力学 (τ_on=10000ms / τ_off=50000ms)
    Batch 0 = Control (0 nM), Batch 1 = Experiment (da_level nM)
    """
    # --- 药理学常数 ---
    EC50_D1 = 4.0
    EC50_D2 = 8.0
    BETA = 1.0
    EPS_D1, EPS_D2 = 0.15, 0.10   # D1: Gain 增强比例; D2: Gain 减弱比例
    BIAS_D1, BIAS_D2 = 3.0, -3.0
    LAM_D1, LAM_D2 = 0.3, 0.2
    # --- D1 Langmuir 动力学参数 ---
    TAU_ON_D1 = 30876.1
    TAU_OFF_D1 = 164472.5
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)   # D1 binding rate (ms⁻¹)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)  # D1 unbinding rate (ms⁻¹)
    # --- D2 Langmuir 动力学参数 ---
    TAU_ON_D2 = 10000.0
    TAU_OFF_D2 = 50000.0
    k_on_d2  = 1.0 / TAU_ON_D2   # D2 binding rate (ms⁻¹)
    k_off_d2 = 1.0 / TAU_OFF_D2  # D2 unbinding rate (ms⁻¹)

    # --- LIF 参数 ---
    # R_base: 基础膜电阻 (GΩ = mV/pA), V in mV, I in pA
    # τ_m = R_base * C_m: E 神经元 τ_m = 0.1 GΩ * 250 pF = 25 ms
    V_rest, V_reset, V_th = -70.0, -75.0, -50.0
    R_base, tau_syn, t_ref = 0.1, 5.0, 5.0  # R_base unit: GΩ (= mV/pA)
    bg_mean, bg_std = 250.0, 50.0  # [Aligned with Gemini] V_ss=-45mV, super-threshold drive
    C_E, C_I = 250.0, 90.0  # membrane capacitance (pF): E=250, I=90

    # --- 初始化 ---
    N = W_t.shape[0]
    batch_size = 2
    steps = int(duration / dt)

    # --- C_m 向量: (1, N), E 神经元=250, I 神经元=90 ---
    C_m = torch.full((1, N), C_I, device=W_t.device)
    C_m[0, :n_exc] = C_E

    # Share identical initial V across batches so baseline is consistent
    V_init = torch.rand((1, N), device=W_t.device) * (V_th - V_reset) + V_reset
    V = V_init.expand(batch_size, -1).clone()
    t_last_spike = torch.full((batch_size, N), -1000.0, device=W_t.device)
    I_syn = torch.zeros((batch_size, N), device=W_t.device)
    alpha_d1 = torch.zeros((batch_size, 1), device=W_t.device)
    alpha_d2 = torch.zeros((batch_size, 1), device=W_t.device)

    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))

    max_spikes = int(steps * N * batch_size * 0.15)
    spike_records = torch.zeros((max_spikes, 3), device=W_t.device, dtype=torch.long)
    spike_count = 0
    num_record = record_indices.shape[0]
    v_traces = torch.zeros((steps, num_record), device=W_t.device)

    # --- 时间步循环 ---
    for i in range(steps):
        current_time = i * dt

        # A. 当前 DA 浓度
        current_da_val = 0.0
        if current_time >= da_onset:
            current_da_val = float(da_level)
        da_t = torch.tensor([[0.0], [current_da_val]], device=W_t.device)

        # B. 更新 alpha_D1 (Langmuir 受体结合动力学)
        alpha_d1 = compute_alpha_d1_step_langmuir(alpha_d1, da_t, current_time, da_onset, dt, k_on_d1, k_off_d1)

        # C. 更新 alpha_D2 (Langmuir 受体结合动力学)
        alpha_d2 = compute_alpha_d2_step_langmuir(alpha_d2, da_t, current_time, da_onset, dt, k_on_d2, k_off_d2)

        # D. 组装调节参数
        # mod_R: D1 区域 R_eff = R_base*(1+EPS_D1*alpha_d1), D2 区域 R_eff = R_base*(1-EPS_D2*alpha_d2)
        # 非受体区域保持 R_base 不变
        mod_R = R_base * (torch.ones((batch_size, N), device=W_t.device)
                          + (EPS_D1 * alpha_d1) * mask_d1
                          - (EPS_D2 * alpha_d2) * mask_d2)
        I_mod = torch.zeros((batch_size, N), device=W_t.device)
        scale_syn = torch.ones((batch_size, N), device=W_t.device)

        I_mod += (BIAS_D1 * alpha_d1) * mask_d1
        scale_syn += (LAM_D1 * alpha_d1) * mask_d1

        I_mod += (BIAS_D2 * alpha_d2) * mask_d2
        scale_syn -= (LAM_D2 * alpha_d2) * mask_d2

        # E. LIF exact integration: V_new = V_inf + (V - V_inf) * exp(-dt/tau_m)
        I_syn = I_syn * decay_factor
        # Share identical noise across batches for consistent baseline
        I_bg = (torch.randn((1, N), device=W_t.device) * bg_std + bg_mean).expand(batch_size, -1)
        I_total = (I_syn * scale_syn) + I_bg + I_mod

        R_eff = mod_R  # mod_R 已包含 R_base，直接作为有效膜电阻
        V_inf = V_rest + R_eff * I_total
        tau_m = R_eff * C_m
        decay_v = torch.exp(-dt / tau_m)
        V_new = V_inf + (V - V_inf) * decay_v
        is_refractory = (current_time - t_last_spike) <= t_ref
        V = torch.where(is_refractory, torch.tensor(V_reset, device=W_t.device), V_new)

        for k in range(num_record):
            v_traces[i, k] = V[record_indices[k, 0], record_indices[k, 1]]

        spikes = V > V_th
        if spikes.any():
            indices = torch.nonzero(spikes)
            num_now = indices.shape[0]
            if spike_count + num_now < max_spikes:
                spike_records[spike_count : spike_count + num_now, 0] = i
                spike_records[spike_count : spike_count + num_now, 1] = indices[:, 0]
                spike_records[spike_count : spike_count + num_now, 2] = indices[:, 1]
                spike_count += num_now
            V[spikes] = V_reset
            t_last_spike[spikes] = torch.tensor(current_time, device=W_t.device)
            I_syn += torch.matmul(spikes.float(), W_t)

    return spike_records[:spike_count], v_traces


# ======================================================================
# 参数一致性校验 (普通 Python 函数, 非 JIT)
#
# 由于 @torch.jit.script 无法访问外部模块, kernel 内部必须硬编码参数。
# 此函数在程序启动时调用, 自动比对 config.py 与 kernel 硬编码值,
# 若不一致则抛出 AssertionError, 强制开发者同步修改。
#
# 调用位置: runners.py 顶部 (import 之后立即调用)
# ======================================================================
def verify_kernel_params_consistent() -> None:
    """
    校验 kernels.py 中硬编码的参数与 config.py 是否一致。
    不一致时抛出 AssertionError, 并打印具体差异。

    在 runners.py / main.py 启动时调用一次即可。
    """
    import sys
    import os
    # 将项目根目录加入 sys.path, 确保能 import config
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import config as cfg  # type: ignore

    # --- kernel 内部硬编码值 (与各 @jit.script 函数保持一致) ---
    _KERNEL_PARAMS: dict = {
        # 受体动力学
        "TAU_ON_D1":  30876.1,
        "TAU_OFF_D1": 164472.5,
        "TAU_ON_D2":  10000.0,
        "TAU_OFF_D2": 50000.0,
        "EC50_D1":    4.0,
        "EC50_D2":    8.0,
        "BETA":       1.0,
        # 调节强度
        "EPS_D1":     0.15,
        "EPS_D2":     0.10,
"BIAS_D1":    3.0,
        "BIAS_D2":   -3.0,
        "LAM_D1":     0.3,
        "LAM_D2":     0.2,
        # LIF 参数
        "V_REST":     -70.0,
        "V_RESET":   -75.0,
        "V_TH":       -50.0,
        "R_BASE":     0.1,
        "C_E":        250.0,
        "C_I":        90.0,
        "TAU_SYN":    5.0,
        "T_REF":      5.0,
"BG_MEAN":    250.0,
        "BG_STD":     50.0,
    }

    errors: list = []
    for name, kernel_val in _KERNEL_PARAMS.items():
        cfg_val = getattr(cfg, name, None)
        if cfg_val is None:
            errors.append(f"  [MISSING] config.{name} not found")
            continue
        if abs(float(cfg_val) - float(kernel_val)) > 1e-6:
            errors.append(
                f"  [MISMATCH] {name}: config={cfg_val}, kernel={kernel_val}"
            )

    if errors:
        msg = (
            "\n\n[kernels.py] Parameter consistency check FAILED!\n"
            "The following parameters differ between config.py and kernels.py.\n"
            "Please update kernels.py to match config.py (or vice versa):\n\n"
            + "\n".join(errors)
            + "\n\nHint: Search for the parameter name in kernels.py and update the literal value.\n"
        )
        raise AssertionError(msg)

    print("[kernels.py] Parameter consistency check PASSED.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import os
    from datetime import datetime

    print("Debugging alpha_d1 / alpha_d2 dynamics (3 methods each)...")

    # ------------------------------------------------------------------ #
    # 公共参数
    # ------------------------------------------------------------------ #
    dt        = 10.0        # ms
    duration  = 1000000.0  # ms = 1000 s
    da_onset  = 100000.0   # ms = 100 s
    da_offset = 300000.0   # ms = 300 s
    da_level  = 20.0       # nM
    BETA      = 1.0

    steps       = int(duration / dt)
    time_points = np.arange(steps) * dt  # ms

    # ------------------------------------------------------------------ #
    # D1 参数
    # ------------------------------------------------------------------ #
    EC50_D1      = 4.0
    TAU_ON_D1    = 30876.1
    TAU_OFF_D1   = 164472.5
    k_on_d1      = 1.0 / (TAU_ON_D1  - 3000)
    k_off_d1     = 1.0 / (TAU_OFF_D1 + 3000)
    Kd_D1        = k_off_d1 / k_on_d1

    # ------------------------------------------------------------------ #
    # D2 参数  (动力学比 D1 快 ~3x; EC50 沿用代码中已有值 8.0 nM)
    # ------------------------------------------------------------------ #
    EC50_D2      = 8.0
    TAU_ON_D2    = 10000.0   # ms — 上升时间常数
    TAU_OFF_D2   = 50000.0   # ms — 下降时间常数
    k_on_d2      = 1.0 / TAU_ON_D2    # ≈ 1.0e-4 ms⁻¹
    k_off_d2     = 1.0 / TAU_OFF_D2   # ≈ 2.0e-5 ms⁻¹
    Kd_D2        = k_off_d2 / k_on_d2

    # ------------------------------------------------------------------ #
    # 初始化状态张量
    # ------------------------------------------------------------------ #
    alpha_d1_tau  = torch.zeros((2, 1))
    alpha_d1_k    = torch.zeros((2, 1))
    alpha_d1_lang = torch.zeros((2, 1))

    alpha_d2_tau  = torch.zeros((2, 1))
    alpha_d2_k    = torch.zeros((2, 1))
    alpha_d2_lang = torch.zeros((2, 1))

    # 记录轨迹
    tr_d1_tau,  tr_d1_k,  tr_d1_lang  = [], [], []
    tr_d2_tau,  tr_d2_k,  tr_d2_lang  = [], [], []
    tr_s_d1,    tr_s_d2               = [], []
    tr_ss_d1_lang, tr_ss_d2_lang      = [], []

    # ------------------------------------------------------------------ #
    # 时间步循环
    # ------------------------------------------------------------------ #
    for i in range(steps):
        current_time = i * dt
        current_da   = da_level if (da_onset <= current_time < da_offset) else 0.0
        da_t         = torch.tensor([[0.0], [current_da]])

        # Sigmoid 目标值
        s_d1_val = (1.0 / (1.0 + math.exp(-BETA * (current_da - EC50_D1)))
                    if current_time >= da_onset else 0.0)
        s_d2_val = (1.0 / (1.0 + math.exp(-BETA * (current_da - EC50_D2)))
                    if current_time >= da_onset else 0.0)
        tr_s_d1.append(s_d1_val)
        tr_s_d2.append(s_d2_val)

        # Langmuir 稳态值
        tr_ss_d1_lang.append(
            (k_on_d1 * s_d1_val) / (k_on_d1 * s_d1_val + k_off_d1) if s_d1_val > 0 else 0.0
        )
        tr_ss_d2_lang.append(
            (k_on_d2 * s_d2_val) / (k_on_d2 * s_d2_val + k_off_d2) if s_d2_val > 0 else 0.0
        )

        # --- D1 三种方法 ---
        alpha_d1_tau = compute_alpha_d1_step(
            alpha_d1_tau, da_t, current_time, da_onset, dt)
        tr_d1_tau.append(alpha_d1_tau[1, 0].item())

        alpha_d1_k = compute_alpha_d1_step_kon_koff(
            alpha_d1_k, da_t, current_time, da_onset, dt, k_on_d1, k_off_d1)
        tr_d1_k.append(alpha_d1_k[1, 0].item())

        alpha_d1_lang = compute_alpha_d1_step_langmuir(
            alpha_d1_lang, da_t, current_time, da_onset, dt, k_on_d1, k_off_d1)
        tr_d1_lang.append(alpha_d1_lang[1, 0].item())

        # --- D2 三种方法 ---
        alpha_d2_tau = compute_alpha_d2_step(
            alpha_d2_tau, da_t, current_time, da_onset, dt)
        tr_d2_tau.append(alpha_d2_tau[1, 0].item())

        alpha_d2_k = compute_alpha_d2_step_kon_koff(
            alpha_d2_k, da_t, current_time, da_onset, dt, k_on_d2, k_off_d2)
        tr_d2_k.append(alpha_d2_k[1, 0].item())

        alpha_d2_lang = compute_alpha_d2_step_langmuir(
            alpha_d2_lang, da_t, current_time, da_onset, dt, k_on_d2, k_off_d2)
        tr_d2_lang.append(alpha_d2_lang[1, 0].item())

    # 转 numpy
    arr_d1_tau,  arr_d1_k,  arr_d1_lang  = map(np.array, [tr_d1_tau,  tr_d1_k,  tr_d1_lang])
    arr_d2_tau,  arr_d2_k,  arr_d2_lang  = map(np.array, [tr_d2_tau,  tr_d2_k,  tr_d2_lang])
    arr_s_d1,    arr_s_d2               = map(np.array, [tr_s_d1,    tr_s_d2])
    arr_ss_d1,   arr_ss_d2              = map(np.array, [tr_ss_d1_lang, tr_ss_d2_lang])

    # ------------------------------------------------------------------ #
    # 绘图: 双子图 — 上 D1, 下 D2
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    fig.suptitle(
        f"Receptor Dynamics Comparison: D1 vs D2  |  DA={da_level} nM  "
        f"|  DA window [{int(da_onset/1000)}s – {int(da_offset/1000)}s]",
        fontsize=13, fontweight='bold'
    )

    def _annotate_peak(ax, arr, t_arr, color, label):
        """在峰值处标注数值。"""
        idx = arr.argmax()
        ax.scatter([t_arr[idx]], [arr[idx]], color=color, zorder=7, s=40)
        ax.annotate(f"peak={arr[idx]:.3f}",
                    xy=(t_arr[idx], arr[idx]),
                    xytext=(t_arr[idx] + 12000, arr[idx] + 0.03),
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                    fontsize=8, color=color)

    for ax_idx, (ax, receptor, ec50,
                 tau_on, tau_off, k_on, k_off, Kd,
                 arr_tau, arr_k, arr_lang,
                 arr_s, arr_ss) in enumerate(zip(
        axes,
        ['D1', 'D2'],
        [EC50_D1, EC50_D2],
        [TAU_ON_D1, TAU_ON_D2],
        [TAU_OFF_D1, TAU_OFF_D2],
        [k_on_d1, k_on_d2],
        [k_off_d1, k_off_d2],
        [Kd_D1, Kd_D2],
        [arr_d1_tau, arr_d2_tau],
        [arr_d1_k,   arr_d2_k],
        [arr_d1_lang, arr_d2_lang],
        [arr_s_d1,   arr_s_d2],
        [arr_ss_d1,  arr_ss_d2],
    )):
        ax.ticklabel_format(style='plain', axis='x')

        # Sigmoid 目标值
        ax.plot(time_points, arr_s, lw=1.5, ls='-', color='black', alpha=0.45, zorder=1,
                label=f"Target s_{receptor} (Sigmoid, EC50={ec50} nM)")

        # Langmuir 稳态线
        ax.plot(time_points, arr_ss, lw=1.5, ls=':', color='red', alpha=0.65, zorder=1,
                label=f"Langmuir α_ss = s/(s+Kd),  Kd={Kd:.4f}")

        # Method 2: k_on/k_off (粗橙色)
        ax.plot(time_points, arr_k, lw=4.0, ls='-', color='darkorange', alpha=0.50, zorder=2,
                label=f"Method 2 (k_on/k_off): k_on={k_on:.2e}, k_off={k_off:.2e} ms⁻¹")

        # Method 1: tau (蓝色虚线)
        ax.plot(time_points, arr_tau, lw=1.8, ls='--', color='steelblue', zorder=3,
                label=f"Method 1 (tau): τ_on={tau_on:,.0f} ms, τ_off={tau_off:,.0f} ms")

        # Method 3: Langmuir (绿色实线)
        ax.plot(time_points, arr_lang, lw=1.8, ls='-', color='seagreen', zorder=4,
                label=f"Method 3 (Langmuir): dα/dt = k_on·s·(1-α) - k_off·α")

        # 给药 / 撤药竖线
        ax.axvline(x=da_onset,  color='green', ls=':', lw=1.5,
                   label=f"DA onset  ({int(da_onset):,} ms)")
        ax.axvline(x=da_offset, color='gray',  ls=':', lw=1.5,
                   label=f"DA offset ({int(da_offset):,} ms)")

        # 峰值标注
        _annotate_peak(ax, arr_tau,  time_points, 'steelblue', 'tau')
        _annotate_peak(ax, arr_lang, time_points, 'seagreen',  'Langmuir')

        # t½ 标注 (tau 方法)
        t_half_on  = tau_on  * math.log(2)
        t_half_off = tau_off * math.log(2)
        for t_abs, arr_ref, color, marker, label_txt in [
            (da_onset  + t_half_on,  arr_tau, 'purple', '^', f"t½ on  = {t_half_on:,.0f} ms"),
            (da_offset + t_half_off, arr_tau, 'brown',  'v', f"t½ off = {t_half_off:,.0f} ms"),
        ]:
            if t_abs < time_points[-1]:
                idx = min(int(t_abs / dt), len(arr_ref) - 1)
                ax.axvline(x=t_abs, color=color, ls='-.', lw=1.2)
                ax.scatter([t_abs], [arr_ref[idx]], color=color, zorder=6, marker=marker, s=50)
                ax.annotate(label_txt,
                            xy=(t_abs, arr_ref[idx]),
                            xytext=(t_abs + 8000, arr_ref[idx] + (0.05 if marker == '^' else -0.07)),
                            arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                            fontsize=8, color=color)

        ax.set_ylabel(f"α_{receptor} activation", fontsize=11)
        ax.set_title(
            f"{receptor} Receptor  |  EC50={ec50} nM  |  "
            f"τ_on={tau_on:,.0f} ms  τ_off={tau_off:,.0f} ms  |  Kd={Kd:.4f}",
            fontsize=10
        )
        ax.legend(fontsize=7.5, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.35)

    axes[-1].set_xlabel("Time (ms)", fontsize=11)
    plt.tight_layout()

    # ------------------------------------------------------------------ #
    # 保存
    # ------------------------------------------------------------------ #
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")
    os.makedirs(output_dir, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"alpha_d1_d2_debug_{timestamp}.png")
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")

    # 打印关键数值
    s_d1_peak = 1.0 / (1.0 + math.exp(-BETA * (da_level - EC50_D1)))
    s_d2_peak = 1.0 / (1.0 + math.exp(-BETA * (da_level - EC50_D2)))
    print(f"\n--- D1 ---")
    print(f"  s_D1 target (peak)   = {s_d1_peak:.4f}")
    print(f"  Langmuir α_ss        = {(k_on_d1*s_d1_peak)/(k_on_d1*s_d1_peak+k_off_d1):.4f}  (Kd={Kd_D1:.4f})")
    print(f"  tau method peak      = {arr_d1_tau.max():.4f}")
    print(f"  Langmuir method peak = {arr_d1_lang.max():.4f}")
    print(f"\n--- D2 ---")
    print(f"  s_D2 target (peak)   = {s_d2_peak:.4f}")
    print(f"  Langmuir α_ss        = {(k_on_d2*s_d2_peak)/(k_on_d2*s_d2_peak+k_off_d2):.4f}  (Kd={Kd_D2:.4f})")
    print(f"  tau method peak      = {arr_d2_tau.max():.4f}")
    print(f"  Langmuir method peak = {arr_d2_lang.max():.4f}")
