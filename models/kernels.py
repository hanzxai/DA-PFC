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

    # 目标值 S(t) — Sigmoid (baseline DA=2nM, both batches)
    s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))

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

    # 目标值 S(t) — Sigmoid (baseline DA=2nM, both batches)
    s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))

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

    # 目标值 S(t) — Sigmoid (baseline DA=2nM, both batches)
    s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))

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

    # 目标值 S(t) — Sigmoid (baseline DA=2nM, both batches)
    s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))

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

    # 目标值 S(t) — Sigmoid (baseline DA=2nM, both batches)
    s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))

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

    # 目标值 S(t) — Sigmoid (baseline DA=2nM, both batches)
    s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))

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
    bg_mean, bg_std = 200.0, 25.0  # V_ss=-50mV=V_th, critical-point drive
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
    bg_mean, bg_std = 200.0, 25.0  # V_ss=-50mV=V_th, critical-point drive
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
    EC50_D1 = 4.0
    EC50_D2 = 8.0
    BETA = 1.0
    EPS_D1, EPS_D2 = 0.15, 0.10   # D1: Gain 增强比例; D2: Gain 减弱比例
    BIAS_D1, BIAS_D2 = 12.0, -10.0
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

    # R_base: 基础膜电阻 (GΩ = mV/pA), V in mV, I in pA
    # τ_m = R_base * C_m: E 神经元 τ_m = 0.1 GΩ * 250 pF = 25 ms
    V_rest, V_reset, V_th = -70.0, -75.0, -50.0
    R_base, tau_syn, t_ref = 0.1, 5.0, 5.0  # R_base unit: GΩ (= mV/pA)
    bg_mean, bg_std = 200.0, 25.0  # V_ss=-50mV=V_th, critical-point drive
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

        # A. 当前 DA 浓度 (baseline=2nM for both batches)
        DA_BASELINE = 2.0
        current_da_val = DA_BASELINE
        if current_time >= da_onset:
            current_da_val = float(da_level)
        da_t = torch.tensor([[DA_BASELINE], [current_da_val]], device=W_t.device)

        # B. 目标值 S(t) — Sigmoid (D1 用于调试参考, D2 由动力学函数内部计算)
        s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))

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
    BIAS_D1, BIAS_D2 = 12.0, -10.0
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
    bg_mean, bg_std = 200.0, 25.0  # V_ss=-50mV=V_th, critical-point drive
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

        # A. 当前 DA 浓度 (baseline=2nM for both batches)
        DA_BASELINE = 2.0
        current_da_val = DA_BASELINE
        if current_time >= da_onset:
            current_da_val = float(da_level)
        da_t = torch.tensor([[DA_BASELINE], [current_da_val]], device=W_t.device)

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

    # Pack final state for checkpoint: V(2,N), I_syn(2,N), t_last_spike(2,N), alpha_d1(2,1), alpha_d2(2,1)
    # Concatenate along dim=1: (2, N+N+N+1+1) = (2, 3N+2)
    final_state = torch.cat([V, I_syn, t_last_spike, alpha_d1, alpha_d2], dim=1)
    return spike_records[:spike_count], v_traces, final_state


# ======================================================================
# 内核 5: D1 + D2 受体动力学 — 两阶段给药
#   Phase 1: [0, da_onset)          → 0 nM (baseline, no DA)
#   Phase 2: [da_onset, phase2_onset) → da_level_1 nM (resting-state DA)
#   Phase 3: [phase2_onset, end)    → da_level_2 nM (new DA challenge)
#
#   Control batch (Batch 0) always stays at 0 nM.
#   Experiment batch (Batch 1) follows the 3-phase schedule above.
#
#   This kernel simulates the scenario where the cortex is first
#   equilibrated under a resting-state DA tone (e.g. 2 nM), and then
#   a sudden DA increase (e.g. 15 nM) is applied on top of the
#   already-active receptor state.
# ======================================================================
@torch.jit.script
def run_dynamic_d1_d2_kernel_two_stage(
    W_t: torch.Tensor,
    mask_d1: torch.Tensor,
    mask_d2: torch.Tensor,
    da_level_1: float,
    da_level_2: float,
    da_onset: float,
    phase2_onset: float,
    duration: float,
    dt: float,
    record_indices: torch.Tensor,
    n_exc: int,
):
    """
    D1 + D2 receptor kinetics kernel with two-stage DA dosing.

    DA schedule for Experiment batch (Batch 1):
      [0, da_onset)          → 0 nM
      [da_onset, phase2_onset) → da_level_1 nM  (resting-state DA)
      [phase2_onset, end)    → da_level_2 nM  (DA challenge)

    Control batch (Batch 0) always receives 0 nM.
    """
    # --- Pharmacology constants ---
    EC50_D1 = 4.0
    EC50_D2 = 8.0
    BETA = 1.0
    EPS_D1, EPS_D2 = 0.15, 0.10
    BIAS_D1, BIAS_D2 = 12.0, -10.0
    LAM_D1, LAM_D2 = 0.3, 0.2
    # --- D1 Langmuir kinetics ---
    TAU_ON_D1 = 30876.1
    TAU_OFF_D1 = 164472.5
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    # --- D2 Langmuir kinetics ---
    TAU_ON_D2 = 10000.0
    TAU_OFF_D2 = 50000.0
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

    # --- LIF parameters ---
    V_rest, V_reset, V_th = -70.0, -75.0, -50.0
    R_base, tau_syn, t_ref = 0.1, 5.0, 5.0
    bg_mean, bg_std = 200.0, 25.0
    C_E, C_I = 250.0, 90.0

    # --- Initialization ---
    N = W_t.shape[0]
    batch_size = 2
    steps = int(duration / dt)

    C_m = torch.full((1, N), C_I, device=W_t.device)
    C_m[0, :n_exc] = C_E

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

    # --- Time-step loop ---
    for i in range(steps):
        current_time = i * dt

        # A. Current DA concentration (two-stage schedule, baseline=2nM)
        DA_BASELINE = 2.0
        current_da_val = DA_BASELINE
        if current_time >= phase2_onset:
            current_da_val = float(da_level_2)
        elif current_time >= da_onset:
            current_da_val = float(da_level_1)
        da_t = torch.tensor([[DA_BASELINE], [current_da_val]], device=W_t.device)

        # B. Update alpha_D1 (Langmuir receptor binding kinetics)
        alpha_d1 = compute_alpha_d1_step_langmuir(alpha_d1, da_t, current_time, da_onset, dt, k_on_d1, k_off_d1)

        # C. Update alpha_D2 (Langmuir receptor binding kinetics)
        alpha_d2 = compute_alpha_d2_step_langmuir(alpha_d2, da_t, current_time, da_onset, dt, k_on_d2, k_off_d2)

        # D. Assemble modulation parameters
        mod_R = R_base * (torch.ones((batch_size, N), device=W_t.device)
                          + (EPS_D1 * alpha_d1) * mask_d1
                          - (EPS_D2 * alpha_d2) * mask_d2)
        I_mod = torch.zeros((batch_size, N), device=W_t.device)
        scale_syn = torch.ones((batch_size, N), device=W_t.device)

        I_mod += (BIAS_D1 * alpha_d1) * mask_d1
        scale_syn += (LAM_D1 * alpha_d1) * mask_d1

        I_mod += (BIAS_D2 * alpha_d2) * mask_d2
        scale_syn -= (LAM_D2 * alpha_d2) * mask_d2

        # E. LIF exact integration
        I_syn = I_syn * decay_factor
        I_bg = (torch.randn((1, N), device=W_t.device) * bg_std + bg_mean).expand(batch_size, -1)
        I_total = (I_syn * scale_syn) + I_bg + I_mod

        R_eff = mod_R
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

    # Pack final state for checkpoint: V(2,N), I_syn(2,N), t_last_spike(2,N), alpha_d1(2,1), alpha_d2(2,1)
    final_state = torch.cat([V, I_syn, t_last_spike, alpha_d1, alpha_d2], dim=1)
    return spike_records[:spike_count], v_traces, final_state


# ======================================================================
# 内核 6: D1 + D2 受体动力学 — 从 checkpoint 恢复仿真
#   从给定的初始状态 (V, I_syn, t_last_spike, alpha_d1, alpha_d2) 开始,
#   施加新的 DA 浓度继续仿真。
#   Control batch (Batch 0) 使用 checkpoint 中的 control 状态继续演化 (0 nM)。
#   Experiment batch (Batch 1) 使用 checkpoint 中的 exp 状态, 施加新 DA。
# ======================================================================
@torch.jit.script
def run_dynamic_d1_d2_kernel_from_state(
    W_t: torch.Tensor,
    mask_d1: torch.Tensor,
    mask_d2: torch.Tensor,
    init_state: torch.Tensor,
    da_level: float,
    da_onset: float,
    duration: float,
    dt: float,
    record_indices: torch.Tensor,
    n_exc: int,
):
    """
    D1 + D2 receptor kinetics kernel resuming from a checkpoint state.

    Args:
        init_state: (2, 3N+2) packed tensor from a previous kernel's final_state.
                    Layout: [V(2,N) | I_syn(2,N) | t_last_spike(2,N) | alpha_d1(2,1) | alpha_d2(2,1)]
        da_level:   New DA concentration (nM) to apply from da_onset.
        da_onset:   Time (ms) when new DA starts (relative to this simulation's t=0).
        duration:   Total simulation time (ms) for this continuation run.
    """
    # --- Pharmacology constants ---
    EC50_D1 = 4.0
    EC50_D2 = 8.0
    BETA = 1.0
    EPS_D1, EPS_D2 = 0.15, 0.10
    BIAS_D1, BIAS_D2 = 12.0, -10.0
    LAM_D1, LAM_D2 = 0.3, 0.2
    # --- D1 Langmuir kinetics ---
    TAU_ON_D1 = 30876.1
    TAU_OFF_D1 = 164472.5
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    # --- D2 Langmuir kinetics ---
    TAU_ON_D2 = 10000.0
    TAU_OFF_D2 = 50000.0
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

    # --- LIF parameters ---
    V_rest, V_reset, V_th = -70.0, -75.0, -50.0
    R_base, tau_syn, t_ref = 0.1, 5.0, 5.0
    bg_mean, bg_std = 200.0, 25.0
    C_E, C_I = 250.0, 90.0

    # --- Unpack initial state ---
    N = W_t.shape[0]
    batch_size = 2
    steps = int(duration / dt)

    C_m = torch.full((1, N), C_I, device=W_t.device)
    C_m[0, :n_exc] = C_E

    # Unpack: [V | I_syn | t_last_spike | alpha_d1 | alpha_d2]
    V = init_state[:, :N].clone()
    I_syn = init_state[:, N:2*N].clone()
    t_last_spike = init_state[:, 2*N:3*N].clone()
    alpha_d1 = init_state[:, 3*N:3*N+1].clone()
    alpha_d2 = init_state[:, 3*N+1:3*N+2].clone()

    # Reset t_last_spike relative to new t=0 (shift by a large negative offset
    # so refractory logic works correctly at the start)
    # The old t_last_spike values are absolute times from the previous run.
    # We set them to -1000 to ensure no neuron is in refractory at t=0 of the new run.
    t_last_spike = torch.full((batch_size, N), -1000.0, device=W_t.device)

    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))

    max_spikes = int(steps * N * batch_size * 0.15)
    spike_records = torch.zeros((max_spikes, 3), device=W_t.device, dtype=torch.long)
    spike_count = 0
    num_record = record_indices.shape[0]
    v_traces = torch.zeros((steps, num_record), device=W_t.device)

    # --- Time-step loop ---
    for i in range(steps):
        current_time = i * dt

        # A. Current DA concentration (baseline=2nM for both batches)
        DA_BASELINE = 2.0
        current_da_val = DA_BASELINE
        if current_time >= da_onset:
            current_da_val = float(da_level)
        da_t = torch.tensor([[DA_BASELINE], [current_da_val]], device=W_t.device)

        # B. Update alpha_D1 (Langmuir receptor binding kinetics)
        alpha_d1 = compute_alpha_d1_step_langmuir(alpha_d1, da_t, current_time, da_onset, dt, k_on_d1, k_off_d1)

        # C. Update alpha_D2 (Langmuir receptor binding kinetics)
        alpha_d2 = compute_alpha_d2_step_langmuir(alpha_d2, da_t, current_time, da_onset, dt, k_on_d2, k_off_d2)

        # D. Assemble modulation parameters
        mod_R = R_base * (torch.ones((batch_size, N), device=W_t.device)
                          + (EPS_D1 * alpha_d1) * mask_d1
                          - (EPS_D2 * alpha_d2) * mask_d2)
        I_mod = torch.zeros((batch_size, N), device=W_t.device)
        scale_syn = torch.ones((batch_size, N), device=W_t.device)

        I_mod += (BIAS_D1 * alpha_d1) * mask_d1
        scale_syn += (LAM_D1 * alpha_d1) * mask_d1

        I_mod += (BIAS_D2 * alpha_d2) * mask_d2
        scale_syn -= (LAM_D2 * alpha_d2) * mask_d2

        # E. LIF exact integration
        I_syn = I_syn * decay_factor
        I_bg = (torch.randn((1, N), device=W_t.device) * bg_std + bg_mean).expand(batch_size, -1)
        I_total = (I_syn * scale_syn) + I_bg + I_mod

        R_eff = mod_R
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

    # Pack final state
    final_state = torch.cat([V, I_syn, t_last_spike, alpha_d1, alpha_d2], dim=1)
    return spike_records[:spike_count], v_traces, final_state


# ======================================================================
# 内核 7: D1 + D2 受体动力学 — DA 脉冲实验 (从 checkpoint 恢复)
#   DA schedule for Experiment batch (Batch 1):
#     [0, pulse_onset)        → da_base nM  (maintain resting DA)
#     [pulse_onset, pulse_off) → da_pulse nM (DA pulse)
#     [pulse_off, end)        → da_base nM  (DA withdrawal)
#   Control batch (Batch 0) always maintains da_base nM.
#
#   Key difference from from_state kernel: BOTH batches maintain da_base,
#   only Exp batch receives the pulse. This allows proper comparison.
# ======================================================================
@torch.jit.script
def run_dynamic_d1_d2_kernel_pulse(
    W_t: torch.Tensor,
    mask_d1: torch.Tensor,
    mask_d2: torch.Tensor,
    init_state: torch.Tensor,
    da_base: float,
    da_pulse: float,
    pulse_onset: float,
    pulse_offset: float,
    duration: float,
    dt: float,
    record_indices: torch.Tensor,
    n_exc: int,
    alpha_record_interval: int,
):
    """
    DA pulse experiment kernel resuming from checkpoint.

    Both batches start from the same checkpoint state (DA=da_base steady-state).
    - Batch 0 (Control): maintains da_base throughout
    - Batch 1 (Experiment): da_base → da_pulse → da_base

    Args:
        init_state: (2, 3N+2) packed tensor from checkpoint.
        da_base:    Baseline DA concentration (nM), e.g. 2.0
        da_pulse:   Pulse DA concentration (nM), e.g. 15.0
        pulse_onset:  Time (ms) when pulse starts
        pulse_offset: Time (ms) when pulse ends
        duration:   Total simulation time (ms)
        alpha_record_interval: Record alpha every N steps (to save memory)

    Returns:
        spike_records, v_traces, final_state, alpha_d1_trace, alpha_d2_trace
    """
    # --- Pharmacology constants ---
    EC50_D1 = 4.0
    EC50_D2 = 8.0
    BETA = 1.0
    EPS_D1, EPS_D2 = 0.15, 0.10
    BIAS_D1, BIAS_D2 = 12.0, -10.0
    LAM_D1, LAM_D2 = 0.3, 0.2
    TAU_ON_D1 = 30876.1
    TAU_OFF_D1 = 164472.5
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    TAU_ON_D2 = 10000.0
    TAU_OFF_D2 = 50000.0
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

    # --- LIF parameters ---
    V_rest, V_reset, V_th = -70.0, -75.0, -50.0
    R_base, tau_syn, t_ref = 0.1, 5.0, 5.0
    bg_mean, bg_std = 200.0, 25.0
    C_E, C_I = 250.0, 90.0

    # --- Unpack initial state ---
    N = W_t.shape[0]
    batch_size = 2
    steps = int(duration / dt)

    C_m = torch.full((1, N), C_I, device=W_t.device)
    C_m[0, :n_exc] = C_E

    V = init_state[:, :N].clone()
    I_syn = init_state[:, N:2*N].clone()
    t_last_spike = torch.full((batch_size, N), -1000.0, device=W_t.device)
    alpha_d1 = init_state[:, 3*N:3*N+1].clone()
    alpha_d2 = init_state[:, 3*N+1:3*N+2].clone()

    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))

    max_spikes = int(steps * N * batch_size * 0.15)
    spike_records = torch.zeros((max_spikes, 3), device=W_t.device, dtype=torch.long)
    spike_count = 0
    num_record = record_indices.shape[0]
    v_traces = torch.zeros((steps, num_record), device=W_t.device)

    # Alpha trace recording (subsampled to save memory)
    n_alpha_records = steps // alpha_record_interval + 1
    alpha_d1_trace = torch.zeros((n_alpha_records, batch_size), device=W_t.device)
    alpha_d2_trace = torch.zeros((n_alpha_records, batch_size), device=W_t.device)
    alpha_record_idx = 0

    # --- Time-step loop ---
    for i in range(steps):
        current_time = i * dt

        # A. DA concentration schedule
        # Control (Batch 0): always da_base
        # Experiment (Batch 1): da_base → da_pulse → da_base
        da_ctrl = da_base
        if current_time >= pulse_onset and current_time < pulse_offset:
            da_exp = da_pulse
        else:
            da_exp = da_base
        da_t = torch.tensor([[da_ctrl], [da_exp]], device=W_t.device)

        # B. Compute Sigmoid targets for BOTH batches (no zeroing of control!)
        s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))
        s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))

        # C. Update alpha_D1 (Langmuir) — manual inline to avoid control-batch zeroing
        bind_d1 = k_on_d1 * s_d1 * (1.0 - alpha_d1)
        unbind_d1 = k_off_d1 * alpha_d1
        alpha_d1 = alpha_d1 + (bind_d1 - unbind_d1) * dt
        alpha_d1 = torch.clamp(alpha_d1, 0.0, 1.0)

        # D. Update alpha_D2 (Langmuir) — manual inline
        bind_d2 = k_on_d2 * s_d2 * (1.0 - alpha_d2)
        unbind_d2 = k_off_d2 * alpha_d2
        alpha_d2 = alpha_d2 + (bind_d2 - unbind_d2) * dt
        alpha_d2 = torch.clamp(alpha_d2, 0.0, 1.0)

        # Record alpha traces
        if i % alpha_record_interval == 0 and alpha_record_idx < n_alpha_records:
            alpha_d1_trace[alpha_record_idx, 0] = alpha_d1[0, 0]
            alpha_d1_trace[alpha_record_idx, 1] = alpha_d1[1, 0]
            alpha_d2_trace[alpha_record_idx, 0] = alpha_d2[0, 0]
            alpha_d2_trace[alpha_record_idx, 1] = alpha_d2[1, 0]
            alpha_record_idx += 1

        # E. Assemble modulation parameters
        mod_R = R_base * (torch.ones((batch_size, N), device=W_t.device)
                          + (EPS_D1 * alpha_d1) * mask_d1
                          - (EPS_D2 * alpha_d2) * mask_d2)
        I_mod = torch.zeros((batch_size, N), device=W_t.device)
        scale_syn = torch.ones((batch_size, N), device=W_t.device)
        I_mod += (BIAS_D1 * alpha_d1) * mask_d1
        scale_syn += (LAM_D1 * alpha_d1) * mask_d1
        I_mod += (BIAS_D2 * alpha_d2) * mask_d2
        scale_syn -= (LAM_D2 * alpha_d2) * mask_d2

        # F. LIF exact integration
        I_syn = I_syn * decay_factor
        I_bg = (torch.randn((1, N), device=W_t.device) * bg_std + bg_mean).expand(batch_size, -1)
        I_total = (I_syn * scale_syn) + I_bg + I_mod

        R_eff = mod_R
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

    final_state = torch.cat([V, I_syn, t_last_spike, alpha_d1, alpha_d2], dim=1)
    return spike_records[:spike_count], v_traces, final_state, alpha_d1_trace[:alpha_record_idx], alpha_d2_trace[:alpha_record_idx]


# ======================================================================
# 内核 8: D1 + D2 受体动力学 — 正弦波 DA 输入 (从 checkpoint 恢复)
#   DA(t) = da_base + amplitude * sin(2π * freq * t)
#   用于测试网络对不同频率 DA 波动的响应 (低通滤波特性)
#   Control batch maintains constant da_base.
# ======================================================================
@torch.jit.script
def run_dynamic_d1_d2_kernel_sine(
    W_t: torch.Tensor,
    mask_d1: torch.Tensor,
    mask_d2: torch.Tensor,
    init_state: torch.Tensor,
    da_base: float,
    da_amplitude: float,
    da_freq_hz: float,
    duration: float,
    dt: float,
    record_indices: torch.Tensor,
    n_exc: int,
    alpha_record_interval: int,
):
    """
    Sinusoidal DA input experiment kernel.

    - Batch 0 (Control): constant da_base
    - Batch 1 (Experiment): da_base + amplitude * sin(2π * freq * t)

    Args:
        da_base:      Baseline DA (nM)
        da_amplitude: Sine wave amplitude (nM)
        da_freq_hz:   Frequency in Hz
        alpha_record_interval: Record alpha every N steps
    """
    # --- Pharmacology constants ---
    EC50_D1 = 4.0
    EC50_D2 = 8.0
    BETA = 1.0
    EPS_D1, EPS_D2 = 0.15, 0.10
    BIAS_D1, BIAS_D2 = 12.0, -10.0
    LAM_D1, LAM_D2 = 0.3, 0.2
    TAU_ON_D1 = 30876.1
    TAU_OFF_D1 = 164472.5
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    TAU_ON_D2 = 10000.0
    TAU_OFF_D2 = 50000.0
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

    # --- LIF parameters ---
    V_rest, V_reset, V_th = -70.0, -75.0, -50.0
    R_base, tau_syn, t_ref = 0.1, 5.0, 5.0
    bg_mean, bg_std = 200.0, 25.0
    C_E, C_I = 250.0, 90.0

    PI = 3.141592653589793

    # --- Unpack initial state ---
    N = W_t.shape[0]
    batch_size = 2
    steps = int(duration / dt)

    C_m = torch.full((1, N), C_I, device=W_t.device)
    C_m[0, :n_exc] = C_E

    V = init_state[:, :N].clone()
    I_syn = init_state[:, N:2*N].clone()
    t_last_spike = torch.full((batch_size, N), -1000.0, device=W_t.device)
    alpha_d1 = init_state[:, 3*N:3*N+1].clone()
    alpha_d2 = init_state[:, 3*N+1:3*N+2].clone()

    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))

    max_spikes = int(steps * N * batch_size * 0.15)
    spike_records = torch.zeros((max_spikes, 3), device=W_t.device, dtype=torch.long)
    spike_count = 0
    num_record = record_indices.shape[0]
    v_traces = torch.zeros((steps, num_record), device=W_t.device)

    n_alpha_records = steps // alpha_record_interval + 1
    alpha_d1_trace = torch.zeros((n_alpha_records, batch_size), device=W_t.device)
    alpha_d2_trace = torch.zeros((n_alpha_records, batch_size), device=W_t.device)
    da_trace = torch.zeros((n_alpha_records,), device=W_t.device)
    alpha_record_idx = 0

    # Convert freq from Hz to ms⁻¹: freq_ms = freq_hz / 1000
    freq_ms = da_freq_hz / 1000.0

    # --- Time-step loop ---
    for i in range(steps):
        current_time = i * dt

        # A. DA concentration: sinusoidal for Exp, constant for Ctrl
        da_sine = da_base + da_amplitude * torch.sin(
            torch.tensor(2.0 * PI * freq_ms * current_time, device=W_t.device))
        da_sine = torch.clamp(da_sine, min=torch.tensor(0.1, device=W_t.device))  # DA >= 0.1 nM
        da_ctrl_val: float = float(da_base)
        da_exp_val: float = float(da_sine.item())
        da_t = torch.tensor([[da_ctrl_val], [da_exp_val]], device=W_t.device)

        # B. Sigmoid targets
        s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))
        s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))

        # C. Update alpha_D1 (Langmuir)
        bind_d1 = k_on_d1 * s_d1 * (1.0 - alpha_d1)
        unbind_d1 = k_off_d1 * alpha_d1
        alpha_d1 = alpha_d1 + (bind_d1 - unbind_d1) * dt
        alpha_d1 = torch.clamp(alpha_d1, 0.0, 1.0)

        # D. Update alpha_D2 (Langmuir)
        bind_d2 = k_on_d2 * s_d2 * (1.0 - alpha_d2)
        unbind_d2 = k_off_d2 * alpha_d2
        alpha_d2 = alpha_d2 + (bind_d2 - unbind_d2) * dt
        alpha_d2 = torch.clamp(alpha_d2, 0.0, 1.0)

        # Record traces
        if i % alpha_record_interval == 0 and alpha_record_idx < n_alpha_records:
            alpha_d1_trace[alpha_record_idx, 0] = alpha_d1[0, 0]
            alpha_d1_trace[alpha_record_idx, 1] = alpha_d1[1, 0]
            alpha_d2_trace[alpha_record_idx, 0] = alpha_d2[0, 0]
            alpha_d2_trace[alpha_record_idx, 1] = alpha_d2[1, 0]
            da_trace[alpha_record_idx] = da_sine.item()
            alpha_record_idx += 1

        # E. Assemble modulation parameters
        mod_R = R_base * (torch.ones((batch_size, N), device=W_t.device)
                          + (EPS_D1 * alpha_d1) * mask_d1
                          - (EPS_D2 * alpha_d2) * mask_d2)
        I_mod = torch.zeros((batch_size, N), device=W_t.device)
        scale_syn = torch.ones((batch_size, N), device=W_t.device)
        I_mod += (BIAS_D1 * alpha_d1) * mask_d1
        scale_syn += (LAM_D1 * alpha_d1) * mask_d1
        I_mod += (BIAS_D2 * alpha_d2) * mask_d2
        scale_syn -= (LAM_D2 * alpha_d2) * mask_d2

        # F. LIF exact integration
        I_syn = I_syn * decay_factor
        I_bg = (torch.randn((1, N), device=W_t.device) * bg_std + bg_mean).expand(batch_size, -1)
        I_total = (I_syn * scale_syn) + I_bg + I_mod

        R_eff = mod_R
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

    final_state = torch.cat([V, I_syn, t_last_spike, alpha_d1, alpha_d2], dim=1)
    return spike_records[:spike_count], v_traces, final_state, alpha_d1_trace[:alpha_record_idx], alpha_d2_trace[:alpha_record_idx], da_trace[:alpha_record_idx]


# ======================================================================
# 内核 9: D1 + D2 受体动力学 — DA 脉冲 + 外部刺激注入 (从 checkpoint 恢复)
#
#   科学目的: 测试 D1 余晖窗口对工作记忆持续性活动的门控作用
#
#   DA schedule (same as kernel 7):
#     Batch 0 (Control): maintains da_base throughout
#     Batch 1 (Experiment): da_base → da_pulse → da_base
#
#   External stimulus:
#     在 [stim_onset, stim_offset) 时间窗内, 给 stim_mask 标记的神经元
#     注入额外电流 stim_amplitude (pA)。
#     Batch 0 和 Batch 1 都接收相同的刺激 (差异仅来自 DA 调制)。
#
#   Key feature: stim_mask 是一个 (N,) float tensor (0.0 or 1.0),
#     允许灵活选择被刺激的神经元子集。
# ======================================================================
@torch.jit.script
def run_dynamic_d1_d2_kernel_pulse_stim(
    W_t: torch.Tensor,
    mask_d1: torch.Tensor,
    mask_d2: torch.Tensor,
    init_state: torch.Tensor,
    da_base: float,
    da_pulse: float,
    pulse_onset: float,
    pulse_offset: float,
    stim_mask: torch.Tensor,
    stim_onset: float,
    stim_offset: float,
    stim_amplitude: float,
    duration: float,
    dt: float,
    record_indices: torch.Tensor,
    n_exc: int,
    alpha_record_interval: int,
):
    """
    DA pulse + external stimulus injection kernel resuming from checkpoint.

    Both batches start from the same checkpoint state (DA=da_base steady-state).
    - Batch 0 (Control): maintains da_base throughout
    - Batch 1 (Experiment): da_base → da_pulse → da_base

    Both batches receive the same external stimulus in [stim_onset, stim_offset).

    Args:
        init_state: (2, 3N+2) packed tensor from checkpoint.
        da_base:    Baseline DA concentration (nM), e.g. 2.0
        da_pulse:   Pulse DA concentration (nM), e.g. 15.0
        pulse_onset:  Time (ms) when DA pulse starts
        pulse_offset: Time (ms) when DA pulse ends
        stim_mask:    (N,) float tensor, 1.0 for stimulated neurons, 0.0 otherwise
        stim_onset:   Time (ms) when external stimulus starts
        stim_offset:  Time (ms) when external stimulus ends
        stim_amplitude: Stimulus current amplitude (pA)
        duration:   Total simulation time (ms)
        alpha_record_interval: Record alpha every N steps (to save memory)

    Returns:
        spike_records, v_traces, final_state, alpha_d1_trace, alpha_d2_trace
    """
    # --- Pharmacology constants ---
    EC50_D1 = 4.0
    EC50_D2 = 8.0
    BETA = 1.0
    EPS_D1, EPS_D2 = 0.15, 0.10
    BIAS_D1, BIAS_D2 = 12.0, -10.0
    LAM_D1, LAM_D2 = 0.3, 0.2
    TAU_ON_D1 = 30876.1
    TAU_OFF_D1 = 164472.5
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    TAU_ON_D2 = 10000.0
    TAU_OFF_D2 = 50000.0
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

    # --- LIF parameters ---
    V_rest, V_reset, V_th = -70.0, -75.0, -50.0
    R_base, tau_syn, t_ref = 0.1, 5.0, 5.0
    bg_mean, bg_std = 200.0, 25.0
    C_E, C_I = 250.0, 90.0

    # --- Unpack initial state ---
    N = W_t.shape[0]
    batch_size = 2
    steps = int(duration / dt)

    C_m = torch.full((1, N), C_I, device=W_t.device)
    C_m[0, :n_exc] = C_E

    V = init_state[:, :N].clone()
    I_syn = init_state[:, N:2*N].clone()
    t_last_spike = torch.full((batch_size, N), -1000.0, device=W_t.device)
    alpha_d1 = init_state[:, 3*N:3*N+1].clone()
    alpha_d2 = init_state[:, 3*N+1:3*N+2].clone()

    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))

    max_spikes = int(steps * N * batch_size * 0.15)
    spike_records = torch.zeros((max_spikes, 3), device=W_t.device, dtype=torch.long)
    spike_count = 0
    num_record = record_indices.shape[0]
    v_traces = torch.zeros((steps, num_record), device=W_t.device)

    # Alpha trace recording (subsampled to save memory)
    n_alpha_records = steps // alpha_record_interval + 1
    alpha_d1_trace = torch.zeros((n_alpha_records, batch_size), device=W_t.device)
    alpha_d2_trace = torch.zeros((n_alpha_records, batch_size), device=W_t.device)
    alpha_record_idx = 0

    # Expand stim_mask to (1, N) for broadcasting with (batch_size, N)
    stim_mask_2d = stim_mask.unsqueeze(0)  # (1, N)

    # --- Time-step loop ---
    for i in range(steps):
        current_time = i * dt

        # A. DA concentration schedule
        da_ctrl = da_base
        if current_time >= pulse_onset and current_time < pulse_offset:
            da_exp = da_pulse
        else:
            da_exp = da_base
        da_t = torch.tensor([[da_ctrl], [da_exp]], device=W_t.device)

        # B. Compute Sigmoid targets for BOTH batches
        s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))
        s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))

        # C. Update alpha_D1 (Langmuir)
        bind_d1 = k_on_d1 * s_d1 * (1.0 - alpha_d1)
        unbind_d1 = k_off_d1 * alpha_d1
        alpha_d1 = alpha_d1 + (bind_d1 - unbind_d1) * dt
        alpha_d1 = torch.clamp(alpha_d1, 0.0, 1.0)

        # D. Update alpha_D2 (Langmuir)
        bind_d2 = k_on_d2 * s_d2 * (1.0 - alpha_d2)
        unbind_d2 = k_off_d2 * alpha_d2
        alpha_d2 = alpha_d2 + (bind_d2 - unbind_d2) * dt
        alpha_d2 = torch.clamp(alpha_d2, 0.0, 1.0)

        # Record alpha traces
        if i % alpha_record_interval == 0 and alpha_record_idx < n_alpha_records:
            alpha_d1_trace[alpha_record_idx, 0] = alpha_d1[0, 0]
            alpha_d1_trace[alpha_record_idx, 1] = alpha_d1[1, 0]
            alpha_d2_trace[alpha_record_idx, 0] = alpha_d2[0, 0]
            alpha_d2_trace[alpha_record_idx, 1] = alpha_d2[1, 0]
            alpha_record_idx += 1

        # E. Assemble modulation parameters
        mod_R = R_base * (torch.ones((batch_size, N), device=W_t.device)
                          + (EPS_D1 * alpha_d1) * mask_d1
                          - (EPS_D2 * alpha_d2) * mask_d2)
        I_mod = torch.zeros((batch_size, N), device=W_t.device)
        scale_syn = torch.ones((batch_size, N), device=W_t.device)
        I_mod += (BIAS_D1 * alpha_d1) * mask_d1
        scale_syn += (LAM_D1 * alpha_d1) * mask_d1
        I_mod += (BIAS_D2 * alpha_d2) * mask_d2
        scale_syn -= (LAM_D2 * alpha_d2) * mask_d2

        # F. External stimulus injection (same for both batches)
        if current_time >= stim_onset and current_time < stim_offset:
            I_mod += stim_amplitude * stim_mask_2d  # broadcast (1,N) -> (batch_size,N)

        # G. LIF exact integration
        I_syn = I_syn * decay_factor
        I_bg = (torch.randn((1, N), device=W_t.device) * bg_std + bg_mean).expand(batch_size, -1)
        I_total = (I_syn * scale_syn) + I_bg + I_mod

        R_eff = mod_R
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

    final_state = torch.cat([V, I_syn, t_last_spike, alpha_d1, alpha_d2], dim=1)
    return spike_records[:spike_count], v_traces, final_state, alpha_d1_trace[:alpha_record_idx], alpha_d2_trace[:alpha_record_idx]


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
        "BIAS_D1":    12.0,
        "BIAS_D2":   -10.0,
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
"BG_MEAN":    200.0,
"BG_STD":     25.0,
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
