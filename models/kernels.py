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
    V_rest, V_reset, V_th = 0.0, -5.0, 20.0
    R_base, tau_mem, t_ref, tau_syn = 1.0, 20.0, 5.0, 5.0
    bg_mean, bg_std = 25.0, 5.0

    # --- 状态变量 ---
    V = torch.rand((batch_size, N), device=W_t.device) * (V_th - V_reset) + V_reset
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
    alpha = dt / tau_mem

    for i in range(steps):
        current_time = i * dt

        # 1. 突触电流衰减
        I_syn = I_syn * decay_factor

        # 2. 总电流
        I_bg = torch.randn((batch_size, N), device=W_t.device) * bg_std + bg_mean
        I_total = (I_syn * scale_syn) + I_bg + I_mod

        # 3. LIF 积分
        V_new = V + alpha * (-(V - V_rest) + (R_base * mod_R) * I_total)
        is_refractory = (current_time - t_last_spike) <= t_ref
        V = torch.where(is_refractory, V, V_new)

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
):
    """分步给药内核: 在 da_onset 时刻瞬时切换调节参数。"""
    mod_R_rest, I_mod_rest, scale_rest = params_rest
    mod_R_act, I_mod_act, scale_act = params_active

    batch_size, N = mod_R_rest.shape
    steps = int(duration / dt)

    # --- LIF 参数 ---
    V_rest, V_reset, V_th = 0.0, -5.0, 20.0
    R_base, tau_mem, t_ref, tau_syn = 1.0, 20.0, 5.0, 5.0
    bg_mean, bg_std = 25.0, 5.0

    # --- 状态变量 ---
    V = torch.rand((batch_size, N), device=W_t.device) * (V_th - V_reset) + V_reset
    t_last_spike = torch.full((batch_size, N), -1000.0, device=W_t.device)
    I_syn = torch.zeros((batch_size, N), device=W_t.device)

    # --- 记录容器 ---
    max_spikes = int(steps * N * batch_size * 0.15)
    spike_records = torch.zeros((max_spikes, 3), device=W_t.device, dtype=torch.long)
    spike_count = 0
    num_record = record_indices.shape[0]
    v_traces = torch.zeros((steps, num_record), device=W_t.device)

    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))
    alpha = dt / tau_mem

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
        I_bg = torch.randn((batch_size, N), device=W_t.device) * bg_std + bg_mean
        I_total = (I_syn * cur_scale) + I_bg + cur_I_mod

        V_new = V + alpha * (-(V - V_rest) + (R_base * cur_mod_R) * I_total)
        is_refractory = (current_time - t_last_spike) <= t_ref
        V = torch.where(is_refractory, V, V_new)

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
):
    """
    D1 受体动力学仿真内核。
    - D1: 一阶动力学 (Tau_on / Tau_off)
    - D2: 瞬时响应
    Batch 0 = Control (0 nM), Batch 1 = Experiment (da_level nM)
    """
    # --- 药理学常数 (与 config.py 一致) ---
    TAU_ON_D1 = 30876.1
    TAU_OFF_D1 = 164472.5
    EC50_D1 = 4.0
    EC50_D2 = 8.0
    BETA = 1.0
    EPS_D1, EPS_D2 = 0.3, 0.2
    BIAS_D1, BIAS_D2 = 3.0, -3.0
    LAM_D1, LAM_D2 = 0.3, 0.2

    # --- LIF 参数 ---
    V_rest, V_reset, V_th = 0.0, -5.0, 20.0
    R_base, tau_mem, tau_syn, t_ref = 1.0, 20.0, 5.0, 5.0
    bg_mean, bg_std = 25.0, 5.0

    # --- 初始化 ---
    N = W_t.shape[0]
    batch_size = 2
    steps = int(duration / dt)

    V = torch.rand((batch_size, N), device=W_t.device) * (V_th - V_reset) + V_reset
    t_last_spike = torch.full((batch_size, N), -1000.0, device=W_t.device)
    I_syn = torch.zeros((batch_size, N), device=W_t.device)
    alpha_d1 = torch.zeros((batch_size, 1), device=W_t.device)

    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))
    lif_alpha = dt / tau_mem

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

        # B. 目标值 S(t) — Sigmoid
        s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))
        if current_time < da_onset:
            s_d1[:] = 0.0
        s_d1[0] = 0.0  # Control 始终为 0

        s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))
        if current_time < da_onset:
            s_d2[:] = 0.0
        s_d2[0] = 0.0

        # C. 更新 alpha_D1 (一阶动力学)
        alpha_d1 = compute_alpha_d1_step(alpha_d1, da_t, current_time, da_onset, dt)

        # D. D2 瞬时
        alpha_d2 = s_d2

        # E. 组装调节参数
        mod_R = torch.ones((batch_size, N), device=W_t.device)
        I_mod = torch.zeros((batch_size, N), device=W_t.device)
        scale_syn = torch.ones((batch_size, N), device=W_t.device)

        mod_R += (EPS_D1 * alpha_d1) * mask_d1
        I_mod += (BIAS_D1 * alpha_d1) * mask_d1
        scale_syn += (LAM_D1 * alpha_d1) * mask_d1

        mod_R -= (EPS_D2 * alpha_d2) * mask_d2
        I_mod += (BIAS_D2 * alpha_d2) * mask_d2
        scale_syn -= (LAM_D2 * alpha_d2) * mask_d2

        # F. LIF 积分
        I_syn = I_syn * decay_factor
        I_bg = torch.randn((batch_size, N), device=W_t.device) * bg_std + bg_mean
        I_total = (I_syn * scale_syn) + I_bg + I_mod

        V_new = V + lif_alpha * (-(V - V_rest) + (R_base * mod_R) * I_total)
        is_refractory = (current_time - t_last_spike) <= t_ref
        V = torch.where(is_refractory, V, V_new)

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import os
    from datetime import datetime

    print("Debugging alpha_d1 dynamics (3 methods)...")

    # --- 参数设置 ---
    dt        = 10.0        # ms
    duration  = 1000000.0  # ms = 1000s
    da_onset  = 100000.0   # ms = 100s
    da_offset = 300000.0   # ms = 300s
    da_level  = 20.0       # nM

    EC50_D1 = 4.0
    BETA    = 1.0

    steps = int(duration / dt)
    time_points = np.arange(steps) * dt  # 单位: ms

    # --- 速率常数 ---
    TAU_ON  = 30876.1
    TAU_OFF = 164472.5
    k_on  = 1.0 / (TAU_ON -3000)   # ms⁻¹ ≈ 3.24e-5
    k_off = 1.0 / (TAU_OFF +3000)  # ms⁻¹ ≈ 6.08e-6
    Kd    = k_off / k_on    # 解离常数 = TAU_ON / TAU_OFF ≈ 0.1877

    # --- 初始化 ---
    alpha_d1_tau  = torch.zeros((2, 1))
    alpha_d1_k    = torch.zeros((2, 1))
    alpha_d1_lang = torch.zeros((2, 1))

    alpha_trace_tau  = []
    alpha_trace_k    = []
    alpha_trace_lang = []
    target_trace     = []   # s_D1 随时间的变化 (target)
    ss_lang_trace    = []   # Langmuir 稳态值 α_ss = s_D1 / (s_D1 + Kd)

    for i in range(steps):
        current_time = i * dt

        # 给药窗口: [da_onset, da_offset)
        current_da = da_level if (da_onset <= current_time < da_offset) else 0.0
        da_t = torch.tensor([[0.0], [current_da]])

        # 计算当前 s_D1 (target)
        s_d1_val = 1.0 / (1.0 + math.exp(-BETA * (current_da - EC50_D1))) \
                   if current_time >= da_onset else 0.0
        target_trace.append(s_d1_val)

        # Langmuir 稳态值
        ss_val = (k_on * s_d1_val) / (k_on * s_d1_val + k_off) if s_d1_val > 0 else 0.0
        ss_lang_trace.append(ss_val)

        # 方法一: tau 版本
        alpha_d1_tau = compute_alpha_d1_step(alpha_d1_tau, da_t, current_time, da_onset, dt)
        alpha_trace_tau.append(alpha_d1_tau[1, 0].item())

        # 方法二: k_on / k_off 版本
        alpha_d1_k = compute_alpha_d1_step_kon_koff(
            alpha_d1_k, da_t, current_time, da_onset, dt, k_on, k_off
        )
        alpha_trace_k.append(alpha_d1_k[1, 0].item())

        # 方法三: Langmuir 受体结合动力学
        alpha_d1_lang = compute_alpha_d1_step_langmuir(
            alpha_d1_lang, da_t, current_time, da_onset, dt, k_on, k_off
        )
        alpha_trace_lang.append(alpha_d1_lang[1, 0].item())

    alpha_arr      = np.array(alpha_trace_tau)
    alpha_arr_k    = np.array(alpha_trace_k)
    alpha_arr_lang = np.array(alpha_trace_lang)
    target_arr     = np.array(target_trace)
    ss_lang_arr    = np.array(ss_lang_trace)

    # --- 计算 t½ (tau 方法) ---
    t_half_on  = TAU_ON  * math.log(2)
    t_half_off = TAU_OFF * math.log(2)
    t_half_on_abs  = da_onset  + t_half_on
    t_half_off_abs = da_offset + t_half_off

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.ticklabel_format(style='plain', axis='x')

    # --- Target: s_D1 (Sigmoid 目标值) ---
    ax.plot(time_points, target_arr, linestyle='-', linewidth=1.5,
            color='black', alpha=0.5, zorder=1,
            label=f"Target s_D1 (Sigmoid, EC50={EC50_D1} nM)")

    # --- Langmuir 稳态线 α_ss = s_D1/(s_D1+Kd) ---
    ax.plot(time_points, ss_lang_arr, linestyle=':', linewidth=1.5,
            color='red', alpha=0.7, zorder=1,
            label=f"Langmuir steady-state α_ss = s_D1/(s_D1+Kd),  Kd={Kd:.4f}")

    # --- 曲线 2: k_on/k_off 方法（粗实线，橙色，底层） ---
    ax.plot(time_points, alpha_arr_k, linestyle='-', linewidth=4.0,
            color='darkorange', alpha=0.55, zorder=2,
            label=f"Method 2 (k_on/k_off): k_on={k_on:.2e}, k_off={k_off:.2e} ms⁻¹")

    # --- 曲线 1: tau 方法（细虚线，蓝色，上层） ---
    ax.plot(time_points, alpha_arr, linestyle='--', linewidth=1.8,
            color='steelblue', zorder=3,
            label=f"Method 1 (tau): τ_on={TAU_ON:,.0f} ms, τ_off={TAU_OFF:,.0f} ms")

    # --- 曲线 3: Langmuir 方法（细实线，绿色，上层） ---
    ax.plot(time_points, alpha_arr_lang, linestyle='-', linewidth=1.8,
            color='seagreen', zorder=4,
            label=f"Method 3 (Langmuir): dα/dt = k_on·s_D1·(1-α) - k_off·α")

    # 给药 / 撤药竖线
    ax.axvline(x=da_onset,  color='green', linestyle=':', linewidth=1.5,
               label=f"DA onset  ({int(da_onset):,} ms)")
    ax.axvline(x=da_offset, color='gray',  linestyle=':', linewidth=1.5,
               label=f"DA offset ({int(da_offset):,} ms)")

    # --- 峰值标注 ---
    # tau 方法峰值
    peak_val = alpha_arr.max()
    peak_t   = time_points[alpha_arr.argmax()]
    ax.scatter([peak_t], [peak_val], color='steelblue', zorder=6)
    ax.annotate(f"Peak(tau)={peak_val:.4f}",
                xy=(peak_t, peak_val),
                xytext=(peak_t + 15000, peak_val - 0.06),
                arrowprops=dict(arrowstyle='->', color='steelblue'),
                fontsize=9, color='steelblue')

    # Langmuir 方法峰值
    peak_val_lang = alpha_arr_lang.max()
    peak_t_lang   = time_points[alpha_arr_lang.argmax()]
    ax.scatter([peak_t_lang], [peak_val_lang], color='seagreen', zorder=6)
    ax.annotate(f"Peak(Langmuir)={peak_val_lang:.4f}",
                xy=(peak_t_lang, peak_val_lang),
                xytext=(peak_t_lang + 15000, peak_val_lang + 0.04),
                arrowprops=dict(arrowstyle='->', color='seagreen'),
                fontsize=9, color='seagreen')

    # target 峰值 (s_D1 稳态)
    s_d1_peak = 1.0 / (1.0 + math.exp(-BETA * (da_level - EC50_D1)))
    ax.axhline(y=s_d1_peak, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(da_offset + 5000, s_d1_peak + 0.01,
            f"s_D1 target = {s_d1_peak:.4f}", fontsize=9, color='black', alpha=0.7)

    # Langmuir 稳态峰值
    ss_peak = (k_on * s_d1_peak) / (k_on * s_d1_peak + k_off)
    ax.axhline(y=ss_peak, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(da_offset + 5000, ss_peak - 0.03,
            f"Langmuir α_ss = {ss_peak:.4f}", fontsize=9, color='red', alpha=0.8)

    # --- t½_on / t½_off 标注 (tau 方法) ---
    if t_half_on_abs < time_points[-1]:
        idx_on  = min(int((t_half_on_abs) / dt), len(alpha_arr) - 1)
        val_on  = alpha_arr[idx_on]
        ax.axvline(x=t_half_on_abs, color='purple', linestyle='-.', linewidth=1.2)
        ax.scatter([t_half_on_abs], [val_on], color='purple', zorder=6, marker='^')
        ax.annotate(f"t½ on = {t_half_on:,.0f} ms",
                    xy=(t_half_on_abs, val_on),
                    xytext=(t_half_on_abs + 8000, val_on - 0.08),
                    arrowprops=dict(arrowstyle='->', color='purple'),
                    fontsize=9, color='purple')

    if t_half_off_abs < time_points[-1]:
        idx_off = min(int((t_half_off_abs) / dt), len(alpha_arr) - 1)
        val_off = alpha_arr[idx_off]
        ax.axvline(x=t_half_off_abs, color='brown', linestyle='-.', linewidth=1.2)
        ax.scatter([t_half_off_abs], [val_off], color='brown', zorder=6, marker='v')
        ax.annotate(f"t½ off = {t_half_off:,.0f} ms",
                    xy=(t_half_off_abs, val_off),
                    xytext=(t_half_off_abs + 8000, val_off + 0.05),
                    arrowprops=dict(arrowstyle='->', color='brown'),
                    fontsize=9, color='brown')

    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("alpha_d1 activation", fontsize=12)
    ax.set_title(
        f"Alpha D1 Dynamics — 3 Methods Comparison  |  DA={da_level} nM  |  Kd={Kd:.4f}",
        fontsize=12
    )
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    # --- 保存到 tmp/ ---
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")
    os.makedirs(output_dir, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"alpha_d1_debug_{timestamp}.png")
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")
    print(f"  s_D1 target (peak)     = {s_d1_peak:.4f}")
    print(f"  Langmuir alpha_ss      = {ss_peak:.4f}  (Kd={Kd:.4f})")
    print(f"  tau method peak        = {peak_val:.4f}")
    print(f"  Langmuir method peak   = {peak_val_lang:.4f}")
