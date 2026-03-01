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
        diff = s_d1 - alpha_d1
        tau_dynamic = torch.where(
            diff > 0,
            torch.tensor(TAU_ON_D1, device=W_t.device),
            torch.tensor(TAU_OFF_D1, device=W_t.device),
        )
        alpha_d1 = alpha_d1 + (diff / tau_dynamic) * dt

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
