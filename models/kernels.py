# models/kernels.py
"""
JIT 编译的仿真内核

所有参数通过 params: torch.Tensor (1-D, 25 elements) 从 config.build_kernel_params() 传入,
消除了硬编码重复, config.py 是唯一的参数定义位置 (Single Source of Truth).

Params tensor layout (see config.py for details):
  [0]  V_REST       [1]  V_RESET      [2]  V_TH
  [3]  R_BASE       [4]  TAU_SYN      [5]  T_REF
  [6]  BG_MEAN      [7]  BG_STD       [8]  C_E
  [9]  C_I          [10] EC50_D1      [11] EC50_D2
  [12] BETA         [13] EPS_D1       [14] EPS_D2
  [15] BIAS_D1      [16] BIAS_D2      [17] LAM_D1
  [18] LAM_D2       [19] TAU_ON_D1    [20] TAU_OFF_D1
  [21] TAU_ON_D2    [22] TAU_OFF_D2   [23] DA_BASELINE
  [24] SPIKE_RATE_ESTIMATE

Deprecated functions (tau/kon_koff alpha steps, verify_kernel_params_consistent)
have been moved to models/_deprecated_kernels.py.
"""
import torch


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
    ec50: float,
    beta: float,
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
        ec50         : D1 半效浓度 (nM), from params[10]
        beta         : Sigmoid 斜率, from params[12]

    Returns:
        alpha_d1_new : (batch, 1) 更新后的 D1 受体激活状态
    """
    # 目标值 S(t) — Sigmoid
    s_d1 = 1.0 / (1.0 + torch.exp(-beta * (da_t - ec50)))

    # Langmuir 动力学: 结合项 - 解离项
    bind_term   = k_on * s_d1 * (1.0 - alpha_d1)   # 结合: 正比于游离受体
    unbind_term = k_off * alpha_d1                   # 解离: 正比于已结合受体

    d_alpha = bind_term - unbind_term
    alpha_d1_new = alpha_d1 + d_alpha * dt
    # 限制在 [0, 1]
    alpha_d1_new = torch.clamp(alpha_d1_new, 0.0, 1.0)
    return alpha_d1_new


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
    ec50: float,
    beta: float,
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
        ec50         : D2 半效浓度 (nM), from params[11]
        beta         : Sigmoid 斜率, from params[12]

    Returns:
        alpha_d2_new : (batch, 1) 更新后的 D2 受体激活状态
    """
    # 目标值 S(t) — Sigmoid
    s_d2 = 1.0 / (1.0 + torch.exp(-beta * (da_t - ec50)))

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
    params: torch.Tensor,
):
    """
    静态并行仿真内核。
    W_t: (N, N), mod_R/I_mod/scale_syn: (Batch, N)
    record_indices: (K, 2) -> [batch_idx, neuron_idx]
    params: (25,) parameter tensor from config.build_kernel_params()
    Returns: (spike_records, v_traces)
    """
    batch_size, N = mod_R.shape
    steps = int(duration / dt)

    # --- Unpack params ---
    V_rest   = float(params[0])
    V_reset  = float(params[1])
    V_th     = float(params[2])
    R_base   = float(params[3])
    tau_syn  = float(params[4])
    t_ref    = float(params[5])
    bg_mean  = float(params[6])
    bg_std   = float(params[7])
    C_E      = float(params[8])
    C_I      = float(params[9])

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
    params: torch.Tensor,
):
    """分步给药内核: 在 da_onset 时刻瞬时切换调节参数。"""
    mod_R_rest, I_mod_rest, scale_rest = params_rest
    mod_R_act, I_mod_act, scale_act = params_active

    batch_size, N = mod_R_rest.shape
    steps = int(duration / dt)

    # --- Unpack params ---
    V_rest   = float(params[0])
    V_reset  = float(params[1])
    V_th     = float(params[2])
    R_base   = float(params[3])
    tau_syn  = float(params[4])
    t_ref    = float(params[5])
    bg_mean  = float(params[6])
    bg_std   = float(params[7])
    C_E      = float(params[8])
    C_I      = float(params[9])

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
    params: torch.Tensor,
):
    """
    D1 受体动力学仿真内核。
    - D1: 一阶动力学 (Langmuir)
    - D2: 一阶动力学 (Langmuir)
    Batch 0 = Control (baseline), Batch 1 = Experiment (da_level nM)
    """
    # --- Unpack params ---
    V_rest   = float(params[0])
    V_reset  = float(params[1])
    V_th     = float(params[2])
    R_base   = float(params[3])
    tau_syn  = float(params[4])
    t_ref    = float(params[5])
    bg_mean  = float(params[6])
    bg_std   = float(params[7])
    C_E      = float(params[8])
    C_I      = float(params[9])
    EC50_D1  = float(params[10])
    EC50_D2  = float(params[11])
    BETA     = float(params[12])
    EPS_D1   = float(params[13])
    EPS_D2   = float(params[14])
    BIAS_D1  = float(params[15])
    BIAS_D2  = float(params[16])
    LAM_D1   = float(params[17])
    LAM_D2   = float(params[18])
    TAU_ON_D1  = float(params[19])
    TAU_OFF_D1 = float(params[20])
    TAU_ON_D2  = float(params[21])
    TAU_OFF_D2 = float(params[22])
    DA_BASELINE = float(params[23])
    # --- Derived Langmuir kinetics ---
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

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

        # A. 当前 DA 浓度 (baseline DA for both batches)
        current_da_val = DA_BASELINE
        if current_time >= da_onset:
            current_da_val = float(da_level)
        da_t = torch.tensor([[DA_BASELINE], [current_da_val]], device=W_t.device)

        # B. 目标值 S(t) — Sigmoid (D1 用于调试参考, D2 由动力学函数内部计算)
        s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))

        # C. 更新 alpha_D1 (Langmuir 受体结合动力学)

        
        
        
        alpha_d1 = compute_alpha_d1_step_langmuir(alpha_d1, da_t, current_time, da_onset, dt, k_on_d1, k_off_d1, EC50_D1, BETA)

        # D. 更新 alpha_D2 (Langmuir 受体结合动力学)

        
        
        
        alpha_d2 = compute_alpha_d2_step_langmuir(alpha_d2, da_t, current_time, da_onset, dt, k_on_d2, k_off_d2, EC50_D2, BETA)

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
    params: torch.Tensor,
):
    """
    D1 + D2 受体动力学仿真内核。
    - D1: 一阶动力学 (Langmuir)
    - D2: 一阶动力学 (Langmuir)
    Batch 0 = Control (baseline), Batch 1 = Experiment (da_level nM)
    """
    # --- Unpack params ---
    V_rest   = float(params[0])
    V_reset  = float(params[1])
    V_th     = float(params[2])
    R_base   = float(params[3])
    tau_syn  = float(params[4])
    t_ref    = float(params[5])
    bg_mean  = float(params[6])
    bg_std   = float(params[7])
    C_E      = float(params[8])
    C_I      = float(params[9])
    EC50_D1  = float(params[10])
    EC50_D2  = float(params[11])
    BETA     = float(params[12])
    EPS_D1   = float(params[13])
    EPS_D2   = float(params[14])
    BIAS_D1  = float(params[15])
    BIAS_D2  = float(params[16])
    LAM_D1   = float(params[17])
    LAM_D2   = float(params[18])
    TAU_ON_D1  = float(params[19])
    TAU_OFF_D1 = float(params[20])
    TAU_ON_D2  = float(params[21])
    TAU_OFF_D2 = float(params[22])
    DA_BASELINE = float(params[23])
    # --- Derived Langmuir kinetics ---
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

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

        # A. 当前 DA 浓度 (baseline DA for both batches)
        current_da_val = DA_BASELINE
        if current_time >= da_onset:
            current_da_val = float(da_level)
        da_t = torch.tensor([[DA_BASELINE], [current_da_val]], device=W_t.device)

        # B. 更新 alpha_D1 (Langmuir 受体结合动力学)
        alpha_d1 = compute_alpha_d1_step_langmuir(alpha_d1, da_t, current_time, da_onset, dt, k_on_d1, k_off_d1, EC50_D1, BETA)

        # C. 更新 alpha_D2 (Langmuir 受体结合动力学)
        alpha_d2 = compute_alpha_d2_step_langmuir(alpha_d2, da_t, current_time, da_onset, dt, k_on_d2, k_off_d2, EC50_D2, BETA)

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
# 内核 4b: D1 + D2 受体动力学 — Checkpoint 专用
#   Batch 0 = Control (0 nM, no DA)
#   Batch 1 = Experiment (da_level nM)
#
#   Purpose: Generate a checkpoint at a given DA baseline (e.g. 2 nM).
#   The control batch stays at 0 nM so the output plot shows the
#   difference between no-DA and the target DA concentration.
#   The final_state from Batch 1 can then be loaded by --resume.
# ======================================================================
@torch.jit.script
def run_dynamic_d1_d2_kernel_ckpt(
    W_t: torch.Tensor,
    mask_d1: torch.Tensor,
    mask_d2: torch.Tensor,
    da_level: float,
    da_onset: float,
    duration: float,
    dt: float,
    record_indices: torch.Tensor,
    n_exc: int,
    params: torch.Tensor,
):
    """
    D1 + D2 receptor kinetics kernel for checkpoint generation.
    - D1: first-order kinetics (Langmuir)
    - D2: first-order kinetics (Langmuir)
    Batch 0 = Control (0 nM, no DA), Batch 1 = Experiment (da_level nM)
    """
    # --- Unpack params ---
    V_rest   = float(params[0])
    V_reset  = float(params[1])
    V_th     = float(params[2])
    R_base   = float(params[3])
    tau_syn  = float(params[4])
    t_ref    = float(params[5])
    bg_mean  = float(params[6])
    bg_std   = float(params[7])
    C_E      = float(params[8])
    C_I      = float(params[9])
    EC50_D1  = float(params[10])
    EC50_D2  = float(params[11])
    BETA     = float(params[12])
    EPS_D1   = float(params[13])
    EPS_D2   = float(params[14])
    BIAS_D1  = float(params[15])
    BIAS_D2  = float(params[16])
    LAM_D1   = float(params[17])
    LAM_D2   = float(params[18])
    TAU_ON_D1  = float(params[19])
    TAU_OFF_D1 = float(params[20])
    TAU_ON_D2  = float(params[21])
    TAU_OFF_D2 = float(params[22])
    DA_BASELINE = float(params[23])
    # --- Derived Langmuir kinetics ---
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

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

        # A. DA concentration: Batch 0 = always 0 nM, Batch 1 = da_level after onset
        exp_da_val = 0.0
        if current_time >= da_onset:
            exp_da_val = float(da_level)
        da_t = torch.tensor([[0.0], [exp_da_val]], device=W_t.device)

        # B. Update alpha_D1 (Langmuir receptor binding kinetics)
        alpha_d1 = compute_alpha_d1_step_langmuir(alpha_d1, da_t, current_time, da_onset, dt, k_on_d1, k_off_d1, EC50_D1, BETA)

        # C. Update alpha_D2 (Langmuir receptor binding kinetics)
        alpha_d2 = compute_alpha_d2_step_langmuir(alpha_d2, da_t, current_time, da_onset, dt, k_on_d2, k_off_d2, EC50_D2, BETA)

        # D. Force Batch 0 alpha = 0 (pure no-DA control)
        #    Sigmoid(0 - EC50) is not exactly 0, so alpha would drift slightly.
        #    Clamp it to guarantee a flat control baseline.
        alpha_d1[0, 0] = 0.0
        alpha_d2[0, 0] = 0.0

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
    params: torch.Tensor,
):
    """
    D1 + D2 receptor kinetics kernel with two-stage DA dosing.
    """
    # --- Unpack params ---
    V_rest   = float(params[0])
    V_reset  = float(params[1])
    V_th     = float(params[2])
    R_base   = float(params[3])
    tau_syn  = float(params[4])
    t_ref    = float(params[5])
    bg_mean  = float(params[6])
    bg_std   = float(params[7])
    C_E      = float(params[8])
    C_I      = float(params[9])
    EC50_D1  = float(params[10])
    EC50_D2  = float(params[11])
    BETA     = float(params[12])
    EPS_D1   = float(params[13])
    EPS_D2   = float(params[14])
    BIAS_D1  = float(params[15])
    BIAS_D2  = float(params[16])
    LAM_D1   = float(params[17])
    LAM_D2   = float(params[18])
    TAU_ON_D1  = float(params[19])
    TAU_OFF_D1 = float(params[20])
    TAU_ON_D2  = float(params[21])
    TAU_OFF_D2 = float(params[22])
    DA_BASELINE = float(params[23])
    # --- Derived Langmuir kinetics ---
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

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

        # A. Current DA concentration (two-stage schedule)
        current_da_val = DA_BASELINE
        if current_time >= phase2_onset:
            current_da_val = float(da_level_2)
        elif current_time >= da_onset:
            current_da_val = float(da_level_1)
        da_t = torch.tensor([[DA_BASELINE], [current_da_val]], device=W_t.device)

        # B. Update alpha_D1 (Langmuir receptor binding kinetics)
        alpha_d1 = compute_alpha_d1_step_langmuir(alpha_d1, da_t, current_time, da_onset, dt, k_on_d1, k_off_d1, EC50_D1, BETA)

        # C. Update alpha_D2 (Langmuir receptor binding kinetics)
        alpha_d2 = compute_alpha_d2_step_langmuir(alpha_d2, da_t, current_time, da_onset, dt, k_on_d2, k_off_d2, EC50_D2, BETA)

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
    params: torch.Tensor,
):
    """
    D1 + D2 receptor kinetics kernel resuming from a checkpoint state.
    """
    # --- Unpack params ---
    V_rest   = float(params[0])
    V_reset  = float(params[1])
    V_th     = float(params[2])
    R_base   = float(params[3])
    tau_syn  = float(params[4])
    t_ref    = float(params[5])
    bg_mean  = float(params[6])
    bg_std   = float(params[7])
    C_E      = float(params[8])
    C_I      = float(params[9])
    EC50_D1  = float(params[10])
    EC50_D2  = float(params[11])
    BETA     = float(params[12])
    EPS_D1   = float(params[13])
    EPS_D2   = float(params[14])
    BIAS_D1  = float(params[15])
    BIAS_D2  = float(params[16])
    LAM_D1   = float(params[17])
    LAM_D2   = float(params[18])
    TAU_ON_D1  = float(params[19])
    TAU_OFF_D1 = float(params[20])
    TAU_ON_D2  = float(params[21])
    TAU_OFF_D2 = float(params[22])
    DA_BASELINE = float(params[23])
    # --- Derived Langmuir kinetics ---
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

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

        # A. Current DA concentration (baseline DA for both batches)
        current_da_val = DA_BASELINE
        if current_time >= da_onset:
            current_da_val = float(da_level)
        da_t = torch.tensor([[DA_BASELINE], [current_da_val]], device=W_t.device)

        # B. Update alpha_D1 (Langmuir receptor binding kinetics)
        alpha_d1 = compute_alpha_d1_step_langmuir(alpha_d1, da_t, current_time, da_onset, dt, k_on_d1, k_off_d1, EC50_D1, BETA)

        # C. Update alpha_D2 (Langmuir receptor binding kinetics)
        alpha_d2 = compute_alpha_d2_step_langmuir(alpha_d2, da_t, current_time, da_onset, dt, k_on_d2, k_off_d2, EC50_D2, BETA)

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
    params: torch.Tensor,
):
    """
    DA pulse experiment kernel resuming from checkpoint.
    Both batches start from the same checkpoint state (DA=da_base steady-state).
    - Batch 0 (Control): maintains da_base throughout
    - Batch 1 (Experiment): da_base → da_pulse → da_base
    """
    # --- Unpack params ---
    V_rest   = float(params[0])
    V_reset  = float(params[1])
    V_th     = float(params[2])
    R_base   = float(params[3])
    tau_syn  = float(params[4])
    t_ref    = float(params[5])
    bg_mean  = float(params[6])
    bg_std   = float(params[7])
    C_E      = float(params[8])
    C_I      = float(params[9])
    EC50_D1  = float(params[10])
    EC50_D2  = float(params[11])
    BETA     = float(params[12])
    EPS_D1   = float(params[13])
    EPS_D2   = float(params[14])
    BIAS_D1  = float(params[15])
    BIAS_D2  = float(params[16])
    LAM_D1   = float(params[17])
    LAM_D2   = float(params[18])
    TAU_ON_D1  = float(params[19])
    TAU_OFF_D1 = float(params[20])
    TAU_ON_D2  = float(params[21])
    TAU_OFF_D2 = float(params[22])
    # --- Derived Langmuir kinetics ---
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

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
    params: torch.Tensor,
):
    """
    Sinusoidal DA input experiment kernel.
    - Batch 0 (Control): constant da_base
    - Batch 1 (Experiment): da_base + amplitude * sin(2π * freq * t)
    """
    # --- Unpack params ---
    V_rest   = float(params[0])
    V_reset  = float(params[1])
    V_th     = float(params[2])
    R_base   = float(params[3])
    tau_syn  = float(params[4])
    t_ref    = float(params[5])
    bg_mean  = float(params[6])
    bg_std   = float(params[7])
    C_E      = float(params[8])
    C_I      = float(params[9])
    EC50_D1  = float(params[10])
    EC50_D2  = float(params[11])
    BETA     = float(params[12])
    EPS_D1   = float(params[13])
    EPS_D2   = float(params[14])
    BIAS_D1  = float(params[15])
    BIAS_D2  = float(params[16])
    LAM_D1   = float(params[17])
    LAM_D2   = float(params[18])
    TAU_ON_D1  = float(params[19])
    TAU_OFF_D1 = float(params[20])
    TAU_ON_D2  = float(params[21])
    TAU_OFF_D2 = float(params[22])
    # --- Derived Langmuir kinetics ---
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

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
    params: torch.Tensor,
):
    """
    DA pulse + external stimulus injection kernel resuming from checkpoint.
    Both batches start from the same checkpoint state (DA=da_base steady-state).
    - Batch 0 (Control): maintains da_base throughout
    - Batch 1 (Experiment): da_base → da_pulse → da_base
    Both batches receive the same external stimulus in [stim_onset, stim_offset).
    """
    # --- Unpack params ---
    V_rest   = float(params[0])
    V_reset  = float(params[1])
    V_th     = float(params[2])
    R_base   = float(params[3])
    tau_syn  = float(params[4])
    t_ref    = float(params[5])
    bg_mean  = float(params[6])
    bg_std   = float(params[7])
    C_E      = float(params[8])
    C_I      = float(params[9])
    EC50_D1  = float(params[10])
    EC50_D2  = float(params[11])
    BETA     = float(params[12])
    EPS_D1   = float(params[13])
    EPS_D2   = float(params[14])
    BIAS_D1  = float(params[15])
    BIAS_D2  = float(params[16])
    LAM_D1   = float(params[17])
    LAM_D2   = float(params[18])
    TAU_ON_D1  = float(params[19])
    TAU_OFF_D1 = float(params[20])
    TAU_ON_D2  = float(params[21])
    TAU_OFF_D2 = float(params[22])
    # --- Derived Langmuir kinetics ---
    k_on_d1  = 1.0 / (TAU_ON_D1 - 3000)
    k_off_d1 = 1.0 / (TAU_OFF_D1 + 3000)
    k_on_d2  = 1.0 / TAU_ON_D2
    k_off_d2 = 1.0 / TAU_OFF_D2

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
