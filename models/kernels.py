# models/kernels.py
import torch

# 计算
@torch.jit.script
def run_batch_network(W_t: torch.Tensor, 
                      mod_R: torch.Tensor, 
                      I_mod: torch.Tensor, 
                      scale_syn: torch.Tensor,
                      duration: float, dt: float,
                      record_indices: torch.Tensor):
    """
    并行仿真内核
    W_t: (N, N)
    Inputs (mod_R, etc): (Batch, N)
    """
    batch_size, N = mod_R.shape
    steps = int(duration / dt)
    
    # 参数
    # V_rest, V_reset, V_th = -60.0, -65.0, -50.0
    # tau, t_ref, tau_syn = 20.0, 3.0, 5.0
    # R_base = 1.0
    V_rest, V_reset, V_th = 0., -5., 20.
    R_base, tau, t_ref, tau_syn = 1., 20., 5., 5.0
    
    # 状态变量初始化 (Batch, N)
    V = torch.rand((batch_size, N), device=W_t.device) * (V_th - V_reset) + V_reset
    t_last_spike = torch.ones((batch_size, N), device=W_t.device) * -1000.0
    I_syn = torch.zeros((batch_size, N), device=W_t.device)
    
    # 记录容器：为了效率，我们分别记录每个 batch 的 spike
    # 这里我们用一个 list of tensors 或者一个大的 tensor 来存
    # 为了 JIT 兼容性，我们用一个大 tensor: (Total_Spikes, 3) -> [time_step, batch_id, neuron_id]
    max_spikes = int(steps * N * batch_size * 0.15)
    spike_records = torch.zeros((max_spikes, 3), device=W_t.device, dtype=torch.long)
    spike_count = 0
    # [新增] 记录电压
    # record_indices shape: (K, 2) -> [batch_idx, neuron_idx]
    # v_traces shape: (steps, K)
    num_record = record_indices.shape[0]
    v_traces = torch.zeros((steps, num_record), device=W_t.device)
    
    # 常量
    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))
    # bg_mean, bg_std = 15.0, 4.0
    bg_mean, bg_std = 25.0, 5.0
    alpha = dt / tau
    
    for i in range(steps):
        current_time = i * dt
        
        # 1. 突触电流衰减
        I_syn = I_syn * decay_factor
        
        # 2. 总电流 (Batch, N)
        # 所有操作都是 element-wise，自动支持 batch 维度
        I_bg = torch.randn((batch_size, N), device=W_t.device) * bg_std + bg_mean
        I_total = (I_syn * scale_syn) + I_bg + I_mod
        
        # 3. LIF 积分
        V_new = V + alpha * (-(V - V_rest) + (R_base * mod_R) * I_total)
        
        # 不应期
        is_refractory = (current_time - t_last_spike) <= t_ref
        V = torch.where(is_refractory, V, V_new)

        # [新增] 在发放判定前，记录指定神经元的电压
        # 这一步开销很小，因为只读取几个值
        for k in range(num_record):
            b_idx = record_indices[k, 0]
            n_idx = record_indices[k, 1]
            v_traces[i, k] = V[b_idx, n_idx]
        
        # 4. 发放
        spikes = V > V_th # (Batch, N) boolean
        
        if spikes.any():
            # 记录 (需要处理 batch 索引)
            # nonzero 返回 (num_spikes, 2) -> [batch_idx, neuron_idx]
            indices = torch.nonzero(spikes) 
            num_now = indices.shape[0]
            
            if spike_count + num_now < max_spikes:
                # 记录 [time, batch, neuron]
                spike_records[spike_count:spike_count+num_now, 0] = i
                spike_records[spike_count:spike_count+num_now, 1] = indices[:, 0]
                spike_records[spike_count:spike_count+num_now, 2] = indices[:, 1]
                spike_count += num_now
            
            # Reset
            V[spikes] = V_reset
            t_last_spike[spikes] = torch.tensor(current_time, device=W_t.device)
            
            # 5. 传播 (Batch Matmul)
            # I_syn (B, N) += spikes (B, N) @ W_t (N, N)
            # PyTorch 会自动广播: (B, N) x (N, N) -> (B, N)
            I_syn += torch.matmul(spikes.float(), W_t)
            
    # return spike_records[:spike_count]
    return spike_records[:spike_count], v_traces

# 计算2 动态给药
@torch.jit.script
def run_batch_network_stepped(W_t: torch.Tensor, 
                              params_rest: tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                              params_active: tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                              duration: float, dt: float,
                              da_onset: float, 
                              record_indices: torch.Tensor):
    
    # 解包参数
    mod_R_rest, I_mod_rest, scale_rest = params_rest
    mod_R_act, I_mod_act, scale_act = params_active
    
    batch_size, N = mod_R_rest.shape
    steps = int(duration / dt)
    
    # 神经元参数
    V_rest, V_reset, V_th = 0., -5., 20.
    R_base, tau, t_ref, tau_syn = 1., 20., 5., 5.0
    
    # 初始化
    V = torch.rand((batch_size, N), device=W_t.device) * (V_th - V_reset) + V_reset
    t_last_spike = torch.ones((batch_size, N), device=W_t.device) * -1000.0
    I_syn = torch.zeros((batch_size, N), device=W_t.device)
    
    # 记录
    max_spikes = int(steps * N * batch_size * 0.15)
    spike_records = torch.zeros((max_spikes, 3), device=W_t.device, dtype=torch.long)
    spike_count = 0
    
    num_record = record_indices.shape[0]
    v_traces = torch.zeros((steps, num_record), device=W_t.device)
    
    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))
    bg_mean, bg_std = 25.0, 5.0
    alpha = dt / tau
    
    for i in range(steps):
        current_time = i * dt
        
        # --- [关键修改] 根据时间切换参数 ---
        if current_time < da_onset:
            # 基线期：大家都用 0nM 参数
            cur_mod_R = mod_R_rest
            cur_I_mod = I_mod_rest
            cur_scale = scale_rest
        else:
            # 给药期：Control 维持 0nM, Exp 变成 10nM
            cur_mod_R = mod_R_act
            cur_I_mod = I_mod_act
            cur_scale = scale_act
            
        # 1. 更新电流
        I_syn = I_syn * decay_factor
        I_bg = torch.randn((batch_size, N), device=W_t.device) * bg_std + bg_mean
        
        # 使用当前选择的参数
        I_total = (I_syn * cur_scale) + I_bg + cur_I_mod
        
        # 2. LIF 积分
        V_new = V + alpha * (-(V - V_rest) + (R_base * cur_mod_R) * I_total)
        
        is_refractory = (current_time - t_last_spike) <= t_ref
        V = torch.where(is_refractory, V, V_new)
        
        # 记录电压
        for k in range(num_record):
            v_traces[i, k] = V[record_indices[k, 0], record_indices[k, 1]]
        
        # 3. 发放
        spikes = V > V_th
        if spikes.any():
            indices = torch.nonzero(spikes)
            num_now = indices.shape[0]
            if spike_count + num_now < max_spikes:
                spike_records[spike_count:spike_count+num_now, 0] = i
                spike_records[spike_count:spike_count+num_now, 1] = indices[:, 0]
                spike_records[spike_count:spike_count+num_now, 2] = indices[:, 1]
                spike_count += num_now
            
            V[spikes] = V_reset
            t_last_spike[spikes] = torch.tensor(current_time, device=W_t.device)
            I_syn += torch.matmul(spikes.float(), W_t)
            
    return spike_records[:spike_count], v_traces



# 2025年12月29日 新增D1R动力学PKA
@torch.jit.script
def run_dynamic_d1_kernel(
    W_t: torch.Tensor,
    mask_d1: torch.Tensor,
    mask_d2: torch.Tensor,
    da_level: float,          # 目标 DA 浓度 (nM)
    da_onset: float,          # 给药时间 (ms)
    duration: float, 
    dt: float,
    record_indices: torch.Tensor
):
    """
    仿真内核：
    - D1: 遵循一阶动力学方程 (Tau_on / Tau_off)
    - D2: 瞬时响应 (Original Step)
    """
    # ==========================================
    # 1. 物理常数定义 (对应 config.py)
    # ==========================================
    # D1 动力学参数
    TAU_ON_D1  = 30876.1
    TAU_OFF_D1 = 164472.5
    
    # 药理学参数 (用于计算 S(t))
    EC50_D1 = 4.0
    EC50_D2 = 8.0
    BETA    = 1.0
    
    # 调节强度 (Gain/Bias/Scaling)
    EPS_D1, EPS_D2 = 0.3, 0.2    # Gain
    BIAS_D1, BIAS_D2 = 3.0, -3.0 # Bias
    LAM_D1, LAM_D2 = 0.3, 0.2    # Synaptic
    
    # ==========================================
    # 2. 初始化
    # ==========================================
    N = W_t.shape[0]
    batch_size = 2 # 固定为2: Batch 0 (Control), Batch 1 (Experiment)
    steps = int(duration / dt)
    
    # 神经元参数 (LIF)
    V_rest, V_reset, V_th = 0.0, -5.0, 20.0
    R_base = 1.0
    tau_mem = 20.0
    tau_syn = 5.0
    t_ref = 5.0
    
    # 状态变量
    # 注意：Batch 0 是 Control (0 nM), Batch 1 是 Exp (da_level nM)
    # 我们在循环里动态判断 DA 浓度
    V = torch.rand((batch_size, N), device=W_t.device) * (V_th - V_reset) + V_reset
    t_last_spike = torch.ones((batch_size, N), device=W_t.device) * -1000.0
    I_syn = torch.zeros((batch_size, N), device=W_t.device)
    
    # --- [核心状态变量] alpha_d1 ---
    # shape: (Batch, 1) 用于广播
    # 初始值为 0 (假设开始时没有 DA)
    alpha_d1 = torch.zeros((batch_size, 1), device=W_t.device)
    
    # 预计算常量
    decay_factor = float(torch.exp(torch.tensor(-dt / tau_syn)))
    lif_alpha = dt / tau_mem
    bg_mean, bg_std = 25.0, 5.0
    
    # 记录容器
    max_spikes = int(steps * N * batch_size * 0.15)
    spike_records = torch.zeros((max_spikes, 3), device=W_t.device, dtype=torch.long)
    spike_count = 0
    num_record = record_indices.shape[0]
    v_traces = torch.zeros((steps, num_record), device=W_t.device)
    
    # ==========================================
    # 3. 时间步循环
    # ==========================================
    for i in range(steps):
        current_time = i * dt
        
        # --------------------------------------
        # A. 计算当前环境多巴胺浓度 [DA](t)
        # --------------------------------------
        # Batch 0: 永远是 0.0
        # Batch 1: < onset 是 0.0, >= onset 是 da_level
        # 我们构建一个 (Batch, 1) 的向量
        current_da_val = 0.0
        if current_time >= da_onset:
            current_da_val = float(da_level) # 确保是 float
            
        # 构造 Tensor: [Control=0, Exp=current_da_val]
        da_t = torch.tensor([[0.0], [current_da_val]], device=W_t.device)
        
        # --------------------------------------
        # B. 计算目标值 S(t) (Sigmoid)
        # --------------------------------------
        # S_D1(t)
        # 加上 1e-6 防止除以 0 (虽然 exp 不会为 0)
        s_d1 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D1)))
        
        # 如果 DA 接近 0，强制 S 为 0 (避免 Sigmoid 在 0 处还有残留值)
        # Sigmoid(0-4) ≈ 0.018，我们希望它是 0
        if current_time < da_onset:
             s_d1[:] = 0.0
        # 对 Batch 0 (Control) 始终强制为 0
        s_d1[0] = 0.0

        # S_D2(t) (瞬时响应，不做动力学，直接算)
        s_d2 = 1.0 / (1.0 + torch.exp(-BETA * (da_t - EC50_D2)))
        if current_time < da_onset: s_d2[:] = 0.0
        s_d2[0] = 0.0
        
        # --------------------------------------
        # C. 更新 alpha_D1 (动力学核心)
        # --------------------------------------
        # 判断上升还是下降: if S > alpha
        # diff > 0 -> Rise, diff <= 0 -> Decay
        diff = s_d1 - alpha_d1
        
        # 选择 Tau (Element-wise selection)
        # torch.where(condition, x, y)
        tau_dynamic = torch.where(diff > 0, 
                                  torch.tensor(TAU_ON_D1, device=W_t.device), 
                                  torch.tensor(TAU_OFF_D1, device=W_t.device))
        
        # 欧拉积分: d_alpha = (S - alpha) / tau * dt
        d_alpha = (diff / tau_dynamic) * dt
        alpha_d1 += d_alpha
        
        # --------------------------------------
        # D. 处理 D2 (保持瞬时，直接赋值)
        # --------------------------------------
        alpha_d2 = s_d2 # 不积分，直接跟随
        
        # --------------------------------------
        # E. 计算调节后的参数 (On-the-fly Modulation)
        # --------------------------------------
        # 此时我们需要根据 alpha_d1 和 alpha_d2 组装 mod_R, I_mod, scale
        # 基础值
        mod_R = torch.ones((batch_size, N), device=W_t.device)
        I_mod = torch.zeros((batch_size, N), device=W_t.device)
        scale_syn = torch.ones((batch_size, N), device=W_t.device)
        
        # 应用 D1 (增强) -> alpha_d1 * mask_d1
        # alpha_d1 是 (B, 1), mask_d1 是 (N,) -> 广播相乘
        mod_R     += (EPS_D1 * alpha_d1) * mask_d1
        I_mod     += (BIAS_D1 * alpha_d1) * mask_d1
        scale_syn += (LAM_D1 * alpha_d1) * mask_d1
        
        # 应用 D2 (抑制) -> alpha_d2 * mask_d2
        # 注意符号：通常 D2 减小 Gain (mod_R)，减小 Scaling，且 Bias 为负
        mod_R     -= (EPS_D2 * alpha_d2) * mask_d2
        I_mod     += (BIAS_D2 * alpha_d2) * mask_d2 # BIAS_D2 本身是负数 (-3.0)
        scale_syn -= (LAM_D2 * alpha_d2) * mask_d2
        
        # --------------------------------------
        # F. LIF 积分 (常规)
        # --------------------------------------
        I_syn = I_syn * decay_factor
        I_bg = torch.randn((batch_size, N), device=W_t.device) * bg_std + bg_mean
        I_total = (I_syn * scale_syn) + I_bg + I_mod
        
        V_new = V + lif_alpha * (-(V - V_rest) + (R_base * mod_R) * I_total)
        
        is_refractory = (current_time - t_last_spike) <= t_ref
        V = torch.where(is_refractory, V, V_new)
        
        # 记录与发放
        for k in range(num_record):
            v_traces[i, k] = V[record_indices[k, 0], record_indices[k, 1]]
            
        spikes = V > V_th
        if spikes.any():
            indices = torch.nonzero(spikes)
            num_now = indices.shape[0]
            if spike_count + num_now < max_spikes:
                spike_records[spike_count:spike_count+num_now, 0] = i
                spike_records[spike_count:spike_count+num_now, 1] = indices[:, 0]
                spike_records[spike_count:spike_count+num_now, 2] = indices[:, 1]
                spike_count += num_now
            V[spikes] = V_reset
            t_last_spike[spikes] = torch.tensor(current_time, device=W_t.device)
            I_syn += torch.matmul(spikes.float(), W_t)
            
    return spike_records[:spike_count], v_traces