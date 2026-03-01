# models/pharmacology.py
import torch
import numpy as np

# 获取参数
def get_batch_modulation_params(N, mask_d1, mask_d2, da_concs, device):
    """
    生成并行计算所需的调节参数矩阵
    da_concs: list of concentrations, e.g., [0.0, 10.0]
    返回形状: (Batch, N)
    """
    batch_size = len(da_concs)
    
    # 初始化 (Batch, N)
    mod_R = torch.ones((batch_size, N), device=device)
    I_mod = torch.zeros((batch_size, N), device=device)
    scale_syn = torch.ones((batch_size, N), device=device)
    
    # 药理学常数
    ec50_d1, ec50_d2, beta = 4.0, 8.0, 1.0
    eps_d1, eps_d2 = 0.3, 0.2
    bias_d1, bias_d2 = 3.0, -3.0
    lam_d1, lam_d2 = 0.3, 0.2
    
    for b, da in enumerate(da_concs):
        if da <= 0.01: continue
        
        # 计算该 batch 的激活度 (Scalar)
        # 注意：这里我们简化处理，假设全脑 DA 浓度均匀
        act_d1 = 1.0 / (1.0 + np.exp(-beta * (da - ec50_d1)))
        act_d2 = 1.0 / (1.0 + np.exp(-beta * (da - ec50_d2)))
        
        # 填充 (Vectorized per batch)
        mod_R[b, mask_d1] = 1.0 + eps_d1 * act_d1
        mod_R[b, mask_d2] = 1.0 - eps_d2 * act_d2
        
        I_mod[b, mask_d1] = bias_d1 * act_d1
        I_mod[b, mask_d2] = bias_d2 * act_d2
        
        scale_syn[b, mask_d1] = 1.0 + lam_d1 * act_d1
        scale_syn[b, mask_d2] = 1.0 - lam_d2 * act_d2
        
    return mod_R, I_mod, scale_syn

def get_stepped_modulation_params(N, mask_d1, mask_d2, da_levels_active, device):
    """
    生成两组参数：
    1. rest: 所有 batch 都是 0nM (基线)
    2. active: 按照 da_levels_active 设定 (实验组变高)
    """
    batch_size = len(da_levels_active)
    
    # --- 辅助函数：计算单次状态 ---
    def compute_state(levels):
        mod_R = torch.ones((batch_size, N), device=device)
        I_mod = torch.zeros((batch_size, N), device=device)
        scale_syn = torch.ones((batch_size, N), device=device)
        
        # 药理学常数
        ec50_d1, ec50_d2, beta = 4.0, 8.0, 1.0
        eps_d1, eps_d2 = 0.3, 0.2
        bias_d1, bias_d2 = 3.0, -3.0
        lam_d1, lam_d2 = 0.3, 0.2
        
        for b, da in enumerate(levels):
            if da <= 0.01: continue
            
            act_d1 = 1.0 / (1.0 + np.exp(-beta * (da - ec50_d1)))
            act_d2 = 1.0 / (1.0 + np.exp(-beta * (da - ec50_d2)))
            
            # Gain
            mod_R[b, mask_d1] = 1.0 + eps_d1 * act_d1
            mod_R[b, mask_d2] = 1.0 - eps_d2 * act_d2
            # Bias
            I_mod[b, mask_d1] = bias_d1 * act_d1
            I_mod[b, mask_d2] = bias_d2 * act_d2
            # Synaptic
            scale_syn[b, mask_d1] = 1.0 + lam_d1 * act_d1
            scale_syn[b, mask_d2] = 1.0 - lam_d2 * act_d2
        return mod_R, I_mod, scale_syn

    # 1. 生成 Rest 参数 (全 0.0)
    levels_rest = [0.0] * batch_size
    R_rest, I_rest, S_rest = compute_state(levels_rest)
    
    # 2. 生成 Active 参数 (例如 [0.0, 10.0])
    R_act, I_act, S_act = compute_state(da_levels_active)
    
    return (R_rest, I_rest, S_rest), (R_act, I_act, S_act)
