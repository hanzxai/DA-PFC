# models/network.py
import torch
import config  # 引用上级配置

# 创建网络结构
def create_network_structure(N_E, N_I, device):
    """构建网络结构 (保持不变)"""
    N = N_E + N_I
    # 注意：为了矩阵乘法，这里 W 需要转置吗？
    # LIF 更新通常是 I += W @ spikes (或者 spikes @ W，取决于形状)
    # 我们这里生成标准的连接矩阵
    W = torch.zeros((N, N), device=device)
    prob = 0.20 
    mask_conn = (torch.rand((N, N), device=device) < prob).float()
    mask_conn.fill_diagonal_(0)

    w_exc = 0.3 
    w_inh = -2.0 
    
    W[:, :N_E] = w_exc * mask_conn[:, :N_E]
    W[:, N_E:] = w_inh * mask_conn[:, N_E:]
    
    # 转置 W，以便后续用 (Batch, N) @ (N, N) 进行计算
    # 这样每一行代表一个 batch 的神经元输入
    W_t = W.t().contiguous()
    
    # Masks
    mask_d1 = torch.zeros(N, dtype=torch.bool, device=device)
    mask_d2 = torch.zeros(N, dtype=torch.bool, device=device)
    
    n_e_d1 = int(N_E * 0.25)
    n_e_d2 = int(N_E * 0.15)
    idx_e_d1_end = n_e_d1
    idx_e_d2_end = n_e_d1 + n_e_d2
    mask_d1[:idx_e_d1_end] = True
    mask_d2[idx_e_d1_end:idx_e_d2_end] = True
    
    n_i_d1 = int(N_I * 0.30)
    n_i_d2 = int(N_I * 0.10)
    idx_i_start = N_E
    idx_i_d1_end = idx_i_start + n_i_d1
    idx_i_d2_end = idx_i_d1_end + n_i_d2
    mask_d1[idx_i_start:idx_i_d1_end] = True
    mask_d2[idx_i_d1_end:idx_i_d2_end] = True
    
    groups_info = {'e_d1_end': idx_e_d1_end, 'e_d2_end': idx_e_d2_end, 'e_other_end': N_E}
    return W_t, mask_d1, mask_d2, groups_info
