# models/network.py
"""网络结构构建模块: 连接矩阵 + D1/D2 受体 Mask 分配"""
import torch
import config


def create_network_structure(N_E: int, N_I: int, device: torch.device):
    """
    构建 E-I 随机稀疏连接网络并分配 D1/D2 受体 Mask。
    
    参数来自 config 模块 (连接概率、权重、受体比例)，
    但 N_E / N_I / device 由调用方传入以保持灵活性。

    Returns:
        W_t       : (N, N) 转置后的连接矩阵，供 (Batch, N) @ (N, N) 使用
        mask_d1   : (N,)   bool, D1 受体 Mask
        mask_d2   : (N,)   bool, D2 受体 Mask
        groups_info: dict   亚群边界索引
    """
    N = N_E + N_I

    # ---- 1. 连接矩阵 ----
    W = torch.zeros((N, N), device=device)
    mask_conn = (torch.rand((N, N), device=device) < config.CONN_PROB).float()
    mask_conn.fill_diagonal_(0)

    W[:, :N_E] = config.W_EXC * mask_conn[:, :N_E]
    W[:, N_E:] = config.W_INH * mask_conn[:, N_E:]

    # 转置: 后续用 spikes @ W_t -> (Batch, N)
    W_t = W.t().contiguous()

    # ---- 2. D1 / D2 受体 Mask ----
    mask_d1 = torch.zeros(N, dtype=torch.bool, device=device)
    mask_d2 = torch.zeros(N, dtype=torch.bool, device=device)

    # 兴奋性神经元
    n_e_d1 = int(N_E * config.FRAC_E_D1)
    n_e_d2 = int(N_E * config.FRAC_E_D2)
    idx_e_d1_end = n_e_d1
    idx_e_d2_end = n_e_d1 + n_e_d2
    mask_d1[:idx_e_d1_end] = True
    mask_d2[idx_e_d1_end:idx_e_d2_end] = True

    # 抑制性神经元
    n_i_d1 = int(N_I * config.FRAC_I_D1)
    n_i_d2 = int(N_I * config.FRAC_I_D2)
    idx_i_start = N_E
    idx_i_d1_end = idx_i_start + n_i_d1
    idx_i_d2_end = idx_i_d1_end + n_i_d2
    mask_d1[idx_i_start:idx_i_d1_end] = True
    mask_d2[idx_i_d1_end:idx_i_d2_end] = True

    groups_info = {
        'e_d1_end': idx_e_d1_end,
        'e_d2_end': idx_e_d2_end,
        'e_other_end': N_E,
    }
    return W_t, mask_d1, mask_d2, groups_info
