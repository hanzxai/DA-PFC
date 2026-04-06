# models/pharmacology.py
"""
药理学参数计算模块
根据 DA 浓度和 Sigmoid 激活函数计算 Gain / Bias / Synaptic Scaling 调节矩阵。
所有药理学常量从 config 模块读取，避免硬编码重复。
"""
import torch
import numpy as np
import config


def _sigmoid_activation(da: float, ec50: float) -> float:
    """计算单个浓度的 Sigmoid 激活度"""
    return 1.0 / (1.0 + np.exp(-config.BETA * (da - ec50)))


def _compute_modulation_state(batch_size, N, mask_d1, mask_d2, da_levels, device):
    """
    核心函数：根据 DA 浓度列表计算一组调节参数矩阵。

    Args:
        batch_size: Batch 数量
        N:          神经元总数
        mask_d1:    D1 受体 Mask (N,)
        mask_d2:    D2 受体 Mask (N,)
        da_levels:  DA 浓度列表, 长度 = batch_size
        device:     计算设备

    Returns:
        mod_R      : (Batch, N)  Gain 调节
        I_mod      : (Batch, N)  偏置电流
        scale_syn  : (Batch, N)  突触缩放
    """
    mod_R = torch.ones((batch_size, N), device=device)
    I_mod = torch.zeros((batch_size, N), device=device)
    scale_syn = torch.ones((batch_size, N), device=device)

    for b, da in enumerate(da_levels):
        if da <= 0.01:
            continue

        act_d1 = _sigmoid_activation(da, config.EC50_D1)
        act_d2 = _sigmoid_activation(da, config.EC50_D2)

        # D1: 增强
        mod_R[b, mask_d1] = 1.0 + config.EPS_D1 * act_d1
        I_mod[b, mask_d1] = config.BIAS_D1 * act_d1
        scale_syn[b, mask_d1] = 1.0 + config.LAM_D1 * act_d1

        # D2: 抑制
        mod_R[b, mask_d2] = 1.0 - config.EPS_D2 * act_d2
        I_mod[b, mask_d2] = config.BIAS_D2 * act_d2
        scale_syn[b, mask_d2] = 1.0 - config.LAM_D2 * act_d2

    return mod_R, I_mod, scale_syn


def get_batch_modulation_params(N, mask_d1, mask_d2, da_concs, device):
    """
    生成并行计算所需的调节参数矩阵 (静态模式)。
    da_concs: 浓度列表, e.g. [0.0, 10.0]
    返回形状: (Batch, N)
    """
    return _compute_modulation_state(len(da_concs), N, mask_d1, mask_d2, da_concs, device)


def get_stepped_modulation_params(N, mask_d1, mask_d2, da_levels_active, device):
    """
    生成分步给药的两组参数:
      1. rest   : 所有 batch 都是 2nM (baseline DA)
      2. active : 按照 da_levels_active 设定 (实验组变高)
    """
    DA_BASELINE = 2.0
    batch_size = len(da_levels_active)
    levels_rest = [DA_BASELINE] * batch_size

    params_rest = _compute_modulation_state(batch_size, N, mask_d1, mask_d2, levels_rest, device)
    params_active = _compute_modulation_state(batch_size, N, mask_d1, mask_d2, da_levels_active, device)

    return params_rest, params_active
