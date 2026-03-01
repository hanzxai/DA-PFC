# simulation/runners.py
"""
仿真运行器: 封装三种仿真模式的完整流程
(网络构建 → 参数计算 → 内核调用 → 数据打包)
"""
import time
import torch
import numpy as np

import config
from models.network import create_network_structure
from models.kernels import (
    run_batch_network,
    run_batch_network_stepped,
    run_dynamic_d1_kernel,
)
from models.pharmacology import get_batch_modulation_params, get_stepped_modulation_params


# ==============================================================================
# 公共辅助函数 (消除三个 Runner 中的重复逻辑)
# ==============================================================================

def _init_network(device: torch.device):
    """固定种子 → 构建网络。返回 (W_t, mask_d1, mask_d2, groups_info)"""
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    return create_network_structure(config.N_E, config.N_I, device)


def _build_record_indices(groups_info: dict, device: torch.device, full: bool = True):
    """
    构建要记录电压的神经元索引。
    full=True  → 4 个记录点 (Control/Exp × D1/D2)
    full=False → 2 个记录点 (Control/Exp × D1 only)
    """
    target_d1 = 0
    target_d2 = groups_info['e_d1_end']

    if full:
        indices = [
            [0, target_d1],  # Control - D1
            [1, target_d1],  # Exp     - D1
            [0, target_d2],  # Control - D2
            [1, target_d2],  # Exp     - D2
        ]
    else:
        indices = [
            [0, target_d1],  # Control - D1
            [1, target_d1],  # Exp     - D1
        ]
    return torch.tensor(indices, device=device, dtype=torch.long)


def _pack_data(cfg_dict, mask_d1, mask_d2, groups_info, spikes, v_traces, record_indices):
    """将仿真结果统一打包为 CPU dict, 供 Analyzer 使用。"""
    return {
        'config': cfg_dict,
        'masks': {
            'd1': mask_d1.cpu(),
            'd2': mask_d2.cpu(),
        },
        'groups_info': groups_info,
        'spikes': spikes.cpu(),
        'v_traces': v_traces.cpu(),
        'record_indices': record_indices.cpu(),
    }


def _sync_and_report(t0: float):
    """GPU 同步 + 打印耗时"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"✅ Finished in {elapsed:.4f}s")


# ==============================================================================
# Runner 1: 静态 DA (一次性跑完)
# ==============================================================================

def run_simulation_in_memory(device_name: str = "cuda:0"):
    """
    静态仿真: 整个时间段 DA 浓度固定不变。
    Batch 0 = Control (0 nM), Batch 1 = Experiment (10 nM)
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"🚀 Simulation running on {device}")

    duration = config.DEFAULT_DURATION
    dt = config.DT
    FIXED_DA = 10.0

    W_t, mask_d1, mask_d2, groups_info = _init_network(device)
    N = config.N_TOTAL
    record_indices = _build_record_indices(groups_info, device, full=True)

    da_conditions = [0.0, FIXED_DA]
    mod_R, I_mod, scale_syn = get_batch_modulation_params(
        N, mask_d1, mask_d2, da_conditions, device
    )

    print("⚡ Running Simulation...")
    t0 = time.time()
    all_spikes, v_traces = run_batch_network(
        W_t, mod_R, I_mod, scale_syn, duration, dt, record_indices
    )
    _sync_and_report(t0)

    return _pack_data(
        cfg_dict={'N_E': config.N_E, 'N_I': config.N_I, 'duration': duration,
                  'dt': dt, 'da_levels': da_conditions},
        mask_d1=mask_d1, mask_d2=mask_d2, groups_info=groups_info,
        spikes=all_spikes, v_traces=v_traces, record_indices=record_indices,
    )


# ==============================================================================
# Runner 2: 分步给药 (Baseline → DA, 参数瞬时切换)
# ==============================================================================

def run_simulation_stepped(device_name: str = "cuda:0", da_level: float = 10.0):
    """
    分步仿真: 在 DA_ONSET 时刻参数瞬时切换。
    Batch 0 = Control (始终 0 nM), Batch 1 = Experiment
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"🚀 Simulation running on {device}")

    duration = 3000.0
    dt = config.DT
    DA_ONSET = 1000.0

    W_t, mask_d1, mask_d2, groups_info = _init_network(device)
    N = config.N_TOTAL
    record_indices = _build_record_indices(groups_info, device, full=False)

    da_levels_active = [0.0, da_level]
    params_rest, params_active = get_stepped_modulation_params(
        N, mask_d1, mask_d2, da_levels_active, device
    )

    print(f"⚡ Running Simulation (Onset at {DA_ONSET}ms)...")
    t0 = time.time()
    all_spikes, v_traces = run_batch_network_stepped(
        W_t, params_rest, params_active, duration, dt, DA_ONSET, record_indices
    )
    _sync_and_report(t0)

    return _pack_data(
        cfg_dict={'N_E': config.N_E, 'N_I': config.N_I, 'duration': duration,
                  'dt': dt, 'da_onset': DA_ONSET, 'da_level': da_level},
        mask_d1=mask_d1, mask_d2=mask_d2, groups_info=groups_info,
        spikes=all_spikes, v_traces=v_traces, record_indices=record_indices,
    )


# ==============================================================================
# Runner 3: D1 受体动力学 (alpha_D1 遵循一阶 ODE)
# ==============================================================================

def run_simulation_d1_kinetics(duration: float = None, target_da: float = None):
    """
    D1 受体动力学仿真:
    alpha_D1 按 Tau_on/Tau_off 缓慢爬升/衰减, D2 保持瞬时。
    """
    device = config.DEVICE

    if duration is None:
        duration = 100000.0  # 100 秒 (因为 Tau ≈ 30s)
    if target_da is None:
        target_da = 10.0

    dt = config.DT
    da_onset = config.DEFAULT_DA_ONSET

    print(f"🚀 Simulation running on {device}")
    print(f"   Mode: Dynamic D1 Kinetics (Tau_rise={config.TAU_ON_D1}ms)")
    print(f"   Duration: {duration}ms, Target DA: {target_da}nM")

    W_t, mask_d1, mask_d2, groups_info = _init_network(device)
    record_indices = _build_record_indices(groups_info, device, full=True)

    print("⚡ Running Kernel...")
    t0 = time.time()
    all_spikes, v_traces = run_dynamic_d1_kernel(
        W_t, mask_d1, mask_d2,
        float(target_da), float(da_onset), float(duration), dt,
        record_indices,
    )
    _sync_and_report(t0)

    return _pack_data(
        cfg_dict={
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': duration, 'dt': dt,
            'da_onset': da_onset, 'da_level': target_da,
            'mode': 'dynamic_d1_kinetics',
        },
        mask_d1=mask_d1, mask_d2=mask_d2, groups_info=groups_info,
        spikes=all_spikes, v_traces=v_traces, record_indices=record_indices,
    )
