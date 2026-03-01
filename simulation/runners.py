# simulation/runners.py
import time
import torch
import config
# 关键：从其他模块导入
from models.network import create_network_structure
from models.kernels import *
from models.pharmacology import *
import numpy as np


# 主函数：运行仿真并返回数据包
# 旧版：一次性全部跑完
def run_simulation_in_memory(device_name="cuda:0"):
    """
    运行仿真并将所有结果打包成字典返回 (不存文件)
    """
    # 1. 硬件配置
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"🚀 Simulation running on {device}")

    # 2. 参数设定 (可以在这里修改，或者作为函数参数传入)
    N_E, N_I = 800, 200
    N = N_E + N_I
    duration, dt = 2000.0, 0.1
    FIXED_DA = 10.0
    
    # 3. 构建网络
    # 为了保证可复现性，固定种子
    torch.manual_seed(42); np.random.seed(42)
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(N_E, N_I, device)
    
    # 4. 设定要记录电压的神经元
    target_neuron_d1 = 0 
    target_neuron_d2 = groups_info['e_d1_end'] 
    
    record_indices = torch.tensor([
        [0, target_neuron_d1], # Control - D1
        [1, target_neuron_d1], # Exp     - D1
        [0, target_neuron_d2], # Control - D2
        [1, target_neuron_d2]  # Exp     - D2
    ], device=device, dtype=torch.long)

    # 5. 准备批处理参数
    da_conditions = [0.0, FIXED_DA]
    mod_R, I_mod, scale_syn = get_batch_modulation_params(N, mask_d1, mask_d2, da_conditions, device)
    
    # 6. 运行仿真
    print("⚡ Running Simulation...")
    t0 = time.time()
    # 调用核心仿真内核
    all_spikes, v_traces = run_batch_network(W_t, mod_R, I_mod, scale_syn, duration, dt, record_indices)
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"✅ Simulation finished in {time.time()-t0:.4f}s")

    # 7. 打包数据 (关键：全部转为 CPU Tensor，方便后续分析，不占显存)
    data_package = {
        'config': {
            'N_E': N_E, 'N_I': N_I, 'duration': duration, 'dt': dt,
            'da_levels': da_conditions
        },
        'masks': {
            'd1': mask_d1.cpu(),
            'd2': mask_d2.cpu()
        },
        'groups_info': groups_info,
        'spikes': all_spikes.cpu(),       # (Time, Batch, NeuronID)
        'v_traces': v_traces.cpu(),       # (Time, TraceID)
        'record_indices': record_indices.cpu()
    }
    
    return data_package


# 新版：分步给药仿真
def run_simulation_stepped(device_name="cuda:0", da_level=10.0):
    """
    运行分步仿真 (Baseline -> DA)
    :param device_name: 计算设备
    :param da_level: 给药阶段的多巴胺浓度 (nM)，默认为 10.0
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"🚀 Simulation running on {device}")

    # 参数
    N_E, N_I = 800, 200
    N = N_E + N_I
    duration, dt = 3000.0, 0.1 # 跑 3秒，看清楚切换过程
    
    # 设定给药时间点
    DA_ONSET = 1000.0 
    # FIXED_DA = 10.0   # 给药后的浓度
    
    torch.manual_seed(42); np.random.seed(42)
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(N_E, N_I, device)
    
    # 记录索引 (这里示例只记第一个 D1 和第一个 D2)
    target_neuron_d1 = 0 
    target_neuron_d2 = groups_info['e_d1_end']
    record_indices = torch.tensor([[0, 0], [1, 0]], device=device, dtype=torch.long)
    # --- [关键修改] 使用传入的 da_level ---
    # Batch 0: 始终是 0.0 (Control)
    # Batch 1: 激活后达到 da_level (Experiment)
    da_levels_active = [0.0, da_level]

    params_rest, params_active = get_stepped_modulation_params(N, mask_d1, mask_d2, da_levels_active, device)
    
    print(f"⚡ Running Simulation (Onset at {DA_ONSET}ms)...")
    t0 = time.time()
    
    # 调用新的 Stepped Kernel
    all_spikes, v_traces = run_batch_network_stepped(
        W_t, params_rest, params_active, duration, dt, DA_ONSET, record_indices
    )
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"✅ Finished in {time.time()-t0:.4f}s")

    return {
        'config': {
            'N_E': N_E, 'N_I': N_I, 'duration': duration, 'dt': dt,
            'da_onset': DA_ONSET, 'da_level': da_level
        },
        'masks': {'d1': mask_d1.cpu(), 'd2': mask_d2.cpu()},
        'groups_info': groups_info,
        'spikes': all_spikes.cpu(),
        'v_traces': v_traces.cpu(),
        'record_indices': record_indices.cpu()
    }


# 2025年12月29日 新增：D1动态PKA
def run_simulation_d1_kinetics(duration=None, target_da=None):
    """
    运行 D1 受体动力学仿真实验
    :param duration: 仿真总时长 (ms)
    :param target_da: 给药后的目标浓度 (nM)
    """
    # 1. 加载配置
    device = config.DEVICE
    
    # 如果没传参数，就用默认值 (但在 main.py 里最好传进来)
    if duration is None: duration = 100000.0 # 默认 100秒，因为 Tau 很长
    if target_da is None: target_da = 10.0   # 默认 10nM
    
    dt = config.DT
    
    # 设定给药时间 (DA_ONSET)
    # 为了看清基线，我们在第 5 秒 (5000ms) 开始给药
    da_onset = 5000.0
    
    print(f"🚀 Simulation running on {device}")
    print(f"   Mode: Dynamic D1 Kinetics (Tau_rise={config.TAU_ON_D1}ms)")
    print(f"   Duration: {duration}ms, Target DA: {target_da}nM")

    # 2. 构建网络 (结构不变)
    # 固定种子保证复现
    torch.manual_seed(42); np.random.seed(42)
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(config.N_E, config.N_I, config.device)
    
    # 3. 设定记录索引
    # 我们记录 Control组(Batch 0) 和 Exp组(Batch 1) 的代表性神经元
    target_neuron_d1 = 0 
    target_neuron_d2 = groups_info['e_d1_end']
    
    # [Batch_idx, Neuron_idx]
    record_indices = torch.tensor([
        [0, target_neuron_d1], # Control - D1
        [1, target_neuron_d1], # Exp     - D1
        [0, target_neuron_d2], # Control - D2
        [1, target_neuron_d2]  # Exp     - D2
    ], device=device, dtype=torch.long)

    # 4. 运行仿真
    # 注意：这里不再需要 get_modulation_params 了
    # 直接把 target_da (float) 传给内核，内核自己会算 Sigmoid 和微分方程
    
    print("⚡ Running Kernel...")
    t0 = time.time()
    
    all_spikes, v_traces = run_dynamic_d1_kernel(
        W_t, 
        mask_d1, 
        mask_d2, 
        float(target_da),   # 目标浓度
        float(da_onset),    # 给药时间点
        float(duration),    # 总时长
        dt,
        record_indices
    )
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    print(f"✅ Finished in {time.time()-t0:.4f}s")

    # 5. 打包数据
    # 保持和 Analyzer 兼容的数据结构
    data_package = {
        'config': {
            'N_E': config.N_E, 
            'N_I': config.N_I, 
            'duration': duration, 
            'dt': dt,
            'da_onset': da_onset, 
            'da_level': target_da,
            'mode': 'dynamic_d1_kinetics'
        },
        'masks': {
            'd1': mask_d1.cpu(),
            'd2': mask_d2.cpu()
        },
        'groups_info': groups_info,
        'spikes': all_spikes.cpu(),   # (Time_step, Batch_id, Neuron_id)
        'v_traces': v_traces.cpu(),
        'record_indices': record_indices.cpu()
    }
    
    return data_package