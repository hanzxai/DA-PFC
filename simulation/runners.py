# simulation/runners.py
"""
仿真运行器: 封装三种仿真模式的完整流程
(网络构建 → 参数计算 → 内核调用 → 数据打包)
"""
import time
import threading
import os
import torch
import numpy as np
from tqdm import tqdm

import config
from models.network import create_network_structure
from models.kernels import (
    run_batch_network,
    run_batch_network_stepped,
    run_dynamic_d1_d2_kernel,
    run_dynamic_d1_d2_kernel_ckpt,
    run_dynamic_d1_d2_kernel_two_stage,
    run_dynamic_d1_d2_kernel_from_state,
    run_wm_kernel,
    run_wm_kernel_dual,
)
from models.pharmacology import get_batch_modulation_params, get_stepped_modulation_params
from simulation.utils import verify_checkpoint_fingerprint


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


def _pack_data(cfg_dict, mask_d1, mask_d2, groups_info, spikes, v_traces, record_indices,
               final_state=None):
    """将仿真结果统一打包为 CPU dict, 供 Analyzer 使用。"""
    data = {
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
    if final_state is not None:
        data['final_state'] = final_state.cpu()
    return data


def _sync_and_report(t0: float):
    """GPU 同步 + 打印耗时"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"✅ Finished in {elapsed:.2f}s")
    return elapsed


def _run_kernel_with_progress(kernel_fn, kernel_args: tuple, duration_ms: float, dt: float):
    """
    在后台线程执行 JIT 内核, 主线程显示 tqdm 进度条。

    因为 @torch.jit.script 内核无法被 Python 直接插桩,
    进度条基于「已用时间 / 预估总时间」来推进, 采用渐进式估算:
      - JIT 编译阶段: 显示 "compiling"
      - 计算阶段: 进度按 smoothstep 曲线推进, 保证不会过早到 100%
      - 内核完成: 跳至 100%
    """
    steps = int(duration_ms / dt)
    result_container = [None]
    error_container = [None]

    def _worker():
        try:
            result_container[0] = kernel_fn(*kernel_args)
        except Exception as e:
            error_container[0] = e

    # 启动内核线程
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    bar = tqdm(total=steps, desc="⚡ Simulating", unit="step",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    poll_interval = 0.3       # 轮询间隔 (秒)
    jit_warmup = 3.0          # 预估 JIT 编译开销 (秒)
    t_start = time.time()
    last_update = 0

    while thread.is_alive():
        thread.join(timeout=poll_interval)
        if not thread.is_alive():
            break

        elapsed = time.time() - t_start

        if elapsed < jit_warmup:
            bar.set_postfix_str("⏳ JIT compiling...")
            continue

        # ── 渐进式进度估算 ──
        # 思路: 不预测"总时间", 而是让进度按 1-1/(1+kt) 曲线自然增长。
        # 这条曲线从 0 单调增到 1, 且永远 <1, 所以进度条不会在完成前到 100%.
        # k 越大进度越快; 这里用 k=0.06 使得:
        #   10s 计算 → 显示 ~30%
        #   30s 计算 → 显示 ~62%
        #   60s 计算 → 显示 ~78%
        #  120s 计算 → 显示 ~88%
        compute_elapsed = elapsed - jit_warmup
        k = 0.06
        pct = compute_elapsed * k / (1.0 + compute_elapsed * k)  # 范围 [0, 1)
        new_pos = int(steps * pct)
        increment = new_pos - last_update
        if increment > 0:
            bar.update(increment)
            last_update = new_pos
        bar.set_postfix_str(f"{elapsed:.1f}s")

    # 完成: 补齐进度条到 100%
    final_gap = steps - last_update
    if final_gap > 0:
        bar.update(final_gap)
    bar.close()

    if error_container[0] is not None:
        raise error_container[0]

    return result_container[0]


# ==============================================================================
# Runner 1: 静态 DA (一次性跑完)
# ==============================================================================

def run_simulation_in_memory(device_name: str = "cuda:0"):
    """
    静态仿真: 整个时间段 DA 浓度固定不变。
    Batch 0 = Control (2 nM baseline), Batch 1 = Experiment (10 nM)
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"🚀 Simulation running on {device}")

    duration = config.DEFAULT_DURATION
    dt = config.DT
    FIXED_DA = 10.0
    DA_BASELINE = config.DA_BASELINE

    W_t, mask_d1, mask_d2, groups_info = _init_network(device)
    N = config.N_TOTAL
    record_indices = _build_record_indices(groups_info, device, full=True)

    da_conditions = [DA_BASELINE, FIXED_DA]
    mod_R, I_mod, scale_syn = get_batch_modulation_params(
        N, mask_d1, mask_d2, da_conditions, device
    )

    t0 = time.time()
    kp = config.build_kernel_params(device)
    all_spikes, v_traces = _run_kernel_with_progress(
        run_batch_network,
        (W_t, mod_R, I_mod, scale_syn, duration, dt, record_indices, config.N_E, kp),
        duration, dt,
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
    Batch 0 = Control (始终 2 nM baseline), Batch 1 = Experiment
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"🚀 Simulation running on {device}")

    duration = 3000.0
    dt = config.DT
    DA_ONSET = 1000.0
    DA_BASELINE = config.DA_BASELINE

    W_t, mask_d1, mask_d2, groups_info = _init_network(device)
    N = config.N_TOTAL
    record_indices = _build_record_indices(groups_info, device, full=False)

    da_levels_active = [DA_BASELINE, da_level]
    params_rest, params_active = get_stepped_modulation_params(
        N, mask_d1, mask_d2, da_levels_active, device
    )

    print(f"   DA Onset at {DA_ONSET}ms")
    t0 = time.time()
    kp = config.build_kernel_params(device)
    all_spikes, v_traces = _run_kernel_with_progress(
        run_batch_network_stepped,
        (W_t, params_rest, params_active, duration, dt, DA_ONSET, record_indices, config.N_E, kp),
        duration, dt,
    )
    _sync_and_report(t0)

    return _pack_data(
        cfg_dict={'N_E': config.N_E, 'N_I': config.N_I, 'duration': duration,
                  'dt': dt, 'da_onset': DA_ONSET, 'da_level': da_level},
        mask_d1=mask_d1, mask_d2=mask_d2, groups_info=groups_info,
        spikes=all_spikes, v_traces=v_traces, record_indices=record_indices,
    )


# ==============================================================================
# Runner 3: D1 + D2 受体动力学 (alpha_D1 和 alpha_D2 均遵循一阶 ODE)
# ==============================================================================

def run_simulation_d1_d2_kinetics(duration: float = None, target_da: float = None, device: torch.device = None):
    """
    D1 + D2 受体动力学仿真:
    - alpha_D1 按 τ_on=30876ms / τ_off=164472ms 缓慢爬升/衰减
    - alpha_D2 按 τ_on=10000ms / τ_off=50000ms 更快响应
    Batch 0 = Control (2 nM baseline), Batch 1 = Experiment (target_da nM)
    """
    if device is None:
        device = config.DEVICE

    if duration is None:
        duration = 100000.0  # 100 秒 (因为 D1 Tau ≈ 30s)
    if target_da is None:
        target_da = 10.0

    dt = config.DT
    da_onset = config.DEFAULT_DA_ONSET

    print(f"🚀 Simulation running on {device}")
    print(f"   Mode: Dynamic D1 + D2 Kinetics")
    print(f"   D1: τ_on={config.TAU_ON_D1}ms, EC50={config.EC50_D1}nM")
    print(f"   D2: τ_on={config.TAU_ON_D2}ms, EC50={config.EC50_D2}nM")
    print(f"   Duration: {duration}ms, Target DA: {target_da}nM, DA onset: {da_onset}ms")

    W_t, mask_d1, mask_d2, groups_info = _init_network(device)
    record_indices = _build_record_indices(groups_info, device, full=True)

    t0 = time.time()
    kp = config.build_kernel_params(device)
    all_spikes, v_traces, final_state = _run_kernel_with_progress(
        run_dynamic_d1_d2_kernel,
        (W_t, mask_d1, mask_d2,
         float(target_da), float(da_onset), float(duration), dt,
         record_indices, config.N_E, kp),
        duration, dt,
    )
    _sync_and_report(t0)

    return _pack_data(
        cfg_dict={
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': duration, 'dt': dt,
            'da_onset': da_onset, 'da_level': target_da,
            'control_da': config.DA_BASELINE,
            'mode': 'dynamic_d1_d2_kinetics',
        },
        mask_d1=mask_d1, mask_d2=mask_d2, groups_info=groups_info,
        spikes=all_spikes, v_traces=v_traces, record_indices=record_indices,
        final_state=final_state,
    )


# ==============================================================================
# Runner 4b: D1 + D2 受体动力学 — Checkpoint 专用
#   Batch 0 = Control (0 nM, no DA)
#   Batch 1 = Experiment (target_da nM)
#   用于生成 checkpoint, 图表展示 0 nM vs target DA 的对比
# ==============================================================================

def run_simulation_d1_d2_ckpt(duration: float = None, target_da: float = None, device: torch.device = None):
    """
    D1 + D2 receptor kinetics simulation for checkpoint generation.
    Batch 0 = Control (0 nM, no DA), Batch 1 = Experiment (target_da nM)

    The plot shows the difference between no-DA and the target DA.
    The final_state from Batch 1 is saved as checkpoint for --resume.
    """
    if device is None:
        device = config.DEVICE

    if duration is None:
        duration = 100000.0
    if target_da is None:
        target_da = config.DA_BASELINE

    dt = config.DT
    da_onset = config.DEFAULT_DA_ONSET

    print(f"🚀 Simulation running on {device}")
    print(f"   Mode: D1+D2 Kinetics (Checkpoint)")
    print(f"   Batch 0 (Control): 0 nM, Batch 1 (Exp): {target_da} nM")
    print(f"   Duration: {duration}ms, DA onset: {da_onset}ms")

    W_t, mask_d1, mask_d2, groups_info = _init_network(device)
    record_indices = _build_record_indices(groups_info, device, full=True)

    t0 = time.time()
    kp = config.build_kernel_params(device)
    all_spikes, v_traces, final_state = _run_kernel_with_progress(
        run_dynamic_d1_d2_kernel_ckpt,
        (W_t, mask_d1, mask_d2,
         float(target_da), float(da_onset), float(duration), dt,
         record_indices, config.N_E, kp),
        duration, dt,
    )
    _sync_and_report(t0)

    return _pack_data(
        cfg_dict={
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': duration, 'dt': dt,
            'da_onset': da_onset, 'da_level': target_da,
            'control_da': 0.0,
            'mode': 'dynamic_d1_d2_ckpt',
        },
        mask_d1=mask_d1, mask_d2=mask_d2, groups_info=groups_info,
        spikes=all_spikes, v_traces=v_traces, record_indices=record_indices,
        final_state=final_state,
    )


# ==============================================================================
# Runner 5: D1 + D2 受体动力学 — 两阶段给药
#   模拟静息态 DA 浓度下的网络稳态, 然后突然施加新的 DA 浓度
# ==============================================================================

def run_simulation_d1_d2_two_stage(
    duration: float = None,
    da_level_1: float = 2.0,
    da_level_2: float = 15.0,
    phase2_onset: float = None,
    device: torch.device = None,
):
    """
    Two-stage DA dosing simulation:
      Phase 1: [0, da_onset)            → 2 nM (baseline DA)
      Phase 2: [da_onset, phase2_onset) → da_level_1 nM (resting-state DA)
      Phase 3: [phase2_onset, end)      → da_level_2 nM (DA challenge)

    Args:
        duration     : Total simulation time (ms). Default 210s.
        da_level_1   : Resting-state DA concentration (nM). Default 2.0.
        da_level_2   : Challenge DA concentration (nM). Default 15.0.
        phase2_onset : Time (ms) when DA switches from level_1 to level_2.
                       Default = da_onset + 100000 (100s after first DA).
        device       : Torch device.
    """
    if device is None:
        device = config.DEVICE

    dt = config.DT
    da_onset = config.DEFAULT_DA_ONSET  # 10000 ms = 10s

    # Default phase2_onset: 20s after da_onset (let resting DA stabilize)
    if phase2_onset is None:
        phase2_onset = da_onset + 20000.0  # 30s mark

    # Default duration: phase2_onset + 100s (observe DA challenge for 100s)
    if duration is None:
        duration = phase2_onset + 100000.0  # 130s total

    print(f"🚀 Simulation running on {device}")
    print(f"   Mode: Two-Stage DA Dosing (D1 + D2 Kinetics)")
    print(f"   Phase 1: [0, {da_onset:.0f}ms) → 2 nM (baseline DA)")
    print(f"   Phase 2: [{da_onset:.0f}ms, {phase2_onset:.0f}ms) → {da_level_1} nM (resting DA)")
    print(f"   Phase 3: [{phase2_onset:.0f}ms, {duration:.0f}ms) → {da_level_2} nM (DA challenge)")
    print(f"   Total duration: {duration/1000:.1f}s")

    W_t, mask_d1, mask_d2, groups_info = _init_network(device)
    record_indices = _build_record_indices(groups_info, device, full=True)

    t0 = time.time()
    kp = config.build_kernel_params(device)
    all_spikes, v_traces, final_state = _run_kernel_with_progress(
        run_dynamic_d1_d2_kernel_two_stage,
        (W_t, mask_d1, mask_d2,
         float(da_level_1), float(da_level_2),
         float(da_onset), float(phase2_onset),
         float(duration), dt,
         record_indices, config.N_E, kp),
        duration, dt,
    )
    _sync_and_report(t0)

    return _pack_data(
        cfg_dict={
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': duration, 'dt': dt,
            'da_onset': phase2_onset,     # Analyzer uses this as the split point
            'da_level': da_level_2,       # primary DA level for analysis
            'control_da': config.DA_BASELINE,  # Control batch DA
            'da_level_1': da_level_1,     # resting-state DA
            'da_level_2': da_level_2,     # challenge DA
            'phase1_da_onset': da_onset,  # when resting DA starts
            'phase2_onset': phase2_onset, # when DA challenge starts
            'mode': 'dynamic_d1_d2_two_stage',
        },
        mask_d1=mask_d1, mask_d2=mask_d2, groups_info=groups_info,
        spikes=all_spikes, v_traces=v_traces, record_indices=record_indices,
        final_state=final_state,
    )


# ==============================================================================
# Runner 6: 从 checkpoint 恢复仿真 (方案二)
#   加载之前仿真保存的 raw_data.pkl 中的 final_state,
#   以此作为初始状态继续仿真, 施加新的 DA 浓度。
# ==============================================================================

def run_simulation_from_checkpoint(
    checkpoint_path: str,
    duration: float = None,
    da_level: float = 15.0,
    da_onset: float = None,
    device: torch.device = None,
):
    """
    Resume simulation from a checkpoint (raw_data.pkl).

    Loads the final_state from a previous simulation and continues
    with a new DA concentration.

    Args:
        checkpoint_path : Path to the raw_data.pkl file from a previous run.
        duration        : Simulation time (ms) for this continuation. Default 100s.
        da_level        : New DA concentration (nM). Default 15.0.
        da_onset        : Time (ms) when new DA starts (relative to t=0 of this run).
                          Default = DEFAULT_DA_ONSET (10s).
        device          : Torch device.
    """
    import pickle

    if device is None:
        device = config.DEVICE

    # Load checkpoint
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        ckpt_data = pickle.load(f)

    if 'final_state' not in ckpt_data:
        raise ValueError(
            "Checkpoint does not contain 'final_state'. "
            "Please re-run the source simulation with the updated code to save final_state."
        )

    # Verify parameter fingerprint — abort if mismatch
    verify_checkpoint_fingerprint(ckpt_data, checkpoint_path)

    ckpt_cfg = ckpt_data['config']
    prev_da = ckpt_cfg.get('da_level', 'unknown')
    prev_mode = ckpt_cfg.get('mode', 'unknown')
    prev_duration = ckpt_cfg.get('duration', 'unknown')

    print(f"   Previous run: mode={prev_mode}, DA={prev_da}nM, duration={prev_duration}ms")

    # Defaults
    if duration is None:
        duration = 100000.0  # 100s
    if da_onset is None:
        da_onset = config.DEFAULT_DA_ONSET  # 10s

    dt = config.DT

    print(f"🚀 Simulation running on {device}")
    print(f"   Mode: Resume from Checkpoint")
    print(f"   Loaded state from: {checkpoint_path}")
    print(f"   Previous DA: {prev_da}nM → New DA: {da_level}nM")
    print(f"   Baseline: [0, {da_onset:.0f}ms) → prev DA state (loaded, no change)")
    print(f"   DA phase: [{da_onset:.0f}ms, {duration:.0f}ms) → {da_level} nM")
    print(f"   Duration: {duration/1000:.1f}s")

    # Rebuild network (same seed → same W_t)
    W_t, mask_d1, mask_d2, groups_info = _init_network(device)
    record_indices = _build_record_indices(groups_info, device, full=True)

    # Move checkpoint state to device
    init_state = ckpt_data['final_state'].to(device)

    # ── Fix: Both batches must resume from the DA-steady-state ──
    # The ckpt kernel uses Batch 0 = 0 nM (pure control) and Batch 1 = target DA.
    # For resume, we need BOTH batches to start from the DA steady-state,
    # so copy Batch 1's state into Batch 0.
    if init_state.shape[0] >= 2:
        print(f"   ⚙️  Overwriting Batch 0 state with Batch 1 (DA steady-state)")
        init_state[0] = init_state[1].clone()

    t0 = time.time()
    kp = config.build_kernel_params(device)
    all_spikes, v_traces, final_state = _run_kernel_with_progress(
        run_dynamic_d1_d2_kernel_from_state,
        (W_t, mask_d1, mask_d2, init_state,
         float(da_level), float(da_onset), float(duration), dt,
         record_indices, config.N_E, kp),
        duration, dt,
    )
    _sync_and_report(t0)

    return _pack_data(
        cfg_dict={
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': duration, 'dt': dt,
            'da_onset': da_onset, 'da_level': da_level,
            'control_da': config.DA_BASELINE,
            'mode': 'resume_from_checkpoint',
            'checkpoint_path': checkpoint_path,
            'prev_da': prev_da,
            'prev_mode': prev_mode,
        },
        mask_d1=mask_d1, mask_d2=mask_d2, groups_info=groups_info,
        spikes=all_spikes, v_traces=v_traces, record_indices=record_indices,
        final_state=final_state,
    )


# ==============================================================================
# Runner WM: Working-Memory task — resume from baseline checkpoint
#   Uses `create_wm_network` (Scheme A: structured selective sub-pools).
#   Both batches share the same network and resume from the same DA-baseline
#   checkpoint (e.g. 2 nM steady state).  Cue is injected into Mem-A, the
#   network must maintain elevated Mem-A activity throughout the delay.
# ==============================================================================

def run_wm_simulation_from_checkpoint(
    checkpoint_path: str,
    duration: float = None,
    da_base: float = None,
    da_pulse: float = None,
    da_pulse_onset: float = None,
    da_pulse_offset: float = None,
    cue_a_onset: float = None,
    cue_a_offset: float = None,
    cue_a_amplitude: float = None,
    cue_b_onset: float = None,
    cue_b_offset: float = None,
    cue_b_amplitude: float = 0.0,
    pool_size: int = None,
    intra_prob: float = None,
    intra_w: float = None,
    use_wta: bool = None,
    iwm_size: int = None,
    e2i_prob: float = None,
    e2i_w: float = None,
    i2e_prob: float = None,
    i2e_w: float = None,
    mem_bg_offset: float = None,
    sfa_b: float = None,
    sfa_tau: float = None,
    da_ramp_up_ms: float = 0.0,
    da_ramp_down_ms: float = 0.0,
    block_d1: bool = False,
    block_d2: bool = False,
    device: torch.device = None,
):
    """
    Working-Memory task simulation resuming from a baseline checkpoint.

    Default protocol (units: ms, relative to t=0 of the WM run):
        baseline   : [0, 5000)
        cue (A)    : [5000, 6000)            amplitude = WM_CUE_AMPLITUDE
        delay      : [6000, 11000)            no input
        probe      : [11000, 11500)
        DA         : held at da_base throughout (set da_pulse if needed)

    Returns the standard data dict (compatible with PFCAnalyzer) augmented
    with the WM groups_info (mem_a/mem_b indices) and protocol metadata.
    """
    import pickle
    from models.network import create_wm_network, build_wm_stim_mask

    if device is None:
        device = config.DEVICE

    # --- Load checkpoint ---
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        ckpt_data = pickle.load(f)
    if 'final_state' not in ckpt_data:
        raise ValueError("Checkpoint missing 'final_state'.")
    verify_checkpoint_fingerprint(ckpt_data, checkpoint_path)

    ckpt_cfg = ckpt_data['config']
    prev_da   = ckpt_cfg.get('da_level', 'unknown')
    prev_mode = ckpt_cfg.get('mode',     'unknown')

    # --- Resolve defaults from config ---
    if da_base is None:
        da_base = config.DA_BASELINE
    if da_pulse is None:
        # By default no DA pulse → keep tone constant at da_base
        da_pulse = da_base
    if da_pulse_onset is None:
        da_pulse_onset = 0.0
    if da_pulse_offset is None:
        # zero-length window ⇒ pulse never fires
        da_pulse_offset = da_pulse_onset

    if cue_a_onset is None:
        cue_a_onset = config.WM_BASELINE_MS
    if cue_a_offset is None:
        cue_a_offset = cue_a_onset + config.WM_CUE_DURATION_MS
    if cue_a_amplitude is None:
        cue_a_amplitude = config.WM_CUE_AMPLITUDE

    if cue_b_onset is None:
        cue_b_onset = 0.0
    if cue_b_offset is None:
        cue_b_offset = 0.0  # disabled by default

    if duration is None:
        # baseline + cue + delay + probe
        duration = (config.WM_BASELINE_MS + config.WM_CUE_DURATION_MS
                    + config.WM_DELAY_MS + config.WM_PROBE_MS)

    dt = config.DT

    # --- Build WM network (uses same RANDOM_SEED → ckpt fingerprint OK) ---
    print(f"🚀 WM simulation running on {device}")
    print(f"   Mode: Working-Memory (Scheme A, resume from ckpt)")
    print(f"   DA: base={da_base}nM, pulse={da_pulse}nM "
          f"(window=[{da_pulse_onset:.0f}ms, {da_pulse_offset:.0f}ms))")
    print(f"   Cue A: [{cue_a_onset:.0f}ms, {cue_a_offset:.0f}ms) "
          f"amp={cue_a_amplitude:.1f}pA")
    print(f"   Cue B: [{cue_b_onset:.0f}ms, {cue_b_offset:.0f}ms) "
          f"amp={cue_b_amplitude:.1f}pA")
    print(f"   Duration: {duration/1000:.2f}s")

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    W_t, mask_d1, mask_d2, groups_info = create_wm_network(
        config.N_E, config.N_I, device,
        pool_size=pool_size, intra_prob=intra_prob, intra_w=intra_w,
        use_wta=use_wta, iwm_size=iwm_size,
        e2i_prob=e2i_prob, e2i_w=e2i_w,
        i2e_prob=i2e_prob, i2e_w=i2e_w,
    )

    if groups_info.get('use_wta', False):
        print(f"   WTA: I-WM=[{groups_info['iwm_start']},{groups_info['iwm_end']}) "
              f"size={groups_info['iwm_size']}, "
              f"E->I p={groups_info['wm_e2i_prob']:.2f}/w={groups_info['wm_e2i_w']:.1f}pA, "
              f"I->E p={groups_info['wm_i2e_prob']:.2f}/w={groups_info['wm_i2e_w']:.1f}pA")
    else:
        print(f"   WTA: disabled (no shared-inhibition loop)")

    # --- Build stim masks for cue-A and cue-B injections ---
    stim_mask_a = build_wm_stim_mask(groups_info, 'A', config.N_TOTAL, device)
    stim_mask_b = build_wm_stim_mask(groups_info, 'B', config.N_TOTAL, device)

    # --- Record indices: monitor one neuron per major sub-population ---
    # NOTE: With Mem-A/B placed in E-D1 segment, mem_a_start = 0 == old
    # target_d1.  We therefore pick the D1 monitor from the D1-BG region
    # (i.e. the slice of E-D1 *outside* both memory pools), so that the
    # "D1" trace in plots reflects pure DA-D1 background dynamics rather
    # than memory-pool persistent activity.
    target_d1   = groups_info['mem_b_end']        # first D1-BG neuron
    target_d2   = groups_info['e_d1_end']         # first E-D2 neuron
    target_memA = groups_info['mem_a_start']
    target_memB = groups_info['mem_b_start']
    record_indices = torch.tensor(
        [
            [0, target_d1],   [1, target_d1],
            [0, target_d2],   [1, target_d2],
            [0, target_memA], [1, target_memA],
            [0, target_memB], [1, target_memB],
        ],
        device=device, dtype=torch.long,
    )

    # --- Move ckpt state to device; copy DA-steady state to both batches ---
    init_state = ckpt_data['final_state'].to(device)
    if init_state.shape[0] >= 2:
        # Both batches resume from the same DA-baseline state.
        # In ckpt-mode the DA-equilibrated state is in Batch 1; copy to Batch 0.
        init_state[0] = init_state[1].clone()

    # ── (Optional) Override alpha_D1 / alpha_D2 with their Langmuir
    #    steady-state values evaluated at DA = da_base.
    #    This bypasses the slow τ_on_D1 ≈ 31 s climb so that the trial
    #    starts in true receptor steady-state — required when probing the
    #    inverted-U on a 12 s trial (otherwise alpha barely climbs).
    #    Enable via env-var WM_ALPHA_INIT_STEADY=1.
    #    State layout: alpha_d1 = init_state[:, 3N:3N+1]
    #                  alpha_d2 = init_state[:, 3N+1:3N+2]
    if os.environ.get('WM_ALPHA_INIT_STEADY', '0') == '1':
        N = config.N_TOTAL
        beta = float(config.BETA)
        ec50_d1 = float(config.EC50_D1)
        ec50_d2 = float(config.EC50_D2)
        # Langmuir steady state: alpha_ss(DA) = 1 / (1 + exp(-beta*(DA - EC50)))
        import math as _m
        alpha_d1_ss = 1.0 / (1.0 + _m.exp(-beta * (float(da_base) - ec50_d1)))
        alpha_d2_ss = 1.0 / (1.0 + _m.exp(-beta * (float(da_base) - ec50_d2)))
        init_state[:, 3 * N:3 * N + 1] = alpha_d1_ss
        init_state[:, 3 * N + 1:3 * N + 2] = alpha_d2_ss
        print(f"   🔬 [alpha-init-steady] DA={da_base} nM → "
              f"α_D1_ss={alpha_d1_ss:.3f}, α_D2_ss={alpha_d2_ss:.3f} "
              f"(both batches; alpha will still evolve dynamically post-cue)")

    alpha_record_interval = max(1, int(50.0 / dt))  # ~50 ms

    # --- Pull dual-channel weight matrices out of groups_info ---
    # `create_wm_network` already built `W_ampa_t` (everything except
    # intra-pool E→E) and `W_nmda_t` (only Mem-A↔A & Mem-B↔B).
    # The dual kernel needs both; W_t (sum) is no longer used here.
    W_ampa_t = groups_info['W_ampa_t']
    W_nmda_t = groups_info['W_nmda_t']

    # --- Build Mem-pool mask for Scheme-B baseline-bias offset ---
    # Mark every neuron belonging to Mem-A ∪ Mem-B with 1.0; everyone else
    # gets 0.0.  The kernel will then add `mem_bg_offset` * mask to I_bg
    # only on memory-pool neurons (not on E-BG, I, D1, D2 etc.).
    mem_pool_mask = torch.zeros(config.N_TOTAL, device=device)
    mem_pool_mask[groups_info['mem_a_start']:groups_info['mem_a_end']] = 1.0
    mem_pool_mask[groups_info['mem_b_start']:groups_info['mem_b_end']] = 1.0
    # Resolve the offset value from config (negative pA expected).
    _mem_bg_offset = float(getattr(config, 'WM_MEM_BG_OFFSET', 0.0))
    if mem_bg_offset is not None:
        _mem_bg_offset = float(mem_bg_offset)
    print(f"   Mem-pool BG offset: {_mem_bg_offset:+.1f} pA "
          f"(applied only to Mem-A∪Mem-B; ΔV_inf = {0.1 * _mem_bg_offset:+.2f} mV)")

    # --- SFA (spike-frequency adaptation) parameters ---
    _sfa_b = float(getattr(config, 'WM_SFA_B', 0.0))
    _sfa_tau = float(getattr(config, 'WM_SFA_TAU', 1500.0))
    if sfa_b is not None:
        _sfa_b = float(sfa_b)
    if sfa_tau is not None:
        _sfa_tau = float(sfa_tau)
    print(f"   SFA: b={_sfa_b:.1f} pA/spike, τ={_sfa_tau:.0f} ms"
          f"  ({'ENABLED' if _sfa_b > 0 else 'disabled'})")

    t0 = time.time()
    kp = config.build_kernel_params(device)
    # ── Pharmacological pathway block (D1 / D2 receptor antagonism) ──
    # Implemented at the parameter-tensor level so the JIT-scripted kernel
    # itself stays untouched.  Setting a pathway's modulation coefficients
    # to zero (EPS, BIAS, LAM) eliminates that receptor's effect on R_eff,
    # I_mod, and scale_syn — equivalent to a saturating antagonist.
    # NOTE: alpha_D1/D2 themselves still evolve normally (Langmuir kinetics)
    # but multiplying them by zero coefficients makes them inert.  We also
    # clone the tensor first so subsequent runs (and config.build_kernel_params)
    # are unaffected.
    if block_d1 or block_d2:
        kp = kp.clone()
        if block_d1:
            # params layout: [13] EPS_D1, [15] BIAS_D1, [17] LAM_D1
            kp[13] = 0.0; kp[15] = 0.0; kp[17] = 0.0
            print("   🚫 D1 pathway BLOCKED (EPS_D1=BIAS_D1=LAM_D1=0)")
        if block_d2:
            # params layout: [14] EPS_D2, [16] BIAS_D2, [18] LAM_D2
            kp[14] = 0.0; kp[16] = 0.0; kp[18] = 0.0
            print("   🚫 D2 pathway BLOCKED (EPS_D2=BIAS_D2=LAM_D2=0)")
    all_spikes, v_traces, final_state, alpha_d1_trace, alpha_d2_trace = (
        _run_kernel_with_progress(
            run_wm_kernel_dual,
            (W_ampa_t, W_nmda_t, mask_d1, mask_d2, init_state,
             float(da_base), float(da_pulse),
             float(da_pulse_onset), float(da_pulse_offset),
             stim_mask_a, stim_mask_b,
             float(cue_a_onset), float(cue_a_offset), float(cue_a_amplitude),
             float(cue_b_onset), float(cue_b_offset), float(cue_b_amplitude),
             float(duration), dt,
             record_indices, config.N_E,
             alpha_record_interval, kp,
             mem_pool_mask, _mem_bg_offset,
             _sfa_b, _sfa_tau,
             float(da_ramp_up_ms), float(da_ramp_down_ms)),
            duration, dt,
        )
    )
    _sync_and_report(t0)

    # --- Pack data dict (PFCAnalyzer-compatible) ---
    cfg_dict = {
        'N_E': config.N_E, 'N_I': config.N_I,
        'duration': duration, 'dt': dt,
        # PFCAnalyzer uses da_onset to split baseline/post-DA windows.
        # We co-opt it as "cue onset" so the existing plot/zoom logic works
        # nicely (baseline before cue, post-cue afterwards).
        'da_onset': cue_a_onset,
        'da_level': da_base,        # display label
        'control_da': da_base,
        'mode': 'working_memory',
        'checkpoint_path': checkpoint_path,
        'prev_da': prev_da,
        'prev_mode': prev_mode,
        'wm_protocol': {
            'cue_a_onset': cue_a_onset,
            'cue_a_offset': cue_a_offset,
            'cue_a_amplitude': cue_a_amplitude,
            'cue_b_onset': cue_b_onset,
            'cue_b_offset': cue_b_offset,
            'cue_b_amplitude': cue_b_amplitude,
            'da_pulse_onset': da_pulse_onset,
            'da_pulse_offset': da_pulse_offset,
            'da_base': da_base,
            'da_pulse': da_pulse,
            'da_ramp_up_ms': da_ramp_up_ms,
            'da_ramp_down_ms': da_ramp_down_ms,
            'baseline_ms': config.WM_BASELINE_MS,
            'cue_ms': config.WM_CUE_DURATION_MS,
            'delay_ms': config.WM_DELAY_MS,
            'probe_ms': config.WM_PROBE_MS,
            # WTA structural metadata (mirrors groups_info)
            'use_wta':       groups_info.get('use_wta', False),
            'iwm_size':      groups_info.get('iwm_size', 0),
            'wm_intra_prob': groups_info.get('wm_intra_prob', None),
            'wm_intra_w':    groups_info.get('wm_intra_w', None),
            'wm_e2i_prob':   groups_info.get('wm_e2i_prob', None),
            'wm_e2i_w':      groups_info.get('wm_e2i_w', None),
            'wm_i2e_prob':   groups_info.get('wm_i2e_prob', None),
            'wm_i2e_w':      groups_info.get('wm_i2e_w', None),
        },
    }

    # Strip the (large) NMDA/AMPA weight tensors from groups_info before
    # packing — they're only needed by the kernel call above and would
    # otherwise bloat every saved pkl by ~8 MB each.
    groups_info_clean = {k: v for k, v in groups_info.items()
                         if k not in ('W_ampa_t', 'W_nmda_t')}

    # Store alpha-record interval in cfg so post-hoc plots can reconstruct
    # the time axis of alpha_d1_trace / alpha_d2_trace.
    cfg_dict['alpha_record_interval'] = alpha_record_interval

    data = _pack_data(
        cfg_dict=cfg_dict,
        mask_d1=mask_d1, mask_d2=mask_d2, groups_info=groups_info_clean,
        spikes=all_spikes, v_traces=v_traces, record_indices=record_indices,
        final_state=final_state,
    )
    # Attach receptor-occupancy traces (recorded by the kernel) so the
    # DA-scan diagnostic figure can plot α_D1(t) and α_D2(t).
    data['alpha_d1_trace'] = alpha_d1_trace.cpu()
    data['alpha_d2_trace'] = alpha_d2_trace.cpu()
    return data
