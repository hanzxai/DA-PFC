#!/usr/bin/env python3
"""
DA Waveform Experiment — Test network response to various DA input patterns.

All modes resume from a checkpoint (DA=2nM steady-state) and apply different
DA waveforms to Batch 1 (Experiment), while Batch 0 (Control) stays at baseline.

Supported modes:
  spike     : Single upward spike (transient DA burst)
  dip       : Single downward dip (transient DA withdrawal)
  sine      : Sinusoidal oscillation around baseline
  burst     : Repeated short pulses (phasic DA firing pattern)
  ramp_up   : Gradual linear increase from baseline to peak
  ramp_down : Gradual linear decrease from baseline to trough
  square    : Square wave oscillation

Usage:
  # Run a single mode:
  python experiments/da_waveform_exp.py --mode spike

  # Run all modes:
  python experiments/da_waveform_exp.py --mode all

  # Custom parameters:
  python experiments/da_waveform_exp.py --mode sine --amplitude 5 --freq 0.1 --duration 100

  # Skip checkpoint generation (reuse existing):
  python experiments/da_waveform_exp.py --mode all --skip-ckpt

  # Specify GPU:
  python experiments/da_waveform_exp.py --mode all --gpu 0
"""
import sys
import os
import argparse
import time
import pickle
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.network import create_network_structure
from models.kernels import run_dynamic_d1_d2_kernel_da_schedule
from simulation.utils import verify_checkpoint_fingerprint
from analysis.analyzer import PFCAnalyzer
from analysis.plotting import (plot_combined_raster,
                                plot_combined_rates_all,
                                plot_combined_rates_E,
                                plot_combined_rates_I)


# ==============================================================================
# DA Waveform Generators
# ==============================================================================

def generate_da_schedule(mode: str, steps: int, dt: float,
                         da_base: float = 2.0,
                         amplitude: float = 10.0,
                         freq_hz: float = 0.05,
                         onset_ms: float = 10000.0,
                         pulse_width_ms: float = 2000.0,
                         burst_interval_ms: float = 5000.0,
                         n_bursts: int = 5,
                         ramp_duration_ms: float = 30000.0,
                         square_period_ms: float = 20000.0,
                         ) -> np.ndarray:
    """
    Generate DA schedule array of shape (steps, 2).
      Column 0 = Batch 0 (Control): constant da_base
      Column 1 = Batch 1 (Experiment): waveform-dependent

    Args:
        mode            : Waveform type
        steps           : Number of simulation timesteps
        dt              : Timestep in ms
        da_base         : Baseline DA concentration (nM)
        amplitude       : Peak deviation from baseline (nM)
        freq_hz         : Frequency for sine mode (Hz)
        onset_ms        : When the waveform starts (ms)
        pulse_width_ms  : Width of each pulse/spike/dip (ms)
        burst_interval_ms: Interval between burst pulses (ms)
        n_bursts        : Number of burst pulses
        ramp_duration_ms: Duration of ramp (ms)
        square_period_ms: Period of square wave (ms)

    Returns:
        da_schedule: (steps, 2) numpy array
    """
    t_ms = np.arange(steps) * dt  # time array in ms
    da_ctrl = np.full(steps, da_base)
    da_exp = np.full(steps, da_base)

    onset_idx = int(onset_ms / dt)

    if mode == 'spike':
        # Single upward spike: baseline → (baseline + amplitude) → baseline
        end_ms = onset_ms + pulse_width_ms
        mask = (t_ms >= onset_ms) & (t_ms < end_ms)
        da_exp[mask] = da_base + amplitude

    elif mode == 'dip':
        # Single downward dip: baseline → (baseline - amplitude) → baseline
        end_ms = onset_ms + pulse_width_ms
        mask = (t_ms >= onset_ms) & (t_ms < end_ms)
        da_exp[mask] = max(0.1, da_base - amplitude)

    elif mode == 'sine':
        # Sinusoidal oscillation starting at onset
        freq_ms = freq_hz / 1000.0  # convert Hz to ms^-1
        mask = t_ms >= onset_ms
        phase = 2.0 * np.pi * freq_ms * (t_ms[mask] - onset_ms)
        da_exp[mask] = da_base + amplitude * np.sin(phase)

    elif mode == 'burst':
        # Repeated short pulses (phasic DA pattern)
        for b in range(n_bursts):
            pulse_start = onset_ms + b * burst_interval_ms
            pulse_end = pulse_start + pulse_width_ms
            mask = (t_ms >= pulse_start) & (t_ms < pulse_end)
            da_exp[mask] = da_base + amplitude

    elif mode == 'ramp_up':
        # Linear ramp from baseline to (baseline + amplitude)
        ramp_end = onset_ms + ramp_duration_ms
        mask = (t_ms >= onset_ms) & (t_ms < ramp_end)
        progress = (t_ms[mask] - onset_ms) / ramp_duration_ms
        da_exp[mask] = da_base + amplitude * progress
        # Hold at peak after ramp
        mask_hold = t_ms >= ramp_end
        da_exp[mask_hold] = da_base + amplitude

    elif mode == 'ramp_down':
        # Linear ramp from baseline to (baseline - amplitude)
        ramp_end = onset_ms + ramp_duration_ms
        mask = (t_ms >= onset_ms) & (t_ms < ramp_end)
        progress = (t_ms[mask] - onset_ms) / ramp_duration_ms
        da_exp[mask] = da_base - amplitude * progress
        # Hold at trough after ramp
        mask_hold = t_ms >= ramp_end
        da_exp[mask_hold] = max(0.1, da_base - amplitude)

    elif mode == 'square':
        # Square wave oscillation
        mask = t_ms >= onset_ms
        period_steps = int(square_period_ms / dt)
        half_period = period_steps // 2
        for i in range(steps):
            if t_ms[i] >= onset_ms:
                phase_idx = int((t_ms[i] - onset_ms) / dt) % period_steps
                if phase_idx < half_period:
                    da_exp[i] = da_base + amplitude
                else:
                    da_exp[i] = max(0.1, da_base - amplitude)

    else:
        raise ValueError(f"Unknown DA waveform mode: {mode}")

    # Clamp DA >= 0.1 nM (physiological minimum)
    da_exp = np.clip(da_exp, 0.1, None)

    schedule = np.stack([da_ctrl, da_exp], axis=1)  # (steps, 2)
    return schedule


# ==============================================================================
# Simulation Runner
# ==============================================================================

def run_waveform_experiment(
    mode: str,
    checkpoint_path: str,
    duration_s: float = 100.0,
    da_base: float = 2.0,
    amplitude: float = 10.0,
    freq_hz: float = 0.05,
    onset_s: float = 10.0,
    pulse_width_s: float = 2.0,
    burst_interval_s: float = 5.0,
    n_bursts: int = 5,
    ramp_duration_s: float = 30.0,
    square_period_s: float = 20.0,
    gpu: int = 0,
):
    """Run a single waveform experiment and return data dict."""

    # Device
    if torch.cuda.is_available() and 0 <= gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
    print(f"🔧 Using device: {device}")

    # Load checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        ckpt_data = pickle.load(f)
    if 'final_state' not in ckpt_data:
        raise ValueError("Checkpoint missing 'final_state'.")
    verify_checkpoint_fingerprint(ckpt_data, checkpoint_path)

    # Convert to ms
    duration_ms = duration_s * 1000.0
    onset_ms = onset_s * 1000.0
    pulse_width_ms = pulse_width_s * 1000.0
    burst_interval_ms = burst_interval_s * 1000.0
    ramp_duration_ms = ramp_duration_s * 1000.0
    square_period_ms = square_period_s * 1000.0
    dt = config.DT
    steps = int(duration_ms / dt)

    # Generate DA schedule
    print(f"📐 Generating DA schedule: mode={mode}, amplitude={amplitude}nM, "
          f"onset={onset_s}s, duration={duration_s}s")
    da_schedule_np = generate_da_schedule(
        mode=mode, steps=steps, dt=dt,
        da_base=da_base, amplitude=amplitude,
        freq_hz=freq_hz, onset_ms=onset_ms,
        pulse_width_ms=pulse_width_ms,
        burst_interval_ms=burst_interval_ms,
        n_bursts=n_bursts,
        ramp_duration_ms=ramp_duration_ms,
        square_period_ms=square_period_ms,
    )
    da_schedule = torch.tensor(da_schedule_np, dtype=torch.float32, device=device)

    # Build network
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(
        config.N_E, config.N_I, device)

    # Record indices (4 points: Ctrl/Exp × D1/D2)
    target_d1 = 0
    target_d2 = groups_info['e_d1_end']
    record_indices = torch.tensor([
        [0, target_d1], [1, target_d1],
        [0, target_d2], [1, target_d2],
    ], device=device, dtype=torch.long)

    # Prepare init_state: both batches from Batch 1 (DA steady-state)
    init_state = ckpt_data['final_state'].to(device)
    if init_state.shape[0] >= 2:
        init_state[0] = init_state[1].clone()

    # Run kernel
    kp = config.build_kernel_params(device)
    alpha_interval = 100  # record alpha every 100 steps

    print(f"🚀 Running simulation: {mode} mode, {duration_s}s...")
    t0 = time.time()
    result = run_dynamic_d1_d2_kernel_da_schedule(
        W_t, mask_d1, mask_d2, init_state,
        da_schedule,
        float(duration_ms), dt,
        record_indices, config.N_E,
        alpha_interval, kp,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"✅ Simulation done in {elapsed:.2f}s")

    all_spikes, v_traces, final_state, alpha_d1_trace, alpha_d2_trace = result

    # Pack data (compatible with PFCAnalyzer)
    data = {
        'config': {
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': duration_ms, 'dt': dt,
            'da_onset': onset_ms,
            'da_level': da_base + amplitude,  # peak DA for labeling
            'control_da': da_base,
            'mode': f'da_waveform_{mode}',
            'waveform_mode': mode,
            'da_base': da_base,
            'amplitude': amplitude,
        },
        'masks': {'d1': mask_d1.cpu(), 'd2': mask_d2.cpu()},
        'groups_info': groups_info,
        'spikes': all_spikes.cpu(),
        'v_traces': v_traces.cpu(),
        'record_indices': record_indices.cpu(),
        'final_state': final_state.cpu(),
        'da_schedule': da_schedule_np,  # save for plotting
        'alpha_d1_trace': alpha_d1_trace.cpu().numpy(),
        'alpha_d2_trace': alpha_d2_trace.cpu().numpy(),
    }
    return data


# ==============================================================================
# Custom Plotting: DA waveform + firing rates
# ==============================================================================

def plot_waveform_summary(data: dict, save_dir: Path):
    """
    Generate a summary figure for a waveform experiment:
      Row 0: DA concentration timeline (Batch 0 vs Batch 1)
      Row 1: Alpha D1 & D2 traces (Batch 0 vs Batch 1)
      Row 2: Firing rates (all populations, Batch 0)
      Row 3: Firing rates (all populations, Batch 1)
    """
    cfg = data['config']
    mode = cfg.get('waveform_mode', 'unknown')
    duration_ms = cfg['duration']
    dt = cfg['dt']
    da_onset = cfg['da_onset']
    steps = int(duration_ms / dt)

    da_schedule = data['da_schedule']
    alpha_d1 = data['alpha_d1_trace']
    alpha_d2 = data['alpha_d2_trace']

    use_seconds = duration_ms > 10000
    scale = 1000.0 if use_seconds else 1.0
    x_label = "Time (s)" if use_seconds else "Time (ms)"

    t_da = np.arange(da_schedule.shape[0]) * dt / scale
    alpha_interval = 100
    t_alpha = np.arange(alpha_d1.shape[0]) * dt * alpha_interval / scale

    fig, axes = plt.subplots(4, 1, figsize=(20, 24), dpi=150,
                              gridspec_kw={'height_ratios': [1, 1, 2, 2]})

    # --- Row 0: DA concentration ---
    ax = axes[0]
    ax.plot(t_da, da_schedule[:, 0], color='#2196F3', linewidth=2, label='Batch 0 (Control)', alpha=0.8)
    ax.plot(t_da, da_schedule[:, 1], color='#E91E63', linewidth=2, label='Batch 1 (Exp)', alpha=0.9)
    ax.fill_between(t_da, da_schedule[:, 0], da_schedule[:, 1], color='#E91E63', alpha=0.1)
    ax.set_ylabel("[DA] (nM)")
    ax.set_xlabel(x_label)
    ax.set_title(f"DA Waveform: {mode.upper()}")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(da_onset / scale, color='gray', linestyle='--', alpha=0.5)

    # --- Row 1: Alpha traces ---
    ax = axes[1]
    ax.plot(t_alpha, alpha_d1[:, 0], color='#2196F3', linewidth=1.5, linestyle='--',
            label='α_D1 Ctrl', alpha=0.7)
    ax.plot(t_alpha, alpha_d1[:, 1], color='#d62728', linewidth=2,
            label='α_D1 Exp', alpha=0.9)
    ax.plot(t_alpha, alpha_d2[:, 0], color='#2196F3', linewidth=1.5, linestyle=':',
            label='α_D2 Ctrl', alpha=0.7)
    ax.plot(t_alpha, alpha_d2[:, 1], color='#9467bd', linewidth=2,
            label='α_D2 Exp', alpha=0.9)
    ax.set_ylabel("Receptor Activation (α)")
    ax.set_xlabel(x_label)
    ax.set_title("Receptor Kinetics Response")
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.axvline(da_onset / scale, color='gray', linestyle='--', alpha=0.5)

    # --- Row 2 & 3: Firing rates ---
    analyzer = PFCAnalyzer(data)
    group_names = ['E-D1', 'E-D2', 'E-Other', 'I-D1', 'I-D2', 'I-Other']

    for row_idx, (batch_idx, batch_name) in enumerate([(0, 'Control'), (1, 'Experiment')]):
        ax = axes[2 + row_idx]
        for grp_name in group_names:
            if grp_name not in analyzer.groups:
                continue
            centers, rate = analyzer.compute_group_rate(batch_idx, grp_name, time_win=100.0)
            if rate is None or len(rate) == 0:
                continue
            x_data = centers / scale if use_seconds else centers
            color = PFCAnalyzer.COLORS.get(grp_name, 'k')
            lw = 2.0 if grp_name.startswith('E') else 1.5
            alpha = 0.85 if grp_name.startswith('E') else 0.65
            ax.plot(x_data, rate, color=color, label=grp_name, lw=lw, alpha=alpha)

        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_xlabel(x_label)
        ax.set_title(f"Firing Rates — {batch_name} (Batch {batch_idx})")
        ax.legend(loc='upper left', ncol=3)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.axvline(da_onset / scale, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = save_dir / f"waveform_summary_{mode}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Saved: {save_path}")


def plot_all_waveforms_comparison(all_schedules: dict, save_dir: Path, dt: float):
    """
    Plot all DA waveform modes side by side for visual comparison.
    """
    n_modes = len(all_schedules)
    if n_modes == 0:
        return

    fig, axes = plt.subplots(n_modes, 1, figsize=(18, 4 * n_modes), dpi=150)
    if n_modes == 1:
        axes = [axes]

    for idx, (mode_name, schedule) in enumerate(all_schedules.items()):
        ax = axes[idx]
        steps = schedule.shape[0]
        t_s = np.arange(steps) * dt / 1000.0
        ax.plot(t_s, schedule[:, 0], color='#2196F3', linewidth=1.5,
                label='Control', alpha=0.7)
        ax.plot(t_s, schedule[:, 1], color='#E91E63', linewidth=2,
                label='Experiment', alpha=0.9)
        ax.fill_between(t_s, schedule[:, 0], schedule[:, 1],
                         color='#E91E63', alpha=0.1)
        ax.set_ylabel("[DA] (nM)")
        ax.set_title(f"Mode: {mode_name.upper()}")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    save_path = save_dir / "all_waveforms_comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Saved: {save_path}")


# ==============================================================================
# Main
# ==============================================================================

ALL_MODES = ['spike', 'dip', 'sine', 'burst', 'ramp_up', 'ramp_down', 'square']

def parse_args():
    parser = argparse.ArgumentParser(
        description="DA Waveform Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", type=str, default="all",
                        choices=ALL_MODES + ['all'],
                        help="Waveform mode (default: all)")
    parser.add_argument("--duration", type=float, default=100.0,
                        help="Simulation duration in seconds (default: 100)")
    parser.add_argument("--da-base", type=float, default=2.0,
                        help="Baseline DA concentration in nM (default: 2.0)")
    parser.add_argument("--amplitude", type=float, default=10.0,
                        help="DA amplitude deviation from baseline in nM (default: 10.0)")
    parser.add_argument("--freq", type=float, default=0.05,
                        help="Sine frequency in Hz (default: 0.05)")
    parser.add_argument("--onset", type=float, default=10.0,
                        help="Waveform onset time in seconds (default: 10)")
    parser.add_argument("--pulse-width", type=float, default=2.0,
                        help="Pulse width in seconds (default: 2)")
    parser.add_argument("--burst-interval", type=float, default=5.0,
                        help="Interval between burst pulses in seconds (default: 5)")
    parser.add_argument("--n-bursts", type=int, default=5,
                        help="Number of burst pulses (default: 5)")
    parser.add_argument("--ramp-duration", type=float, default=30.0,
                        help="Ramp duration in seconds (default: 30)")
    parser.add_argument("--square-period", type=float, default=20.0,
                        help="Square wave period in seconds (default: 20)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU card number (default: 0)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path (auto-detected if not specified)")
    parser.add_argument("--skip-ckpt", action="store_true",
                        help="Skip checkpoint generation, reuse existing")
    parser.add_argument("--base-dur", type=float, default=500.0,
                        help="Baseline checkpoint duration in seconds (default: 500)")
    return parser.parse_args()


def find_or_create_checkpoint(args) -> str:
    """Find existing checkpoint or create one."""
    if args.ckpt:
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
        return args.ckpt

    # Auto-detect checkpoint
    bg_mean = config.BG_MEAN
    da_str = f"{args.da_base:g}"
    dur_str = f"{int(args.base_dur)}"
    bg_str = f"{bg_mean:g}"
    ckpt_path = PROJECT_ROOT / "checkpoints" / f"ckpt_DA{da_str}nM_bg{bg_str}_{dur_str}s.pkl"

    if ckpt_path.exists() and args.skip_ckpt:
        print(f"✅ Using existing checkpoint: {ckpt_path}")
        return str(ckpt_path)

    if ckpt_path.exists():
        print(f"✅ Found existing checkpoint: {ckpt_path}")
        return str(ckpt_path)

    # Need to generate checkpoint
    print(f"📦 Generating baseline checkpoint (DA={args.da_base}nM, {args.base_dur}s)...")
    import subprocess
    cmd = [
        sys.executable, str(PROJECT_ROOT / "main.py"),
        "--da", str(args.da_base),
        "--duration", str(args.base_dur),
        "--gpu", str(args.gpu),
        "--save-ckpt",
    ]
    print(f"   Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError("Checkpoint generation failed!")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {ckpt_path}")
    print(f"✅ Checkpoint saved: {ckpt_path}")
    return str(ckpt_path)


def main():
    args = parse_args()
    t_total_start = time.time()

    # Determine modes to run
    modes = ALL_MODES if args.mode == 'all' else [args.mode]

    # Setup output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    mode_tag = args.mode if args.mode != 'all' else "all_waveforms"
    exp_dir = PROJECT_ROOT / "outputs" / f"exp_{timestamp}_waveform_{mode_tag}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Find or create checkpoint
    ckpt_path = find_or_create_checkpoint(args)

    # Save experiment config
    exp_config = {
        'modes': modes,
        'duration_s': args.duration,
        'da_base': args.da_base,
        'amplitude': args.amplitude,
        'freq_hz': args.freq,
        'onset_s': args.onset,
        'pulse_width_s': args.pulse_width,
        'burst_interval_s': args.burst_interval,
        'n_bursts': args.n_bursts,
        'ramp_duration_s': args.ramp_duration,
        'square_period_s': args.square_period,
        'checkpoint': ckpt_path,
    }
    with open(exp_dir / "experiment_config.json", 'w') as f:
        json.dump(exp_config, f, indent=2)

    print("\n" + "=" * 60)
    print(f"🧪 DA Waveform Experiment")
    print(f"   Modes: {', '.join(modes)}")
    print(f"   Duration: {args.duration}s, Baseline DA: {args.da_base}nM")
    print(f"   Amplitude: {args.amplitude}nM")
    print(f"   Output: {exp_dir}")
    print("=" * 60 + "\n")

    all_schedules = {}

    for mode in modes:
        print(f"\n{'━' * 60}")
        print(f"🔬 Running mode: {mode.upper()}")
        print(f"{'━' * 60}")

        # Create mode subdirectory
        mode_dir = exp_dir / mode
        mode_dir.mkdir(exist_ok=True)

        # Run experiment
        data = run_waveform_experiment(
            mode=mode,
            checkpoint_path=ckpt_path,
            duration_s=args.duration,
            da_base=args.da_base,
            amplitude=args.amplitude,
            freq_hz=args.freq,
            onset_s=args.onset,
            pulse_width_s=args.pulse_width,
            burst_interval_s=args.burst_interval,
            n_bursts=args.n_bursts,
            ramp_duration_s=args.ramp_duration,
            square_period_s=args.square_period,
            gpu=args.gpu,
        )

        # Save raw data
        with open(mode_dir / "raw_data.pkl", 'wb') as f:
            pickle.dump(data, f)

        # Save DA schedule for comparison plot
        all_schedules[mode] = data['da_schedule']

        # Generate plots
        print(f"🎨 Generating plots for {mode}...")

        # 1. Custom waveform summary (DA + alpha + rates)
        plot_waveform_summary(data, mode_dir)

        # 2. Standard combined plots (reuse existing infrastructure)
        # Patch data config for plotting compatibility
        data['config']['mode'] = 'resume_from_checkpoint'
        analyzer = PFCAnalyzer(data)
        plot_combined_raster(analyzer, save_dir=mode_dir)
        plot_combined_rates_all(analyzer, save_dir=mode_dir)

        # Save analysis report
        analyzer.save_report(str(mode_dir / "analysis_report.txt"))

        print(f"✅ Mode {mode} complete → {mode_dir}")

    # Generate comparison plot of all waveforms
    if len(all_schedules) > 1:
        print(f"\n📊 Generating waveform comparison plot...")
        plot_all_waveforms_comparison(all_schedules, exp_dir, config.DT)

    # Summary
    t_total = time.time() - t_total_start
    print("\n" + "=" * 60)
    print(f"✅ All experiments complete!")
    print(f"   Total time: {t_total:.1f}s")
    print(f"   Results: {exp_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
