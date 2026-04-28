#!/usr/bin/env python3
"""
Experiment E: D1/D2 Tau Segregation — Temporal Filtering by Receptor Kinetics

Scientific Question:
  D1 receptors have τ_on ≈ 31s (slow), D2 receptors have τ_on ≈ 10s (fast, ~3x).
  How does this ~3x difference in receptor kinetics create temporal filtering
  that differentially shapes network responses to DA signals of varying timescales?

Key Predictions:
  1. Short DA pulses (< 10s): D2 activates substantially, D1 barely responds
     → Network effect dominated by D2 (inhibitory bias)
  2. Long DA pulses (> 60s): Both D1 and D2 approach steady-state
     → Full D1+D2 interaction emerges
  3. Fast pulse trains: D2 tracks individual pulses, D1 integrates (low-pass)
     → D1 sees "average DA", D2 sees "pulsatile DA"
  4. After DA withdrawal: D1 decays much slower than D2
     → "D1 afterglow window" where D1 > D2

Sub-experiments:
  A. Pulse Duration Scan: 5s, 15s, 30s, 60s, 120s DA pulses
  B. Pulse Train Frequency Scan: 0.01, 0.02, 0.05, 0.1, 0.2 Hz pulse trains
  C. Onset/Offset Latency: Measure time-to-half-peak for D1 vs D2

Protocol:
  All sub-experiments resume from DA=2nM steady-state checkpoint.
  Batch 0 = Control (constant 2 nM), Batch 1 = Experiment (DA waveform)

Usage:
  python -m experiments.exp_e_tau_segregation
  python -m experiments.exp_e_tau_segregation --sub a         # Pulse duration scan only
  python -m experiments.exp_e_tau_segregation --sub b         # Pulse train scan only
  python -m experiments.exp_e_tau_segregation --sub c         # Latency measurement only
  python -m experiments.exp_e_tau_segregation --sub all       # All sub-experiments
  python -m experiments.exp_e_tau_segregation --gpu 1
  python -m experiments.exp_e_tau_segregation --da-pulse 20   # Custom pulse amplitude
"""
import argparse
import json
import time
import os
import sys
import math
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.network import create_network_structure
from models.kernels import run_dynamic_d1_d2_kernel_pulse
from analysis.analyzer import PFCAnalyzer
from analysis.plotting import (plot_combined_raster,
                                plot_combined_rates_all,
                                plot_combined_rates_E,
                                plot_combined_rates_I)

RATE_GROUPS = ['E-D1', 'E-D2', 'E-Other', 'All-E', 'I-D1', 'I-D2', 'I-Other', 'All-I']


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment E: D1/D2 Tau Segregation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sub", type=str, default="all",
                        choices=["a", "b", "c", "all"],
                        help="Sub-experiment: a=pulse duration, b=pulse train, c=latency, all=run all")
    parser.add_argument("--da-base", type=float, default=2.0,
                        help="Baseline DA concentration (nM), default 2.0")
    parser.add_argument("--da-pulse", type=float, default=15.0,
                        help="Pulse DA concentration (nM), default 15.0")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path (auto-detected if not specified)")
    parser.add_argument("--skip-ckpt", action="store_true",
                        help="Skip checkpoint generation, reuse existing")
    parser.add_argument("--base-dur", type=float, default=500.0,
                        help="Baseline checkpoint duration in seconds (default: 500)")
    return parser.parse_args()


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:05.2f}s"
    return f"{int(m)}m {s:05.2f}s" if m > 0 else f"{s:.2f}s"


# ==============================================================================
# Checkpoint Management
# ==============================================================================

def find_or_create_checkpoint(args) -> str:
    """Find existing checkpoint or create one."""
    if args.ckpt:
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
        return args.ckpt

    bg_str = f"{config.BG_MEAN:g}"
    da_str = f"{args.da_base:g}"
    dur_str = f"{int(args.base_dur)}"
    ckpt_path = PROJECT_ROOT / "checkpoints" / f"ckpt_DA{da_str}nM_bg{bg_str}_{dur_str}s.pkl"

    if ckpt_path.exists():
        print(f"✅ Found existing checkpoint: {ckpt_path}")
        return str(ckpt_path)

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


# ==============================================================================
# Common Simulation Helper
# ==============================================================================

def run_pulse_sim(ckpt_path: str, device: torch.device,
                  da_base: float, da_pulse: float,
                  pre_pulse_s: float, pulse_duration_s: float, post_pulse_s: float,
                  alpha_record_interval: int = 100):
    """
    Run a single DA pulse simulation from checkpoint.

    Returns:
        data: dict with spikes, alpha traces, config, etc.
    """
    # Load checkpoint
    with open(ckpt_path, 'rb') as f:
        ckpt_data = pickle.load(f)
    init_state = ckpt_data['final_state'].to(device)
    if init_state.shape[0] >= 2:
        init_state[0] = init_state[1].clone()

    # Build network
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(config.N_E, config.N_I, device)

    record_indices = torch.tensor([
        [0, 0], [1, 0],
        [0, groups_info['e_d1_end']], [1, groups_info['e_d1_end']],
    ], device=device, dtype=torch.long)

    # Timing
    total_ms = (pre_pulse_s + pulse_duration_s + post_pulse_s) * 1000.0
    pulse_onset_ms = pre_pulse_s * 1000.0
    pulse_offset_ms = (pre_pulse_s + pulse_duration_s) * 1000.0

    kp = config.build_kernel_params(device)

    spikes, v_traces, final_state, alpha_d1_trace, alpha_d2_trace = run_dynamic_d1_d2_kernel_pulse(
        W_t, mask_d1, mask_d2, init_state,
        float(da_base), float(da_pulse),
        float(pulse_onset_ms), float(pulse_offset_ms),
        float(total_ms), float(config.DT),
        record_indices, config.N_E,
        alpha_record_interval, kp,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    data = {
        'config': {
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': total_ms, 'dt': config.DT,
            'da_onset': pulse_onset_ms, 'da_offset': pulse_offset_ms,
            'da_level': da_pulse,
            'control_da': da_base,
            'mode': 'pulse_response',
        },
        'masks': {'d1': mask_d1.cpu(), 'd2': mask_d2.cpu()},
        'groups_info': groups_info,
        'spikes': spikes.cpu(),
        'v_traces': v_traces.cpu(),
        'record_indices': record_indices.cpu(),
        'alpha_d1_trace': alpha_d1_trace.cpu().numpy(),
        'alpha_d2_trace': alpha_d2_trace.cpu().numpy(),
        'pulse_onset_s': pre_pulse_s,
        'pulse_offset_s': pre_pulse_s + pulse_duration_s,
        'total_s': pre_pulse_s + pulse_duration_s + post_pulse_s,
    }
    return data


def _generate_standard_plots(data: dict, save_dir: Path, label: str):
    """
    Generate standard main.py-style plots (combined raster + rates) for a simulation.
    Saves into a sub-directory named by label.
    """
    sub_dir = save_dir / label
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎨 Generating standard plots for '{label}'...")
    analyzer = PFCAnalyzer(data)

    plot_combined_raster(analyzer, save_dir=sub_dir)
    plot_combined_rates_all(analyzer, save_dir=sub_dir)
    plot_combined_rates_E(analyzer, save_dir=sub_dir)
    plot_combined_rates_I(analyzer, save_dir=sub_dir)

    # Save analysis report
    analyzer.save_report(str(sub_dir / "analysis_report.txt"))
    print(f"📁 Standard plots saved in: {sub_dir}")


def compute_group_rates(data: dict, time_win_ms: float = 1000.0) -> dict:
    """Compute time-resolved firing rates for all subgroups."""
    analyzer = PFCAnalyzer(data)
    results = {}
    for grp_name in RATE_GROUPS:
        results[grp_name] = {}
        for batch_id in [0, 1]:
            centers, rate = analyzer.compute_group_rate(batch_id, grp_name, time_win=time_win_ms)
            if rate is not None:
                results[grp_name][batch_id] = {
                    'times_s': centers / 1000.0,
                    'rates': rate,
                }
            else:
                results[grp_name][batch_id] = {'times_s': np.array([]), 'rates': np.array([])}
    return results


# ==============================================================================
# Sub-Experiment A: Pulse Duration Scan
# ==============================================================================

def run_sub_a(args, ckpt_path: str, device: torch.device, save_dir: Path):
    """
    Sub-Experiment A: Pulse Duration Scan

    Apply DA pulses of varying duration (5s, 15s, 30s, 60s, 120s) and measure:
    - Peak alpha_D1 and alpha_D2 achieved during the pulse
    - Ratio of peak alpha to steady-state alpha (fractional activation)
    - Firing rate changes for D1 vs D2 subgroups
    """
    print("\n" + "=" * 70)
    print("  🔬 Sub-Experiment A: Pulse Duration Scan")
    print("=" * 70)

    pulse_durations = [5.0, 15.0, 30.0, 60.0, 120.0]  # seconds
    pre_pulse = 20.0   # seconds baseline before pulse
    post_pulse = 60.0  # seconds recovery after pulse

    results = {}

    for dur_s in pulse_durations:
        print(f"\n  ── Pulse duration: {dur_s:.0f}s ──")
        t0 = time.time()

        data = run_pulse_sim(
            ckpt_path, device,
            da_base=args.da_base, da_pulse=args.da_pulse,
            pre_pulse_s=pre_pulse, pulse_duration_s=dur_s, post_pulse_s=post_pulse,
        )

        elapsed = time.time() - t0
        print(f"     Sim time: {_fmt_elapsed(elapsed)}")

        alpha_d1 = data['alpha_d1_trace'][:, 1]  # Exp batch
        alpha_d2 = data['alpha_d2_trace'][:, 1]

        # Peak alpha during pulse window
        alpha_times = np.linspace(0, data['total_s'], len(alpha_d1))
        pulse_mask = (alpha_times >= pre_pulse) & (alpha_times <= pre_pulse + dur_s)

        peak_d1 = float(alpha_d1[pulse_mask].max()) if np.any(pulse_mask) else 0.0
        peak_d2 = float(alpha_d2[pulse_mask].max()) if np.any(pulse_mask) else 0.0

        # Baseline alpha (before pulse)
        bl_mask = alpha_times < pre_pulse
        bl_d1 = float(alpha_d1[bl_mask].mean()) if np.any(bl_mask) else 0.0
        bl_d2 = float(alpha_d2[bl_mask].mean()) if np.any(bl_mask) else 0.0

        # Delta alpha (peak - baseline)
        delta_d1 = peak_d1 - bl_d1
        delta_d2 = peak_d2 - bl_d2

        # Compute firing rates
        rate_data = compute_group_rates(data, time_win_ms=1000.0)

        # Mean rate during pulse (Exp batch)
        pulse_rates = {}
        for grp in ['E-D1', 'E-D2', 'All-E', 'I-D1', 'I-D2', 'All-I']:
            rd = rate_data[grp][1]
            if len(rd['rates']) > 0:
                mask = (rd['times_s'] >= pre_pulse) & (rd['times_s'] <= pre_pulse + dur_s)
                pulse_rates[grp] = float(np.mean(rd['rates'][mask])) if np.any(mask) else 0.0
            else:
                pulse_rates[grp] = 0.0

        results[dur_s] = {
            'peak_d1': peak_d1, 'peak_d2': peak_d2,
            'bl_d1': bl_d1, 'bl_d2': bl_d2,
            'delta_d1': delta_d1, 'delta_d2': delta_d2,
            'pulse_rates': pulse_rates,
            'alpha_d1_trace': alpha_d1,
            'alpha_d2_trace': alpha_d2,
            'alpha_times': alpha_times,
            'sim_time': elapsed,
        }

        print(f"     α_D1: baseline={bl_d1:.4f} → peak={peak_d1:.4f} (Δ={delta_d1:.4f})")
        print(f"     α_D2: baseline={bl_d2:.4f} → peak={peak_d2:.4f} (Δ={delta_d2:.4f})")
        print(f"     D2/D1 peak ratio: {peak_d2/peak_d1:.2f}x" if peak_d1 > 0.001 else "     D1 peak ≈ 0")

        # Generate standard main.py-style plots for this pulse duration
        _generate_standard_plots(data, save_dir, label=f"sub_a_pulse_{dur_s:.0f}s")

    # ---- Plot results ----
    _plot_sub_a(results, pulse_durations, args, save_dir)

    # ---- Print summary table ----
    _print_sub_a_table(results, pulse_durations)

    return results


def _plot_sub_a(results, pulse_durations, args, save_dir):
    """Generate plots for Sub-Experiment A."""

    # Figure 1: Alpha traces overlay — D1 (solid) and D2 (dashed) on same axes
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    fig.suptitle(
        f"Sub-Exp A: Pulse Duration Scan — α Response to {args.da_pulse}nM DA Pulses\n"
        f"D1 (solid, τ_on≈{config.TAU_ON_D1/1000:.0f}s)  vs  "
        f"D2 (dashed, τ_on≈{config.TAU_ON_D2/1000:.0f}s)  —  D2 is ~3x faster",
        fontsize=14, fontweight='bold'
    )

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(pulse_durations)))

    for i, dur_s in enumerate(pulse_durations):
        r = results[dur_s]
        t = r['alpha_times']
        # D1: solid line
        ax.plot(t, r['alpha_d1_trace'], color=colors[i], linewidth=2.5,
                linestyle='-', label=f'{dur_s:.0f}s — D1')
        # D2: dashed line
        ax.plot(t, r['alpha_d2_trace'], color=colors[i], linewidth=2.5,
                linestyle='--', label=f'{dur_s:.0f}s — D2')
        # Mark pulse window
        ax.axvspan(20.0, 20.0 + dur_s, alpha=0.04, color=colors[i])

    ax.axvline(20.0, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='DA onset')
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Receptor Activation (α)', fontsize=13)
    ax.legend(fontsize=9, loc='upper right', ncol=2,
              title='Pulse Duration — Receptor', title_fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "sub_a_alpha_traces.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: sub_a_alpha_traces.png")

    # Figure 2: Peak alpha vs pulse duration (key result)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Sub-Exp A: Peak Receptor Activation vs Pulse Duration\n"
        "(Short pulses → D2 dominates; Long pulses → D1 catches up)",
        fontsize=14, fontweight='bold'
    )

    durs = np.array(pulse_durations)
    peaks_d1 = np.array([results[d]['delta_d1'] for d in pulse_durations])
    peaks_d2 = np.array([results[d]['delta_d2'] for d in pulse_durations])

    # Panel A: Absolute peak alpha
    ax = axes[0]
    ax.plot(durs, peaks_d1, 'o-', color='#d62728', linewidth=2.5, markersize=10,
            label=f'Δα_D1 (τ_on≈{config.TAU_ON_D1/1000:.0f}s)')
    ax.plot(durs, peaks_d2, 's-', color='#1f77b4', linewidth=2.5, markersize=10,
            label=f'Δα_D2 (τ_on≈{config.TAU_ON_D2/1000:.0f}s)')
    ax.set_xlabel('Pulse Duration (s)', fontsize=13)
    ax.set_ylabel('Δα (peak − baseline)', fontsize=13)
    ax.set_title('A. Absolute Activation Gain', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel B: D2/D1 ratio
    ax = axes[1]
    ratio = np.where(peaks_d1 > 0.0001, peaks_d2 / peaks_d1, 0)
    ax.plot(durs, ratio, 'D-', color='purple', linewidth=2.5, markersize=10)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Pulse Duration (s)', fontsize=13)
    ax.set_ylabel('Δα_D2 / Δα_D1 Ratio', fontsize=13)
    ax.set_title('B. D2/D1 Activation Ratio', fontsize=13)
    ax.grid(True, alpha=0.3)
    # Annotate
    for i, d in enumerate(durs):
        if ratio[i] > 0:
            ax.annotate(f'{ratio[i]:.1f}x', (d, ratio[i]),
                        textcoords="offset points", xytext=(0, 12),
                        ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_dir / "sub_a_peak_vs_duration.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: sub_a_peak_vs_duration.png")

    # Figure 3: Firing rate changes
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Sub-Exp A: Firing Rate During DA Pulse vs Pulse Duration",
                 fontsize=14, fontweight='bold')

    e_groups = ['E-D1', 'E-D2', 'E-Other']
    i_groups = ['I-D1', 'I-D2', 'I-Other']
    e_colors = ['#d62728', '#1f77b4', 'gray']
    i_colors = ['#ff7f0e', '#9467bd', '#2ca02c']

    for ax, groups, colors, title in zip(
        axes, [e_groups, i_groups], [e_colors, i_colors],
        ['Excitatory Subgroups', 'Inhibitory Subgroups']
    ):
        for grp, color in zip(groups, colors):
            rates = [results[d]['pulse_rates'].get(grp, 0) for d in pulse_durations]
            ax.plot(durs, rates, 'o-', color=color, linewidth=2, markersize=8, label=grp)
        ax.set_xlabel('Pulse Duration (s)', fontsize=13)
        ax.set_ylabel('Mean Firing Rate During Pulse (Hz)', fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "sub_a_rates_vs_duration.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: sub_a_rates_vs_duration.png")


def _print_sub_a_table(results, pulse_durations):
    """Print summary table for Sub-Experiment A."""
    w = 90
    print("\n" + "=" * w)
    print("  📊 Sub-Exp A: Pulse Duration Scan — Summary")
    print("=" * w)
    print(f"  {'Duration':>10} │ {'Δα_D1':>10} {'Δα_D2':>10} {'D2/D1':>8} │ {'E-D1 Hz':>8} {'E-D2 Hz':>8} {'I-D1 Hz':>8} {'I-D2 Hz':>8}")
    print(f"  {'─'*10}─┼─{'─'*10}─{'─'*10}─{'─'*8}─┼─{'─'*8}─{'─'*8}─{'─'*8}─{'─'*8}")

    for dur_s in pulse_durations:
        r = results[dur_s]
        ratio = r['delta_d2'] / r['delta_d1'] if r['delta_d1'] > 0.0001 else float('inf')
        pr = r['pulse_rates']
        print(f"  {dur_s:>8.0f}s │ {r['delta_d1']:>10.4f} {r['delta_d2']:>10.4f} {ratio:>7.1f}x │"
              f" {pr.get('E-D1',0):>8.2f} {pr.get('E-D2',0):>8.2f}"
              f" {pr.get('I-D1',0):>8.2f} {pr.get('I-D2',0):>8.2f}")

    print("=" * w)
    print("  Key: Δα = peak − baseline during pulse. D2/D1 > 1 means D2 activates more.")
    print("  Prediction: Short pulses → high D2/D1 ratio (D2 dominates)")
    print("=" * w)


# ==============================================================================
# Sub-Experiment B: Pulse Train Frequency Scan
# ==============================================================================

def run_sub_b(args, ckpt_path: str, device: torch.device, save_dir: Path):
    """
    Sub-Experiment B: Pulse Train Frequency Scan

    Apply repeated DA pulses at different frequencies and measure:
    - Alpha modulation depth (peak-to-trough) for D1 vs D2
    - D2 can track fast pulses, D1 only tracks slow pulses (low-pass filter)
    """
    print("\n" + "=" * 70)
    print("  🔬 Sub-Experiment B: Pulse Train Frequency Scan")
    print("=" * 70)

    # Pulse train parameters
    # Frequency = 1 / (pulse_on + pulse_off)
    # We fix pulse_on = 5s and vary pulse_off to get different frequencies
    pulse_on = 5.0  # seconds, each pulse duration
    frequencies = [0.01, 0.02, 0.05, 0.1]  # Hz
    # period = 1/freq; pulse_off = period - pulse_on
    n_cycles = 5  # number of complete cycles per frequency

    pre_pulse = 20.0  # baseline before first pulse
    post_pulse = 40.0  # recovery after last pulse

    results = {}

    for freq in frequencies:
        period = 1.0 / freq
        pulse_off = period - pulse_on
        if pulse_off < 1.0:
            print(f"  ⚠️ Skipping freq={freq}Hz: pulse_off={pulse_off:.1f}s too short")
            continue

        total_train = n_cycles * period
        total_s = pre_pulse + total_train + post_pulse

        print(f"\n  ── Freq: {freq}Hz (period={period:.0f}s, on={pulse_on}s, off={pulse_off:.0f}s, {n_cycles} cycles) ──")
        print(f"     Total duration: {total_s:.0f}s")

        # We need to run multiple pulses. Use the pulse kernel with the first pulse,
        # but actually we need a custom approach. Let's use sequential pulses by
        # running one long simulation with the first pulse only, then measure the
        # alpha modulation analytically.
        #
        # Better approach: run a single long simulation and construct the DA schedule
        # using the da_waveform infrastructure.
        t0 = time.time()

        # Build DA schedule manually
        dt = config.DT
        steps = int(total_s * 1000.0 / dt)
        t_ms = np.arange(steps) * dt
        t_s = t_ms / 1000.0

        da_ctrl = np.full(steps, args.da_base)
        da_exp = np.full(steps, args.da_base)

        # Generate pulse train
        for c in range(n_cycles):
            cycle_start = pre_pulse + c * period
            cycle_end = cycle_start + pulse_on
            mask = (t_s >= cycle_start) & (t_s < cycle_end)
            da_exp[mask] = args.da_pulse

        da_schedule = np.stack([da_ctrl, da_exp], axis=1)
        da_schedule_tensor = torch.tensor(da_schedule, dtype=torch.float32, device=device)

        # Load checkpoint and run
        with open(ckpt_path, 'rb') as f:
            ckpt_data = pickle.load(f)
        init_state = ckpt_data['final_state'].to(device)
        if init_state.shape[0] >= 2:
            init_state[0] = init_state[1].clone()

        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        W_t, mask_d1, mask_d2, groups_info = create_network_structure(config.N_E, config.N_I, device)

        record_indices = torch.tensor([
            [0, 0], [1, 0],
            [0, groups_info['e_d1_end']], [1, groups_info['e_d1_end']],
        ], device=device, dtype=torch.long)

        kp = config.build_kernel_params(device)
        alpha_interval = 100

        from models.kernels import run_dynamic_d1_d2_kernel_da_schedule
        result = run_dynamic_d1_d2_kernel_da_schedule(
            W_t, mask_d1, mask_d2, init_state,
            da_schedule_tensor,
            float(total_s * 1000.0), dt,
            record_indices, config.N_E,
            alpha_interval, kp,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        all_spikes, v_traces, final_state, alpha_d1_trace, alpha_d2_trace = result
        alpha_d1 = alpha_d1_trace.cpu().numpy()[:, 1]  # Exp batch
        alpha_d2 = alpha_d2_trace.cpu().numpy()[:, 1]
        alpha_times = np.linspace(0, total_s, len(alpha_d1))

        # Measure modulation depth in the last 3 cycles (skip first 2 for transient)
        train_start = pre_pulse + 2 * period
        train_end = pre_pulse + total_train
        train_mask = (alpha_times >= train_start) & (alpha_times <= train_end)

        if np.any(train_mask):
            mod_d1 = float(alpha_d1[train_mask].max() - alpha_d1[train_mask].min())
            mod_d2 = float(alpha_d2[train_mask].max() - alpha_d2[train_mask].min())
        else:
            mod_d1 = mod_d2 = 0.0

        results[freq] = {
            'mod_depth_d1': mod_d1,
            'mod_depth_d2': mod_d2,
            'alpha_d1_trace': alpha_d1,
            'alpha_d2_trace': alpha_d2,
            'alpha_times': alpha_times,
            'da_schedule': da_schedule,
            'period': period,
            'sim_time': elapsed,
        }

        print(f"     Sim time: {_fmt_elapsed(elapsed)}")
        print(f"     Modulation depth — D1: {mod_d1:.4f}, D2: {mod_d2:.4f}")
        print(f"     D2/D1 modulation ratio: {mod_d2/mod_d1:.2f}x" if mod_d1 > 0.0001 else "     D1 mod ≈ 0")

        # Generate standard main.py-style plots for this frequency
        # Build a PFCAnalyzer-compatible data dict
        sub_b_data = {
            'config': {
                'N_E': config.N_E, 'N_I': config.N_I,
                'duration': total_s * 1000.0, 'dt': config.DT,
                'da_onset': pre_pulse * 1000.0, 'da_level': args.da_pulse,
                'control_da': args.da_base,
                'mode': 'pulse_train',
            },
            'masks': {'d1': mask_d1.cpu(), 'd2': mask_d2.cpu()},
            'groups_info': groups_info,
            'spikes': all_spikes.cpu(),
            'v_traces': v_traces.cpu(),
            'record_indices': record_indices.cpu(),
            'da_schedule': da_schedule,
        }
        _generate_standard_plots(sub_b_data, save_dir, label=f"sub_b_freq_{freq}Hz")

    # ---- Plot results ----
    _plot_sub_b(results, frequencies, args, save_dir)
    _print_sub_b_table(results, frequencies)

    return results


def _plot_sub_b(results, frequencies, args, save_dir):
    """Generate plots for Sub-Experiment B."""
    valid_freqs = [f for f in frequencies if f in results]
    if not valid_freqs:
        return

    # Figure 1: Alpha traces for each frequency
    n_freqs = len(valid_freqs)
    fig, axes = plt.subplots(n_freqs, 1, figsize=(20, 5 * n_freqs), sharex=False)
    if n_freqs == 1:
        axes = [axes]

    fig.suptitle(
        f"Sub-Exp B: Pulse Train Frequency Scan — α Response\n"
        f"(DA pulses: {args.da_base}→{args.da_pulse}nM, 5s on)",
        fontsize=14, fontweight='bold'
    )

    for ax, freq in zip(axes, valid_freqs):
        r = results[freq]
        t = r['alpha_times']

        # Plot DA schedule (background)
        da_t = np.linspace(0, t[-1], r['da_schedule'].shape[0])
        da_norm = (r['da_schedule'][:, 1] - args.da_base) / (args.da_pulse - args.da_base)
        ax_da = ax.twinx()
        ax_da.fill_between(da_t, 0, da_norm, alpha=0.1, color='green')
        ax_da.set_ylabel('[DA] (norm)', fontsize=10, color='green', alpha=0.5)
        ax_da.set_ylim(-0.1, 1.5)
        ax_da.tick_params(axis='y', labelcolor='green', labelsize=9)

        # Plot alpha traces
        ax.plot(t, r['alpha_d1_trace'], color='#d62728', linewidth=2,
                label=f'α_D1 (mod={r["mod_depth_d1"]:.4f})')
        ax.plot(t, r['alpha_d2_trace'], color='#1f77b4', linewidth=2,
                label=f'α_D2 (mod={r["mod_depth_d2"]:.4f})')
        ax.set_ylabel('α', fontsize=12)
        ax.set_title(f'Freq = {freq} Hz (period = {r["period"]:.0f}s)', fontsize=12)
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "sub_b_alpha_traces.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: sub_b_alpha_traces.png")

    # Figure 2: Bode-like plot — Modulation depth vs frequency
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Sub-Exp B: Receptor Modulation Depth vs Pulse Frequency\n"
        "(D1 = low-pass filter, D2 = wider bandwidth)",
        fontsize=14, fontweight='bold'
    )

    freqs_arr = np.array(valid_freqs)
    mod_d1 = np.array([results[f]['mod_depth_d1'] for f in valid_freqs])
    mod_d2 = np.array([results[f]['mod_depth_d2'] for f in valid_freqs])

    # Panel A: Absolute modulation depth
    ax = axes[0]
    ax.plot(freqs_arr, mod_d1, 'o-', color='#d62728', linewidth=2.5, markersize=10,
            label=f'D1 (τ_on≈{config.TAU_ON_D1/1000:.0f}s)')
    ax.plot(freqs_arr, mod_d2, 's-', color='#1f77b4', linewidth=2.5, markersize=10,
            label=f'D2 (τ_on≈{config.TAU_ON_D2/1000:.0f}s)')
    ax.set_xscale('log')
    ax.set_xlabel('Pulse Frequency (Hz)', fontsize=13)
    ax.set_ylabel('Modulation Depth (peak−trough of α)', fontsize=13)
    ax.set_title('A. Modulation Depth (Bode-like)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    # Panel B: D2/D1 ratio
    ax = axes[1]
    ratio = np.where(mod_d1 > 0.0001, mod_d2 / mod_d1, 0)
    ax.plot(freqs_arr, ratio, 'D-', color='purple', linewidth=2.5, markersize=10)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Pulse Frequency (Hz)', fontsize=13)
    ax.set_ylabel('D2/D1 Modulation Ratio', fontsize=13)
    ax.set_title('B. D2 Advantage at High Frequencies', fontsize=13)
    ax.grid(True, alpha=0.3, which='both')
    for i, f in enumerate(freqs_arr):
        if ratio[i] > 0:
            ax.annotate(f'{ratio[i]:.1f}x', (f, ratio[i]),
                        textcoords="offset points", xytext=(0, 12),
                        ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_dir / "sub_b_bode_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: sub_b_bode_plot.png")


def _print_sub_b_table(results, frequencies):
    """Print summary table for Sub-Experiment B."""
    valid_freqs = [f for f in frequencies if f in results]
    w = 70
    print("\n" + "=" * w)
    print("  📊 Sub-Exp B: Pulse Train Frequency Scan — Summary")
    print("=" * w)
    print(f"  {'Freq (Hz)':>10} {'Period (s)':>12} │ {'Mod D1':>10} {'Mod D2':>10} {'D2/D1':>8}")
    print(f"  {'─'*10}─{'─'*12}─┼─{'─'*10}─{'─'*10}─{'─'*8}")

    for freq in valid_freqs:
        r = results[freq]
        ratio = r['mod_depth_d2'] / r['mod_depth_d1'] if r['mod_depth_d1'] > 0.0001 else float('inf')
        print(f"  {freq:>10.3f} {r['period']:>11.0f}s │ {r['mod_depth_d1']:>10.4f} {r['mod_depth_d2']:>10.4f} {ratio:>7.1f}x")

    print("=" * w)
    print("  Key: Mod = peak−trough of α during steady-state pulse train.")
    print("  Prediction: High freq → D1 mod ≈ 0 (can't track), D2 mod still significant.")
    print("=" * w)


# ==============================================================================
# Sub-Experiment C: Onset/Offset Latency Measurement
# ==============================================================================

def run_sub_c(args, ckpt_path: str, device: torch.device, save_dir: Path):
    """
    Sub-Experiment C: Onset/Offset Latency Measurement

    Apply a long DA pulse (120s) and precisely measure:
    - Time for α to reach 10%, 50%, 90% of its peak (onset latency)
    - Time for α to decay to 90%, 50%, 10% of its peak after DA withdrawal (offset latency)
    - Compare D1 vs D2 latencies quantitatively
    """
    print("\n" + "=" * 70)
    print("  🔬 Sub-Experiment C: Onset/Offset Latency Measurement")
    print("=" * 70)

    pre_pulse = 20.0
    pulse_duration = 120.0  # Long pulse to ensure near-steady-state
    post_pulse = 300.0      # Long recovery to observe full decay

    print(f"  Protocol: {args.da_base}→{args.da_pulse}→{args.da_base} nM")
    print(f"  Pre-pulse: {pre_pulse}s, Pulse: {pulse_duration}s, Post-pulse: {post_pulse}s")
    print(f"  Total: {pre_pulse + pulse_duration + post_pulse:.0f}s")

    t0 = time.time()
    data = run_pulse_sim(
        ckpt_path, device,
        da_base=args.da_base, da_pulse=args.da_pulse,
        pre_pulse_s=pre_pulse, pulse_duration_s=pulse_duration, post_pulse_s=post_pulse,
        alpha_record_interval=10,  # Higher resolution for latency measurement
    )
    elapsed = time.time() - t0
    print(f"  Sim time: {_fmt_elapsed(elapsed)}")

    alpha_d1 = data['alpha_d1_trace'][:, 1]  # Exp batch
    alpha_d2 = data['alpha_d2_trace'][:, 1]
    alpha_times = np.linspace(0, data['total_s'], len(alpha_d1))

    # Baseline values
    bl_mask = alpha_times < pre_pulse
    bl_d1 = float(alpha_d1[bl_mask].mean())
    bl_d2 = float(alpha_d2[bl_mask].mean())

    # Peak values (at end of pulse)
    pulse_end_mask = (alpha_times >= pre_pulse + pulse_duration - 5) & (alpha_times <= pre_pulse + pulse_duration)
    peak_d1 = float(alpha_d1[pulse_end_mask].mean()) if np.any(pulse_end_mask) else bl_d1
    peak_d2 = float(alpha_d2[pulse_end_mask].mean()) if np.any(pulse_end_mask) else bl_d2

    range_d1 = peak_d1 - bl_d1
    range_d2 = peak_d2 - bl_d2

    # Measure onset latencies (time from pulse onset to reach X% of range)
    def measure_onset_latency(alpha, times, baseline, peak_val, threshold_pct):
        """Time from pulse onset to reach threshold_pct of (peak - baseline)."""
        target = baseline + (peak_val - baseline) * threshold_pct
        onset_mask = times >= pre_pulse
        alpha_onset = alpha[onset_mask]
        times_onset = times[onset_mask]
        crossed = np.where(alpha_onset >= target)[0]
        if len(crossed) > 0:
            return float(times_onset[crossed[0]] - pre_pulse)
        return float('inf')

    # Measure offset latencies (time from pulse offset to decay to X% of range)
    def measure_offset_latency(alpha, times, baseline, peak_val, threshold_pct):
        """Time from pulse offset to decay to threshold_pct of (peak - baseline)."""
        target = baseline + (peak_val - baseline) * threshold_pct
        offset_time = pre_pulse + pulse_duration
        offset_mask = times >= offset_time
        alpha_offset = alpha[offset_mask]
        times_offset = times[offset_mask]
        crossed = np.where(alpha_offset <= target)[0]
        if len(crossed) > 0:
            return float(times_offset[crossed[0]] - offset_time)
        return float('inf')

    thresholds = [0.1, 0.5, 0.9]
    latencies = {'onset': {}, 'offset': {}}

    for pct in thresholds:
        pct_label = f'{int(pct*100)}%'
        lat_d1_on = measure_onset_latency(alpha_d1, alpha_times, bl_d1, peak_d1, pct)
        lat_d2_on = measure_onset_latency(alpha_d2, alpha_times, bl_d2, peak_d2, pct)
        latencies['onset'][pct_label] = {'D1': lat_d1_on, 'D2': lat_d2_on}

        lat_d1_off = measure_offset_latency(alpha_d1, alpha_times, bl_d1, peak_d1, 1.0 - pct)
        lat_d2_off = measure_offset_latency(alpha_d2, alpha_times, bl_d2, peak_d2, 1.0 - pct)
        latencies['offset'][pct_label] = {'D1': lat_d1_off, 'D2': lat_d2_off}

    results = {
        'bl_d1': bl_d1, 'bl_d2': bl_d2,
        'peak_d1': peak_d1, 'peak_d2': peak_d2,
        'range_d1': range_d1, 'range_d2': range_d2,
        'latencies': latencies,
        'alpha_d1_trace': alpha_d1,
        'alpha_d2_trace': alpha_d2,
        'alpha_times': alpha_times,
        'sim_time': elapsed,
    }

    # Generate standard main.py-style plots for Sub-C
    _generate_standard_plots(data, save_dir, label="sub_c_latency")

    # ---- Plot results ----
    _plot_sub_c(results, args, save_dir, pre_pulse, pulse_duration)
    _print_sub_c_table(results)

    return results


def _plot_sub_c(results, args, save_dir, pre_pulse, pulse_duration):
    """Generate plots for Sub-Experiment C."""
    alpha_d1 = results['alpha_d1_trace']
    alpha_d2 = results['alpha_d2_trace']
    t = results['alpha_times']
    pulse_offset = pre_pulse + pulse_duration

    # Figure 1: Full alpha trace with latency annotations
    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    fig.suptitle(
        f"Sub-Exp C: Onset/Offset Latency — D1 vs D2\n"
        f"DA: {args.da_base}→{args.da_pulse}→{args.da_base} nM  |  "
        f"D1 τ_on≈{config.TAU_ON_D1/1000:.0f}s, D2 τ_on≈{config.TAU_ON_D2/1000:.0f}s",
        fontsize=14, fontweight='bold'
    )

    # Panel A: Onset zoom
    ax = axes[0]
    zoom_start = pre_pulse - 5
    zoom_end = pre_pulse + 150
    mask = (t >= zoom_start) & (t <= zoom_end)

    ax.plot(t[mask], alpha_d1[mask], color='#d62728', linewidth=2.5, label='α_D1')
    ax.plot(t[mask], alpha_d2[mask], color='#1f77b4', linewidth=2.5, label='α_D2')
    ax.axvline(pre_pulse, color='green', linestyle='--', linewidth=2, label='DA ON')

    # Draw threshold lines
    for pct, ls in zip([0.1, 0.5, 0.9], [':', '--', '-.']):
        target_d1 = results['bl_d1'] + results['range_d1'] * pct
        target_d2 = results['bl_d2'] + results['range_d2'] * pct
        ax.axhline(target_d1, color='#d62728', linestyle=ls, alpha=0.3, linewidth=1)
        ax.axhline(target_d2, color='#1f77b4', linestyle=ls, alpha=0.3, linewidth=1)

    # Annotate 50% latency
    lat = results['latencies']['onset']['50%']
    for receptor, color, lat_val, bl, rng in [
        ('D1', '#d62728', lat['D1'], results['bl_d1'], results['range_d1']),
        ('D2', '#1f77b4', lat['D2'], results['bl_d2'], results['range_d2']),
    ]:
        if lat_val < float('inf'):
            target = bl + rng * 0.5
            ax.annotate(f'{receptor}: {lat_val:.1f}s',
                        xy=(pre_pulse + lat_val, target),
                        xytext=(pre_pulse + lat_val + 10, target + rng * 0.15),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                        fontsize=11, fontweight='bold', color=color)

    ax.set_ylabel('α', fontsize=12)
    ax.set_title('A. Onset Latency (DA pulse ON)', fontsize=13)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Panel B: Offset zoom
    ax = axes[1]
    zoom_start = pulse_offset - 5
    zoom_end = min(pulse_offset + 250, t[-1])
    mask = (t >= zoom_start) & (t <= zoom_end)

    ax.plot(t[mask], alpha_d1[mask], color='#d62728', linewidth=2.5, label='α_D1')
    ax.plot(t[mask], alpha_d2[mask], color='#1f77b4', linewidth=2.5, label='α_D2')
    ax.axvline(pulse_offset, color='red', linestyle='--', linewidth=2, label='DA OFF')

    # Shade D1 afterglow window
    post_mask = t >= pulse_offset
    afterglow = post_mask & (alpha_d1 > alpha_d2 * 1.05)
    if np.any(afterglow):
        ag_start = t[afterglow][0]
        ag_end = t[afterglow][-1]
        ax.axvspan(ag_start, ag_end, alpha=0.15, color='orange',
                   label=f'D1 afterglow [{ag_start:.0f}s–{ag_end:.0f}s]')

    # Annotate 50% offset latency
    lat = results['latencies']['offset']['50%']
    for receptor, color, lat_val, bl, rng in [
        ('D1', '#d62728', lat['D1'], results['bl_d1'], results['range_d1']),
        ('D2', '#1f77b4', lat['D2'], results['bl_d2'], results['range_d2']),
    ]:
        if lat_val < float('inf'):
            target = bl + rng * 0.5
            ax.annotate(f'{receptor}: {lat_val:.1f}s',
                        xy=(pulse_offset + lat_val, target),
                        xytext=(pulse_offset + lat_val + 15, target + rng * 0.15),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                        fontsize=11, fontweight='bold', color=color)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('α', fontsize=12)
    ax.set_title('B. Offset Latency (DA pulse OFF) — D1 Afterglow Window', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "sub_c_latency_traces.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: sub_c_latency_traces.png")

    # Figure 2: Latency bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Sub-Exp C: D1 vs D2 Latency Comparison",
                 fontsize=14, fontweight='bold')

    thresholds = ['10%', '50%', '90%']
    x = np.arange(len(thresholds))
    bar_width = 0.35

    for ax, phase, title in zip(axes, ['onset', 'offset'],
                                 ['Onset Latency (time to reach X%)',
                                  'Offset Latency (time to decay to 100−X%)']):
        d1_vals = [results['latencies'][phase][t]['D1'] for t in thresholds]
        d2_vals = [results['latencies'][phase][t]['D2'] for t in thresholds]

        # Cap inf values for display
        d1_vals = [v if v < 1e6 else 0 for v in d1_vals]
        d2_vals = [v if v < 1e6 else 0 for v in d2_vals]

        bars1 = ax.bar(x - bar_width/2, d1_vals, bar_width, color='#d62728',
                        label=f'D1 (τ_on≈{config.TAU_ON_D1/1000:.0f}s)', alpha=0.8)
        bars2 = ax.bar(x + bar_width/2, d2_vals, bar_width, color='#1f77b4',
                        label=f'D2 (τ_on≈{config.TAU_ON_D2/1000:.0f}s)', alpha=0.8)

        # Add value labels
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                        f'{h:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 1,
                        f'{h:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Threshold', fontsize=13)
        ax.set_ylabel('Latency (s)', fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(thresholds, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / "sub_c_latency_bars.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: sub_c_latency_bars.png")


def _print_sub_c_table(results):
    """Print summary table for Sub-Experiment C."""
    w = 80
    print("\n" + "=" * w)
    print("  📊 Sub-Exp C: Onset/Offset Latency — Summary")
    print("=" * w)

    print(f"\n  Baseline: α_D1={results['bl_d1']:.4f}, α_D2={results['bl_d2']:.4f}")
    print(f"  Peak:     α_D1={results['peak_d1']:.4f}, α_D2={results['peak_d2']:.4f}")
    print(f"  Range:    Δα_D1={results['range_d1']:.4f}, Δα_D2={results['range_d2']:.4f}")

    print(f"\n  {'Phase':<8} {'Threshold':<12} │ {'D1 (s)':>10} {'D2 (s)':>10} {'D1/D2':>8}")
    print(f"  {'─'*8}─{'─'*12}─┼─{'─'*10}─{'─'*10}─{'─'*8}")

    for phase in ['onset', 'offset']:
        for pct in ['10%', '50%', '90%']:
            lat = results['latencies'][phase][pct]
            d1_val = lat['D1']
            d2_val = lat['D2']
            ratio = d1_val / d2_val if d2_val > 0.01 and d1_val < 1e6 else float('inf')
            d1_str = f"{d1_val:.1f}" if d1_val < 1e6 else "N/A"
            d2_str = f"{d2_val:.1f}" if d2_val < 1e6 else "N/A"
            ratio_str = f"{ratio:.1f}x" if ratio < 1e6 else "N/A"
            print(f"  {phase:<8} {pct:<12} │ {d1_str:>10} {d2_str:>10} {ratio_str:>8}")

    print("=" * w)
    print("  Key: D1/D2 > 1 means D1 is slower. Expected ratio ≈ 3x (matching τ ratio).")
    print("=" * w)


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    t_total_start = time.time()

    # Device
    if torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"🔧 Device: {device}")

    # Output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sub_tag = args.sub if args.sub != "all" else "abc"
    save_dir = PROJECT_ROOT / "outputs" / f"exp_e_tau_segregation_{timestamp}_{sub_tag}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint
    ckpt_path = find_or_create_checkpoint(args)

    # Save experiment config
    exp_config = {
        'sub_experiments': args.sub,
        'da_base': args.da_base,
        'da_pulse': args.da_pulse,
        'checkpoint': ckpt_path,
        'device': str(device),
        'D1_tau_on_ms': config.TAU_ON_D1,
        'D1_tau_off_ms': config.TAU_OFF_D1,
        'D2_tau_on_ms': config.TAU_ON_D2,
        'D2_tau_off_ms': config.TAU_OFF_D2,
        'D1_EC50': config.EC50_D1,
        'D2_EC50': config.EC50_D2,
    }
    with open(save_dir / "experiment_config.json", 'w') as f:
        json.dump(exp_config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  🧪 Experiment E: D1/D2 Tau Segregation")
    print(f"{'='*70}")
    print(f"  D1: τ_on={config.TAU_ON_D1:.0f}ms ({config.TAU_ON_D1/1000:.0f}s), "
          f"τ_off={config.TAU_OFF_D1:.0f}ms ({config.TAU_OFF_D1/1000:.0f}s)")
    print(f"  D2: τ_on={config.TAU_ON_D2:.0f}ms ({config.TAU_ON_D2/1000:.0f}s), "
          f"τ_off={config.TAU_OFF_D2:.0f}ms ({config.TAU_OFF_D2/1000:.0f}s)")
    print(f"  τ ratio (D1/D2): on={config.TAU_ON_D1/config.TAU_ON_D2:.1f}x, "
          f"off={config.TAU_OFF_D1/config.TAU_OFF_D2:.1f}x")
    print(f"  DA: baseline={args.da_base}nM, pulse={args.da_pulse}nM")
    print(f"  Output: {save_dir}")
    print(f"{'='*70}\n")

    all_results = {}

    # Sub-Experiment A: Pulse Duration Scan
    if args.sub in ['a', 'all']:
        all_results['sub_a'] = run_sub_a(args, ckpt_path, device, save_dir)

    # Sub-Experiment B: Pulse Train Frequency Scan
    if args.sub in ['b', 'all']:
        all_results['sub_b'] = run_sub_b(args, ckpt_path, device, save_dir)

    # Sub-Experiment C: Onset/Offset Latency
    if args.sub in ['c', 'all']:
        all_results['sub_c'] = run_sub_c(args, ckpt_path, device, save_dir)

    # Save all results (excluding large traces)
    summary = {}
    if 'sub_a' in all_results:
        summary['sub_a'] = {
            str(k): {
                'peak_d1': v['peak_d1'], 'peak_d2': v['peak_d2'],
                'delta_d1': v['delta_d1'], 'delta_d2': v['delta_d2'],
                'pulse_rates': v['pulse_rates'],
            } for k, v in all_results['sub_a'].items()
        }
    if 'sub_b' in all_results:
        summary['sub_b'] = {
            str(k): {
                'mod_depth_d1': v['mod_depth_d1'],
                'mod_depth_d2': v['mod_depth_d2'],
                'period': v['period'],
            } for k, v in all_results['sub_b'].items()
        }
    if 'sub_c' in all_results:
        summary['sub_c'] = {
            'latencies': all_results['sub_c']['latencies'],
            'bl_d1': all_results['sub_c']['bl_d1'],
            'bl_d2': all_results['sub_c']['bl_d2'],
            'peak_d1': all_results['sub_c']['peak_d1'],
            'peak_d2': all_results['sub_c']['peak_d2'],
        }

    with open(save_dir / "exp_e_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📄 Results saved: {save_dir / 'exp_e_results.json'}")

    # Total time
    t_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"  ⏱️  Total time: {_fmt_elapsed(t_total)}")
    print(f"  📁 All results in: {save_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
