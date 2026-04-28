#!/usr/bin/env python3
"""
Experiment H: D2-only Window — Network-Level Emergent Effects of D1/D2 Tau Asymmetry

Scientific Question:
  After a short DA pulse (e.g. 5s), D2 receptors activate rapidly (τ_on≈10s)
  while D1 receptors barely respond (τ_on≈31s). During this "D2-only window":

  1. E/I Balance Shift: D2 suppresses both E-D2 and I-D2, but E-D2 is 25% of E
     while I-D2 is only 8% of I. Does this asymmetry cause a transient E/I shift?

  2. Disinhibition: When I-D2 firing drops, its inhibition on E neurons weakens.
     Does E-D1 show a rebound increase due to this disinhibition effect?

  3. After DA withdrawal, D2 decays fast but D1 retains activation.
     In this "D1-afterglow window", is the network in a pure D1-enhanced state?

  These are EMERGENT network effects that cannot be predicted from receptor
  kinetics equations alone — they depend on recurrent connectivity.

Sub-experiments:
  A. Time-Window Analysis: Apply 5s DA pulse, compute firing rates and E/I ratio
     in 4 time windows (Baseline / D2-only / Transition / D1-afterglow).
     Repeat for 5s, 15s, 30s pulses to show how the effect changes.

  B. Causal Verification: Block I-D2 D2-modulation (set mask_d2=False for I-D2
     neurons), re-run 5s pulse. If E-D1 rebound disappears, it confirms the
     disinhibition mechanism.

Usage:
  python -m experiments.exp_h_d2_window
  python -m experiments.exp_h_d2_window --sub a
  python -m experiments.exp_h_d2_window --sub b
  python -m experiments.exp_h_d2_window --sub all
  python -m experiments.exp_h_d2_window --da-pulse 15 --gpu 0
"""
import argparse
import json
import time
import os
import sys
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
                                plot_rates_simple_all,
                                plot_rates_simple_E,
                                plot_rates_simple_I)

RATE_GROUPS = ['E-D1', 'E-D2', 'E-Other', 'All-E', 'I-D1', 'I-D2', 'I-Other', 'All-I']

# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment H: D2-only Window Network Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sub", type=str, default="all",
                        choices=["a", "b", "all"],
                        help="Sub-experiment: a=time-window analysis, b=causal verification, all=both")
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
    if m > 0:
        return f"{m}m {s:05.2f}s"
    return f"{s:.2f}s"


# ==============================================================================
# Checkpoint Management (reuse from exp_e)
# ==============================================================================

def find_or_create_checkpoint(args, device):
    """Find existing checkpoint or create one."""
    if args.ckpt:
        if os.path.exists(args.ckpt):
            return args.ckpt
        print(f"❌ Checkpoint not found: {args.ckpt}")
        sys.exit(1)

    # Auto-detect
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    pattern = f"ckpt_DA{args.da_base:g}nM_bg{int(config.BG_MEAN)}_{int(args.base_dur)}s.pkl"
    ckpt_path = ckpt_dir / pattern

    if ckpt_path.exists() and not args.skip_ckpt:
        print(f"✅ Found existing checkpoint: {ckpt_path}")
        return str(ckpt_path)

    if args.skip_ckpt:
        print(f"❌ --skip-ckpt but no checkpoint found at {ckpt_path}")
        sys.exit(1)

    # Create checkpoint
    print(f"🔧 Creating baseline checkpoint: DA={args.da_base}nM, {args.base_dur}s...")
    from simulation.runners import run_simulation_d1_d2_ckpt
    from simulation.utils import save_checkpoint
    data = run_simulation_d1_d2_ckpt(
        duration=args.base_dur * 1000.0,
        target_da=args.da_base,
        device=device,
    )
    save_checkpoint(data, da_level=args.da_base, duration_s=args.base_dur)
    print(f"✅ Checkpoint saved: {ckpt_path}")
    return str(ckpt_path)


# ==============================================================================
# Core Simulation
# ==============================================================================

def run_pulse_sim(ckpt_path: str, device: torch.device,
                  da_base: float, da_pulse: float,
                  pre_pulse_s: float, pulse_duration_s: float, post_pulse_s: float,
                  alpha_record_interval: int = 100,
                  block_i_d2: bool = False):
    """
    Run a single DA pulse simulation from checkpoint.

    Args:
        block_i_d2: If True, remove D2 modulation from I-D2 neurons (causal test).
    """
    with open(ckpt_path, 'rb') as f:
        ckpt_data = pickle.load(f)
    init_state = ckpt_data['final_state'].to(device)
    if init_state.shape[0] >= 2:
        init_state[0] = init_state[1].clone()

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(config.N_E, config.N_I, device)

    # Causal manipulation: block D2 modulation on I-D2 neurons
    if block_i_d2:
        N_E = config.N_E
        n_i_d1 = int(config.N_I * config.FRAC_I_D1)
        n_i_d2 = int(config.N_I * config.FRAC_I_D2)
        i_d2_start = N_E + n_i_d1
        i_d2_end = i_d2_start + n_i_d2
        mask_d2[i_d2_start:i_d2_end] = False
        print(f"  ⚠️  I-D2 D2-modulation BLOCKED (neurons {i_d2_start}–{i_d2_end})")

    record_indices = torch.tensor([
        [0, 0], [1, 0],
        [0, groups_info['e_d1_end']], [1, groups_info['e_d1_end']],
    ], device=device, dtype=torch.long)

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


def generate_standard_plots(data: dict, save_dir: Path, label: str):
    """Generate simplified 2×2 rate plots (DA timeline + full rates, no zoom rows)."""
    sub_dir = save_dir / label
    sub_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎨 Generating plots for '{label}'...")
    analyzer = PFCAnalyzer(data)

    plot_combined_raster(analyzer, save_dir=sub_dir)
    plot_rates_simple_all(analyzer, save_dir=sub_dir)
    plot_rates_simple_E(analyzer, save_dir=sub_dir)
    plot_rates_simple_I(analyzer, save_dir=sub_dir)

    analyzer.save_report(str(sub_dir / "analysis_report.txt"))
    print(f"📁 Plots saved in: {sub_dir}")


# ==============================================================================
# Time-Window Analysis Utilities
# ==============================================================================

def compute_window_rates(data: dict, windows: dict, time_win_ms: float = 500.0) -> dict:
    """
    Compute mean firing rates for each subgroup in each time window.

    Args:
        data: simulation data dict
        windows: dict of {window_name: (start_s, end_s)}
        time_win_ms: bin width for rate computation

    Returns:
        {window_name: {group_name: {batch_id: mean_rate_hz}}}
    """
    analyzer = PFCAnalyzer.__new__(PFCAnalyzer)
    analyzer.data = data
    analyzer.cfg = data['config']
    analyzer.dt = data['config']['dt']
    analyzer.duration = data['config']['duration']
    analyzer.N_E = data['config']['N_E']
    analyzer.N_I = data['config']['N_I']
    analyzer.N = analyzer.N_E + analyzer.N_I
    analyzer.da_onset = data['config'].get('da_onset', 0)
    analyzer.da_level = data['config'].get('da_level', 0)
    analyzer.control_da = data['config'].get('control_da', 2.0)
    analyzer._build_group_masks()

    results = {}
    for wname, (start_s, end_s) in windows.items():
        start_ms = start_s * 1000.0
        end_ms = end_s * 1000.0
        results[wname] = {}
        for grp_name in RATE_GROUPS:
            results[wname][grp_name] = {}
            for batch_id in [0, 1]:
                centers, rate = analyzer.compute_group_rate(batch_id, grp_name, time_win=time_win_ms)
                if rate is None or len(rate) == 0:
                    results[wname][grp_name][batch_id] = 0.0
                    continue
                mask = (centers >= start_ms) & (centers < end_ms)
                if np.any(mask):
                    results[wname][grp_name][batch_id] = float(np.mean(rate[mask]))
                else:
                    results[wname][grp_name][batch_id] = 0.0
    return results


# ==============================================================================
# Sub-Experiment A: Time-Window Analysis
# ==============================================================================

def run_sub_a(args, ckpt_path: str, device: torch.device, save_dir: Path):
    """
    Sub-A: Time-Window Analysis for different pulse durations.

    For each pulse duration (5s, 15s, 30s):
      1. Run simulation
      2. Generate simplified rate plots
      3. Compute rates in 4 time windows
      4. Compute E/I ratio and disinhibition index
    """
    print("\n" + "=" * 70)
    print("  🔬 Sub-Experiment A: Time-Window Analysis")
    print("=" * 70)

    pulse_durations = [5.0, 15.0, 30.0]
    pre_pulse = 20.0
    post_pulse = 100.0  # Long post-pulse to observe D1-afterglow

    all_results = {}

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

        # Generate simplified rate plots
        generate_standard_plots(data, save_dir, label=f"sub_a_pulse_{dur_s:.0f}s")

        # Define 4 time windows (relative to simulation start)
        pulse_end = pre_pulse + dur_s
        windows = {
            'Baseline':      (pre_pulse - 10.0, pre_pulse),
            'D2-only':       (pulse_end, pulse_end + 10.0),
            'Transition':    (pulse_end + 10.0, pulse_end + 30.0),
            'D1-afterglow':  (pulse_end + 30.0, pulse_end + 80.0),
        }

        # Compute rates in each window
        win_rates = compute_window_rates(data, windows)

        # Compute E/I ratio and disinhibition index
        analysis = {}
        for wname in windows:
            wr = win_rates[wname]
            # E/I ratio (Exp batch)
            e_rate = wr['All-E'][1]
            i_rate = wr['All-I'][1]
            ei_ratio = e_rate / i_rate if i_rate > 0.01 else 0.0

            # E/I ratio (Ctrl batch)
            e_rate_ctrl = wr['All-E'][0]
            i_rate_ctrl = wr['All-I'][0]
            ei_ratio_ctrl = e_rate_ctrl / i_rate_ctrl if i_rate_ctrl > 0.01 else 0.0

            # Disinhibition index: ΔR(E-D1) / ΔR(I-D2)
            # where ΔR = Exp - Ctrl
            delta_e_d1 = wr['E-D1'][1] - wr['E-D1'][0]
            delta_i_d2 = wr['I-D2'][1] - wr['I-D2'][0]
            di_index = delta_e_d1 / delta_i_d2 if abs(delta_i_d2) > 0.01 else 0.0

            analysis[wname] = {
                'ei_ratio_exp': ei_ratio,
                'ei_ratio_ctrl': ei_ratio_ctrl,
                'ei_ratio_shift': ei_ratio - ei_ratio_ctrl,
                'delta_e_d1': delta_e_d1,
                'delta_i_d2': delta_i_d2,
                'disinhibition_index': di_index,
                'rates': {grp: {'ctrl': wr[grp][0], 'exp': wr[grp][1]} for grp in RATE_GROUPS},
            }

        all_results[dur_s] = {
            'windows': {k: list(v) for k, v in windows.items()},
            'analysis': analysis,
        }

        # Print per-duration summary
        _print_window_table(dur_s, windows, analysis)

    # Plot comparison across pulse durations
    _plot_sub_a_comparison(all_results, pulse_durations, args, save_dir)

    return all_results


def _print_window_table(dur_s, windows, analysis):
    """Print a summary table for one pulse duration."""
    w = 110
    print(f"\n  {'─'*w}")
    print(f"  📊 Pulse={dur_s:.0f}s — Time-Window Analysis")
    print(f"  {'─'*w}")
    print(f"  {'Window':<16} │ {'E-D1':>8} {'E-D2':>8} {'I-D1':>8} {'I-D2':>8} │"
          f" {'E/I Exp':>8} {'E/I Ctrl':>8} {'ΔE/I':>8} │ {'DI Index':>9}")
    print(f"  {'─'*16}─┼─{'─'*8}─{'─'*8}─{'─'*8}─{'─'*8}─┼─"
          f"{'─'*8}─{'─'*8}─{'─'*8}─┼─{'─'*9}")

    for wname in windows:
        a = analysis[wname]
        r = a['rates']
        print(f"  {wname:<16} │"
              f" {r['E-D1']['exp']:>8.2f} {r['E-D2']['exp']:>8.2f}"
              f" {r['I-D1']['exp']:>8.2f} {r['I-D2']['exp']:>8.2f} │"
              f" {a['ei_ratio_exp']:>8.3f} {a['ei_ratio_ctrl']:>8.3f}"
              f" {a['ei_ratio_shift']:>+8.4f} │"
              f" {a['disinhibition_index']:>+9.3f}")

    print(f"  {'─'*w}")
    print(f"  Key: DI Index = ΔR(E-D1) / ΔR(I-D2). Negative = disinhibition (I-D2↓ → E-D1↑)")


def _plot_sub_a_comparison(all_results, pulse_durations, args, save_dir):
    """Plot comparison of E/I ratio shift and disinhibition across pulse durations."""
    window_names = ['Baseline', 'D2-only', 'Transition', 'D1-afterglow']
    window_colors = ['#9E9E9E', '#1f77b4', '#ff7f0e', '#d62728']

    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle(
        f"Exp H Sub-A: D2-only Window Network Effects — DA Pulse {args.da_base}→{args.da_pulse}→{args.da_base} nM\n"
        f"D1 τ_on≈{config.TAU_ON_D1/1000:.0f}s, D2 τ_on≈{config.TAU_ON_D2/1000:.0f}s",
        fontsize=16, fontweight='bold'
    )

    durs = np.array(pulse_durations)

    # Panel A: E/I ratio shift per window across pulse durations
    ax = axes[0, 0]
    for i, wname in enumerate(window_names):
        shifts = [all_results[d]['analysis'][wname]['ei_ratio_shift'] for d in pulse_durations]
        ax.plot(durs, shifts, 'o-', color=window_colors[i], linewidth=2.5, markersize=10, label=wname)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Pulse Duration (s)', fontsize=13)
    ax.set_ylabel('ΔE/I Ratio (Exp − Ctrl)', fontsize=13)
    ax.set_title('A. E/I Balance Shift by Time Window', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel B: Disinhibition index per window
    ax = axes[0, 1]
    for i, wname in enumerate(window_names):
        di_vals = [all_results[d]['analysis'][wname]['disinhibition_index'] for d in pulse_durations]
        ax.plot(durs, di_vals, 's-', color=window_colors[i], linewidth=2.5, markersize=10, label=wname)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Pulse Duration (s)', fontsize=13)
    ax.set_ylabel('Disinhibition Index\n(ΔE-D1 / ΔI-D2)', fontsize=13)
    ax.set_title('B. Disinhibition Index\n(Negative = I-D2↓ causes E-D1↑)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel C: ΔRate for key subgroups (5s pulse, bar chart by window)
    ax = axes[1, 0]
    dur_key = pulse_durations[0]  # 5s pulse
    key_groups = ['E-D1', 'E-D2', 'I-D1', 'I-D2']
    group_colors = ['#d62728', '#1f77b4', '#ff7f0e', '#9467bd']
    x = np.arange(len(window_names))
    bar_w = 0.18
    for j, grp in enumerate(key_groups):
        delta_rates = []
        for wname in window_names:
            r = all_results[dur_key]['analysis'][wname]['rates'][grp]
            delta_rates.append(r['exp'] - r['ctrl'])
        ax.bar(x + j * bar_w, delta_rates, bar_w, color=group_colors[j], alpha=0.8, label=grp)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks(x + 1.5 * bar_w)
    ax.set_xticklabels(window_names, fontsize=11)
    ax.set_ylabel('ΔRate (Exp − Ctrl) Hz', fontsize=13)
    ax.set_title(f'C. Rate Changes by Window ({dur_key:.0f}s Pulse)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: E-D1 rate across windows for different pulse durations
    ax = axes[1, 1]
    for i, dur in enumerate(pulse_durations):
        e_d1_rates = [all_results[dur]['analysis'][wname]['rates']['E-D1']['exp']
                      for wname in window_names]
        ax.plot(window_names, e_d1_rates, 'o-', linewidth=2.5, markersize=10,
                label=f'{dur:.0f}s pulse')
    # Also plot ctrl baseline
    bl_ctrl = all_results[pulse_durations[0]]['analysis']['Baseline']['rates']['E-D1']['ctrl']
    ax.axhline(bl_ctrl, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Ctrl baseline')
    ax.set_ylabel('E-D1 Firing Rate (Hz)', fontsize=13)
    ax.set_title('D. E-D1 Rate Across Windows\n(Does disinhibition cause rebound?)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "sub_a_window_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: sub_a_window_comparison.png")


# ==============================================================================
# Sub-Experiment B: Causal Verification — Block I-D2 D2-Modulation
# ==============================================================================

def run_sub_b(args, ckpt_path: str, device: torch.device, save_dir: Path):
    """
    Sub-B: Causal verification by blocking I-D2 D2-modulation.

    Run two simulations with 5s DA pulse:
      1. Normal: all D2 modulation intact
      2. Block I-D2: D2 modulation removed from I-D2 neurons only

    If E-D1 rebound in D2-only window disappears when I-D2 is blocked,
    it confirms the disinhibition mechanism.
    """
    print("\n" + "=" * 70)
    print("  🔬 Sub-Experiment B: Causal Verification — Block I-D2")
    print("=" * 70)

    pulse_dur = 5.0
    pre_pulse = 20.0
    post_pulse = 100.0

    # Run 1: Normal
    print("\n  ── Run 1: Normal (all D2 modulation intact) ──")
    t0 = time.time()
    data_normal = run_pulse_sim(
        ckpt_path, device,
        da_base=args.da_base, da_pulse=args.da_pulse,
        pre_pulse_s=pre_pulse, pulse_duration_s=pulse_dur, post_pulse_s=post_pulse,
        block_i_d2=False,
    )
    print(f"     Sim time: {_fmt_elapsed(time.time() - t0)}")
    generate_standard_plots(data_normal, save_dir, label="sub_b_normal")

    # Run 2: Block I-D2
    print("\n  ── Run 2: I-D2 D2-modulation BLOCKED ──")
    t0 = time.time()
    data_blocked = run_pulse_sim(
        ckpt_path, device,
        da_base=args.da_base, da_pulse=args.da_pulse,
        pre_pulse_s=pre_pulse, pulse_duration_s=pulse_dur, post_pulse_s=post_pulse,
        block_i_d2=True,
    )
    print(f"     Sim time: {_fmt_elapsed(time.time() - t0)}")
    generate_standard_plots(data_blocked, save_dir, label="sub_b_block_i_d2")

    # Compute window rates for both
    pulse_end = pre_pulse + pulse_dur
    windows = {
        'Baseline':      (pre_pulse - 10.0, pre_pulse),
        'D2-only':       (pulse_end, pulse_end + 10.0),
        'Transition':    (pulse_end + 10.0, pulse_end + 30.0),
        'D1-afterglow':  (pulse_end + 30.0, pulse_end + 80.0),
    }

    rates_normal = compute_window_rates(data_normal, windows)
    rates_blocked = compute_window_rates(data_blocked, windows)

    # Print comparison
    _print_causal_comparison(windows, rates_normal, rates_blocked)

    # Plot comparison
    _plot_sub_b_comparison(data_normal, data_blocked, windows,
                           rates_normal, rates_blocked, args, save_dir)

    return {
        'normal': rates_normal,
        'blocked': rates_blocked,
        'windows': {k: list(v) for k, v in windows.items()},
    }


def _print_causal_comparison(windows, rates_normal, rates_blocked):
    """Print comparison table: Normal vs I-D2 Blocked."""
    w = 100
    print(f"\n  {'='*w}")
    print(f"  📊 Causal Verification: Normal vs I-D2-Blocked (5s pulse)")
    print(f"  {'='*w}")
    print(f"  {'Window':<16} │ {'E-D1 Normal':>12} {'E-D1 Block':>12} {'Δ':>8} │"
          f" {'I-D2 Normal':>12} {'I-D2 Block':>12} {'Δ':>8}")
    print(f"  {'─'*16}─┼─{'─'*12}─{'─'*12}─{'─'*8}─┼─{'─'*12}─{'─'*12}─{'─'*8}")

    for wname in windows:
        rn = rates_normal[wname]
        rb = rates_blocked[wname]
        e_d1_n = rn['E-D1'][1]
        e_d1_b = rb['E-D1'][1]
        i_d2_n = rn['I-D2'][1]
        i_d2_b = rb['I-D2'][1]
        print(f"  {wname:<16} │"
              f" {e_d1_n:>12.2f} {e_d1_b:>12.2f} {e_d1_b - e_d1_n:>+8.2f} │"
              f" {i_d2_n:>12.2f} {i_d2_b:>12.2f} {i_d2_b - i_d2_n:>+8.2f}")

    print(f"  {'='*w}")
    print(f"  If E-D1 rebound disappears when I-D2 is blocked → disinhibition confirmed.")


def _plot_sub_b_comparison(data_normal, data_blocked, windows,
                           rates_normal, rates_blocked, args, save_dir):
    """Plot Normal vs I-D2-Blocked comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.suptitle(
        f"Exp H Sub-B: Causal Verification — Normal vs I-D2 D2-Blocked\n"
        f"DA: {args.da_base}→{args.da_pulse}→{args.da_base} nM, 5s pulse",
        fontsize=16, fontweight='bold'
    )

    window_names = list(windows.keys())

    # Panel A: E-D1 rate comparison (Normal vs Blocked)
    ax = axes[0, 0]
    e_d1_normal = [rates_normal[w]['E-D1'][1] for w in window_names]
    e_d1_blocked = [rates_blocked[w]['E-D1'][1] for w in window_names]
    x = np.arange(len(window_names))
    bar_w = 0.35
    ax.bar(x - bar_w/2, e_d1_normal, bar_w, color='#d62728', alpha=0.8, label='Normal')
    ax.bar(x + bar_w/2, e_d1_blocked, bar_w, color='#d62728', alpha=0.4,
           hatch='//', edgecolor='#d62728', label='I-D2 Blocked')
    ax.set_xticks(x)
    ax.set_xticklabels(window_names, fontsize=11)
    ax.set_ylabel('E-D1 Firing Rate (Hz)', fontsize=13)
    ax.set_title('A. E-D1 Rate: Normal vs I-D2 Blocked', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: I-D2 rate comparison
    ax = axes[0, 1]
    i_d2_normal = [rates_normal[w]['I-D2'][1] for w in window_names]
    i_d2_blocked = [rates_blocked[w]['I-D2'][1] for w in window_names]
    ax.bar(x - bar_w/2, i_d2_normal, bar_w, color='#9467bd', alpha=0.8, label='Normal')
    ax.bar(x + bar_w/2, i_d2_blocked, bar_w, color='#9467bd', alpha=0.4,
           hatch='//', edgecolor='#9467bd', label='I-D2 Blocked')
    ax.set_xticks(x)
    ax.set_xticklabels(window_names, fontsize=11)
    ax.set_ylabel('I-D2 Firing Rate (Hz)', fontsize=13)
    ax.set_title('B. I-D2 Rate: Normal vs Blocked\n(Blocked = no D2 suppression on I-D2)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Time-resolved E-D1 rate overlay
    ax = axes[1, 0]
    analyzer_n = PFCAnalyzer.__new__(PFCAnalyzer)
    analyzer_n.data = data_normal
    analyzer_n.cfg = data_normal['config']
    analyzer_n.dt = data_normal['config']['dt']
    analyzer_n.duration = data_normal['config']['duration']
    analyzer_n.N_E = data_normal['config']['N_E']
    analyzer_n.N_I = data_normal['config']['N_I']
    analyzer_n.N = analyzer_n.N_E + analyzer_n.N_I
    analyzer_n.da_onset = data_normal['config'].get('da_onset', 0)
    analyzer_n._build_group_masks()

    analyzer_b = PFCAnalyzer.__new__(PFCAnalyzer)
    analyzer_b.data = data_blocked
    analyzer_b.cfg = data_blocked['config']
    analyzer_b.dt = data_blocked['config']['dt']
    analyzer_b.duration = data_blocked['config']['duration']
    analyzer_b.N_E = data_blocked['config']['N_E']
    analyzer_b.N_I = data_blocked['config']['N_I']
    analyzer_b.N = analyzer_b.N_E + analyzer_b.N_I
    analyzer_b.da_onset = data_blocked['config'].get('da_onset', 0)
    analyzer_b._build_group_masks()

    tw = 500.0  # 500ms bins
    centers_n, rate_n = analyzer_n.compute_group_rate(1, 'E-D1', time_win=tw)
    centers_b, rate_b = analyzer_b.compute_group_rate(1, 'E-D1', time_win=tw)
    if rate_n is not None and rate_b is not None:
        ax.plot(centers_n / 1000.0, rate_n, color='#d62728', linewidth=2.5, label='E-D1 Normal')
        ax.plot(centers_b / 1000.0, rate_b, color='#d62728', linewidth=2.5, linestyle='--',
                alpha=0.6, label='E-D1 I-D2-Blocked')
        # Mark pulse window
        pulse_onset_s = data_normal['pulse_onset_s']
        pulse_offset_s = data_normal['pulse_offset_s']
        ax.axvspan(pulse_onset_s, pulse_offset_s, alpha=0.1, color='green', label='DA pulse')
        ax.axvline(pulse_onset_s, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axvline(pulse_offset_s, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('E-D1 Firing Rate (Hz)', fontsize=13)
    ax.set_title('C. E-D1 Time Course: Normal vs Blocked', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel D: ΔRate difference (Normal - Blocked) for key groups
    ax = axes[1, 1]
    key_groups = ['E-D1', 'E-D2', 'I-D1', 'I-D2']
    group_colors = ['#d62728', '#1f77b4', '#ff7f0e', '#9467bd']
    for j, grp in enumerate(key_groups):
        deltas = []
        for wname in window_names:
            rn = rates_normal[wname][grp][1]
            rb = rates_blocked[wname][grp][1]
            deltas.append(rn - rb)
        ax.bar(x + j * 0.18, deltas, 0.18, color=group_colors[j], alpha=0.8, label=grp)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks(x + 0.27)
    ax.set_xticklabels(window_names, fontsize=11)
    ax.set_ylabel('ΔRate (Normal − Blocked) Hz', fontsize=13)
    ax.set_title('D. Effect of Blocking I-D2\n(Positive = Normal has higher rate)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / "sub_b_causal_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: sub_b_causal_comparison.png")


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

    # Checkpoint
    ckpt_path = find_or_create_checkpoint(args, device)

    # Output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sub_tag = args.sub if args.sub != 'all' else 'ab'
    save_dir = PROJECT_ROOT / "outputs" / f"exp_h_d2_window_{timestamp}_{sub_tag}"
    save_dir.mkdir(parents=True, exist_ok=True)

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
        'EPS_D1': config.EPS_D1,
        'EPS_D2': config.EPS_D2,
        'BIAS_D1': config.BIAS_D1,
        'BIAS_D2': config.BIAS_D2,
        'LAM_D1': config.LAM_D1,
        'LAM_D2': config.LAM_D2,
        'N_E': config.N_E,
        'N_I': config.N_I,
        'FRAC_E_D1': config.FRAC_E_D1,
        'FRAC_E_D2': config.FRAC_E_D2,
        'FRAC_I_D1': config.FRAC_I_D1,
        'FRAC_I_D2': config.FRAC_I_D2,
    }
    with open(save_dir / "experiment_config.json", 'w') as f:
        json.dump(exp_config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  🧪 Experiment H: D2-only Window — Network Emergent Effects")
    print(f"{'='*70}")
    print(f"  D1: τ_on={config.TAU_ON_D1:.0f}ms ({config.TAU_ON_D1/1000:.0f}s), "
          f"τ_off={config.TAU_OFF_D1:.0f}ms ({config.TAU_OFF_D1/1000:.0f}s)")
    print(f"  D2: τ_on={config.TAU_ON_D2:.0f}ms ({config.TAU_ON_D2/1000:.0f}s), "
          f"τ_off={config.TAU_OFF_D2:.0f}ms ({config.TAU_OFF_D2/1000:.0f}s)")
    print(f"  DA: baseline={args.da_base}nM, pulse={args.da_pulse}nM")
    print(f"  Network: E-D2={int(config.N_E*config.FRAC_E_D2)} neurons (25% of E), "
          f"I-D2={int(config.N_I*config.FRAC_I_D2)} neurons (8% of I)")
    print(f"  Output: {save_dir}")
    print(f"{'='*70}\n")

    all_results = {}

    if args.sub in ['a', 'all']:
        all_results['sub_a'] = run_sub_a(args, ckpt_path, device, save_dir)

    if args.sub in ['b', 'all']:
        all_results['sub_b'] = run_sub_b(args, ckpt_path, device, save_dir)

    # Save all results
    with open(save_dir / "exp_h_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n📄 Results saved: {save_dir / 'exp_h_results.json'}")

    t_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"  ⏱️  Total time: {_fmt_elapsed(t_total)}")
    print(f"  📁 Results in: {save_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
