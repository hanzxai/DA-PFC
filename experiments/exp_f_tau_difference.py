#!/usr/bin/env python3
"""
Experiment F: D1/D2 Tau Difference — DA Staircase & Hysteresis

Scientific Question:
  D1 receptors have τ_on ≈ 31s / τ_off ≈ 164s (slow),
  D2 receptors have τ_on ≈ 10s / τ_off ≈ 50s  (fast, ~3x).
  How does this kinetic difference manifest when DA concentration
  changes in a staircase pattern (ascending then descending)?

Key Predictions:
  1. Ascending staircase: D2 rapidly reaches each new steady-state level,
     while D1 lags behind and accumulates slowly.
  2. Descending staircase: D2 quickly tracks each step down,
     while D1 retains a "memory" of the peak DA and decays slowly.
  3. Hysteresis: At the same DA concentration, α values differ between
     ascending and descending phases — the hysteresis loop is wider for D1.
  4. Network effect: E-D1 and E-D2 subgroup firing rates show different
     temporal profiles, with E-D2 tracking DA changes more faithfully.

Protocol:
  From DA=2nM steady-state checkpoint:
    Phase 0: [0, 20s)       → 2 nM  (pre-staircase baseline)
    Phase 1: [20s, 50s)     → 5 nM  (step 1 up)
    Phase 2: [50s, 80s)     → 10 nM (step 2 up)
    Phase 3: [80s, 110s)    → 15 nM (step 3 up — peak)
    Phase 4: [110s, 140s)   → 10 nM (step 1 down)
    Phase 5: [140s, 170s)   → 5 nM  (step 2 down)
    Phase 6: [170s, 270s)   → 2 nM  (recovery — observe slow D1 decay)

Sub-experiments:
  A. Staircase Alpha Dynamics: Full α_D1 / α_D2 traces with DA protocol overlay
  B. Hysteresis Loop: α vs DA concentration, showing ascending vs descending paths
  C. Network Firing Rate Response: Subgroup rate time courses + rate hysteresis

Usage:
  python -m experiments.exp_f_tau_difference
  python -m experiments.exp_f_tau_difference --sub a         # Staircase only
  python -m experiments.exp_f_tau_difference --sub b         # Hysteresis only
  python -m experiments.exp_f_tau_difference --sub c         # Network rates only
  python -m experiments.exp_f_tau_difference --sub all       # All sub-experiments
  python -m experiments.exp_f_tau_difference --gpu 1
  python -m experiments.exp_f_tau_difference --da-peak 20    # Custom peak DA
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
from models.kernels import run_dynamic_d1_d2_kernel_da_schedule
from analysis.analyzer import PFCAnalyzer

RATE_GROUPS = ['E-D1', 'E-D2', 'E-Other', 'All-E', 'I-D1', 'I-D2', 'I-Other', 'All-I']


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment F: D1/D2 Tau Difference — DA Staircase & Hysteresis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sub", type=str, default="all",
                        choices=["a", "b", "c", "all"],
                        help="Sub-experiment: a=staircase, b=hysteresis, c=network rates, all=run all")
    parser.add_argument("--da-base", type=float, default=2.0,
                        help="Baseline DA concentration (nM), default 2.0")
    parser.add_argument("--da-peak", type=float, default=15.0,
                        help="Peak DA concentration (nM), default 15.0")
    parser.add_argument("--step-duration", type=float, default=30.0,
                        help="Duration of each staircase step (s), default 30")
    parser.add_argument("--recovery-duration", type=float, default=100.0,
                        help="Duration of recovery phase after staircase (s), default 100")
    parser.add_argument("--n-steps", type=int, default=3,
                        help="Number of ascending steps (default 3: base→5→10→15)")
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

    if args.skip_ckpt:
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run without --skip-ckpt to auto-generate, or provide --ckpt path."
        )

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
# DA Staircase Schedule Builder
# ==============================================================================

def build_staircase_schedule(args, dt: float):
    """
    Build a DA staircase schedule: ascending steps → descending steps → recovery.

    Returns:
        da_levels  : list of (start_s, end_s, da_nM) tuples describing each phase
        da_schedule: (steps, 2) numpy array, col 0 = control (constant baseline),
                     col 1 = experiment (staircase)
        total_s    : total duration in seconds
    """
    da_base = args.da_base
    da_peak = args.da_peak
    step_dur = args.step_duration
    recovery_dur = args.recovery_duration
    n_steps = args.n_steps

    # Build DA level sequence: base → intermediate steps → peak → reverse → base
    da_values = np.linspace(da_base, da_peak, n_steps + 1)  # e.g. [2, 5.33, 8.67, 12] for n=3
    # Use nicer round values
    da_values = np.round(da_values, 1)
    da_values[0] = da_base
    da_values[-1] = da_peak

    # Phase list: pre-baseline, ascending, descending, recovery
    pre_baseline = 20.0  # seconds
    phases = []

    # Phase 0: Pre-baseline
    phases.append((0.0, pre_baseline, da_base))

    # Ascending phases
    t = pre_baseline
    for i in range(1, len(da_values)):
        phases.append((t, t + step_dur, float(da_values[i])))
        t += step_dur

    # Descending phases (skip the peak, go from n_steps-1 down to 1)
    for i in range(len(da_values) - 2, 0, -1):
        phases.append((t, t + step_dur, float(da_values[i])))
        t += step_dur

    # Recovery phase (back to baseline)
    phases.append((t, t + recovery_dur, da_base))
    t += recovery_dur

    total_s = t
    total_steps = int(total_s * 1000.0 / dt)

    # Build schedule array
    t_ms = np.arange(total_steps) * dt
    t_s = t_ms / 1000.0

    da_ctrl = np.full(total_steps, da_base)
    da_exp = np.full(total_steps, da_base)

    for start_s, end_s, da_val in phases:
        mask = (t_s >= start_s) & (t_s < end_s)
        da_exp[mask] = da_val

    da_schedule = np.stack([da_ctrl, da_exp], axis=1)

    return phases, da_schedule, total_s, da_values


# ==============================================================================
# Common Simulation Runner
# ==============================================================================

def run_staircase_sim(ckpt_path: str, device: torch.device, args,
                      alpha_record_interval: int = 100):
    """
    Run the DA staircase simulation from checkpoint.

    Returns:
        data: dict with spikes, alpha traces, phases, config, etc.
    """
    dt = config.DT
    phases, da_schedule, total_s, da_values = build_staircase_schedule(args, dt)

    print(f"\n  📋 DA Staircase Protocol:")
    print(f"  {'Phase':<8} {'Time Window':<20} {'DA (nM)':<10}")
    print(f"  {'─'*8} {'─'*20} {'─'*10}")
    for i, (start, end, da) in enumerate(phases):
        label = "baseline" if i == 0 else ("recovery" if i == len(phases) - 1 else
                ("↑ step" if da > phases[i-1][2] else "↓ step"))
        print(f"  {label:<8} [{start:>6.0f}s, {end:>6.0f}s)  {da:>6.1f} nM")
    print(f"  Total duration: {total_s:.0f}s")

    # Load checkpoint
    with open(ckpt_path, 'rb') as f:
        ckpt_data = pickle.load(f)
    init_state = ckpt_data['final_state'].to(device)
    if init_state.shape[0] >= 2:
        init_state[0] = init_state[1].clone()

    # Build network (same seed)
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(config.N_E, config.N_I, device)

    record_indices = torch.tensor([
        [0, 0], [1, 0],
        [0, groups_info['e_d1_end']], [1, groups_info['e_d1_end']],
    ], device=device, dtype=torch.long)

    da_schedule_tensor = torch.tensor(da_schedule, dtype=torch.float32, device=device)
    kp = config.build_kernel_params(device)

    print(f"\n  ⚡ Running staircase simulation ({total_s:.0f}s)...")
    t0 = time.time()

    result = run_dynamic_d1_d2_kernel_da_schedule(
        W_t, mask_d1, mask_d2, init_state,
        da_schedule_tensor,
        float(total_s * 1000.0), dt,
        record_indices, config.N_E,
        alpha_record_interval, kp,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"  ✅ Simulation done in {_fmt_elapsed(elapsed)}")

    all_spikes, v_traces, final_state, alpha_d1_trace, alpha_d2_trace = result

    data = {
        'config': {
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': total_s * 1000.0, 'dt': config.DT,
            'da_onset': phases[1][0] * 1000.0,  # first step onset
            'da_level': args.da_peak,
            'control_da': args.da_base,
            'mode': 'staircase',
        },
        'masks': {'d1': mask_d1.cpu(), 'd2': mask_d2.cpu()},
        'groups_info': groups_info,
        'spikes': all_spikes.cpu(),
        'v_traces': v_traces.cpu(),
        'record_indices': record_indices.cpu(),
        'alpha_d1_trace': alpha_d1_trace.cpu().numpy(),
        'alpha_d2_trace': alpha_d2_trace.cpu().numpy(),
        'phases': phases,
        'da_schedule': da_schedule,
        'da_values': da_values,
        'total_s': total_s,
        'sim_time': elapsed,
    }
    return data


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


def _get_alpha_at_da_levels(alpha_trace, alpha_times, phases, ascending=True):
    """
    Extract mean alpha values at each DA level during ascending or descending phase.

    For each step, take the mean alpha in the last 1/3 of the step duration
    (to allow partial equilibration).

    Returns:
        da_levels: list of DA concentrations
        alpha_vals: list of corresponding mean alpha values
    """
    da_levels = []
    alpha_vals = []

    for i, (start, end, da) in enumerate(phases):
        if i == 0 or i == len(phases) - 1:
            continue  # skip pre-baseline and recovery

        # Determine if this is ascending or descending
        prev_da = phases[i - 1][2]
        is_ascending = (da >= prev_da)

        if ascending and not is_ascending:
            continue
        if not ascending and is_ascending:
            continue

        # Take mean alpha in the last 1/3 of the step
        step_dur = end - start
        sample_start = end - step_dur / 3.0
        mask = (alpha_times >= sample_start) & (alpha_times < end)

        if np.any(mask):
            da_levels.append(da)
            alpha_vals.append(float(np.mean(alpha_trace[mask])))

    return da_levels, alpha_vals


# ==============================================================================
# Sub-Experiment A: Staircase Alpha Dynamics
# ==============================================================================

def run_sub_a(data: dict, args, save_dir: Path):
    """
    Sub-Experiment A: Staircase Alpha Dynamics

    Visualize the full α_D1 / α_D2 traces overlaid with the DA staircase protocol.
    Key observation: D2 tracks each step closely, D1 lags behind.
    """
    print("\n" + "=" * 70)
    print("  🔬 Sub-Experiment A: Staircase Alpha Dynamics")
    print("=" * 70)

    alpha_d1 = data['alpha_d1_trace']
    alpha_d2 = data['alpha_d2_trace']
    phases = data['phases']
    total_s = data['total_s']

    alpha_d1_exp = alpha_d1[:, 1]  # Experiment batch
    alpha_d2_exp = alpha_d2[:, 1]
    alpha_d1_ctrl = alpha_d1[:, 0]  # Control batch
    alpha_d2_ctrl = alpha_d2[:, 0]
    alpha_times = np.linspace(0, total_s, len(alpha_d1_exp))

    # ================================================================
    # Figure 1: Main result — DA protocol + Alpha dynamics (3 panels)
    # ================================================================
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 1, figure=fig, hspace=0.3, height_ratios=[1, 2, 2])
    fig.suptitle(
        f"Experiment F-A: DA Staircase — α_D1 vs α_D2 Dynamics\n"
        f"D1 τ_on≈{config.TAU_ON_D1/1000:.0f}s (slow)  |  "
        f"D2 τ_on≈{config.TAU_ON_D2/1000:.0f}s (fast, ~{config.TAU_ON_D1/config.TAU_ON_D2:.0f}x faster)",
        fontsize=15, fontweight='bold'
    )

    # --- Panel A: DA protocol ---
    ax_da = fig.add_subplot(gs[0])
    da_t = np.linspace(0, total_s, data['da_schedule'].shape[0])
    da_exp = data['da_schedule'][:, 1]
    ax_da.fill_between(da_t, 0, da_exp, alpha=0.3, color='green', step='post')
    ax_da.step(da_t, da_exp, color='green', linewidth=2.5, where='post')
    ax_da.set_ylabel('DA (nM)', fontsize=13)
    ax_da.set_title('A. DA Staircase Protocol', fontsize=14, fontweight='bold')
    ax_da.set_xlim(0, total_s)
    ax_da.set_ylim(0, args.da_peak * 1.15)
    ax_da.grid(True, alpha=0.3)

    # Mark ascending / descending
    asc_end = None
    for i, (start, end, da) in enumerate(phases):
        if i > 0 and i < len(phases) - 1:
            prev_da = phases[i - 1][2]
            if da < prev_da and asc_end is None:
                asc_end = start
    if asc_end:
        ax_da.axvline(asc_end, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
        ax_da.text(asc_end - 2, args.da_peak * 1.05, '← ascending', fontsize=10,
                   ha='right', color='#2196F3', fontweight='bold')
        ax_da.text(asc_end + 2, args.da_peak * 1.05, 'descending →', fontsize=10,
                   ha='left', color='#F44336', fontweight='bold')

    # --- Panel B: Alpha D1 dynamics ---
    ax_d1 = fig.add_subplot(gs[1])
    ax_d1.plot(alpha_times, alpha_d1_exp, '-', color='#d62728', linewidth=2.5,
               label='α_D1 (Exp) — SLOW')
    ax_d1.plot(alpha_times, alpha_d1_ctrl, '--', color='#d62728', linewidth=1.5,
               alpha=0.4, label='α_D1 (Control)')

    # Shade staircase steps
    for i, (start, end, da) in enumerate(phases):
        if i == 0 or i == len(phases) - 1:
            continue
        intensity = (da - args.da_base) / (args.da_peak - args.da_base)
        ax_d1.axvspan(start, end, alpha=0.08 * intensity, color='green')
        ax_d1.text((start + end) / 2, ax_d1.get_ylim()[0] if ax_d1.get_ylim()[0] != 0 else 0,
                   f'{da:.0f}nM', fontsize=8, ha='center', va='bottom', color='gray')

    ax_d1.set_ylabel('α_D1', fontsize=13, color='#d62728')
    ax_d1.set_title('B. D1 Receptor Activation (τ_on≈31s — SLOW response)',
                     fontsize=14, fontweight='bold')
    ax_d1.legend(fontsize=11, loc='upper right')
    ax_d1.set_xlim(0, total_s)
    ax_d1.grid(True, alpha=0.3)

    # --- Panel C: Alpha D2 dynamics ---
    ax_d2 = fig.add_subplot(gs[2])
    ax_d2.plot(alpha_times, alpha_d2_exp, '-', color='#1f77b4', linewidth=2.5,
               label='α_D2 (Exp) — FAST')
    ax_d2.plot(alpha_times, alpha_d2_ctrl, '--', color='#1f77b4', linewidth=1.5,
               alpha=0.4, label='α_D2 (Control)')

    for i, (start, end, da) in enumerate(phases):
        if i == 0 or i == len(phases) - 1:
            continue
        intensity = (da - args.da_base) / (args.da_peak - args.da_base)
        ax_d2.axvspan(start, end, alpha=0.08 * intensity, color='green')

    ax_d2.set_ylabel('α_D2', fontsize=13, color='#1f77b4')
    ax_d2.set_xlabel('Time (s)', fontsize=13)
    ax_d2.set_title('C. D2 Receptor Activation (τ_on≈10s — FAST response)',
                     fontsize=14, fontweight='bold')
    ax_d2.legend(fontsize=11, loc='upper right')
    ax_d2.set_xlim(0, total_s)
    ax_d2.grid(True, alpha=0.3)

    plt.savefig(save_dir / "sub_a_staircase_alpha.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: sub_a_staircase_alpha.png")

    # ================================================================
    # Figure 2: D1 vs D2 overlay + Δα
    # ================================================================
    fig, axes = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
    fig.suptitle(
        "D1 vs D2 Receptor Activation — Direct Comparison\n"
        "(Same staircase protocol, different kinetics)",
        fontsize=14, fontweight='bold'
    )

    # Panel A: Overlay
    ax = axes[0]
    ax.plot(alpha_times, alpha_d1_exp, '-', color='#d62728', linewidth=2.5,
            label=f'α_D1 (τ_on≈{config.TAU_ON_D1/1000:.0f}s)')
    ax.plot(alpha_times, alpha_d2_exp, '-', color='#1f77b4', linewidth=2.5,
            label=f'α_D2 (τ_on≈{config.TAU_ON_D2/1000:.0f}s)')

    # DA protocol on twin axis
    ax_twin = ax.twinx()
    ax_twin.step(da_t, da_exp, color='green', linewidth=1.5, alpha=0.4, where='post')
    ax_twin.fill_between(da_t, 0, da_exp, alpha=0.08, color='green', step='post')
    ax_twin.set_ylabel('[DA] (nM)', fontsize=11, color='green', alpha=0.6)
    ax_twin.set_ylim(0, args.da_peak * 2)

    ax.set_ylabel('Receptor Activation (α)', fontsize=13)
    ax.set_title('A. α_D1 vs α_D2 — D2 tracks DA steps, D1 lags behind', fontsize=13)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel B: Δα = D1 - D2
    ax = axes[1]
    delta_alpha = alpha_d1_exp - alpha_d2_exp
    ax.plot(alpha_times, delta_alpha, '-', color='purple', linewidth=2.5,
            label='Δα = α_D1 − α_D2')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.fill_between(alpha_times, 0, delta_alpha,
                    where=(delta_alpha > 0), alpha=0.3, color='red', label='D1 > D2')
    ax.fill_between(alpha_times, 0, delta_alpha,
                    where=(delta_alpha < 0), alpha=0.3, color='blue', label='D2 > D1')

    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Δα (D1 − D2)', fontsize=13)
    ax.set_title('B. Temporal Segregation — Δα reveals kinetic difference', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, total_s)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "sub_a_alpha_overlay.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: sub_a_alpha_overlay.png")

    # ---- Print quantitative summary ----
    _print_sub_a_summary(data, alpha_times, alpha_d1_exp, alpha_d2_exp)


def _print_sub_a_summary(data, alpha_times, alpha_d1_exp, alpha_d2_exp):
    """Print quantitative summary of alpha dynamics at each staircase step."""
    phases = data['phases']
    w = 90
    print("\n" + "=" * w)
    print("  📊 Sub-Exp A: Alpha Values at Each Staircase Step")
    print("=" * w)
    print(f"  {'Phase':<12} {'DA (nM)':>8} │ {'α_D1':>10} {'α_D2':>10} │ {'Δα(D1-D2)':>10} {'D2/D1':>8}")
    print(f"  {'─'*12}─{'─'*8}─┼─{'─'*10}─{'─'*10}─┼─{'─'*10}─{'─'*8}")

    for i, (start, end, da) in enumerate(phases):
        # Sample from last 1/3 of each step
        step_dur = end - start
        sample_start = end - step_dur / 3.0
        mask = (alpha_times >= sample_start) & (alpha_times < end)
        if not np.any(mask):
            continue

        mean_d1 = float(np.mean(alpha_d1_exp[mask]))
        mean_d2 = float(np.mean(alpha_d2_exp[mask]))
        delta = mean_d1 - mean_d2
        ratio = mean_d2 / mean_d1 if mean_d1 > 0.001 else float('inf')

        if i == 0:
            label = "baseline"
        elif i == len(phases) - 1:
            label = "recovery"
        else:
            prev_da = phases[i - 1][2]
            label = "↑ ascend" if da > prev_da else ("↓ descend" if da < prev_da else "= hold")

        print(f"  {label:<12} {da:>7.1f}  │ {mean_d1:>10.4f} {mean_d2:>10.4f} │ {delta:>+10.4f} {ratio:>7.2f}x")

    print("=" * w)
    print("  Key: D2/D1 > 1 means D2 is more activated. During ascending, D2 leads.")
    print("       During descending, D1 retains higher values (slower decay).")
    print("=" * w)


# ==============================================================================
# Sub-Experiment B: Hysteresis Loop Analysis
# ==============================================================================

def run_sub_b(data: dict, args, save_dir: Path):
    """
    Sub-Experiment B: Hysteresis Loop Analysis

    Plot α vs DA concentration for ascending and descending phases separately.
    The gap between the two curves reveals the kinetic "memory" effect.
    D1 shows a wider hysteresis loop (slower kinetics → more memory).
    """
    print("\n" + "=" * 70)
    print("  🔬 Sub-Experiment B: Hysteresis Loop Analysis")
    print("=" * 70)

    alpha_d1_exp = data['alpha_d1_trace'][:, 1]
    alpha_d2_exp = data['alpha_d2_trace'][:, 1]
    alpha_times = np.linspace(0, data['total_s'], len(alpha_d1_exp))
    phases = data['phases']

    # Extract alpha values at each DA level for ascending and descending
    da_asc_d1, alpha_asc_d1 = _get_alpha_at_da_levels(alpha_d1_exp, alpha_times, phases, ascending=True)
    da_desc_d1, alpha_desc_d1 = _get_alpha_at_da_levels(alpha_d1_exp, alpha_times, phases, ascending=False)
    da_asc_d2, alpha_asc_d2 = _get_alpha_at_da_levels(alpha_d2_exp, alpha_times, phases, ascending=True)
    da_desc_d2, alpha_desc_d2 = _get_alpha_at_da_levels(alpha_d2_exp, alpha_times, phases, ascending=False)

    # Add baseline point to both ascending and descending
    # Use the last 1/3 of the pre-staircase baseline period for averaging
    bl_start = phases[1][0] * 2.0 / 3.0  # last 1/3 of baseline
    bl_end = phases[1][0]
    bl_mask = (alpha_times >= bl_start) & (alpha_times < bl_end)
    if np.any(bl_mask):
        bl_d1 = float(np.mean(alpha_d1_exp[bl_mask]))
        bl_d2 = float(np.mean(alpha_d2_exp[bl_mask]))
        da_asc_d1 = [args.da_base] + da_asc_d1
        alpha_asc_d1 = [bl_d1] + alpha_asc_d1
        da_asc_d2 = [args.da_base] + da_asc_d2
        alpha_asc_d2 = [bl_d2] + alpha_asc_d2

    # Add recovery point to descending
    rec_phase = phases[-1]
    rec_mask = (alpha_times >= rec_phase[1] - rec_phase[1] / 3.0) & (alpha_times < rec_phase[1])
    # Use the last portion of recovery
    rec_end_mask = (alpha_times >= data['total_s'] - 10) & (alpha_times < data['total_s'])
    if np.any(rec_end_mask):
        rec_d1 = float(np.mean(alpha_d1_exp[rec_end_mask]))
        rec_d2 = float(np.mean(alpha_d2_exp[rec_end_mask]))
        da_desc_d1 = da_desc_d1 + [args.da_base]
        alpha_desc_d1 = alpha_desc_d1 + [rec_d1]
        da_desc_d2 = da_desc_d2 + [args.da_base]
        alpha_desc_d2 = alpha_desc_d2 + [rec_d2]

    # ================================================================
    # Figure: Hysteresis loops — D1 vs D2 side by side
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(
        "Experiment F-B: DA-α Hysteresis Loops — D1 vs D2\n"
        "Wider loop = slower kinetics = more temporal \"memory\"",
        fontsize=15, fontweight='bold'
    )

    # Panel A: D1 hysteresis
    ax = axes[0]
    ax.plot(da_asc_d1, alpha_asc_d1, 'o-', color='#d62728', linewidth=2.5, markersize=10,
            label='Ascending (DA ↑)', zorder=3)
    ax.plot(da_desc_d1, alpha_desc_d1, 's--', color='#d62728', linewidth=2.5, markersize=10,
            alpha=0.7, label='Descending (DA ↓)', zorder=3)

    # Fill hysteresis area
    if len(da_asc_d1) > 1 and len(da_desc_d1) > 1:
        # Interpolate to common DA grid for fill
        da_common = sorted(set(da_asc_d1) | set(da_desc_d1))
        asc_interp = np.interp(da_common, da_asc_d1, alpha_asc_d1)
        desc_interp = np.interp(da_common, da_desc_d1[::-1], alpha_desc_d1[::-1])
        ax.fill_between(da_common, asc_interp, desc_interp, alpha=0.15, color='#d62728')

    # Add arrows to show direction
    for i in range(len(da_asc_d1) - 1):
        ax.annotate('', xy=(da_asc_d1[i+1], alpha_asc_d1[i+1]),
                    xytext=(da_asc_d1[i], alpha_asc_d1[i]),
                    arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5))

    ax.set_xlabel('DA Concentration (nM)', fontsize=13)
    ax.set_ylabel('α_D1', fontsize=13, color='#d62728')
    ax.set_title(f'A. D1 Hysteresis (τ_on≈{config.TAU_ON_D1/1000:.0f}s)\nWIDE loop = slow kinetics',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel B: D2 hysteresis
    ax = axes[1]
    ax.plot(da_asc_d2, alpha_asc_d2, 'o-', color='#1f77b4', linewidth=2.5, markersize=10,
            label='Ascending (DA ↑)', zorder=3)
    ax.plot(da_desc_d2, alpha_desc_d2, 's--', color='#1f77b4', linewidth=2.5, markersize=10,
            alpha=0.7, label='Descending (DA ↓)', zorder=3)

    if len(da_asc_d2) > 1 and len(da_desc_d2) > 1:
        da_common = sorted(set(da_asc_d2) | set(da_desc_d2))
        asc_interp = np.interp(da_common, da_asc_d2, alpha_asc_d2)
        desc_interp = np.interp(da_common, da_desc_d2[::-1], alpha_desc_d2[::-1])
        ax.fill_between(da_common, asc_interp, desc_interp, alpha=0.15, color='#1f77b4')

    for i in range(len(da_asc_d2) - 1):
        ax.annotate('', xy=(da_asc_d2[i+1], alpha_asc_d2[i+1]),
                    xytext=(da_asc_d2[i], alpha_asc_d2[i]),
                    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5))

    ax.set_xlabel('DA Concentration (nM)', fontsize=13)
    ax.set_ylabel('α_D2', fontsize=13, color='#1f77b4')
    ax.set_title(f'B. D2 Hysteresis (τ_on≈{config.TAU_ON_D2/1000:.0f}s)\nNARROW loop = fast kinetics',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel C: Hysteresis width comparison
    ax = axes[2]

    # Compute hysteresis width at each shared DA level
    shared_da = sorted(set(da_asc_d1[1:]) & set(da_desc_d1[:-1]))  # exclude baseline endpoints
    if shared_da:
        hyst_d1 = []
        hyst_d2 = []
        for da_val in shared_da:
            # Find alpha at this DA level in ascending and descending
            if da_val in da_asc_d1 and da_val in da_desc_d1:
                idx_asc = da_asc_d1.index(da_val)
                idx_desc = da_desc_d1.index(da_val)
                hyst_d1.append(alpha_desc_d1[idx_desc] - alpha_asc_d1[idx_asc])
            else:
                hyst_d1.append(0)

            if da_val in da_asc_d2 and da_val in da_desc_d2:
                idx_asc = da_asc_d2.index(da_val)
                idx_desc = da_desc_d2.index(da_val)
                hyst_d2.append(alpha_desc_d2[idx_desc] - alpha_asc_d2[idx_asc])
            else:
                hyst_d2.append(0)

        x = np.arange(len(shared_da))
        bar_width = 0.35
        bars1 = ax.bar(x - bar_width/2, hyst_d1, bar_width, color='#d62728',
                        label=f'D1 (τ_on≈{config.TAU_ON_D1/1000:.0f}s)', alpha=0.8)
        bars2 = ax.bar(x + bar_width/2, hyst_d2, bar_width, color='#1f77b4',
                        label=f'D2 (τ_on≈{config.TAU_ON_D2/1000:.0f}s)', alpha=0.8)

        # Value labels
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='#d62728')
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='#1f77b4')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{da:.0f} nM' for da in shared_da], fontsize=11)

    ax.set_xlabel('DA Concentration (nM)', fontsize=13)
    ax.set_ylabel('Hysteresis Width (α_desc − α_asc)', fontsize=13)
    ax.set_title('C. Hysteresis Width Comparison\nD1 >> D2 (slower = wider)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / "sub_b_hysteresis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: sub_b_hysteresis.png")

    # ---- Print hysteresis summary ----
    _print_sub_b_summary(da_asc_d1, alpha_asc_d1, da_desc_d1, alpha_desc_d1,
                         da_asc_d2, alpha_asc_d2, da_desc_d2, alpha_desc_d2)

    return {
        'da_asc_d1': da_asc_d1, 'alpha_asc_d1': alpha_asc_d1,
        'da_desc_d1': da_desc_d1, 'alpha_desc_d1': alpha_desc_d1,
        'da_asc_d2': da_asc_d2, 'alpha_asc_d2': alpha_asc_d2,
        'da_desc_d2': da_desc_d2, 'alpha_desc_d2': alpha_desc_d2,
    }


def _print_sub_b_summary(da_asc_d1, alpha_asc_d1, da_desc_d1, alpha_desc_d1,
                          da_asc_d2, alpha_asc_d2, da_desc_d2, alpha_desc_d2):
    """Print hysteresis summary table."""
    w = 85
    print("\n" + "=" * w)
    print("  📊 Sub-Exp B: Hysteresis Loop — Summary")
    print("=" * w)
    print(f"  {'DA (nM)':>8} │ {'D1 asc':>10} {'D1 desc':>10} {'D1 Δ':>10} │"
          f" {'D2 asc':>10} {'D2 desc':>10} {'D2 Δ':>10}")
    print(f"  {'─'*8}─┼─{'─'*10}─{'─'*10}─{'─'*10}─┼─{'─'*10}─{'─'*10}─{'─'*10}")

    # Merge all DA levels
    all_da = sorted(set(da_asc_d1) | set(da_desc_d1))
    for da_val in all_da:
        d1_asc = alpha_asc_d1[da_asc_d1.index(da_val)] if da_val in da_asc_d1 else float('nan')
        d1_desc = alpha_desc_d1[da_desc_d1.index(da_val)] if da_val in da_desc_d1 else float('nan')
        d2_asc = alpha_asc_d2[da_asc_d2.index(da_val)] if da_val in da_asc_d2 else float('nan')
        d2_desc = alpha_desc_d2[da_desc_d2.index(da_val)] if da_val in da_desc_d2 else float('nan')

        d1_delta = d1_desc - d1_asc if not (np.isnan(d1_asc) or np.isnan(d1_desc)) else float('nan')
        d2_delta = d2_desc - d2_asc if not (np.isnan(d2_asc) or np.isnan(d2_desc)) else float('nan')

        def _fmt(v):
            return f"{v:>10.4f}" if not np.isnan(v) else f"{'N/A':>10}"

        print(f"  {da_val:>7.1f}  │ {_fmt(d1_asc)} {_fmt(d1_desc)} {_fmt(d1_delta)} │"
              f" {_fmt(d2_asc)} {_fmt(d2_desc)} {_fmt(d2_delta)}")

    print("=" * w)
    print("  Key: Δ = desc − asc. Positive Δ = descending α > ascending α (hysteresis).")
    print("  Prediction: D1 Δ >> D2 Δ because D1 decays much slower.")
    print("=" * w)


# ==============================================================================
# Sub-Experiment C: Network Firing Rate Response
# ==============================================================================

def run_sub_c(data: dict, args, save_dir: Path):
    """
    Sub-Experiment C: Network Firing Rate Response

    Compute and plot firing rates for D1 and D2 subgroups during the staircase.
    Key observation: E-D2 firing rate tracks DA steps more faithfully than E-D1.
    Also plot rate hysteresis (rate vs DA concentration).
    """
    print("\n" + "=" * 70)
    print("  🔬 Sub-Experiment C: Network Firing Rate Response")
    print("=" * 70)

    rate_data = compute_group_rates(data, time_win_ms=2000.0)
    phases = data['phases']
    total_s = data['total_s']
    da_t = np.linspace(0, total_s, data['da_schedule'].shape[0])
    da_exp = data['da_schedule'][:, 1]

    # ================================================================
    # Figure 1: Firing rate time courses
    # ================================================================
    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(4, 1, figure=fig, hspace=0.35, height_ratios=[1, 2, 2, 2])
    fig.suptitle(
        "Experiment F-C: Network Firing Rate Response to DA Staircase\n"
        "D2 subgroups track DA changes faster than D1 subgroups",
        fontsize=15, fontweight='bold'
    )

    # Panel A: DA protocol
    ax_da = fig.add_subplot(gs[0])
    ax_da.fill_between(da_t, 0, da_exp, alpha=0.3, color='green', step='post')
    ax_da.step(da_t, da_exp, color='green', linewidth=2.5, where='post')
    ax_da.set_ylabel('DA (nM)', fontsize=13)
    ax_da.set_title('A. DA Staircase Protocol', fontsize=14, fontweight='bold')
    ax_da.set_xlim(0, total_s)
    ax_da.grid(True, alpha=0.3)

    # Panel B: E-D1 vs E-D2 (Experiment batch)
    ax = fig.add_subplot(gs[1])
    for grp, color, label in [
        ('E-D1', '#d62728', 'E-D1 (slow D1 receptor)'),
        ('E-D2', '#1f77b4', 'E-D2 (fast D2 receptor)'),
        ('E-Other', 'gray', 'E-Other (no DA receptor)'),
    ]:
        rd = rate_data[grp][1]
        if len(rd['rates']) > 0:
            ax.plot(rd['times_s'], rd['rates'], '-', color=color, linewidth=2, label=label)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=13)
    ax.set_title('B. Excitatory Subgroups — Experiment Batch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, total_s)
    ax.grid(True, alpha=0.3)

    # Panel C: I-D1 vs I-D2 (Experiment batch)
    ax = fig.add_subplot(gs[2])
    for grp, color, label in [
        ('I-D1', '#ff7f0e', 'I-D1 (slow D1 receptor)'),
        ('I-D2', '#9467bd', 'I-D2 (fast D2 receptor)'),
        ('I-Other', '#2ca02c', 'I-Other (no DA receptor)'),
    ]:
        rd = rate_data[grp][1]
        if len(rd['rates']) > 0:
            ax.plot(rd['times_s'], rd['rates'], '-', color=color, linewidth=2, label=label)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=13)
    ax.set_title('C. Inhibitory Subgroups — Experiment Batch', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, total_s)
    ax.grid(True, alpha=0.3)

    # Panel D: Control vs Experiment for All-E and All-I
    ax = fig.add_subplot(gs[3])
    for grp, color in [('All-E', '#e377c2'), ('All-I', '#17becf')]:
        rd_exp = rate_data[grp][1]
        rd_ctrl = rate_data[grp][0]
        if len(rd_exp['rates']) > 0:
            ax.plot(rd_exp['times_s'], rd_exp['rates'], '-', color=color, linewidth=2,
                    label=f'{grp} (Exp)')
        if len(rd_ctrl['rates']) > 0:
            ax.plot(rd_ctrl['times_s'], rd_ctrl['rates'], ':', color=color, linewidth=1.5,
                    alpha=0.5, label=f'{grp} (Ctrl)')
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=13)
    ax.set_title('D. Overall E/I Population — Control vs Experiment', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, total_s)
    ax.grid(True, alpha=0.3)

    plt.savefig(save_dir / "sub_c_firing_rates.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: sub_c_firing_rates.png")

    # ================================================================
    # Figure 2: Rate hysteresis — firing rate vs DA concentration
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(
        "Experiment F-C: Firing Rate Hysteresis — Rate vs DA Concentration\n"
        "D1 subgroups show wider hysteresis (slower kinetics)",
        fontsize=14, fontweight='bold'
    )

    for ax, groups, colors, title in [
        (axes[0],
         ['E-D1', 'E-D2', 'E-Other'],
         ['#d62728', '#1f77b4', 'gray'],
         'Excitatory Subgroups'),
        (axes[1],
         ['I-D1', 'I-D2', 'I-Other'],
         ['#ff7f0e', '#9467bd', '#2ca02c'],
         'Inhibitory Subgroups'),
    ]:
        for grp, color in zip(groups, colors):
            # Compute mean rate at each DA level for ascending and descending
            da_asc_rates = []
            da_desc_rates = []
            da_asc_levels = []
            da_desc_levels = []

            for i, (start, end, da) in enumerate(phases):
                if i == 0 or i == len(phases) - 1:
                    continue

                prev_da = phases[i - 1][2]
                is_ascending = (da >= prev_da)

                # Get mean rate in last 1/3 of step
                step_dur = end - start
                sample_start = end - step_dur / 3.0
                rd = rate_data[grp][1]
                if len(rd['rates']) > 0:
                    mask = (rd['times_s'] >= sample_start) & (rd['times_s'] < end)
                    if np.any(mask):
                        mean_rate = float(np.mean(rd['rates'][mask]))
                        if is_ascending:
                            da_asc_levels.append(da)
                            da_asc_rates.append(mean_rate)
                        else:
                            da_desc_levels.append(da)
                            da_desc_rates.append(mean_rate)

            # Add baseline
            bl_rd = rate_data[grp][1]
            if len(bl_rd['rates']) > 0:
                bl_mask = bl_rd['times_s'] < phases[1][0]
                if np.any(bl_mask):
                    bl_rate = float(np.mean(bl_rd['rates'][bl_mask]))
                    da_asc_levels = [args.da_base] + da_asc_levels
                    da_asc_rates = [bl_rate] + da_asc_rates

            if da_asc_levels:
                ax.plot(da_asc_levels, da_asc_rates, 'o-', color=color, linewidth=2,
                        markersize=8, label=f'{grp} ↑')
            if da_desc_levels:
                ax.plot(da_desc_levels, da_desc_rates, 's--', color=color, linewidth=2,
                        markersize=8, alpha=0.7, label=f'{grp} ↓')

        ax.set_xlabel('DA Concentration (nM)', fontsize=13)
        ax.set_ylabel('Firing Rate (Hz)', fontsize=13)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "sub_c_rate_hysteresis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: sub_c_rate_hysteresis.png")

    # ---- Print rate summary ----
    _print_sub_c_summary(rate_data, data)

    return rate_data


def _print_sub_c_summary(rate_data, data):
    """Print firing rate summary at each staircase step."""
    phases = data['phases']
    key_groups = ['E-D1', 'E-D2', 'I-D1', 'I-D2']
    w = 95
    print("\n" + "=" * w)
    print("  📊 Sub-Exp C: Mean Firing Rate at Each Staircase Step (Exp Batch)")
    print("=" * w)
    print(f"  {'Phase':<10} {'DA':>6} │ {'E-D1':>8} {'E-D2':>8} {'I-D1':>8} {'I-D2':>8} │ {'E-D1/D2':>8}")
    print(f"  {'─'*10}─{'─'*6}─┼─{'─'*8}─{'─'*8}─{'─'*8}─{'─'*8}─┼─{'─'*8}")

    for i, (start, end, da) in enumerate(phases):
        step_dur = end - start
        sample_start = end - step_dur / 3.0

        rates = {}
        for grp in key_groups:
            rd = rate_data[grp][1]
            if len(rd['rates']) > 0:
                mask = (rd['times_s'] >= sample_start) & (rd['times_s'] < end)
                rates[grp] = float(np.mean(rd['rates'][mask])) if np.any(mask) else 0.0
            else:
                rates[grp] = 0.0

        if i == 0:
            label = "baseline"
        elif i == len(phases) - 1:
            label = "recovery"
        else:
            prev_da = phases[i - 1][2]
            label = "↑ up" if da > prev_da else "↓ down"

        ratio = rates['E-D1'] / rates['E-D2'] if rates['E-D2'] > 0.01 else float('inf')
        print(f"  {label:<10} {da:>5.1f}  │ {rates['E-D1']:>8.2f} {rates['E-D2']:>8.2f}"
              f" {rates['I-D1']:>8.2f} {rates['I-D2']:>8.2f} │ {ratio:>7.2f}x")

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
    save_dir = PROJECT_ROOT / "outputs" / f"exp_f_tau_difference_{timestamp}_{sub_tag}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint
    ckpt_path = find_or_create_checkpoint(args)

    # Save experiment config
    da_values = np.linspace(args.da_base, args.da_peak, args.n_steps + 1)
    da_values = np.round(da_values, 1)
    exp_config = {
        'sub_experiments': args.sub,
        'da_base': args.da_base,
        'da_peak': args.da_peak,
        'step_duration_s': args.step_duration,
        'recovery_duration_s': args.recovery_duration,
        'n_steps': args.n_steps,
        'da_staircase_levels': da_values.tolist(),
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
    print(f"  🧪 Experiment F: D1/D2 Tau Difference — DA Staircase & Hysteresis")
    print(f"{'='*70}")
    print(f"  D1: τ_on={config.TAU_ON_D1:.0f}ms ({config.TAU_ON_D1/1000:.0f}s), "
          f"τ_off={config.TAU_OFF_D1:.0f}ms ({config.TAU_OFF_D1/1000:.0f}s)")
    print(f"  D2: τ_on={config.TAU_ON_D2:.0f}ms ({config.TAU_ON_D2/1000:.0f}s), "
          f"τ_off={config.TAU_OFF_D2:.0f}ms ({config.TAU_OFF_D2/1000:.0f}s)")
    print(f"  τ ratio (D1/D2): on={config.TAU_ON_D1/config.TAU_ON_D2:.1f}x, "
          f"off={config.TAU_OFF_D1/config.TAU_OFF_D2:.1f}x")
    print(f"  DA staircase: {da_values.tolist()} nM")
    print(f"  Step duration: {args.step_duration}s, Recovery: {args.recovery_duration}s")
    print(f"  Output: {save_dir}")
    print(f"{'='*70}")

    # Run simulation (shared across all sub-experiments)
    print("\n  ⏳ Running staircase simulation...")
    t_sim_start = time.time()
    data = run_staircase_sim(ckpt_path, device, args)
    t_sim_elapsed = time.time() - t_sim_start

    all_results = {}

    # Sub-Experiment A: Staircase Alpha Dynamics
    if args.sub in ['a', 'all']:
        run_sub_a(data, args, save_dir)

    # Sub-Experiment B: Hysteresis Loop Analysis
    if args.sub in ['b', 'all']:
        all_results['sub_b'] = run_sub_b(data, args, save_dir)

    # Sub-Experiment C: Network Firing Rate Response
    if args.sub in ['c', 'all']:
        all_results['sub_c'] = run_sub_c(data, args, save_dir)

    # Save summary results
    summary = {
        'sim_time': data['sim_time'],
        'total_duration_s': data['total_s'],
        'da_staircase_levels': data['da_values'].tolist(),
        'phases': [(s, e, d) for s, e, d in data['phases']],
    }
    if 'sub_b' in all_results:
        summary['hysteresis'] = {
            'da_asc_d1': all_results['sub_b']['da_asc_d1'],
            'alpha_asc_d1': all_results['sub_b']['alpha_asc_d1'],
            'da_desc_d1': all_results['sub_b']['da_desc_d1'],
            'alpha_desc_d1': all_results['sub_b']['alpha_desc_d1'],
            'da_asc_d2': all_results['sub_b']['da_asc_d2'],
            'alpha_asc_d2': all_results['sub_b']['alpha_asc_d2'],
            'da_desc_d2': all_results['sub_b']['da_desc_d2'],
            'alpha_desc_d2': all_results['sub_b']['alpha_desc_d2'],
        }

    with open(save_dir / "exp_f_results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n📄 Results saved: {save_dir / 'exp_f_results.json'}")

    # Save alpha traces as numpy
    np.savez(save_dir / "alpha_traces.npz",
             times=np.linspace(0, data['total_s'], len(data['alpha_d1_trace'][:, 0])),
             alpha_d1=data['alpha_d1_trace'],
             alpha_d2=data['alpha_d2_trace'])

    # Total time
    t_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"  ⏱️  Timing Report")
    print(f"{'='*70}")
    print(f"  Simulation:  {_fmt_elapsed(t_sim_elapsed)}")
    print(f"  Total:       {_fmt_elapsed(t_total)}")
    print(f"  📁 Results in: {save_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
