#!/usr/bin/env python3
# sweep_da.py
"""
DA Dose-Response Sweep — Inverted-U Hypothesis Test

Systematically sweep DA concentrations to test whether PFC network
activity follows the classic Inverted-U (Arnsten 2011) dose-response curve.

============================================================
 Usage
============================================================
  python sweep_da.py                          # default 13 concentrations, 100s each
  python sweep_da.py --duration 200           # longer per-concentration run
  python sweep_da.py --gpu 1                  # use GPU 1
  python sweep_da.py --concentrations 1 3 5 8 15  # custom concentration list

============================================================
 Method
============================================================
  For each DA concentration:
    1. Load checkpoint: checkpoints/ckpt_DA2nM_500s.pkl (DA=2nM steady-state)
    2. Run 100s simulation (10s baseline + 90s with target DA)
    3. Extract steady-state firing rates from the last 30s
    4. Record rates for all 8 subgroups

  After all concentrations are done:
    - Save results to JSON
    - Generate dose-response curve plots
"""
import argparse
import json
import time
import os
import sys
import numpy as np
import torch
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from simulation.runners import run_simulation_from_checkpoint
from analysis.analyzer import PFCAnalyzer


# ======================================================================
# Default DA concentrations to sweep (nM)
# ======================================================================
DEFAULT_DA_CONCENTRATIONS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]

# Checkpoint path
CHECKPOINT_PATH = "checkpoints/ckpt_DA2nM_500s.pkl"

# Subgroups to analyze
RATE_GROUPS = ['E-D1', 'E-D2', 'E-Other', 'All-E',
               'I-D1', 'I-D2', 'I-Other', 'All-I']


def parse_args():
    parser = argparse.ArgumentParser(description="DA Dose-Response Sweep (Inverted-U)")
    parser.add_argument("--duration", type=float, default=100.0,
                        help="Simulation duration per concentration (s), default 100")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID, default 0")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="Checkpoint path for steady-state initial condition")
    parser.add_argument("--concentrations", type=float, nargs='+', default=None,
                        help="Custom DA concentration list (nM). Default: 13 points from 0.5 to 50")
    parser.add_argument("--steady-window", type=float, default=30.0,
                        help="Time window (s) at the end of simulation to compute steady-state rate, default 30")
    return parser.parse_args()


def compute_steady_state_rates(data: dict, steady_window_ms: float, time_win: float = 50.0) -> dict:
    """
    Compute mean firing rates in the steady-state window (last N seconds).

    Args:
        data: Simulation data dict from runner
        steady_window_ms: Duration of steady-state window (ms) from the end
        time_win: Bin width for rate computation (ms)

    Returns:
        dict: {group_name: {'ctrl': float, 'exp': float, 'delta': float, 'pct': float}}
    """
    analyzer = PFCAnalyzer(data)
    duration = analyzer.duration
    ss_start = duration - steady_window_ms
    ss_end = duration

    results = {}
    for grp_name in RATE_GROUPS:
        grp_results = {}
        for batch_id, batch_label in [(0, 'ctrl'), (1, 'exp')]:
            centers, rate = analyzer.compute_group_rate(batch_id, grp_name, time_win=time_win)
            if rate is None or len(rate) == 0:
                grp_results[batch_label] = 0.0
                continue
            # Select bins within steady-state window
            mask_ss = (centers >= ss_start) & (centers <= ss_end)
            if np.any(mask_ss):
                grp_results[batch_label] = float(np.mean(rate[mask_ss]))
            else:
                grp_results[batch_label] = 0.0

        # DA effect: Exp - Ctrl
        grp_results['delta'] = grp_results['exp'] - grp_results['ctrl']
        if grp_results['ctrl'] > 0.01:
            grp_results['pct'] = grp_results['delta'] / grp_results['ctrl'] * 100.0
        else:
            grp_results['pct'] = 0.0

        results[grp_name] = grp_results

    return results


def plot_dose_response(sweep_results: dict, da_concentrations: list, save_dir: str):
    """
    Generate dose-response curve plots.

    Produces 3 figures:
      1. All-E and All-I dose-response (main result)
      2. E subgroups (E-D1, E-D2, E-Other) dose-response
      3. I subgroups (I-D1, I-D2, I-Other) dose-response
    """
    da_arr = np.array(da_concentrations)

    # ---- Figure 1: All-E and All-I (main Inverted-U test) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("DA Dose-Response Curve — PFC Network (Inverted-U Test)",
                 fontsize=16, fontweight='bold')

    for ax, grp, color, title in zip(
        axes,
        ['All-E', 'All-I'],
        ['#e377c2', '#17becf'],
        ['Excitatory Population (All-E)', 'Inhibitory Population (All-I)'],
    ):
        rates_exp = [sweep_results[da]['rates'][grp]['exp'] for da in da_concentrations]
        rates_ctrl = [sweep_results[da]['rates'][grp]['ctrl'] for da in da_concentrations]

        ax.plot(da_arr, rates_exp, 'o-', color=color, linewidth=2.5, markersize=8,
                label='Exp (DA applied)', zorder=3)
        ax.plot(da_arr, rates_ctrl, 's--', color='gray', linewidth=1.5, markersize=6,
                alpha=0.6, label='Ctrl (0 nM)', zorder=2)

        # Mark EC50 lines
        ax.axvline(config.EC50_D1, color='red', linestyle=':', alpha=0.5, linewidth=1.5,
                   label=f'D1 EC50={config.EC50_D1} nM')
        ax.axvline(config.EC50_D2, color='blue', linestyle=':', alpha=0.5, linewidth=1.5,
                   label=f'D2 EC50={config.EC50_D2} nM')

        # Find and mark peak
        peak_idx = np.argmax(rates_exp)
        ax.scatter([da_arr[peak_idx]], [rates_exp[peak_idx]], color='red', s=150,
                   zorder=5, marker='*', label=f'Peak @ {da_arr[peak_idx]:.1f} nM')

        ax.set_xscale('log')
        ax.set_xlabel('DA Concentration (nM)', fontsize=13)
        ax.set_ylabel('Steady-State Firing Rate (Hz)', fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(da_arr)
        ax.set_xticklabels([f'{d:.1f}' if d < 1 else f'{d:.0f}' for d in da_arr],
                           rotation=45, fontsize=10)

    plt.tight_layout()
    path1 = os.path.join(save_dir, "dose_response_main.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: {path1}")

    # ---- Figure 2: E subgroups ----
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("DA Dose-Response — Excitatory Subgroups", fontsize=16, fontweight='bold')

    e_groups = ['E-D1', 'E-D2', 'E-Other', 'All-E']
    e_colors = ['#d62728', '#1f77b4', 'gray', '#e377c2']
    e_markers = ['o', 's', '^', 'D']

    for grp, color, marker in zip(e_groups, e_colors, e_markers):
        rates = [sweep_results[da]['rates'][grp]['exp'] for da in da_concentrations]
        ax.plot(da_arr, rates, f'{marker}-', color=color, linewidth=2, markersize=7,
                label=grp)

    ax.axvline(config.EC50_D1, color='red', linestyle=':', alpha=0.4, linewidth=1.5,
               label=f'D1 EC50={config.EC50_D1} nM')
    ax.axvline(config.EC50_D2, color='blue', linestyle=':', alpha=0.4, linewidth=1.5,
               label=f'D2 EC50={config.EC50_D2} nM')

    ax.set_xscale('log')
    ax.set_xlabel('DA Concentration (nM)', fontsize=13)
    ax.set_ylabel('Steady-State Firing Rate (Hz)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(da_arr)
    ax.set_xticklabels([f'{d:.1f}' if d < 1 else f'{d:.0f}' for d in da_arr],
                       rotation=45, fontsize=10)

    plt.tight_layout()
    path2 = os.path.join(save_dir, "dose_response_E_subgroups.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: {path2}")

    # ---- Figure 3: I subgroups ----
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("DA Dose-Response — Inhibitory Subgroups", fontsize=16, fontweight='bold')

    i_groups = ['I-D1', 'I-D2', 'I-Other', 'All-I']
    i_colors = ['#ff7f0e', '#9467bd', '#2ca02c', '#17becf']
    i_markers = ['o', 's', '^', 'D']

    for grp, color, marker in zip(i_groups, i_colors, i_markers):
        rates = [sweep_results[da]['rates'][grp]['exp'] for da in da_concentrations]
        ax.plot(da_arr, rates, f'{marker}-', color=color, linewidth=2, markersize=7,
                label=grp)

    ax.axvline(config.EC50_D1, color='red', linestyle=':', alpha=0.4, linewidth=1.5,
               label=f'D1 EC50={config.EC50_D1} nM')
    ax.axvline(config.EC50_D2, color='blue', linestyle=':', alpha=0.4, linewidth=1.5,
               label=f'D2 EC50={config.EC50_D2} nM')

    ax.set_xscale('log')
    ax.set_xlabel('DA Concentration (nM)', fontsize=13)
    ax.set_ylabel('Steady-State Firing Rate (Hz)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(da_arr)
    ax.set_xticklabels([f'{d:.1f}' if d < 1 else f'{d:.0f}' for d in da_arr],
                       rotation=45, fontsize=10)

    plt.tight_layout()
    path3 = os.path.join(save_dir, "dose_response_I_subgroups.png")
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: {path3}")

    # ---- Figure 4: DA effect (delta from control) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("DA Effect (Δ from Control) — Dose-Response",
                 fontsize=16, fontweight='bold')

    for ax, groups, colors, markers, title in zip(
        axes,
        [e_groups, i_groups],
        [e_colors, i_colors],
        [e_markers, i_markers],
        ['Excitatory Subgroups', 'Inhibitory Subgroups'],
    ):
        for grp, color, marker in zip(groups, colors, markers):
            deltas = [sweep_results[da]['rates'][grp]['delta'] for da in da_concentrations]
            ax.plot(da_arr, deltas, f'{marker}-', color=color, linewidth=2, markersize=7,
                    label=grp)

        ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.axvline(config.EC50_D1, color='red', linestyle=':', alpha=0.4, linewidth=1.5)
        ax.axvline(config.EC50_D2, color='blue', linestyle=':', alpha=0.4, linewidth=1.5)

        ax.set_xscale('log')
        ax.set_xlabel('DA Concentration (nM)', fontsize=13)
        ax.set_ylabel('Δ Firing Rate (Hz)', fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(da_arr)
        ax.set_xticklabels([f'{d:.1f}' if d < 1 else f'{d:.0f}' for d in da_arr],
                           rotation=45, fontsize=10)

    plt.tight_layout()
    path4 = os.path.join(save_dir, "dose_response_delta.png")
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: {path4}")


def print_sweep_summary(sweep_results: dict, da_concentrations: list):
    """Print a formatted summary table of the sweep results."""
    w = 130
    print("\n" + "=" * w)
    print("  📊 DA Dose-Response Sweep — Summary Table")
    print("=" * w)

    # Header
    header = f"  {'DA (nM)':<10} │"
    for grp in RATE_GROUPS:
        header += f" {grp:>10}"
    print(header)
    print(f"  {'─'*10}─┼─" + "─" * (11 * len(RATE_GROUPS)))

    # Data rows (Exp rates)
    for da in da_concentrations:
        row = f"  {da:<10.1f} │"
        for grp in RATE_GROUPS:
            rate = sweep_results[da]['rates'][grp]['exp']
            row += f" {rate:>10.2f}"
        print(row)

    print("=" * w)

    # Find peak for All-E
    all_e_rates = [sweep_results[da]['rates']['All-E']['exp'] for da in da_concentrations]
    peak_idx = np.argmax(all_e_rates)
    peak_da = da_concentrations[peak_idx]
    peak_rate = all_e_rates[peak_idx]

    print(f"\n  🏔️  All-E Peak: DA = {peak_da:.1f} nM → {peak_rate:.2f} Hz")
    print(f"  📈 All-E range: {min(all_e_rates):.2f} – {max(all_e_rates):.2f} Hz")

    # Check if inverted-U shape exists
    if peak_idx > 0 and peak_idx < len(da_concentrations) - 1:
        print(f"  ✅ Inverted-U shape detected! Peak at interior point ({peak_da:.1f} nM)")
    elif peak_idx == 0:
        print(f"  ⚠️  Peak at lowest concentration — may need lower DA points")
    else:
        print(f"  ⚠️  Peak at highest concentration — may need higher DA points or longer simulation")

    print("=" * w)


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as HH:MM:SS.ss"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:05.2f}s"
    if m > 0:
        return f"{m}m {s:05.2f}s"
    return f"{s:.2f}s"


def main():
    args = parse_args()
    t_total_start = time.time()

    # Device setup
    if torch.cuda.is_available():
        if 0 <= args.gpu < torch.cuda.device_count():
            device = torch.device(f"cuda:{args.gpu}")
        else:
            print(f"⚠️ GPU {args.gpu} not available, falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"🔧 Using device: {device}")

    # DA concentrations
    da_concentrations = args.concentrations or DEFAULT_DA_CONCENTRATIONS
    da_concentrations = sorted(da_concentrations)
    n_conc = len(da_concentrations)

    # Verify checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print(f"   Please run: python main.py --da 2.0 --duration 500 --save-ckpt")
        sys.exit(1)

    # Setup output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("outputs", f"sweep_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    duration_ms = args.duration * 1000.0
    steady_window_ms = args.steady_window * 1000.0

    print(f"\n{'='*70}")
    print(f"  🧪 DA Dose-Response Sweep — Inverted-U Hypothesis Test")
    print(f"{'='*70}")
    print(f"  Checkpoint:      {args.checkpoint}")
    print(f"  Concentrations:  {da_concentrations} nM ({n_conc} points)")
    print(f"  Duration/conc:   {args.duration}s")
    print(f"  Steady window:   last {args.steady_window}s")
    print(f"  Output dir:      {save_dir}")
    print(f"  Device:          {device}")
    print(f"{'='*70}\n")

    # ---- Run sweep ----
    sweep_results = {}

    for i, da in enumerate(da_concentrations):
        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{n_conc}] DA = {da:.1f} nM")
        print(f"{'─'*60}")

        t_start = time.time()

        # Run simulation from checkpoint
        data = run_simulation_from_checkpoint(
            checkpoint_path=args.checkpoint,
            duration=duration_ms,
            da_level=da,
            device=device,
        )

        t_sim = time.time() - t_start

        # Compute steady-state rates
        rates = compute_steady_state_rates(data, steady_window_ms)

        sweep_results[da] = {
            'rates': rates,
            'sim_time': t_sim,
        }

        # Print quick summary for this concentration
        all_e = rates['All-E']
        all_i = rates['All-I']
        print(f"  ⏱️  Sim time: {_fmt_elapsed(t_sim)}")
        print(f"  📊 All-E: Ctrl={all_e['ctrl']:.2f} Hz, Exp={all_e['exp']:.2f} Hz, Δ={all_e['delta']:+.2f} Hz ({all_e['pct']:+.1f}%)")
        print(f"  📊 All-I: Ctrl={all_i['ctrl']:.2f} Hz, Exp={all_i['exp']:.2f} Hz, Δ={all_i['delta']:+.2f} Hz ({all_i['pct']:+.1f}%)")

        # Free GPU memory
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Save results ----
    print(f"\n{'='*70}")
    print(f"  💾 Saving results...")
    print(f"{'='*70}")

    # Convert to JSON-serializable format
    json_results = {}
    for da in da_concentrations:
        json_results[str(da)] = sweep_results[da]
    json_results['_meta'] = {
        'concentrations': da_concentrations,
        'duration_s': args.duration,
        'steady_window_s': args.steady_window,
        'checkpoint': args.checkpoint,
        'device': str(device),
        'timestamp': timestamp,
    }

    json_path = os.path.join(save_dir, "sweep_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  📄 Results saved: {json_path}")

    # ---- Print summary ----
    print_sweep_summary(sweep_results, da_concentrations)

    # ---- Generate plots ----
    print(f"\n🎨 Generating dose-response plots...")
    plot_dose_response(sweep_results, da_concentrations, save_dir)

    # ---- Total time ----
    t_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"  ⏱️  Total sweep time: {_fmt_elapsed(t_total)}")
    print(f"  ✅ All results saved in: {save_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
