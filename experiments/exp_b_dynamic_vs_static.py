#!/usr/bin/env python3
"""
Experiment B: Dynamic vs Static Model Comparison

Scientific Question:
  What are the observable differences between the dynamic α (Langmuir kinetics)
  model and the static α (instantaneous) model?

Key Predictions:
  1. Response latency: Dynamic model has a delay, static model responds instantly
  2. Overshoot/undershoot: Dynamic model may show transient over/undershoot
  3. Steady-state convergence: Both models should converge to similar final states
  4. The dynamic model captures temporal features that the static model misses

Protocol:
  Same DA step protocol (2nM → 15nM) applied to both models:
  - Dynamic model: from checkpoint, DA=15nM, Langmuir kinetics
  - Static model: from checkpoint, DA=15nM, instantaneous α switch

Usage:
  python -m experiments.exp_b_dynamic_vs_static
  python -m experiments.exp_b_dynamic_vs_static --da-target 20 --duration 200
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
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.network import create_network_structure
from models.kernels import (
    run_dynamic_d1_d2_kernel_pulse,
    run_batch_network_stepped,
)
from models.pharmacology import get_stepped_modulation_params
from analysis.analyzer import PFCAnalyzer

CHECKPOINT_PATH = "checkpoints/ckpt_DA2nM_500s.pkl"
RATE_GROUPS = ['E-D1', 'E-D2', 'E-Other', 'All-E', 'I-D1', 'I-D2', 'I-Other', 'All-I']


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment B: Dynamic vs Static Comparison")
    parser.add_argument("--da-base", type=float, default=2.0,
                        help="Baseline DA (nM), default 2.0")
    parser.add_argument("--da-target", type=float, default=15.0,
                        help="Target DA after step (nM), default 15.0")
    parser.add_argument("--baseline", type=float, default=20.0,
                        help="Baseline period before DA step (s), default 20")
    parser.add_argument("--duration", type=float, default=150.0,
                        help="Total simulation duration (s), default 150")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    return parser.parse_args()


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:05.2f}s"
    return f"{int(m)}m {s:05.2f}s" if m > 0 else f"{s:.2f}s"


def compute_time_resolved_rates(data: dict, time_win_ms: float = 1000.0) -> dict:
    """Compute time-resolved firing rates for all subgroups."""
    analyzer = PFCAnalyzer(data)
    results = {}
    for grp_name in RATE_GROUPS:
        results[grp_name] = {}
        for batch_id in [0, 1]:
            centers, rate = analyzer.compute_group_rate(batch_id, grp_name, time_win=time_win_ms)
            if rate is not None:
                results[grp_name][batch_id] = {
                    'times': centers / 1000.0,
                    'rates': rate,
                }
            else:
                results[grp_name][batch_id] = {'times': np.array([]), 'rates': np.array([])}
    return results


def run_dynamic_model(args, device, W_t, mask_d1, mask_d2, groups_info, init_state):
    """Run the dynamic (Langmuir kinetics) model."""
    total_ms = args.duration * 1000.0
    baseline_ms = args.baseline * 1000.0

    target_d1 = 0
    target_d2 = groups_info['e_d1_end']
    record_indices = torch.tensor([
        [0, target_d1], [1, target_d1],
        [0, target_d2], [1, target_d2],
    ], device=device, dtype=torch.long)

    alpha_record_interval = 100

    print(f"  ⚡ Running DYNAMIC model...")
    t0 = time.time()

    # Use pulse kernel with pulse that never ends (pulse_offset = total_ms + 1)
    spikes, v_traces, final_state, alpha_d1_trace, alpha_d2_trace = run_dynamic_d1_d2_kernel_pulse(
        W_t, mask_d1, mask_d2, init_state,
        float(args.da_base), float(args.da_target),
        float(baseline_ms), float(total_ms + 1000.0),  # pulse never ends
        float(total_ms), float(config.DT),
        record_indices, config.N_E,
        alpha_record_interval,
        config.build_kernel_params(device),
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"  ✅ Dynamic model done in {_fmt_elapsed(elapsed)}")

    data = {
        'config': {
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': total_ms, 'dt': config.DT,
            'da_onset': baseline_ms, 'da_level': args.da_target,
        },
        'masks': {'d1': mask_d1.cpu(), 'd2': mask_d2.cpu()},
        'groups_info': groups_info,
        'spikes': spikes.cpu(),
        'v_traces': v_traces.cpu(),
        'record_indices': record_indices.cpu(),
    }

    return data, alpha_d1_trace.cpu().numpy(), alpha_d2_trace.cpu().numpy(), elapsed


def run_static_model(args, device, W_t, mask_d1, mask_d2, groups_info):
    """Run the static (instantaneous α) model."""
    total_ms = args.duration * 1000.0
    baseline_ms = args.baseline * 1000.0
    N = config.N_TOTAL

    target_d1 = 0
    target_d2 = groups_info['e_d1_end']
    record_indices = torch.tensor([
        [0, target_d1], [1, target_d1],
        [0, target_d2], [1, target_d2],
    ], device=device, dtype=torch.long)

    # Static model: use stepped kernel with instantaneous parameter switch
    da_levels_active = [0.0, args.da_target]
    params_rest, params_active = get_stepped_modulation_params(
        N, mask_d1, mask_d2, da_levels_active, device
    )

    print(f"  ⚡ Running STATIC model...")
    t0 = time.time()

    spikes, v_traces = run_batch_network_stepped(
        W_t, params_rest, params_active,
        float(total_ms), float(config.DT), float(baseline_ms),
        record_indices, config.N_E,
        config.build_kernel_params(device),
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"  ✅ Static model done in {_fmt_elapsed(elapsed)}")

    data = {
        'config': {
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': total_ms, 'dt': config.DT,
            'da_onset': baseline_ms, 'da_level': args.da_target,
        },
        'masks': {'d1': mask_d1.cpu(), 'd2': mask_d2.cpu()},
        'groups_info': groups_info,
        'spikes': spikes.cpu(),
        'v_traces': v_traces.cpu(),
        'record_indices': record_indices.cpu(),
    }

    return data, elapsed


def compute_response_metrics(rate_data, da_onset_s, total_s):
    """
    Compute response metrics: latency, overshoot, steady-state.

    Args:
        rate_data: {group: {batch: {times, rates}}}
        da_onset_s: DA step time (s)
        total_s: Total duration (s)

    Returns:
        dict with metrics for each group
    """
    metrics = {}
    for grp in ['All-E', 'All-I', 'E-D1', 'E-D2']:
        if grp not in rate_data or 1 not in rate_data[grp]:
            continue
        rd = rate_data[grp][1]
        times = rd['times']
        rates = rd['rates']

        if len(times) == 0:
            continue

        # Baseline rate (before DA step)
        bl_mask = times < da_onset_s
        if np.any(bl_mask):
            bl_rate = np.mean(rates[bl_mask])
        else:
            bl_rate = rates[0] if len(rates) > 0 else 0

        # Post-DA rates
        post_mask = times >= da_onset_s
        post_times = times[post_mask]
        post_rates = rates[post_mask]

        if len(post_rates) == 0:
            continue

        # Steady-state rate (last 20% of post-DA period)
        ss_start = da_onset_s + 0.8 * (total_s - da_onset_s)
        ss_mask = post_times >= ss_start
        ss_rate = np.mean(post_rates[ss_mask]) if np.any(ss_mask) else post_rates[-1]

        # Response latency: time to reach 10% of total change
        total_change = ss_rate - bl_rate
        if abs(total_change) > 0.1:
            threshold = bl_rate + 0.1 * total_change
            if total_change > 0:
                crossed = post_rates >= threshold
            else:
                crossed = post_rates <= threshold
            if np.any(crossed):
                latency = float(post_times[crossed][0] - da_onset_s)
            else:
                latency = float('inf')
        else:
            latency = float('inf')

        # Peak/trough (overshoot/undershoot)
        peak_rate = np.max(post_rates)
        trough_rate = np.min(post_rates)

        metrics[grp] = {
            'baseline_rate': float(bl_rate),
            'steady_state_rate': float(ss_rate),
            'total_change': float(total_change),
            'latency_10pct_s': latency,
            'peak_rate': float(peak_rate),
            'trough_rate': float(trough_rate),
            'overshoot': float(peak_rate - ss_rate) if total_change > 0 else 0.0,
            'undershoot': float(ss_rate - trough_rate) if total_change < 0 else 0.0,
        }

    return metrics


def plot_comparison(dyn_rates, static_rates, dyn_alpha_d1, dyn_alpha_d2,
                    dyn_metrics, static_metrics, args, save_dir):
    """Generate comparison plots."""
    da_onset_s = args.baseline
    total_s = args.duration

    # Alpha trace time axis
    alpha_times = np.linspace(0, total_s, len(dyn_alpha_d1))

    # ================================================================
    # Figure 1: Main comparison — Dynamic vs Static firing rates
    # ================================================================
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(
        f"Experiment B: Dynamic vs Static Model Comparison\n"
        f"DA Step: {args.da_base} → {args.da_target} nM at t={da_onset_s}s",
        fontsize=15, fontweight='bold'
    )

    # --- Panel A: Alpha dynamics (dynamic model only) ---
    ax = fig.add_subplot(gs[0, :])
    ax.plot(alpha_times, dyn_alpha_d1[:, 1], '-', color='#d62728', linewidth=2.5,
            label='α_D1 (Dynamic)')
    ax.plot(alpha_times, dyn_alpha_d2[:, 1], '-', color='#1f77b4', linewidth=2.5,
            label='α_D2 (Dynamic)')

    # Static alpha values (instantaneous)
    import math
    s_d1_static = 1.0 / (1.0 + math.exp(-config.BETA * (args.da_target - config.EC50_D1)))
    s_d2_static = 1.0 / (1.0 + math.exp(-config.BETA * (args.da_target - config.EC50_D2)))
    ax.axhline(s_d1_static, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.6,
               label=f'α_D1 (Static) = {s_d1_static:.3f}')
    ax.axhline(s_d2_static, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.6,
               label=f'α_D2 (Static) = {s_d2_static:.3f}')

    ax.axvline(da_onset_s, color='green', linestyle=':', linewidth=2, label='DA step')
    ax.set_ylabel('α', fontsize=12)
    ax.set_title('A. Receptor Activation: Dynamic (solid) vs Static (dashed)', fontsize=13)
    ax.legend(fontsize=10, ncol=3)
    ax.set_xlim(0, total_s)
    ax.grid(True, alpha=0.3)

    # --- Panels B-E: Firing rate comparisons for key groups ---
    compare_groups = [
        ('All-E', '#e377c2', 'B. All Excitatory (All-E)'),
        ('All-I', '#17becf', 'C. All Inhibitory (All-I)'),
        ('E-D1', '#d62728', 'D. Excitatory D1 (E-D1)'),
        ('E-D2', '#1f77b4', 'E. Excitatory D2 (E-D2)'),
    ]

    for idx, (grp, color, title) in enumerate(compare_groups):
        row = 1 + idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])

        # Dynamic model (Exp batch)
        if grp in dyn_rates and 1 in dyn_rates[grp]:
            rd = dyn_rates[grp][1]
            ax.plot(rd['times'], rd['rates'], '-', color=color, linewidth=2.5,
                    label='Dynamic (Langmuir)', zorder=3)

        # Static model (Exp batch)
        if grp in static_rates and 1 in static_rates[grp]:
            rd = static_rates[grp][1]
            ax.plot(rd['times'], rd['rates'], '--', color='black', linewidth=2,
                    alpha=0.7, label='Static (Instantaneous)', zorder=2)

        # Control (from dynamic model)
        if grp in dyn_rates and 0 in dyn_rates[grp]:
            rd = dyn_rates[grp][0]
            ax.plot(rd['times'], rd['rates'], ':', color='gray', linewidth=1,
                    alpha=0.5, label='Control (0 nM)')

        ax.axvline(da_onset_s, color='green', linestyle=':', linewidth=1.5)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Rate (Hz)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim(0, total_s)
        ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(save_dir, "dynamic_vs_static_main.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: dynamic_vs_static_main.png")

    # ================================================================
    # Figure 2: Metrics comparison table (as bar chart)
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Dynamic vs Static — Response Metrics Comparison",
                 fontsize=14, fontweight='bold')

    groups_to_compare = ['All-E', 'All-I', 'E-D1', 'E-D2']
    x = np.arange(len(groups_to_compare))
    width = 0.35

    # Latency comparison
    ax = axes[0]
    dyn_lat = [dyn_metrics.get(g, {}).get('latency_10pct_s', 0) for g in groups_to_compare]
    stat_lat = [static_metrics.get(g, {}).get('latency_10pct_s', 0) for g in groups_to_compare]
    # Cap infinite latencies for display
    dyn_lat = [min(l, 100) for l in dyn_lat]
    stat_lat = [min(l, 100) for l in stat_lat]
    ax.bar(x - width/2, dyn_lat, width, label='Dynamic', color='#ff7f0e')
    ax.bar(x + width/2, stat_lat, width, label='Static', color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels(groups_to_compare, fontsize=10)
    ax.set_ylabel('Latency (s)', fontsize=11)
    ax.set_title('Response Latency (10% threshold)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Steady-state comparison
    ax = axes[1]
    dyn_ss = [dyn_metrics.get(g, {}).get('steady_state_rate', 0) for g in groups_to_compare]
    stat_ss = [static_metrics.get(g, {}).get('steady_state_rate', 0) for g in groups_to_compare]
    ax.bar(x - width/2, dyn_ss, width, label='Dynamic', color='#ff7f0e')
    ax.bar(x + width/2, stat_ss, width, label='Static', color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels(groups_to_compare, fontsize=10)
    ax.set_ylabel('Rate (Hz)', fontsize=11)
    ax.set_title('Steady-State Firing Rate', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Overshoot comparison
    ax = axes[2]
    dyn_os = [dyn_metrics.get(g, {}).get('overshoot', 0) for g in groups_to_compare]
    stat_os = [static_metrics.get(g, {}).get('overshoot', 0) for g in groups_to_compare]
    ax.bar(x - width/2, dyn_os, width, label='Dynamic', color='#ff7f0e')
    ax.bar(x + width/2, stat_os, width, label='Static', color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels(groups_to_compare, fontsize=10)
    ax.set_ylabel('Overshoot (Hz)', fontsize=11)
    ax.set_title('Overshoot (Peak - Steady State)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: metrics_comparison.png")

    # ================================================================
    # Figure 3: Zoomed onset region
    # ================================================================
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f"DA Step Onset Region — Dynamic vs Static\n"
                 f"(Zoomed: {da_onset_s-5}s to {da_onset_s+60}s)",
                 fontsize=14, fontweight='bold')

    for ax, grp, color, title in zip(
        axes,
        ['All-E', 'All-I'],
        ['#e377c2', '#17becf'],
        ['All-E (Excitatory)', 'All-I (Inhibitory)'],
    ):
        if grp in dyn_rates and 1 in dyn_rates[grp]:
            rd = dyn_rates[grp][1]
            ax.plot(rd['times'], rd['rates'], '-', color=color, linewidth=2.5,
                    label='Dynamic')
        if grp in static_rates and 1 in static_rates[grp]:
            rd = static_rates[grp][1]
            ax.plot(rd['times'], rd['rates'], '--', color='black', linewidth=2,
                    alpha=0.7, label='Static')

        ax.axvline(da_onset_s, color='green', linestyle=':', linewidth=2, label='DA step')
        ax.set_xlim(max(0, da_onset_s - 5), min(total_s, da_onset_s + 60))
        ax.set_ylabel('Rate (Hz)', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel('Time (s)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "onset_zoom.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: onset_zoom.png")


def main():
    args = parse_args()
    t_total_start = time.time()

    # Device
    if torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"🔧 Device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("outputs", f"exp_b_compare_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  🧪 Experiment B: Dynamic vs Static Model Comparison")
    print(f"{'='*70}")
    print(f"  DA step: {args.da_base} → {args.da_target} nM at t={args.baseline}s")
    print(f"  Duration: {args.duration}s")
    print(f"  Output: {save_dir}")
    print(f"{'='*70}\n")

    # Build network
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(config.N_E, config.N_I, device)

    # Load checkpoint for dynamic model
    print(f"📂 Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        ckpt_data = pickle.load(f)
    init_state = ckpt_data['final_state'].to(device)

    # ---- Run both models ----
    print(f"\n--- Dynamic Model (Langmuir Kinetics) ---")
    dyn_data, dyn_alpha_d1, dyn_alpha_d2, dyn_time = run_dynamic_model(
        args, device, W_t, mask_d1, mask_d2, groups_info, init_state)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n--- Static Model (Instantaneous α) ---")
    static_data, static_time = run_static_model(
        args, device, W_t, mask_d1, mask_d2, groups_info)

    # ---- Compute rates ----
    print(f"\n📊 Computing firing rates...")
    dyn_rates = compute_time_resolved_rates(dyn_data, time_win_ms=1000.0)
    static_rates = compute_time_resolved_rates(static_data, time_win_ms=1000.0)

    # ---- Compute metrics ----
    print(f"📐 Computing response metrics...")
    dyn_metrics = compute_response_metrics(dyn_rates, args.baseline, args.duration)
    static_metrics = compute_response_metrics(static_rates, args.baseline, args.duration)

    # ---- Generate plots ----
    print(f"\n🎨 Generating comparison plots...")
    plot_comparison(dyn_rates, static_rates, dyn_alpha_d1, dyn_alpha_d2,
                    dyn_metrics, static_metrics, args, save_dir)

    # ---- Save results ----
    results = {
        'args': vars(args),
        'dynamic_model': {
            'sim_time': dyn_time,
            'metrics': dyn_metrics,
        },
        'static_model': {
            'sim_time': static_time,
            'metrics': static_metrics,
        },
        'timestamp': timestamp,
    }

    json_path = os.path.join(save_dir, "exp_b_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📄 Results saved: {json_path}")

    # ---- Print summary ----
    t_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"  📊 Experiment B — Summary")
    print(f"{'='*70}")
    print(f"  {'Group':<10} │ {'Dyn Latency':>12} {'Stat Latency':>12} │ {'Dyn SS Rate':>12} {'Stat SS Rate':>12}")
    print(f"  {'─'*10}─┼─{'─'*12}─{'─'*12}─┼─{'─'*12}─{'─'*12}")
    for grp in ['All-E', 'All-I', 'E-D1', 'E-D2']:
        dm = dyn_metrics.get(grp, {})
        sm = static_metrics.get(grp, {})
        dl = dm.get('latency_10pct_s', float('inf'))
        sl = sm.get('latency_10pct_s', float('inf'))
        dss = dm.get('steady_state_rate', 0)
        sss = sm.get('steady_state_rate', 0)
        dl_str = f"{dl:.1f}s" if dl < 100 else "N/A"
        sl_str = f"{sl:.1f}s" if sl < 100 else "N/A"
        print(f"  {grp:<10} │ {dl_str:>12} {sl_str:>12} │ {dss:>11.2f}  {sss:>11.2f}")
    print(f"{'='*70}")
    print(f"  ⏱️  Total time: {_fmt_elapsed(t_total)}")
    print(f"  📁 Results in: {save_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
