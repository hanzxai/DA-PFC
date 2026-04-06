#!/usr/bin/env python3
"""
Experiment A: DA Pulse Response — Temporal Segregation of D1/D2 Receptors

Scientific Question:
  When PFC receives a transient DA pulse (mimicking VTA phasic firing),
  how does the D1/D2 temporal segregation shape the network's transient response?

Key Predictions:
  1. D2 responds faster than D1 during both DA onset and offset
  2. After DA withdrawal, a "D1 afterglow window" exists where D1 is still
     active but D2 has already decayed — this is UNIQUE to the dynamic model
  3. The network firing rate shows a complex temporal profile that cannot
     be captured by a static (instantaneous) model

Protocol:
  From DA=2nM steady-state checkpoint:
    [0, 20s)     → DA = 2 nM  (pre-pulse baseline)
    [20s, 50s)   → DA = 15 nM (DA pulse, 30s duration)
    [50s, 300s)  → DA = 2 nM  (post-pulse recovery, observe D1 afterglow)

Usage:
  python -m experiments.exp_a_pulse_response
  python -m experiments.exp_a_pulse_response --da-pulse 20 --pulse-duration 60
  python -m experiments.exp_a_pulse_response --gpu 1
"""
import argparse
import json
import time
import os
import sys
import math
import numpy as np
import torch
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.network import create_network_structure
from models.kernels import run_dynamic_d1_d2_kernel_pulse
from analysis.analyzer import PFCAnalyzer

CHECKPOINT_PATH = "checkpoints/ckpt_DA2nM_500s.pkl"
RATE_GROUPS = ['E-D1', 'E-D2', 'E-Other', 'All-E', 'I-D1', 'I-D2', 'I-Other', 'All-I']


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment A: DA Pulse Response")
    parser.add_argument("--da-base", type=float, default=2.0,
                        help="Baseline DA concentration (nM), default 2.0")
    parser.add_argument("--da-pulse", type=float, default=15.0,
                        help="Pulse DA concentration (nM), default 15.0")
    parser.add_argument("--pre-pulse", type=float, default=20.0,
                        help="Pre-pulse baseline duration (s), default 20")
    parser.add_argument("--pulse-duration", type=float, default=30.0,
                        help="DA pulse duration (s), default 30")
    parser.add_argument("--post-pulse", type=float, default=250.0,
                        help="Post-pulse recovery duration (s), default 250")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    return parser.parse_args()


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:05.2f}s"
    return f"{int(m)}m {s:05.2f}s" if m > 0 else f"{s:.2f}s"


def compute_time_resolved_rates(data: dict, time_win_ms: float = 1000.0) -> dict:
    """
    Compute time-resolved firing rates for all subgroups.

    Args:
        data: Simulation data dict
        time_win_ms: Time bin width (ms) for rate computation

    Returns:
        dict: {group_name: {batch_id: {'times': array, 'rates': array}}}
    """
    analyzer = PFCAnalyzer(data)
    results = {}
    for grp_name in RATE_GROUPS:
        results[grp_name] = {}
        for batch_id in [0, 1]:
            centers, rate = analyzer.compute_group_rate(batch_id, grp_name, time_win=time_win_ms)
            if rate is not None:
                results[grp_name][batch_id] = {
                    'times': centers / 1000.0,  # Convert to seconds
                    'rates': rate,
                }
            else:
                results[grp_name][batch_id] = {'times': np.array([]), 'rates': np.array([])}
    return results


def plot_pulse_response(alpha_d1_trace, alpha_d2_trace, rate_data, args, save_dir):
    """
    Generate comprehensive pulse response figures.

    Figure 1: Alpha dynamics + DA protocol (top panel)
    Figure 2: Firing rate time courses for all subgroups
    Figure 3: D1 afterglow window analysis
    """
    pulse_onset_s = args.pre_pulse
    pulse_offset_s = args.pre_pulse + args.pulse_duration
    total_s = args.pre_pulse + args.pulse_duration + args.post_pulse

    # Alpha trace time axis
    alpha_times = np.linspace(0, total_s, len(alpha_d1_trace))

    # ================================================================
    # Figure 1: Main result — Alpha dynamics + Firing rates
    # ================================================================
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        f"Experiment A: DA Pulse Response — D1/D2 Temporal Segregation\n"
        f"DA: {args.da_base}→{args.da_pulse}→{args.da_base} nM  |  "
        f"Pulse: [{pulse_onset_s:.0f}s, {pulse_offset_s:.0f}s)",
        fontsize=15, fontweight='bold'
    )

    # --- Panel A: DA protocol ---
    ax_da = fig.add_subplot(gs[0, :])
    da_profile = np.full_like(alpha_times, args.da_base)
    mask_pulse = (alpha_times >= pulse_onset_s) & (alpha_times < pulse_offset_s)
    da_profile[mask_pulse] = args.da_pulse
    ax_da.fill_between(alpha_times, 0, da_profile, alpha=0.3, color='green', label='DA concentration')
    ax_da.plot(alpha_times, da_profile, color='green', linewidth=2)
    ax_da.set_ylabel('DA (nM)', fontsize=12)
    ax_da.set_title('A. DA Pulse Protocol', fontsize=13, fontweight='bold')
    ax_da.legend(fontsize=10)
    ax_da.set_xlim(0, total_s)
    ax_da.grid(True, alpha=0.3)

    # --- Panel B: Alpha D1 and D2 dynamics ---
    ax_alpha = fig.add_subplot(gs[1, :])

    # Control batch (dashed)
    ax_alpha.plot(alpha_times, alpha_d1_trace[:, 0], '--', color='#d62728', linewidth=1.5,
                  alpha=0.5, label='α_D1 (Control)')
    ax_alpha.plot(alpha_times, alpha_d2_trace[:, 0], '--', color='#1f77b4', linewidth=1.5,
                  alpha=0.5, label='α_D2 (Control)')

    # Experiment batch (solid)
    ax_alpha.plot(alpha_times, alpha_d1_trace[:, 1], '-', color='#d62728', linewidth=2.5,
                  label='α_D1 (Exp) — SLOW')
    ax_alpha.plot(alpha_times, alpha_d2_trace[:, 1], '-', color='#1f77b4', linewidth=2.5,
                  label='α_D2 (Exp) — FAST')

    # Mark pulse window
    ax_alpha.axvspan(pulse_onset_s, pulse_offset_s, alpha=0.1, color='green', label='DA pulse')
    ax_alpha.axvline(pulse_onset_s, color='green', linestyle=':', linewidth=1.5)
    ax_alpha.axvline(pulse_offset_s, color='green', linestyle=':', linewidth=1.5)

    # Mark D1 afterglow window (where D1 > D2 after pulse offset)
    exp_d1 = alpha_d1_trace[:, 1]
    exp_d2 = alpha_d2_trace[:, 1]
    post_pulse_mask = alpha_times >= pulse_offset_s
    afterglow_mask = post_pulse_mask & (exp_d1 > exp_d2 * 1.1)  # D1 > 1.1*D2
    if np.any(afterglow_mask):
        ag_start = alpha_times[afterglow_mask][0]
        ag_end = alpha_times[afterglow_mask][-1]
        ax_alpha.axvspan(ag_start, ag_end, alpha=0.15, color='orange',
                         label=f'D1 afterglow [{ag_start:.0f}s–{ag_end:.0f}s]')

    ax_alpha.set_ylabel('Receptor Activation (α)', fontsize=12)
    ax_alpha.set_title('B. D1/D2 Receptor Activation Dynamics', fontsize=13, fontweight='bold')
    ax_alpha.legend(fontsize=9, ncol=3, loc='upper right')
    ax_alpha.set_xlim(0, total_s)
    ax_alpha.grid(True, alpha=0.3)

    # --- Panel C: Excitatory subgroup firing rates ---
    ax_e = fig.add_subplot(gs[2, :])
    e_groups = ['E-D1', 'E-D2', 'E-Other', 'All-E']
    e_colors = ['#d62728', '#1f77b4', 'gray', '#e377c2']
    e_styles = ['-', '-', '-', '--']

    for grp, color, ls in zip(e_groups, e_colors, e_styles):
        if grp in rate_data and 1 in rate_data[grp]:
            rd = rate_data[grp][1]
            ax_e.plot(rd['times'], rd['rates'], ls, color=color, linewidth=2, label=f'{grp} (Exp)')
        if grp in rate_data and 0 in rate_data[grp]:
            rd = rate_data[grp][0]
            ax_e.plot(rd['times'], rd['rates'], ':', color=color, linewidth=1, alpha=0.5,
                      label=f'{grp} (Ctrl)')

    ax_e.axvspan(pulse_onset_s, pulse_offset_s, alpha=0.1, color='green')
    ax_e.axvline(pulse_onset_s, color='green', linestyle=':', linewidth=1)
    ax_e.axvline(pulse_offset_s, color='green', linestyle=':', linewidth=1)
    ax_e.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax_e.set_title('C. Excitatory Subgroup Firing Rates', fontsize=13, fontweight='bold')
    ax_e.legend(fontsize=9, ncol=4, loc='upper right')
    ax_e.set_xlim(0, total_s)
    ax_e.grid(True, alpha=0.3)

    # --- Panel D: Inhibitory subgroup firing rates ---
    ax_i = fig.add_subplot(gs[3, :])
    i_groups = ['I-D1', 'I-D2', 'I-Other', 'All-I']
    i_colors = ['#ff7f0e', '#9467bd', '#2ca02c', '#17becf']

    for grp, color in zip(i_groups, i_colors):
        if grp in rate_data and 1 in rate_data[grp]:
            rd = rate_data[grp][1]
            ax_i.plot(rd['times'], rd['rates'], '-', color=color, linewidth=2, label=f'{grp} (Exp)')
        if grp in rate_data and 0 in rate_data[grp]:
            rd = rate_data[grp][0]
            ax_i.plot(rd['times'], rd['rates'], ':', color=color, linewidth=1, alpha=0.5)

    ax_i.axvspan(pulse_onset_s, pulse_offset_s, alpha=0.1, color='green')
    ax_i.axvline(pulse_onset_s, color='green', linestyle=':', linewidth=1)
    ax_i.axvline(pulse_offset_s, color='green', linestyle=':', linewidth=1)
    ax_i.set_xlabel('Time (s)', fontsize=12)
    ax_i.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax_i.set_title('D. Inhibitory Subgroup Firing Rates', fontsize=13, fontweight='bold')
    ax_i.legend(fontsize=9, ncol=4, loc='upper right')
    ax_i.set_xlim(0, total_s)
    ax_i.grid(True, alpha=0.3)

    plt.savefig(os.path.join(save_dir, "pulse_response_main.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: pulse_response_main.png")

    # ================================================================
    # Figure 2: D1 Afterglow Window — Zoomed analysis
    # ================================================================
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(
        "D1 Afterglow Window Analysis\n"
        f"(Post-pulse period: DA returns to {args.da_base} nM)",
        fontsize=14, fontweight='bold'
    )

    zoom_start = pulse_offset_s - 10  # 10s before pulse end
    zoom_end = min(pulse_offset_s + 200, total_s)  # 200s after pulse end

    # Panel A: Alpha zoom
    ax = axes[0]
    ax.plot(alpha_times, exp_d1, '-', color='#d62728', linewidth=2.5, label='α_D1 (Exp)')
    ax.plot(alpha_times, exp_d2, '-', color='#1f77b4', linewidth=2.5, label='α_D2 (Exp)')

    # Shade afterglow region
    if np.any(afterglow_mask):
        ax.axvspan(ag_start, ag_end, alpha=0.2, color='orange', label='D1 Afterglow Window')

    ax.axvline(pulse_offset_s, color='green', linestyle='--', linewidth=2, label='DA pulse OFF')
    ax.set_xlim(zoom_start, zoom_end)
    ax.set_ylabel('α', fontsize=12)
    ax.set_title('A. Receptor Activation After DA Withdrawal', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel B: Firing rate zoom
    ax = axes[1]
    for grp, color in zip(['All-E', 'All-I'], ['#e377c2', '#17becf']):
        if grp in rate_data and 1 in rate_data[grp]:
            rd = rate_data[grp][1]
            ax.plot(rd['times'], rd['rates'], '-', color=color, linewidth=2, label=f'{grp} (Exp)')
        if grp in rate_data and 0 in rate_data[grp]:
            rd = rate_data[grp][0]
            ax.plot(rd['times'], rd['rates'], ':', color=color, linewidth=1, alpha=0.5,
                    label=f'{grp} (Ctrl)')

    if np.any(afterglow_mask):
        ax.axvspan(ag_start, ag_end, alpha=0.15, color='orange')
    ax.axvline(pulse_offset_s, color='green', linestyle='--', linewidth=2)
    ax.set_xlim(zoom_start, zoom_end)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax.set_title('B. Network Firing Rate After DA Withdrawal', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pulse_afterglow_zoom.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: pulse_afterglow_zoom.png")

    # ================================================================
    # Figure 3: Temporal segregation quantification
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("D1/D2 Temporal Segregation — Δα = α_D1 - α_D2",
                 fontsize=14, fontweight='bold')

    delta_alpha = exp_d1 - exp_d2
    ax.plot(alpha_times, delta_alpha, '-', color='purple', linewidth=2.5, label='Δα = α_D1 - α_D2')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.fill_between(alpha_times, 0, delta_alpha,
                    where=(delta_alpha > 0), alpha=0.3, color='red', label='D1 > D2')
    ax.fill_between(alpha_times, 0, delta_alpha,
                    where=(delta_alpha < 0), alpha=0.3, color='blue', label='D2 > D1')

    ax.axvspan(pulse_onset_s, pulse_offset_s, alpha=0.1, color='green')
    ax.axvline(pulse_onset_s, color='green', linestyle=':', linewidth=1.5, label='Pulse ON')
    ax.axvline(pulse_offset_s, color='green', linestyle=':', linewidth=1.5, label='Pulse OFF')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Δα (D1 - D2)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, total_s)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temporal_segregation.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: temporal_segregation.png")


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
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("outputs", f"exp_a_pulse_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Timing parameters (convert to ms)
    pre_pulse_ms = args.pre_pulse * 1000.0
    pulse_duration_ms = args.pulse_duration * 1000.0
    post_pulse_ms = args.post_pulse * 1000.0
    total_ms = pre_pulse_ms + pulse_duration_ms + post_pulse_ms
    pulse_onset_ms = pre_pulse_ms
    pulse_offset_ms = pre_pulse_ms + pulse_duration_ms

    print(f"\n{'='*70}")
    print(f"  🧪 Experiment A: DA Pulse Response")
    print(f"{'='*70}")
    print(f"  DA protocol: {args.da_base} → {args.da_pulse} → {args.da_base} nM")
    print(f"  Pre-pulse:   [0, {args.pre_pulse}s) = {args.da_base} nM")
    print(f"  Pulse:       [{args.pre_pulse}s, {args.pre_pulse + args.pulse_duration}s) = {args.da_pulse} nM")
    print(f"  Post-pulse:  [{args.pre_pulse + args.pulse_duration}s, {args.pre_pulse + args.pulse_duration + args.post_pulse}s) = {args.da_base} nM")
    print(f"  Total:       {total_ms/1000:.0f}s")
    print(f"  Output:      {save_dir}")
    print(f"{'='*70}\n")

    # Load checkpoint
    print(f"📂 Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        ckpt_data = pickle.load(f)
    init_state = ckpt_data['final_state'].to(device)

    # Build network (same seed)
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(config.N_E, config.N_I, device)

    # Record indices
    target_d1 = 0
    target_d2 = groups_info['e_d1_end']
    record_indices = torch.tensor([
        [0, target_d1], [1, target_d1],
        [0, target_d2], [1, target_d2],
    ], device=device, dtype=torch.long)

    # Alpha recording interval (every 100 steps = 100ms)
    alpha_record_interval = 100

    # Run simulation
    print(f"⚡ Running pulse simulation ({total_ms/1000:.0f}s)...")
    t0 = time.time()

    spikes, v_traces, final_state, alpha_d1_trace, alpha_d2_trace = run_dynamic_d1_d2_kernel_pulse(
        W_t, mask_d1, mask_d2, init_state,
        float(args.da_base), float(args.da_pulse),
        float(pulse_onset_ms), float(pulse_offset_ms),
        float(total_ms), float(config.DT),
        record_indices, config.N_E,
        alpha_record_interval,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sim_time = time.time() - t0
    print(f"✅ Simulation done in {_fmt_elapsed(sim_time)}")

    # Pack data for analyzer
    data = {
        'config': {
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': total_ms, 'dt': config.DT,
            'da_onset': pulse_onset_ms, 'da_level': args.da_pulse,
            'mode': 'pulse_response',
        },
        'masks': {'d1': mask_d1.cpu(), 'd2': mask_d2.cpu()},
        'groups_info': groups_info,
        'spikes': spikes.cpu(),
        'v_traces': v_traces.cpu(),
        'record_indices': record_indices.cpu(),
    }

    # Compute time-resolved rates
    print(f"📊 Computing firing rates...")
    rate_data = compute_time_resolved_rates(data, time_win_ms=1000.0)

    # Convert alpha traces to numpy
    alpha_d1_np = alpha_d1_trace.cpu().numpy()
    alpha_d2_np = alpha_d2_trace.cpu().numpy()

    # Generate plots
    print(f"🎨 Generating plots...")
    plot_pulse_response(alpha_d1_np, alpha_d2_np, rate_data, args, save_dir)

    # Save results
    results = {
        'args': vars(args),
        'sim_time': sim_time,
        'alpha_d1_exp_peak': float(alpha_d1_np[:, 1].max()),
        'alpha_d2_exp_peak': float(alpha_d2_np[:, 1].max()),
        'timestamp': timestamp,
    }

    # Quantify D1 afterglow
    alpha_times = np.linspace(0, total_ms / 1000.0, len(alpha_d1_np))
    post_mask = alpha_times >= (pulse_offset_ms / 1000.0)
    exp_d1_post = alpha_d1_np[post_mask, 1]
    exp_d2_post = alpha_d2_np[post_mask, 1]
    afterglow = exp_d1_post > exp_d2_post * 1.1
    if np.any(afterglow):
        ag_times = alpha_times[post_mask][afterglow]
        results['afterglow_start_s'] = float(ag_times[0])
        results['afterglow_end_s'] = float(ag_times[-1])
        results['afterglow_duration_s'] = float(ag_times[-1] - ag_times[0])
    else:
        results['afterglow_duration_s'] = 0.0

    json_path = os.path.join(save_dir, "exp_a_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📄 Results saved: {json_path}")

    # Save alpha traces as numpy
    np.savez(os.path.join(save_dir, "alpha_traces.npz"),
             times=alpha_times,
             alpha_d1=alpha_d1_np,
             alpha_d2=alpha_d2_np)

    t_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"  ⏱️  Total time: {_fmt_elapsed(t_total)}")
    print(f"  📁 Results in: {save_dir}")
    if results.get('afterglow_duration_s', 0) > 0:
        print(f"  🌅 D1 Afterglow Window: {results['afterglow_start_s']:.0f}s – {results['afterglow_end_s']:.0f}s "
              f"(duration: {results['afterglow_duration_s']:.0f}s)")
    else:
        print(f"  ⚠️  No D1 afterglow window detected")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
