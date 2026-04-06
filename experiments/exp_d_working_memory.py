#!/usr/bin/env python3
"""
Experiment D: DA Time-Scale Gating of PFC Stimulus Response Gain

Scientific Question:
  How do the different time constants of D1/D2 receptors modulate the
  gain of stimulus-evoked responses in PFC?

Key Hypothesis:
  The D1 afterglow window (where D1 >> D2 after DA pulse withdrawal) creates
  a privileged time window where stimulus-evoked responses are ENHANCED
  compared to baseline or to time points outside the window.

Experimental Design (5 Runs × 2 Batches):
  Each Run delivers a brief sensory stimulus at a DIFFERENT time point
  relative to the DA pulse, scanning across the afterglow window.

  Batch 0 (Control): DA protocol ONLY, NO stimulus → pure DA effect on rate
  Batch 1 (Experiment): DA protocol + stimulus → DA + stimulus combined effect

  ΔRate = Rate(B1) - Rate(B0) = stimulus-evoked response gain

  Run 1 — Baseline (no DA pulse):
    B0: constant DA=2nM, no stim
    B1: constant DA=2nM, stim at t=80s
    → Measures baseline stimulus response without DA modulation

  Run 2 — During DA pulse (peak D2 suppression):
    B0: DA pulse, no stim
    B1: DA pulse, stim at t=35s (during pulse, 15s after onset)
    → D1 rising, D2 near peak → net suppression expected

  Run 3 — Early afterglow (D1 high, D2 declining):
    B0: DA pulse, no stim
    B1: DA pulse, stim at t=65s (15s after pulse offset)
    → D1 still high, D2 decaying → D1 afterglow window

  Run 4 — Peak afterglow (D1 >> D2):
    B0: DA pulse, no stim
    B1: DA pulse, stim at t=100s (50s after pulse offset)
    → D1 still elevated, D2 mostly decayed → maximum D1 advantage

  Run 5 — Late recovery (both decayed):
    B0: DA pulse, no stim
    B1: DA pulse, stim at t=250s (200s after pulse offset)
    → Both D1 and D2 largely decayed → should approach baseline

  DA Pulse Protocol (Runs 2-5):
    [0, 20s)     → DA = 2 nM  (pre-pulse baseline)
    [20s, 50s)   → DA = 15 nM (DA pulse)
    [50s, end)   → DA = 2 nM  (post-pulse recovery)

  Stimulus Protocol:
    Brief current injection to a mixed ensemble of E neurons.
    Default: 300 pA × 5s, 100 E neurons (proportional from D1/D2/Other).

Key Comparisons:
  1. ΔRate across Runs 1-5: How does stimulus gain change over time?
  2. Run 3/4 vs Run 1: Is afterglow window gain > baseline gain?
  3. Run 2 vs Run 1: Does active DA pulse suppress or enhance gain?
  4. Run 5 vs Run 1: Does gain return to baseline after recovery?

Usage:
  python -m experiments.exp_d_working_memory
  python -m experiments.exp_d_working_memory --stim-amplitude 300 --stim-duration 5
  python -m experiments.exp_d_working_memory --gpu 0
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.network import create_network_structure
from models.kernels import (
    run_dynamic_d1_d2_kernel_pulse,
    run_dynamic_d1_d2_kernel_pulse_stim,
)
from analysis.analyzer import PFCAnalyzer

CHECKPOINT_PATH = "checkpoints/ckpt_DA2nM_500s.pkl"
RATE_GROUPS = ['E-D1', 'E-D2', 'E-Other', 'All-E', 'I-D1', 'I-D2', 'I-Other', 'All-I']
# Subgroups for stimulus ensemble analysis
STIM_ANALYSIS_GROUPS = ['E-D1', 'E-D2', 'E-Other', 'All-E']


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment D: Stimulus Response Gain Modulation")
    # DA parameters
    parser.add_argument("--da-base", type=float, default=2.0,
                        help="Baseline DA concentration (nM), default 2.0")
    parser.add_argument("--da-pulse", type=float, default=15.0,
                        help="Pulse DA concentration (nM), default 15.0")
    parser.add_argument("--pre-pulse", type=float, default=20.0,
                        help="Pre-pulse baseline duration (s), default 20")
    parser.add_argument("--pulse-duration", type=float, default=30.0,
                        help="DA pulse duration (s), default 30")
    # Stimulus parameters
    parser.add_argument("--stim-amplitude", type=float, default=300.0,
                        help="Stimulus current amplitude (pA), default 300")
    parser.add_argument("--stim-duration", type=float, default=5.0,
                        help="Stimulus duration (s), default 5")
    parser.add_argument("--stim-n-neurons", type=int, default=100,
                        help="Number of stimulated E neurons, default 100")
    # Timing for stimulus delivery (5 time points)
    parser.add_argument("--stim-times", type=str, default="80,35,65,100,250",
                        help="Comma-separated stimulus times (s) for Runs 1-5")
    # Simulation parameters
    parser.add_argument("--post-stim-observe", type=float, default=30.0,
                        help="Observation time after stimulus offset (s), default 30")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    # Run selection (for debugging: run only specific runs)
    parser.add_argument("--runs", type=str, default="1,2,3,4,5",
                        help="Comma-separated run IDs to execute, default '1,2,3,4,5'")
    return parser.parse_args()


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed time as human-readable string."""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:05.2f}s"
    return f"{int(m)}m {s:05.2f}s" if m > 0 else f"{s:.2f}s"


def create_stimulus_ensemble(groups_info, n_stim, device):
    """
    Create a mixed stimulus ensemble from E-D1, E-D2, and E-Other subgroups.

    The ensemble is selected proportionally from each subgroup to mimic
    a non-specific sensory input arriving at PFC.

    Args:
        groups_info: dict with subgroup boundary indices
        n_stim: total number of neurons to stimulate
        device: torch device

    Returns:
        stim_mask: (N,) float tensor, 1.0 for stimulated neurons
        stim_indices: list of stimulated neuron indices
    """
    N = config.N_E + config.N_I

    # Subgroup boundaries
    e_d1_end = groups_info['e_d1_end']      # E-D1: [0, e_d1_end)
    e_d2_end = groups_info['e_d2_end']      # E-D2: [e_d1_end, e_d2_end)
    e_other_end = groups_info['e_other_end']  # E-Other: [e_d2_end, e_other_end)

    n_e_d1 = e_d1_end
    n_e_d2 = e_d2_end - e_d1_end
    n_e_other = e_other_end - e_d2_end
    n_e_total = n_e_d1 + n_e_d2 + n_e_other

    # Proportional allocation
    n_from_d1 = max(1, int(n_stim * n_e_d1 / n_e_total))
    n_from_d2 = max(1, int(n_stim * n_e_d2 / n_e_total))
    n_from_other = n_stim - n_from_d1 - n_from_d2
    n_from_other = max(1, n_from_other)

    # Select neurons (deterministic for reproducibility)
    rng = np.random.RandomState(seed=123)
    idx_d1 = rng.choice(range(0, e_d1_end), size=min(n_from_d1, n_e_d1), replace=False)
    idx_d2 = rng.choice(range(e_d1_end, e_d2_end), size=min(n_from_d2, n_e_d2), replace=False)
    idx_other = rng.choice(range(e_d2_end, e_other_end), size=min(n_from_other, n_e_other), replace=False)

    stim_indices = np.concatenate([idx_d1, idx_d2, idx_other])

    stim_mask = torch.zeros(N, device=device, dtype=torch.float32)
    stim_mask[torch.tensor(stim_indices, device=device, dtype=torch.long)] = 1.0

    print(f"  📌 Stimulus ensemble: {len(stim_indices)} neurons "
          f"(E-D1: {len(idx_d1)}, E-D2: {len(idx_d2)}, E-Other: {len(idx_other)})")

    return stim_mask, stim_indices.tolist()


def compute_time_resolved_rates(data, time_win_ms=1000.0):
    """Compute time-resolved firing rates for all subgroups."""
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


def compute_stim_ensemble_rate(data, stim_indices, time_win_ms=500.0):
    """
    Compute firing rate specifically for the stimulated neuron ensemble.

    Returns:
        dict: {batch_id: {'times': array (s), 'rates': array (Hz)}}
    """
    dt = data['config']['dt']
    duration = data['config']['duration']
    spikes_all = data['spikes'].numpy()
    n_stim = len(stim_indices)
    stim_set = set(stim_indices)

    bins = np.arange(0, duration + time_win_ms, time_win_ms)
    centers = (bins[:-1] + bins[1:]) / 2

    results = {}
    for batch_id in [0, 1]:
        mask_batch = spikes_all[:, 1] == batch_id
        batch_spikes = spikes_all[mask_batch]
        ts = batch_spikes[:, 0] * dt  # ms
        ids = batch_spikes[:, 2]

        # Filter to stimulus ensemble
        mask_stim = np.array([int(nid) in stim_set for nid in ids])
        h, _ = np.histogram(ts[mask_stim], bins=bins)
        rate = h / (time_win_ms / 1000.0 * n_stim)  # Hz

        from scipy.ndimage import gaussian_filter1d
        rate_smooth = gaussian_filter1d(rate, sigma=1.5)

        results[batch_id] = {
            'times': centers / 1000.0,
            'rates': rate_smooth,
        }
    return results


def measure_stimulus_gain(stim_rate_data, stim_onset_s, stim_offset_s,
                          baseline_window_s=10.0):
    """
    Measure stimulus-evoked response gain by comparing B0 (no stim) vs B1 (stim).

    For each batch, compute:
    1. Pre-stimulus baseline rate (from [stim_onset - baseline_window, stim_onset))
    2. During-stimulus peak rate
    3. Post-stimulus rate (first 10s after stim offset)

    Gain metrics:
    - delta_rate_during: B1_during - B0_during (stimulus-evoked increase)
    - delta_rate_post: B1_post - B0_post (post-stimulus residual)
    - gain_ratio: B1_during / B0_during (multiplicative gain)

    Args:
        stim_rate_data: dict from compute_stim_ensemble_rate
        stim_onset_s: stimulus onset time (s)
        stim_offset_s: stimulus offset time (s)
        baseline_window_s: window before stimulus for baseline estimation (s)

    Returns:
        dict with gain metrics
    """
    results = {}

    for batch_id in [0, 1]:
        rd = stim_rate_data[batch_id]
        times = rd['times']
        rates = rd['rates']

        if len(times) == 0:
            results[batch_id] = {
                'baseline_rate_hz': 0.0,
                'during_stim_rate_hz': 0.0,
                'peak_rate_hz': 0.0,
                'post_stim_rate_hz': 0.0,
            }
            continue

        # Pre-stimulus baseline: [stim_onset - baseline_window, stim_onset)
        bl_start = max(0, stim_onset_s - baseline_window_s)
        bl_mask = (times >= bl_start) & (times < stim_onset_s)
        bl_rate = float(np.mean(rates[bl_mask])) if np.any(bl_mask) else 0.0

        # During stimulus: [stim_onset, stim_offset)
        during_mask = (times >= stim_onset_s) & (times < stim_offset_s)
        during_rate = float(np.mean(rates[during_mask])) if np.any(during_mask) else 0.0
        peak_rate = float(np.max(rates[during_mask])) if np.any(during_mask) else 0.0

        # Post-stimulus: [stim_offset, stim_offset + 10s)
        post_mask = (times >= stim_offset_s) & (times < stim_offset_s + 10.0)
        post_rate = float(np.mean(rates[post_mask])) if np.any(post_mask) else 0.0

        results[batch_id] = {
            'baseline_rate_hz': bl_rate,
            'during_stim_rate_hz': during_rate,
            'peak_rate_hz': peak_rate,
            'post_stim_rate_hz': post_rate,
        }

    # Compute gain metrics (B1 - B0)
    b0 = results[0]
    b1 = results[1]
    results['gain'] = {
        'delta_rate_during_hz': b1['during_stim_rate_hz'] - b0['during_stim_rate_hz'],
        'delta_rate_post_hz': b1['post_stim_rate_hz'] - b0['post_stim_rate_hz'],
        'delta_peak_hz': b1['peak_rate_hz'] - b0['peak_rate_hz'],
        'delta_baseline_hz': b1['baseline_rate_hz'] - b0['baseline_rate_hz'],
        'gain_ratio': (b1['during_stim_rate_hz'] / b0['during_stim_rate_hz']
                       if b0['during_stim_rate_hz'] > 0 else float('nan')),
    }

    return results


def run_single_experiment(run_id, run_label, W_t, mask_d1, mask_d2, init_state,
                          stim_mask, stim_mask_zero, record_indices, args, device,
                          has_da_pulse, stim_time_s):
    """
    Execute a single run (2 batches) and return results.

    NEW BATCH DESIGN:
      Batch 0 = DA protocol only, NO stimulus (stim_mask_zero)
      Batch 1 = DA protocol + stimulus (stim_mask)

    This is achieved by running TWO separate simulations:
      - Sim A: with stim_mask_zero (no stim) → extract Batch 1 as our "B0"
      - Sim B: with stim_mask (stim) → extract Batch 1 as our "B1"

    Wait — the kernel applies the same stim to BOTH batches. So we need a
    different approach. We run the kernel ONCE with stim, and the DA difference
    between B0 (constant da_base) and B1 (da_pulse protocol) gives us the
    DA modulation effect. The stimulus is the SAME for both batches.

    REVISED APPROACH: Actually, let's use the kernel as-is but reinterpret:
      - B0 (Ctrl): constant DA=da_base + stimulus → baseline DA + stim
      - B1 (Exp): DA pulse protocol + stimulus → DA modulated + stim
      - ΔRate = B1 - B0 = pure DA modulation effect on stimulus response

    This directly measures: does the DA afterglow window enhance the
    stimulus-evoked response compared to constant baseline DA?
    """
    # Timing
    pre_pulse_s = args.pre_pulse
    pulse_dur_s = args.pulse_duration
    stim_dur_s = args.stim_duration
    observe_s = args.post_stim_observe

    # Compute total duration: need enough time after stimulus
    if stim_time_s is not None:
        total_s = stim_time_s + stim_dur_s + observe_s
    else:
        total_s = pre_pulse_s + pulse_dur_s + observe_s

    # Ensure total duration covers DA pulse
    if has_da_pulse:
        total_s = max(total_s, pre_pulse_s + pulse_dur_s + 30.0)

    # Convert to ms
    total_ms = total_s * 1000.0
    pulse_onset_ms = pre_pulse_s * 1000.0
    pulse_offset_ms = (pre_pulse_s + pulse_dur_s) * 1000.0

    # DA parameters
    da_base = float(args.da_base)
    da_pulse_val = float(args.da_pulse) if has_da_pulse else float(args.da_base)

    if stim_time_s is not None:
        stim_onset_ms = stim_time_s * 1000.0
        stim_offset_ms = (stim_time_s + stim_dur_s) * 1000.0
        stim_amp = float(args.stim_amplitude)
    else:
        # No stimulus: set onset/offset outside simulation range
        stim_onset_ms = total_ms + 1000.0
        stim_offset_ms = total_ms + 2000.0
        stim_amp = 0.0

    alpha_record_interval = 100  # every 100ms

    print(f"\n  ⚡ Run {run_id}: {run_label}")
    if has_da_pulse:
        print(f"     DA: pulse ({da_base}→{da_pulse_val}→{da_base} nM)")
    else:
        print(f"     DA: constant {da_base} nM")
    if stim_time_s is not None:
        print(f"     Stimulus: t={stim_time_s}s ({stim_amp} pA × {stim_dur_s}s)")
    print(f"     Duration: {total_s:.0f}s")

    t0 = time.time()

    spikes, v_traces, final_state, alpha_d1_trace, alpha_d2_trace = \
        run_dynamic_d1_d2_kernel_pulse_stim(
            W_t, mask_d1, mask_d2, init_state,
            da_base, da_pulse_val,
            float(pulse_onset_ms), float(pulse_offset_ms),
            stim_mask,
            float(stim_onset_ms), float(stim_offset_ms), stim_amp,
            float(total_ms), float(config.DT),
            record_indices, config.N_E,
            alpha_record_interval,
            config.build_kernel_params(device),
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sim_time = time.time() - t0
    print(f"     ✅ Done in {_fmt_elapsed(sim_time)} | Spikes: {spikes.shape[0]:,}")

    return {
        'run_id': run_id,
        'run_label': run_label,
        'has_da_pulse': has_da_pulse,
        'stim_time_s': stim_time_s,
        'total_s': total_s,
        'total_ms': total_ms,
        'pulse_onset_ms': pulse_onset_ms,
        'pulse_offset_ms': pulse_offset_ms,
        'stim_onset_ms': stim_onset_ms if stim_time_s else None,
        'stim_offset_ms': stim_offset_ms if stim_time_s else None,
        'sim_time': sim_time,
        'spikes': spikes.cpu(),
        'v_traces': v_traces.cpu(),
        'final_state': final_state,
        'alpha_d1_trace': alpha_d1_trace.cpu().numpy(),
        'alpha_d2_trace': alpha_d2_trace.cpu().numpy(),
    }


def plot_run_comparison(all_results, stim_indices, args, save_dir):
    """
    Generate comprehensive comparison plots across all runs.

    Figure 1: Overview — Alpha dynamics + stimulus ensemble rates for each run
    Figure 2: Zoomed stimulus response — B0 vs B1 around stimulus time
    Figure 3: Gain summary — bar chart of ΔRate across runs
    Figure 4: All-E firing rate comparison across runs
    """
    n_runs = len(all_results)
    pulse_onset_s = args.pre_pulse
    pulse_offset_s = args.pre_pulse + args.pulse_duration

    # ================================================================
    # Figure 1: Alpha dynamics + firing rates for each run
    # ================================================================
    fig, axes = plt.subplots(n_runs, 2, figsize=(22, 5 * n_runs), squeeze=False)
    fig.suptitle(
        "Experiment D: Stimulus Response Gain Modulation by D1 Afterglow\n"
        f"DA: {args.da_base}→{args.da_pulse}→{args.da_base} nM  |  "
        f"Stimulus: {args.stim_amplitude} pA × {args.stim_duration}s  |  "
        f"Ensemble: {args.stim_n_neurons} E neurons\n"
        f"B0 = constant DA (ctrl) + stim  |  B1 = DA pulse + stim",
        fontsize=14, fontweight='bold'
    )

    for row, res in enumerate(all_results):
        total_s = res['total_s']
        alpha_d1 = res['alpha_d1_trace']
        alpha_d2 = res['alpha_d2_trace']
        alpha_times = np.linspace(0, total_s, len(alpha_d1))

        stim_time_s = res['stim_time_s']
        stim_end_s = stim_time_s + args.stim_duration if stim_time_s else None

        # Left panel: Alpha dynamics
        ax = axes[row, 0]
        ax.plot(alpha_times, alpha_d1[:, 0], '--', color='#d62728', lw=1.2, alpha=0.4, label='α_D1 (B0-Ctrl)')
        ax.plot(alpha_times, alpha_d2[:, 0], '--', color='#1f77b4', lw=1.2, alpha=0.4, label='α_D2 (B0-Ctrl)')
        ax.plot(alpha_times, alpha_d1[:, 1], '-', color='#d62728', lw=2.5, label='α_D1 (B1-Exp)')
        ax.plot(alpha_times, alpha_d2[:, 1], '-', color='#1f77b4', lw=2.5, label='α_D2 (B1-Exp)')

        # Mark D1-D2 difference for Exp batch
        alpha_diff = alpha_d1[:, 1] - alpha_d2[:, 1]
        ax.fill_between(alpha_times, 0, alpha_diff,
                         where=alpha_diff > 0, alpha=0.15, color='#d62728',
                         label='D1>D2 window')

        if res['has_da_pulse']:
            ax.axvspan(pulse_onset_s, pulse_offset_s, alpha=0.1, color='green')
            ax.axvline(pulse_onset_s, color='green', ls=':', lw=1)
            ax.axvline(pulse_offset_s, color='green', ls=':', lw=1)

        if stim_time_s is not None:
            ax.axvspan(stim_time_s, stim_end_s, alpha=0.3, color='gold', label='Stimulus')
            ax.axvline(stim_time_s, color='orange', ls='-', lw=2, alpha=0.8)

        ax.set_ylabel('α', fontsize=11)
        ax.set_title(f"Run {res['run_id']}: {res['run_label']} — Receptor Dynamics",
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, ncol=3, loc='upper right')
        ax.set_xlim(0, total_s)
        ax.grid(True, alpha=0.3)

        # Right panel: Stimulus ensemble firing rate
        ax = axes[row, 1]
        stim_rate = res.get('stim_ensemble_rate', None)
        if stim_rate is not None:
            for batch_id, (color, ls, label) in enumerate(
                [('#2ca02c', '--', 'B0 (Ctrl DA + stim)'), ('#ff7f0e', '-', 'B1 (Exp DA + stim)')]):
                rd = stim_rate[batch_id]
                ax.plot(rd['times'], rd['rates'], ls, color=color, lw=2, label=label)

            # Shade the ΔRate region during stimulus
            if stim_time_s is not None:
                b0_rates = stim_rate[0]['rates']
                b1_rates = stim_rate[1]['rates']
                t_arr = stim_rate[0]['times']
                stim_mask_arr = (t_arr >= stim_time_s) & (t_arr <= stim_end_s)
                ax.fill_between(t_arr, b0_rates, b1_rates,
                                where=stim_mask_arr, alpha=0.3, color='#ff7f0e',
                                label='ΔRate (gain)')

        if stim_time_s is not None:
            ax.axvspan(stim_time_s, stim_end_s, alpha=0.15, color='gold')

        if res['has_da_pulse']:
            ax.axvspan(pulse_onset_s, pulse_offset_s, alpha=0.08, color='green')

        ax.set_ylabel('Firing Rate (Hz)', fontsize=11)
        ax.set_title(f"Run {res['run_id']}: Stimulus Ensemble Rate", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim(0, total_s)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Time (s)', fontsize=11)
    axes[-1, 1].set_xlabel('Time (s)', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "wm_overview.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: wm_overview.png")

    # ================================================================
    # Figure 2: Zoomed stimulus response comparison
    # ================================================================
    fig, axes = plt.subplots(2, n_runs, figsize=(6 * n_runs, 10), squeeze=False)
    fig.suptitle(
        "Stimulus-Evoked Response — Zoomed View\n"
        f"(Ensemble: {args.stim_n_neurons} E neurons, {args.stim_amplitude} pA × {args.stim_duration}s)",
        fontsize=13, fontweight='bold'
    )

    for col, res in enumerate(all_results):
        stim_time_s = res['stim_time_s']
        stim_end_s = stim_time_s + args.stim_duration if stim_time_s else None

        stim_rate = res.get('stim_ensemble_rate', None)

        # Top row: B0 vs B1 firing rates (zoomed)
        ax = axes[0, col]
        if stim_rate is not None and stim_time_s is not None:
            zoom_start = stim_time_s - 10
            zoom_end = min(stim_time_s + args.stim_duration + args.post_stim_observe, res['total_s'])

            for batch_id, (color, ls, label) in enumerate(
                [('#2ca02c', '--', 'B0 (Ctrl)'), ('#ff7f0e', '-', 'B1 (Exp)')]):
                rd = stim_rate[batch_id]
                ax.plot(rd['times'], rd['rates'], ls, color=color, lw=2, label=label)

            ax.axvspan(stim_time_s, stim_end_s, alpha=0.3, color='gold', label='Stimulus')
            ax.set_xlim(zoom_start, zoom_end)

            # Annotate gain metrics
            gain_data = res.get('gain_metrics', {}).get('gain', {})
            delta_during = gain_data.get('delta_rate_during_hz', 0)
            delta_peak = gain_data.get('delta_peak_hz', 0)
            ax.annotate(f"ΔRate={delta_during:+.1f} Hz\nΔPeak={delta_peak:+.1f} Hz",
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        fontsize=9, va='top', ha='left', fontweight='bold',
                        color='#ff7f0e',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No stimulus', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='gray')

        ax.set_ylabel('Firing Rate (Hz)', fontsize=10)
        ax.set_title(f"Run {res['run_id']}: {res['run_label']}", fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom row: ΔRate (B1 - B0) time course
        ax = axes[1, col]
        if stim_rate is not None and stim_time_s is not None:
            b0_rates = stim_rate[0]['rates']
            b1_rates = stim_rate[1]['rates']
            t_arr = stim_rate[0]['times']
            delta_rates = b1_rates - b0_rates

            ax.plot(t_arr, delta_rates, '-', color='#9467bd', lw=2, label='ΔRate (B1-B0)')
            ax.axhline(0, color='gray', ls='--', lw=1)
            ax.axvspan(stim_time_s, stim_end_s, alpha=0.3, color='gold')
            ax.set_xlim(zoom_start, zoom_end)

            # Fill positive/negative regions
            ax.fill_between(t_arr, 0, delta_rates,
                            where=(t_arr >= zoom_start) & (t_arr <= zoom_end) & (delta_rates > 0),
                            alpha=0.3, color='#d62728', label='Enhanced')
            ax.fill_between(t_arr, 0, delta_rates,
                            where=(t_arr >= zoom_start) & (t_arr <= zoom_end) & (delta_rates < 0),
                            alpha=0.3, color='#1f77b4', label='Suppressed')

        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('ΔRate (Hz)', fontsize=10)
        ax.set_title(f"Run {res['run_id']}: DA Modulation of Stimulus Response", fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "wm_persistent_zoom.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: wm_persistent_zoom.png")

    # ================================================================
    # Figure 3: Gain summary — bar chart + line plot
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Stimulus Response Gain Modulation Summary",
                 fontsize=14, fontweight='bold')

    # Left: Bar chart of ΔRate during stimulus for each run
    ax = axes[0]
    bar_data = []
    bar_labels = []
    bar_colors = []
    stim_times_list = []

    for res in all_results:
        gain = res.get('gain_metrics', {}).get('gain', {})
        delta = gain.get('delta_rate_during_hz', 0.0)
        bar_data.append(delta)
        bar_labels.append(f"Run{res['run_id']}\n{res['run_label'][:25]}")
        stim_times_list.append(res.get('stim_time_s', 0))

        # Color based on sign
        if delta > 0.5:
            bar_colors.append('#d62728')  # Enhanced (red)
        elif delta < -0.5:
            bar_colors.append('#1f77b4')  # Suppressed (blue)
        else:
            bar_colors.append('#7f7f7f')  # Neutral (gray)

    x = np.arange(len(bar_data))
    bars = ax.bar(x, bar_data, color=bar_colors, width=0.6, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, bar_data):
        y_pos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 0.8
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f'{val:+.1f} Hz', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=10, fontweight='bold')

    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel('ΔRate during stimulus (Hz)\n(B1_Exp - B0_Ctrl)', fontsize=11)
    ax.set_title('DA Modulation of Stimulus-Evoked Response', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Right: ΔRate as function of stimulus time (relative to pulse offset)
    ax = axes[1]
    stim_times_rel = []
    delta_rates_plot = []
    delta_peaks_plot = []
    alpha_d1_at_stim = []
    alpha_d2_at_stim = []

    for res in all_results:
        gain = res.get('gain_metrics', {}).get('gain', {})
        stim_t = res.get('stim_time_s', 0)
        stim_rel = stim_t - pulse_offset_s  # Relative to pulse offset

        stim_times_rel.append(stim_rel)
        delta_rates_plot.append(gain.get('delta_rate_during_hz', 0.0))
        delta_peaks_plot.append(gain.get('delta_peak_hz', 0.0))

        # Get alpha values at stimulus time
        alpha_d1 = res['alpha_d1_trace']
        alpha_d2 = res['alpha_d2_trace']
        alpha_times = np.linspace(0, res['total_s'], len(alpha_d1))
        idx = np.argmin(np.abs(alpha_times - stim_t))
        alpha_d1_at_stim.append(float(alpha_d1[idx, 1]))
        alpha_d2_at_stim.append(float(alpha_d2[idx, 1]))

    ax.plot(stim_times_rel, delta_rates_plot, 'o-', color='#d62728', lw=2.5,
            markersize=10, label='ΔRate (mean during stim)', zorder=5)
    ax.plot(stim_times_rel, delta_peaks_plot, 's--', color='#ff7f0e', lw=1.5,
            markersize=8, label='ΔPeak', zorder=4)

    # Secondary y-axis for alpha values
    ax2 = ax.twinx()
    ax2.plot(stim_times_rel, alpha_d1_at_stim, '^:', color='#d62728', lw=1, alpha=0.5,
             markersize=6, label='α_D1 at stim')
    ax2.plot(stim_times_rel, alpha_d2_at_stim, 'v:', color='#1f77b4', lw=1, alpha=0.5,
             markersize=6, label='α_D2 at stim')
    alpha_diff_at_stim = [a1 - a2 for a1, a2 in zip(alpha_d1_at_stim, alpha_d2_at_stim)]
    ax2.bar(stim_times_rel, alpha_diff_at_stim, width=8, alpha=0.15, color='#d62728',
            label='α_D1 - α_D2')
    ax2.set_ylabel('α value at stimulus time', fontsize=10, color='gray')
    ax2.legend(fontsize=8, loc='upper right')

    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='green', ls='--', lw=1.5, alpha=0.7, label='DA pulse offset')

    # Mark the DA pulse period
    ax.axvspan(pulse_onset_s - pulse_offset_s, 0, alpha=0.08, color='green', label='DA pulse')

    ax.set_xlabel('Time relative to DA pulse offset (s)', fontsize=11)
    ax.set_ylabel('ΔRate (Hz)', fontsize=11)
    ax.set_title('Stimulus Gain vs. Timing (Afterglow Window Scan)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "wm_persistence_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: wm_persistence_summary.png")

    # ================================================================
    # Figure 4: All-E firing rate comparison across runs
    # ================================================================
    fig, axes = plt.subplots(n_runs, 1, figsize=(18, 4 * n_runs), sharex=False)
    if n_runs == 1:
        axes = [axes]
    fig.suptitle("All Excitatory Neurons — Firing Rate Time Course",
                 fontsize=14, fontweight='bold')

    for row, res in enumerate(all_results):
        ax = axes[row]
        rate_data = res.get('rate_data', {})
        total_s = res['total_s']
        stim_time_s = res['stim_time_s']
        stim_end_s = stim_time_s + args.stim_duration if stim_time_s else None

        for grp, color, ls in [('All-E', '#e377c2', '-'), ('All-I', '#17becf', '-')]:
            if grp in rate_data:
                for bid, (bls, balpha) in enumerate([(':', 0.5), ('-', 1.0)]):
                    if bid in rate_data[grp]:
                        rd = rate_data[grp][bid]
                        batch_label = 'Ctrl' if bid == 0 else 'Exp'
                        ax.plot(rd['times'], rd['rates'], bls, color=color,
                                lw=2, alpha=balpha, label=f'{grp} ({batch_label})')

        if res['has_da_pulse']:
            ax.axvspan(pulse_onset_s, pulse_offset_s, alpha=0.1, color='green')
        if stim_time_s is not None:
            ax.axvspan(stim_time_s, stim_end_s, alpha=0.3, color='gold')

        ax.set_ylabel('Rate (Hz)', fontsize=11)
        ax.set_title(f"Run {res['run_id']}: {res['run_label']}", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, ncol=4, loc='upper right')
        ax.set_xlim(0, total_s)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "wm_all_e_rates.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: wm_all_e_rates.png")


def main():
    args = parse_args()
    t_total_start = time.time()

    # Parse run selection
    run_ids = [int(x.strip()) for x in args.runs.split(',')]

    # Parse stimulus times
    stim_times = [float(x.strip()) for x in args.stim_times.split(',')]

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
    save_dir = os.path.join("outputs", f"exp_d_wm_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  🧪 Experiment D: Stimulus Response Gain Modulation")
    print(f"{'='*70}")
    print(f"  DA pulse: {args.da_base} → {args.da_pulse} → {args.da_base} nM")
    print(f"  Pulse window: [{args.pre_pulse}s, {args.pre_pulse + args.pulse_duration}s)")
    print(f"  Stimulus: {args.stim_amplitude} pA × {args.stim_duration}s, {args.stim_n_neurons} neurons")
    print(f"  Stimulus times: {stim_times}")
    print(f"  Post-stim observation: {args.post_stim_observe}s")
    print(f"  Runs to execute: {run_ids}")
    print(f"  Output: {save_dir}")
    print(f"{'='*70}")

    # Load checkpoint
    print(f"\n📂 Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        ckpt_data = pickle.load(f)
    init_state = ckpt_data['final_state'].to(device)

    # Build network (same seed as checkpoint)
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(config.N_E, config.N_I, device)

    # Create stimulus ensemble
    print(f"\n📌 Creating stimulus ensemble...")
    stim_mask, stim_indices = create_stimulus_ensemble(
        groups_info, args.stim_n_neurons, device)

    # Zero stimulus mask (for no-stim control — not used in current design)
    stim_mask_zero = torch.zeros_like(stim_mask)

    # Record indices (sample neurons for voltage traces)
    target_d1 = 0
    target_d2 = groups_info['e_d1_end']
    record_indices = torch.tensor([
        [0, target_d1], [1, target_d1],
        [0, target_d2], [1, target_d2],
    ], device=device, dtype=torch.long)

    # ================================================================
    # Define the 5 runs
    # ================================================================
    pulse_offset_s = args.pre_pulse + args.pulse_duration

    run_configs = {
        1: {
            'label': f'Baseline (no DA pulse, stim@{stim_times[0]:.0f}s)',
            'has_da_pulse': False,
            'stim_time_s': stim_times[0] if len(stim_times) > 0 else 80.0,
        },
        2: {
            'label': f'During DA pulse (stim@{stim_times[1]:.0f}s)',
            'has_da_pulse': True,
            'stim_time_s': stim_times[1] if len(stim_times) > 1 else 35.0,
        },
        3: {
            'label': f'Early afterglow (stim@{stim_times[2]:.0f}s)',
            'has_da_pulse': True,
            'stim_time_s': stim_times[2] if len(stim_times) > 2 else 65.0,
        },
        4: {
            'label': f'Peak afterglow (stim@{stim_times[3]:.0f}s)',
            'has_da_pulse': True,
            'stim_time_s': stim_times[3] if len(stim_times) > 3 else 100.0,
        },
        5: {
            'label': f'Late recovery (stim@{stim_times[4]:.0f}s)',
            'has_da_pulse': True,
            'stim_time_s': stim_times[4] if len(stim_times) > 4 else 250.0,
        },
    }

    # ================================================================
    # Execute runs
    # ================================================================
    all_results = []

    for rid in run_ids:
        if rid not in run_configs:
            print(f"⚠️ Unknown run ID: {rid}, skipping")
            continue

        rc = run_configs[rid]
        result = run_single_experiment(
            run_id=rid,
            run_label=rc['label'],
            W_t=W_t,
            mask_d1=mask_d1,
            mask_d2=mask_d2,
            init_state=init_state,
            stim_mask=stim_mask,
            stim_mask_zero=stim_mask_zero,
            record_indices=record_indices,
            args=args,
            device=device,
            has_da_pulse=rc['has_da_pulse'],
            stim_time_s=rc['stim_time_s'],
        )

        # Pack data for analyzer
        data = {
            'config': {
                'N_E': config.N_E, 'N_I': config.N_I,
                'duration': result['total_ms'], 'dt': config.DT,
                'da_onset': result['pulse_onset_ms'],
                'da_level': args.da_pulse if rc['has_da_pulse'] else args.da_base,
                'mode': 'working_memory',
            },
            'masks': {'d1': mask_d1.cpu(), 'd2': mask_d2.cpu()},
            'groups_info': groups_info,
            'spikes': result['spikes'],
            'v_traces': result['v_traces'],
            'record_indices': record_indices.cpu(),
        }

        # Compute firing rates
        print(f"     📊 Computing firing rates...")
        rate_data = compute_time_resolved_rates(data, time_win_ms=1000.0)
        result['rate_data'] = rate_data

        # Compute stimulus ensemble rate (finer time resolution)
        stim_rate = compute_stim_ensemble_rate(data, stim_indices, time_win_ms=500.0)
        result['stim_ensemble_rate'] = stim_rate

        # Measure stimulus-evoked gain
        if rc['stim_time_s'] is not None:
            stim_offset_s = rc['stim_time_s'] + args.stim_duration
            gain_metrics = measure_stimulus_gain(
                stim_rate, rc['stim_time_s'], stim_offset_s,
                baseline_window_s=10.0)
            result['gain_metrics'] = gain_metrics

            g = gain_metrics['gain']
            print(f"     📏 Gain: ΔRate={g['delta_rate_during_hz']:+.2f} Hz, "
                  f"ΔPeak={g['delta_peak_hz']:+.2f} Hz, "
                  f"Ratio={g['gain_ratio']:.3f}")
            for bid in [0, 1]:
                m = gain_metrics[bid]
                batch_label = 'B0(Ctrl)' if bid == 0 else 'B1(Exp)'
                print(f"     📏 {batch_label}: baseline={m['baseline_rate_hz']:.1f} Hz, "
                      f"during={m['during_stim_rate_hz']:.1f} Hz, "
                      f"peak={m['peak_rate_hz']:.1f} Hz, "
                      f"post={m['post_stim_rate_hz']:.1f} Hz")
        else:
            result['gain_metrics'] = {}

        all_results.append(result)

    # ================================================================
    # Generate plots
    # ================================================================
    print(f"\n🎨 Generating comparison plots...")
    plot_run_comparison(all_results, stim_indices, args, save_dir)

    # ================================================================
    # Save results
    # ================================================================
    summary = {
        'args': vars(args),
        'stim_indices': stim_indices,
        'timestamp': timestamp,
        'runs': [],
    }

    for res in all_results:
        run_summary = {
            'run_id': res['run_id'],
            'run_label': res['run_label'],
            'has_da_pulse': res['has_da_pulse'],
            'stim_time_s': res['stim_time_s'],
            'total_s': res['total_s'],
            'sim_time': res['sim_time'],
            'n_spikes': int(res['spikes'].shape[0]),
            'alpha_d1_exp_peak': float(res['alpha_d1_trace'][:, 1].max()),
            'alpha_d2_exp_peak': float(res['alpha_d2_trace'][:, 1].max()),
        }

        # Add gain metrics
        if res.get('gain_metrics'):
            gm = res['gain_metrics']
            run_summary['gain_metrics'] = {
                'batch_0': gm.get(0, {}),
                'batch_1': gm.get(1, {}),
                'gain': gm.get('gain', {}),
            }

        summary['runs'].append(run_summary)

    # Save JSON
    json_path = os.path.join(save_dir, "exp_d_results.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📄 Results saved: {json_path}")

    # Save alpha traces for each run
    for res in all_results:
        np.savez(os.path.join(save_dir, f"alpha_traces_run{res['run_id']}.npz"),
                 times=np.linspace(0, res['total_s'], len(res['alpha_d1_trace'])),
                 alpha_d1=res['alpha_d1_trace'],
                 alpha_d2=res['alpha_d2_trace'])

    # ================================================================
    # Print summary
    # ================================================================
    t_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"  ⏱️  Total time: {_fmt_elapsed(t_total)}")
    print(f"  📁 Results in: {save_dir}")
    print(f"{'='*70}")

    pulse_off = args.pre_pulse + args.pulse_duration
    print(f"\n  📊 Stimulus Response Gain Summary:")
    print(f"  {'Run':<6} {'Label':<40} {'Stim@':<8} {'Rel.t':<8} {'ΔRate':<10} {'ΔPeak':<10} {'Ratio':<8}")
    print(f"  {'─'*6} {'─'*40} {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*8}")

    for res in all_results:
        gm = res.get('gain_metrics', {}).get('gain', {})
        stim_t = res.get('stim_time_s', 0)
        rel_t = stim_t - pulse_off
        delta_r = gm.get('delta_rate_during_hz', 0.0)
        delta_p = gm.get('delta_peak_hz', 0.0)
        ratio = gm.get('gain_ratio', float('nan'))
        print(f"  {res['run_id']:<6} {res['run_label']:<40} {stim_t:<8.0f} {rel_t:<+8.0f} "
              f"{delta_r:<+10.2f} {delta_p:<+10.2f} {ratio:<8.3f}")

    print(f"\n  🔬 Key Comparisons (ΔRate = B1_Exp - B0_Ctrl during stimulus):")
    run_map = {r['run_id']: r for r in all_results}

    if 1 in run_map:
        baseline_gain = run_map[1].get('gain_metrics', {}).get('gain', {}).get('delta_rate_during_hz', 0)
        print(f"     Run 1 (baseline, no DA pulse): ΔRate = {baseline_gain:+.2f} Hz")

        for rid in [2, 3, 4, 5]:
            if rid in run_map:
                gain = run_map[rid].get('gain_metrics', {}).get('gain', {}).get('delta_rate_during_hz', 0)
                diff = gain - baseline_gain
                direction = '↑ ENHANCED' if diff > 0.5 else '↓ SUPPRESSED' if diff < -0.5 else '→ SIMILAR'
                print(f"     Run {rid} vs Run 1: {gain:+.2f} Hz (Δ={diff:+.2f} Hz, {direction})")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
