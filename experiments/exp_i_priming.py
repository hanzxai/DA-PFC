#!/usr/bin/env python3
"""
Experiment I: Priming Effect between Stimulus and DA Pulse
===========================================================

Scientific Question
-------------------
Does the temporal ORDER of stimulus and DA pulse produce an ASYMMETRIC
priming effect on the E-D1 ensemble?

Key Hypothesis
--------------
Due to the slow D1 receptor off-kinetics (tau_off ~164 s), a DA pulse
creates a prolonged "D1-only afterglow window" where R_eff is elevated
and synaptic scaling is enhanced for D1-expressing neurons. Therefore:

    Cond B (DA -> Stim):  stimulus lands in the D1 afterglow -> ENHANCED
    Cond A (Stim -> DA):  stimulus precedes DA              -> WEAK or none

Priming Index = DeltaRate(Cond B) - DeltaRate(Cond A) > 0
would demonstrate an ASYMMETRIC D1-mediated priming gate.

Experimental Design (4 Runs x 2 Batches)
----------------------------------------
All runs start from the same DA=2 nM steady-state checkpoint.
Each run has Batch 0 = constant DA=2 nM (control) and Batch 1 = the
DA schedule of that run.  The stimulus, when present, is applied to
BOTH batches so that DeltaRate = B1 - B0 isolates the pure DA effect.

Run 1 (Cond A: Stim -> DA)   stim [20, 25) s  ;  DA pulse [45, 75) s
Run 2 (Cond B: DA -> Stim)   DA pulse [20, 50) s  ;  stim [70, 75) s
Run 3 (Ctrl1: Stim only)     stim [20, 25) s  ;  DA flat 2 nM
Run 4 (Ctrl2: DA only)       DA pulse [20, 50) s  ;  no stimulus

Stimulus
--------
Target:    100 randomly selected E-D1 neurons (subgroup [0, e_d1_end))
Amplitude: 150 pA      (reduced from 300 pA -> V_inf approx -55 mV, sub-threshold)
Duration:  5 s
Shape:     square pulse

I_ext in the equations (see models/kernels.py run_dynamic_d1_d2_kernel_pulse_stim):

    I_total(t) = I_syn * scale_syn + I_bg + I_mod + I_stim
    I_stim_i(t) = A_stim * stim_mask_i   if t in [stim_on, stim_off)
                = 0                      otherwise
    R_eff_i(t) = R_base * (1 + eps_D1*alpha_D1*mask_D1_i - eps_D2*alpha_D2*mask_D2_i)
    V_inf_i    = V_rest + R_eff_i * I_total_i

Because E-D1 neurons simultaneously receive I_stim AND have R_eff boosted
by alpha_D1, an afterglow-timed stimulus is multiplicatively amplified.

Usage
-----
    python -m experiments.exp_i_priming
    python -m experiments.exp_i_priming --stim-amplitude 150 --stim-duration 5
    python -m experiments.exp_i_priming --runs 1,2,3,4 --gpu 0
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.network import create_network_structure
from models.kernels import run_dynamic_d1_d2_kernel_pulse_stim
from analysis.analyzer import PFCAnalyzer
from analysis.plotting import (
    plot_combined_raster,
    plot_combined_rates_all,
    plot_combined_rates_E,
    plot_combined_rates_I,
)

CHECKPOINT_PATH = "checkpoints/ckpt_DA2nM_bg200_500s.pkl"
RATE_GROUPS = ['E-D1', 'E-D2', 'E-Other', 'All-E', 'I-D1', 'I-D2', 'I-Other', 'All-I']


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Experiment I: Priming effect (Stim vs DA order)")
    # DA parameters
    parser.add_argument("--da-base", type=float, default=2.0,
                        help="Baseline DA concentration (nM)")
    parser.add_argument("--da-pulse", type=float, default=15.0,
                        help="Pulse DA concentration (nM)")
    parser.add_argument("--pulse-duration", type=float, default=30.0,
                        help="DA pulse duration (s)")
    # Stimulus parameters
    parser.add_argument("--stim-amplitude", type=float, default=150.0,
                        help="Stimulus current amplitude (pA), default 150")
    parser.add_argument("--stim-duration", type=float, default=5.0,
                        help="Stimulus duration (s), default 5")
    parser.add_argument("--stim-n-neurons", type=int, default=100,
                        help="Number of stimulated E-D1 neurons, default 100")
    # Total simulation duration (covers afterglow decay: D1 tau_off ~164s, need ~2x)
    parser.add_argument("--total-duration", type=float, default=300.0,
                        help="Total simulation duration per run (s), default 300")
    # Run selection
    parser.add_argument("--runs", type=str, default="1,2,3,4",
                        help="Comma-separated run IDs to execute")
    # Misc
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--seed", type=int, default=123,
                        help="RNG seed for stimulus ensemble selection")
    return parser.parse_args()


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:05.2f}s"
    return f"{int(m)}m {s:05.2f}s" if m > 0 else f"{s:.2f}s"


# =============================================================================
# Stimulus ensemble (E-D1 ONLY)
# =============================================================================
def create_ed1_stimulus_ensemble(groups_info, n_stim, device, seed=123):
    """
    Select n_stim neurons exclusively from the E-D1 subgroup [0, e_d1_end).

    Returns:
        stim_mask:    (N,) float tensor, 1.0 at stimulated neuron indices
        stim_indices: python list of neuron indices
    """
    N = config.N_E + config.N_I
    e_d1_end = groups_info['e_d1_end']
    n_available = e_d1_end
    n_actual = min(n_stim, n_available)
    if n_actual < n_stim:
        print(f"  WARNING: requested {n_stim} E-D1 neurons but only "
              f"{n_available} available; using {n_actual}")

    rng = np.random.RandomState(seed=seed)
    stim_indices = rng.choice(np.arange(0, e_d1_end), size=n_actual, replace=False)
    stim_indices = np.sort(stim_indices)

    stim_mask = torch.zeros(N, device=device, dtype=torch.float32)
    stim_mask[torch.tensor(stim_indices, device=device, dtype=torch.long)] = 1.0

    print(f"  Stimulus ensemble: {n_actual} neurons, ALL from E-D1 "
          f"(E-D1 subgroup size = {n_available})")
    return stim_mask, stim_indices.tolist()


# =============================================================================
# Rate analysis helpers
# =============================================================================
def compute_time_resolved_rates(data, time_win_ms=1000.0):
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


def compute_stim_ensemble_rate(data, stim_indices, time_win_ms=500.0):
    dt = data['config']['dt']
    duration = data['config']['duration']
    spikes_all = data['spikes'].numpy()
    n_stim = max(1, len(stim_indices))
    stim_set = set(stim_indices)

    bins = np.arange(0, duration + time_win_ms, time_win_ms)
    centers = (bins[:-1] + bins[1:]) / 2

    results = {}
    for batch_id in [0, 1]:
        mask_batch = spikes_all[:, 1] == batch_id
        batch_spikes = spikes_all[mask_batch]
        ts = batch_spikes[:, 0] * dt
        ids = batch_spikes[:, 2]

        mask_stim = np.array([int(nid) in stim_set for nid in ids])
        h, _ = np.histogram(ts[mask_stim], bins=bins)
        rate = h / (time_win_ms / 1000.0 * n_stim)

        from scipy.ndimage import gaussian_filter1d
        rate_smooth = gaussian_filter1d(rate, sigma=1.5)

        results[batch_id] = {
            'times': centers / 1000.0,
            'rates': rate_smooth,
        }
    return results


def measure_stimulus_gain(stim_rate_data, stim_onset_s, stim_offset_s,
                          baseline_window_s=10.0, post_window_s=10.0):
    """Per-run gain metrics.  If stim_onset_s is None, returns zeros."""
    results = {}
    for batch_id in [0, 1]:
        rd = stim_rate_data[batch_id]
        times = rd['times']
        rates = rd['rates']

        if len(times) == 0 or stim_onset_s is None:
            results[batch_id] = {
                'baseline_rate_hz': 0.0,
                'during_stim_rate_hz': 0.0,
                'peak_rate_hz': 0.0,
                'post_stim_rate_hz': 0.0,
            }
            continue

        bl_start = max(0, stim_onset_s - baseline_window_s)
        bl_mask = (times >= bl_start) & (times < stim_onset_s)
        bl_rate = float(np.mean(rates[bl_mask])) if np.any(bl_mask) else 0.0

        during_mask = (times >= stim_onset_s) & (times < stim_offset_s)
        during_rate = float(np.mean(rates[during_mask])) if np.any(during_mask) else 0.0
        peak_rate = float(np.max(rates[during_mask])) if np.any(during_mask) else 0.0

        post_mask = (times >= stim_offset_s) & (times < stim_offset_s + post_window_s)
        post_rate = float(np.mean(rates[post_mask])) if np.any(post_mask) else 0.0

        results[batch_id] = {
            'baseline_rate_hz': bl_rate,
            'during_stim_rate_hz': during_rate,
            'peak_rate_hz': peak_rate,
            'post_stim_rate_hz': post_rate,
        }

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


# =============================================================================
# Single run executor
# =============================================================================
def run_single(run_id, run_label,
               W_t, mask_d1, mask_d2, init_state,
               stim_mask, record_indices,
               da_base, da_pulse_val,
               pulse_onset_s, pulse_offset_s,
               stim_onset_s, stim_offset_s, stim_amp,
               total_s, device):
    """
    Execute one run (2 batches sharing the same kernel call).

    Batch 0 (Control) always keeps DA = da_base (achieved by reusing the same
    kernel path but with da_pulse_val = da_base for control-like runs).
    In this exp we rely on the kernel's built-in design:
      Batch 0 = constant da_base throughout
      Batch 1 = da_base -> da_pulse_val -> da_base (pulse_onset, pulse_offset)

    For a run WITHOUT a DA pulse (Ctrl1), we set da_pulse_val = da_base so
    Batch 1 also stays flat, giving us a pure "no-DA-effect" control.

    For a run WITHOUT stimulus (Ctrl2), we shift stim_onset beyond total to
    effectively disable it.
    """
    total_ms = total_s * 1000.0

    # DA timing (ms) — if the run has no pulse, set pulse window outside total
    if pulse_onset_s is None:
        pulse_onset_ms = total_ms + 1000.0
        pulse_offset_ms = total_ms + 2000.0
    else:
        pulse_onset_ms = pulse_onset_s * 1000.0
        pulse_offset_ms = pulse_offset_s * 1000.0

    # Stimulus timing (ms) — if no stimulus, shift outside total
    if stim_onset_s is None:
        stim_onset_ms = total_ms + 1000.0
        stim_offset_ms = total_ms + 2000.0
        stim_amp_used = 0.0
    else:
        stim_onset_ms = stim_onset_s * 1000.0
        stim_offset_ms = stim_offset_s * 1000.0
        stim_amp_used = float(stim_amp)

    alpha_record_interval = 100  # every 100 ms

    print(f"\n  >> Run {run_id}: {run_label}")
    if pulse_onset_s is not None:
        print(f"     DA   : pulse {da_base}->{da_pulse_val}->{da_base} nM  "
              f"window [{pulse_onset_s:.0f}, {pulse_offset_s:.0f}) s")
    else:
        print(f"     DA   : constant {da_base} nM (no pulse)")
    if stim_onset_s is not None:
        print(f"     Stim : {stim_amp_used:.0f} pA x {stim_offset_s - stim_onset_s:.1f} s  "
              f"window [{stim_onset_s:.0f}, {stim_offset_s:.0f}) s  (E-D1 only)")
    else:
        print(f"     Stim : none")
    print(f"     Duration: {total_s:.0f} s")

    t0 = time.time()
    spikes, v_traces, final_state, alpha_d1_trace, alpha_d2_trace = \
        run_dynamic_d1_d2_kernel_pulse_stim(
            W_t, mask_d1, mask_d2, init_state,
            float(da_base), float(da_pulse_val),
            float(pulse_onset_ms), float(pulse_offset_ms),
            stim_mask,
            float(stim_onset_ms), float(stim_offset_ms), float(stim_amp_used),
            float(total_ms), float(config.DT),
            record_indices, config.N_E,
            alpha_record_interval,
            config.build_kernel_params(device),
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sim_time = time.time() - t0
    print(f"     Done in {_fmt_elapsed(sim_time)} | Spikes: {spikes.shape[0]:,}")

    return {
        'run_id': run_id,
        'run_label': run_label,
        'has_da_pulse': pulse_onset_s is not None,
        'has_stim': stim_onset_s is not None,
        'pulse_onset_s': pulse_onset_s,
        'pulse_offset_s': pulse_offset_s,
        'stim_onset_s': stim_onset_s,
        'stim_offset_s': stim_offset_s,
        'stim_amp': stim_amp_used,
        'total_s': total_s,
        'total_ms': total_ms,
        'pulse_onset_ms': pulse_onset_ms,
        'pulse_offset_ms': pulse_offset_ms,
        'stim_onset_ms': stim_onset_ms,
        'stim_offset_ms': stim_offset_ms,
        'sim_time': sim_time,
        'spikes': spikes.cpu(),
        'v_traces': v_traces.cpu(),
        'final_state': final_state,
        'alpha_d1_trace': alpha_d1_trace.cpu().numpy(),
        'alpha_d2_trace': alpha_d2_trace.cpu().numpy(),
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_all(all_results, stim_indices, args, save_dir):
    n_runs = len(all_results)

    # -------------------------------------------------------------------------
    # Figure 1: Overview — alpha dynamics + stim-ensemble firing rate per run
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(n_runs, 2, figsize=(22, 4.5 * n_runs), squeeze=False)
    fig.suptitle(
        "Experiment I: Priming Effect between Stimulus and DA Pulse\n"
        f"DA pulse: {args.da_base}->{args.da_pulse}->{args.da_base} nM  |  "
        f"Stim: {args.stim_amplitude} pA x {args.stim_duration} s on "
        f"{args.stim_n_neurons} E-D1 neurons\n"
        f"B0 = constant DA={args.da_base} nM  |  B1 = run-specific DA schedule",
        fontsize=14, fontweight='bold'
    )

    for row, res in enumerate(all_results):
        total_s = res['total_s']
        alpha_d1 = res['alpha_d1_trace']
        alpha_d2 = res['alpha_d2_trace']
        alpha_times = np.linspace(0, total_s, len(alpha_d1))

        # Left: alpha dynamics
        ax = axes[row, 0]
        ax.plot(alpha_times, alpha_d1[:, 0], '--', color='#d62728', lw=1.2, alpha=0.4,
                label='alpha_D1 (B0 Ctrl)')
        ax.plot(alpha_times, alpha_d2[:, 0], '--', color='#1f77b4', lw=1.2, alpha=0.4,
                label='alpha_D2 (B0 Ctrl)')
        ax.plot(alpha_times, alpha_d1[:, 1], '-', color='#d62728', lw=2.5,
                label='alpha_D1 (B1 Exp)')
        ax.plot(alpha_times, alpha_d2[:, 1], '-', color='#1f77b4', lw=2.5,
                label='alpha_D2 (B1 Exp)')
        diff = alpha_d1[:, 1] - alpha_d2[:, 1]
        ax.fill_between(alpha_times, 0, diff, where=diff > 0, alpha=0.15,
                        color='#d62728', label='D1>D2 window')

        if res['has_da_pulse']:
            ax.axvspan(res['pulse_onset_s'], res['pulse_offset_s'],
                       alpha=0.1, color='green', label='DA pulse')
        if res['has_stim']:
            ax.axvspan(res['stim_onset_s'], res['stim_offset_s'],
                       alpha=0.3, color='gold', label='Stimulus')

        ax.set_ylabel('alpha', fontsize=11)
        ax.set_title(f"Run {res['run_id']}: {res['run_label']}  —  Receptor Dynamics",
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, ncol=3, loc='upper right')
        ax.set_xlim(0, total_s)
        ax.grid(True, alpha=0.3)

        # Right: stim-ensemble firing rate
        ax = axes[row, 1]
        sr = res.get('stim_ensemble_rate')
        if sr is not None:
            ax.plot(sr[0]['times'], sr[0]['rates'], '--', color='#2ca02c', lw=2,
                    label='B0 (Ctrl, flat DA)')
            ax.plot(sr[1]['times'], sr[1]['rates'], '-', color='#ff7f0e', lw=2,
                    label='B1 (Exp, DA schedule)')
            if res['has_stim']:
                t_arr = sr[0]['times']
                sm = (t_arr >= res['stim_onset_s']) & (t_arr <= res['stim_offset_s'])
                ax.fill_between(t_arr, sr[0]['rates'], sr[1]['rates'],
                                where=sm, alpha=0.3, color='#ff7f0e',
                                label='DeltaRate (gain)')

        if res['has_da_pulse']:
            ax.axvspan(res['pulse_onset_s'], res['pulse_offset_s'],
                       alpha=0.08, color='green')
        if res['has_stim']:
            ax.axvspan(res['stim_onset_s'], res['stim_offset_s'],
                       alpha=0.15, color='gold')

        ax.set_ylabel('Firing Rate (Hz)', fontsize=11)
        ax.set_title(f"Run {res['run_id']}: E-D1 Stimulus Ensemble Rate",
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim(0, total_s)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Time (s)', fontsize=11)
    axes[-1, 1].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "priming_overview.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: priming_overview.png")

    # -------------------------------------------------------------------------
    # Figure 2: Zoom on stimulus window (only for runs with stim)
    # -------------------------------------------------------------------------
    runs_with_stim = [r for r in all_results if r['has_stim']]
    n_z = len(runs_with_stim)
    if n_z > 0:
        fig, axes = plt.subplots(2, n_z, figsize=(6.5 * n_z, 10), squeeze=False)
        fig.suptitle("Stimulus-Evoked Response — Zoomed View (E-D1 ensemble)",
                     fontsize=13, fontweight='bold')

        for col, res in enumerate(runs_with_stim):
            son, soff = res['stim_onset_s'], res['stim_offset_s']
            zoom_start = max(0, son - 10)
            zoom_end = min(soff + 30, res['total_s'])

            # Top: B0 vs B1 rate
            ax = axes[0, col]
            sr = res.get('stim_ensemble_rate')
            if sr is not None:
                ax.plot(sr[0]['times'], sr[0]['rates'], '--', color='#2ca02c',
                        lw=2, label='B0 (Ctrl)')
                ax.plot(sr[1]['times'], sr[1]['rates'], '-', color='#ff7f0e',
                        lw=2, label='B1 (Exp)')
                ax.axvspan(son, soff, alpha=0.3, color='gold', label='Stimulus')
                if res['has_da_pulse']:
                    ax.axvspan(res['pulse_onset_s'], res['pulse_offset_s'],
                               alpha=0.08, color='green', label='DA pulse')
                ax.set_xlim(zoom_start, zoom_end)

                g = res.get('gain_metrics', {}).get('gain', {})
                txt = (f"DeltaRate during={g.get('delta_rate_during_hz', 0):+.2f} Hz\n"
                       f"DeltaPeak      ={g.get('delta_peak_hz', 0):+.2f} Hz\n"
                       f"DeltaRate post ={g.get('delta_rate_post_hz', 0):+.2f} Hz")
                ax.annotate(txt, xy=(0.02, 0.98), xycoords='axes fraction',
                            fontsize=9, va='top', ha='left', fontweight='bold',
                            color='#333',
                            bbox=dict(boxstyle='round', facecolor='lightyellow',
                                      alpha=0.85))

            ax.set_ylabel('Firing Rate (Hz)', fontsize=10)
            ax.set_title(f"Run {res['run_id']}: {res['run_label']}",
                         fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Bottom: DeltaRate time course
            ax = axes[1, col]
            if sr is not None:
                b0r, b1r = sr[0]['rates'], sr[1]['rates']
                t_arr = sr[0]['times']
                dr = b1r - b0r
                ax.plot(t_arr, dr, '-', color='#9467bd', lw=2, label='DeltaRate (B1-B0)')
                ax.axhline(0, color='gray', ls='--', lw=1)
                ax.axvspan(son, soff, alpha=0.3, color='gold')
                if res['has_da_pulse']:
                    ax.axvspan(res['pulse_onset_s'], res['pulse_offset_s'],
                               alpha=0.08, color='green')
                ax.fill_between(t_arr, 0, dr, where=(dr > 0) & (t_arr >= zoom_start) &
                                (t_arr <= zoom_end), alpha=0.3, color='#d62728',
                                label='Enhanced')
                ax.fill_between(t_arr, 0, dr, where=(dr < 0) & (t_arr >= zoom_start) &
                                (t_arr <= zoom_end), alpha=0.3, color='#1f77b4',
                                label='Suppressed')
                ax.set_xlim(zoom_start, zoom_end)

            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('DeltaRate (Hz)', fontsize=10)
            ax.set_title(f"Run {res['run_id']}: DA modulation of stim response",
                         fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "priming_zoom.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: priming_zoom.png")

    # -------------------------------------------------------------------------
    # Figure 3: Priming summary — Cond A vs Cond B bar chart + Priming Index
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Priming Index: Cond B (DA->Stim) vs Cond A (Stim->DA)",
                 fontsize=13, fontweight='bold')

    run_map = {r['run_id']: r for r in all_results}
    labels = []
    deltas_during = []
    deltas_peak = []
    deltas_post = []
    alpha_d1_at_stim = []
    alpha_d2_at_stim = []
    for rid in [1, 2, 3]:  # Cond A, Cond B, Ctrl1
        if rid not in run_map:
            continue
        res = run_map[rid]
        if not res['has_stim']:
            continue
        g = res.get('gain_metrics', {}).get('gain', {})
        labels.append(f"R{rid}\n{res['run_label']}")
        deltas_during.append(g.get('delta_rate_during_hz', 0))
        deltas_peak.append(g.get('delta_peak_hz', 0))
        deltas_post.append(g.get('delta_rate_post_hz', 0))

        a1 = res['alpha_d1_trace'][:, 1]
        a2 = res['alpha_d2_trace'][:, 1]
        at = np.linspace(0, res['total_s'], len(a1))
        idx = np.argmin(np.abs(at - res['stim_onset_s']))
        alpha_d1_at_stim.append(float(a1[idx]))
        alpha_d2_at_stim.append(float(a2[idx]))

    # Left: bar chart of DeltaRate during / peak / post
    ax = axes[0]
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, deltas_during, width, label='DeltaRate (during)',
           color='#d62728', edgecolor='black')
    ax.bar(x, deltas_peak, width, label='DeltaPeak',
           color='#ff7f0e', edgecolor='black')
    ax.bar(x + width, deltas_post, width, label='DeltaRate (post 10 s)',
           color='#9467bd', edgecolor='black')
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('DeltaRate (Hz)  (B1 - B0)', fontsize=11)
    ax.set_title('Stimulus-Evoked Response by Condition', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Priming Index = DeltaRate(B) - DeltaRate(A)
    ax = axes[1]
    if 1 in run_map and 2 in run_map:
        dA = run_map[1].get('gain_metrics', {}).get('gain', {}).get('delta_rate_during_hz', 0)
        dB = run_map[2].get('gain_metrics', {}).get('gain', {}).get('delta_rate_during_hz', 0)
        pi = dB - dA
        bars = ax.bar(['Cond A\n(Stim->DA)', 'Cond B\n(DA->Stim)', 'Priming\nIndex'],
                      [dA, dB, pi],
                      color=['#1f77b4', '#d62728', '#2ca02c'],
                      edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, [dA, dB, pi]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.1 if val >= 0 else -0.3),
                    f'{val:+.2f} Hz', ha='center',
                    va='bottom' if val >= 0 else 'top',
                    fontsize=11, fontweight='bold')
        ax.axhline(0, color='black', lw=1)
        ax.set_ylabel('DeltaRate (Hz)', fontsize=11)
        ax.set_title(f'Priming Index = DeltaRate(B) - DeltaRate(A) = {pi:+.2f} Hz',
                     fontsize=12, fontweight='bold')

        # Annotate alpha values at stim time
        if len(alpha_d1_at_stim) >= 2:
            txt = (f"At stim onset:\n"
                   f"Cond A: alpha_D1={alpha_d1_at_stim[0]:.3f}, alpha_D2={alpha_d2_at_stim[0]:.3f}\n"
                   f"Cond B: alpha_D1={alpha_d1_at_stim[1]:.3f}, alpha_D2={alpha_d2_at_stim[1]:.3f}")
            ax.annotate(txt, xy=(0.02, 0.02), xycoords='axes fraction',
                        fontsize=9, va='bottom', ha='left',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))

        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'Need Run 1 (Cond A) and Run 2 (Cond B)',
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "priming_summary.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: priming_summary.png")

    # -------------------------------------------------------------------------
    # Figure 4: All-E / All-I rates
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(n_runs, 1, figsize=(18, 4 * n_runs), sharex=False,
                             squeeze=False)
    fig.suptitle("All Excitatory / Inhibitory Rates", fontsize=14, fontweight='bold')
    for row, res in enumerate(all_results):
        ax = axes[row, 0]
        rate_data = res.get('rate_data', {})
        for grp, color in [('All-E', '#e377c2'), ('All-I', '#17becf')]:
            if grp in rate_data:
                for bid, (ls, alpha) in enumerate([(':', 0.5), ('-', 1.0)]):
                    if bid in rate_data[grp]:
                        rd = rate_data[grp][bid]
                        bl = 'Ctrl' if bid == 0 else 'Exp'
                        ax.plot(rd['times'], rd['rates'], ls, color=color, lw=2,
                                alpha=alpha, label=f'{grp} ({bl})')
        if res['has_da_pulse']:
            ax.axvspan(res['pulse_onset_s'], res['pulse_offset_s'],
                       alpha=0.1, color='green')
        if res['has_stim']:
            ax.axvspan(res['stim_onset_s'], res['stim_offset_s'],
                       alpha=0.3, color='gold')
        ax.set_ylabel('Rate (Hz)', fontsize=11)
        ax.set_title(f"Run {res['run_id']}: {res['run_label']}",
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, ncol=4, loc='upper right')
        ax.set_xlim(0, res['total_s'])
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('Time (s)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "priming_all_rates.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: priming_all_rates.png")


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    t_total_start = time.time()

    run_ids = [int(x.strip()) for x in args.runs.split(',')]

    if torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("outputs", f"exp_i_priming_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 72)
    print("  Experiment I: Priming Effect between Stimulus and DA Pulse")
    print("=" * 72)
    print(f"  DA     : {args.da_base} -> {args.da_pulse} -> {args.da_base} nM, "
          f"pulse duration {args.pulse_duration} s")
    print(f"  Stim   : {args.stim_amplitude} pA x {args.stim_duration} s "
          f"on {args.stim_n_neurons} E-D1 neurons")
    print(f"  Total  : {args.total_duration} s per run")
    print(f"  Runs   : {run_ids}")
    print(f"  Output : {save_dir}")
    print("=" * 72)

    # Load checkpoint (DA=2 nM steady state)
    print(f"\nLoading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        ckpt_data = pickle.load(f)
    init_state = ckpt_data['final_state'].to(device)

    # ── Fix: Both batches must resume from the DA=da_base steady-state ──
    # The source checkpoint (ckpt_DA2nM_bg200_500s.pkl) was produced by the
    # ckpt kernel, which uses Batch 0 = 0 nM (pure control, DA-free) and
    # Batch 1 = target DA (= 2 nM).  Only Batch 1 holds the true 2-nM
    # steady state.  Without this copy, Batch 0 (our "Ctrl") would start
    # from the 0-nM state and slowly drift toward 2 nM over the slow D1
    # tau_off ~164 s, which is exactly what the spurious Col1 drift in
    # the Ctrl1 firing-rate plot showed.  This matches main.py's behaviour
    # (simulation/runners.py::run_simulation_from_checkpoint).
    if init_state.shape[0] >= 2:
        print(f"   Overwriting Batch 0 state with Batch 1 (DA={args.da_base} nM steady-state)")
        init_state[0] = init_state[1].clone()

    # Build network with same seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(
        config.N_E, config.N_I, device)

    # Stimulus ensemble: E-D1 only
    print("\nCreating E-D1 stimulus ensemble...")
    stim_mask, stim_indices = create_ed1_stimulus_ensemble(
        groups_info, args.stim_n_neurons, device, seed=args.seed)

    # Voltage record indices: one E-D1 + one E-D2 in each batch
    target_d1 = 0
    target_d2 = groups_info['e_d1_end']
    record_indices = torch.tensor([
        [0, target_d1], [1, target_d1],
        [0, target_d2], [1, target_d2],
    ], device=device, dtype=torch.long)

    # -------------------------------------------------------------------------
    # Run configurations (priming design)
    # -------------------------------------------------------------------------
    pulse_dur = args.pulse_duration
    stim_dur = args.stim_duration

    run_configs = {
        1: {  # Cond A: Stim -> DA
            'label': 'Cond A: Stim -> DA (stim@20s, DA pulse@45-75s)',
            'pulse_onset_s':  45.0,
            'pulse_offset_s': 45.0 + pulse_dur,
            'stim_onset_s':   20.0,
            'stim_offset_s':  20.0 + stim_dur,
            'has_da_pulse':   True,
        },
        2: {  # Cond B: DA -> Stim
            'label': 'Cond B: DA -> Stim (DA pulse@20-50s, stim@70s)',
            'pulse_onset_s':  20.0,
            'pulse_offset_s': 20.0 + pulse_dur,
            'stim_onset_s':   70.0,
            'stim_offset_s':  70.0 + stim_dur,
            'has_da_pulse':   True,
        },
        3: {  # Ctrl1: Stim only
            'label': 'Ctrl1: Stim only (DA flat 2 nM, stim@20s)',
            'pulse_onset_s':  None,
            'pulse_offset_s': None,
            'stim_onset_s':   20.0,
            'stim_offset_s':  20.0 + stim_dur,
            'has_da_pulse':   False,
        },
        4: {  # Ctrl2: DA only
            'label': 'Ctrl2: DA only (DA pulse@20-50s, no stim)',
            'pulse_onset_s':  20.0,
            'pulse_offset_s': 20.0 + pulse_dur,
            'stim_onset_s':   None,
            'stim_offset_s':  None,
            'has_da_pulse':   True,
        },
    }

    # -------------------------------------------------------------------------
    # Execute
    # -------------------------------------------------------------------------
    all_results = []
    for rid in run_ids:
        if rid not in run_configs:
            print(f"WARNING: unknown run ID {rid}, skipping")
            continue
        rc = run_configs[rid]

        # For Ctrl1 (no DA pulse) force da_pulse_val = da_base so B1 also stays flat
        da_pulse_val = args.da_pulse if rc['has_da_pulse'] else args.da_base

        res = run_single(
            run_id=rid,
            run_label=rc['label'],
            W_t=W_t, mask_d1=mask_d1, mask_d2=mask_d2,
            init_state=init_state,
            stim_mask=stim_mask,
            record_indices=record_indices,
            da_base=args.da_base,
            da_pulse_val=da_pulse_val,
            pulse_onset_s=rc['pulse_onset_s'],
            pulse_offset_s=rc['pulse_offset_s'],
            stim_onset_s=rc['stim_onset_s'],
            stim_offset_s=rc['stim_offset_s'],
            stim_amp=args.stim_amplitude,
            total_s=args.total_duration,
            device=device,
        )

        # Analyze
        # For PFCAnalyzer: da_onset / da_offset define the DA pulse window (ms).
        # Use 'pulse_response' mode so plot_da_concentration draws a square
        # pulse (baseline -> pulse -> baseline) rather than a step.
        # For runs without DA pulse (Ctrl1), set da_level = da_base so B1 also
        # appears as a flat line at da_base, and keep da_onset at the midpoint
        # purely for the "Before DA / After DA" zoom-window heuristics.
        if rc['has_da_pulse']:
            da_onset_ms_for_analyzer = rc['pulse_onset_s'] * 1000.0
            da_offset_ms_for_analyzer = rc['pulse_offset_s'] * 1000.0
            da_level_for_plot = da_pulse_val
        else:
            da_onset_ms_for_analyzer = res['total_ms'] * 0.5  # midpoint
            da_offset_ms_for_analyzer = None
            da_level_for_plot = args.da_base  # flat line at baseline

        data = {
            'config': {
                'N_E': config.N_E, 'N_I': config.N_I,
                'duration': res['total_ms'], 'dt': config.DT,
                'da_onset': da_onset_ms_for_analyzer,
                'da_offset': da_offset_ms_for_analyzer,
                'da_level': da_level_for_plot,
                'control_da': args.da_base,
                'mode': 'pulse_response',
            },
            'masks': {'d1': mask_d1.cpu(), 'd2': mask_d2.cpu()},
            'groups_info': groups_info,
            'spikes': res['spikes'],
            'v_traces': res['v_traces'],
            'record_indices': record_indices.cpu(),
        }
        res['analyzer_data'] = data  # save for standard plots

        print("     Computing firing rates...")
        res['rate_data'] = compute_time_resolved_rates(data, time_win_ms=1000.0)
        res['stim_ensemble_rate'] = compute_stim_ensemble_rate(
            data, stim_indices, time_win_ms=500.0)

        if rc['stim_onset_s'] is not None:
            gm = measure_stimulus_gain(
                res['stim_ensemble_rate'],
                rc['stim_onset_s'], rc['stim_offset_s'],
                baseline_window_s=10.0, post_window_s=10.0)
            res['gain_metrics'] = gm
            g = gm['gain']
            print(f"     Gain: DeltaRate={g['delta_rate_during_hz']:+.2f} Hz  "
                  f"DeltaPeak={g['delta_peak_hz']:+.2f} Hz  "
                  f"Ratio={g['gain_ratio']:.3f}")
            for bid in [0, 1]:
                m = gm[bid]
                bl = 'B0(Ctrl)' if bid == 0 else 'B1(Exp)'
                print(f"       {bl}: baseline={m['baseline_rate_hz']:.2f} "
                      f"during={m['during_stim_rate_hz']:.2f} "
                      f"peak={m['peak_rate_hz']:.2f} "
                      f"post={m['post_stim_rate_hz']:.2f} Hz")
        else:
            res['gain_metrics'] = {}

        all_results.append(res)

    # -------------------------------------------------------------------------
    # Standard spike rate / raster plots (per run, like main.py output)
    # -------------------------------------------------------------------------
    print("\nGenerating standard spike rate & raster plots (per run)...")
    from pathlib import Path
    for res in all_results:
        run_sub_dir = Path(save_dir) / f"run{res['run_id']}_{res['run_label'].split(':')[0].strip().replace(' ', '_')}"
        run_sub_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Run {res['run_id']}: {run_sub_dir}")
        analyzer = PFCAnalyzer(res['analyzer_data'])
        plot_combined_raster(analyzer, save_dir=run_sub_dir)
        plot_combined_rates_all(analyzer, save_dir=run_sub_dir)
        plot_combined_rates_E(analyzer, save_dir=run_sub_dir)
        plot_combined_rates_I(analyzer, save_dir=run_sub_dir)
        analyzer.save_report(str(run_sub_dir / "analysis_report.txt"))

    # -------------------------------------------------------------------------
    # Priming-specific plots
    # -------------------------------------------------------------------------
    print("\nGenerating priming analysis plots...")
    plot_all(all_results, stim_indices, args, save_dir)

    # -------------------------------------------------------------------------
    # Save summary JSON
    # -------------------------------------------------------------------------
    summary = {
        'args': vars(args),
        'stim_indices_count': len(stim_indices),
        'stim_indices_first20': stim_indices[:20],
        'timestamp': timestamp,
        'runs': [],
    }
    for res in all_results:
        rs = {
            'run_id': res['run_id'],
            'run_label': res['run_label'],
            'has_da_pulse': res['has_da_pulse'],
            'has_stim': res['has_stim'],
            'pulse_onset_s': res['pulse_onset_s'],
            'pulse_offset_s': res['pulse_offset_s'],
            'stim_onset_s': res['stim_onset_s'],
            'stim_offset_s': res['stim_offset_s'],
            'stim_amp': res['stim_amp'],
            'total_s': res['total_s'],
            'sim_time': res['sim_time'],
            'n_spikes': int(res['spikes'].shape[0]),
            'alpha_d1_exp_peak': float(res['alpha_d1_trace'][:, 1].max()),
            'alpha_d2_exp_peak': float(res['alpha_d2_trace'][:, 1].max()),
        }
        if res.get('gain_metrics'):
            gm = res['gain_metrics']
            rs['gain_metrics'] = {
                'batch_0': gm.get(0, {}),
                'batch_1': gm.get(1, {}),
                'gain': gm.get('gain', {}),
            }
        summary['runs'].append(rs)

    # Priming Index
    run_map = {r['run_id']: r for r in all_results}
    if 1 in run_map and 2 in run_map:
        dA = run_map[1].get('gain_metrics', {}).get('gain', {}).get('delta_rate_during_hz', 0)
        dB = run_map[2].get('gain_metrics', {}).get('gain', {}).get('delta_rate_during_hz', 0)
        summary['priming_index_hz'] = dB - dA
        summary['cond_a_delta_hz'] = dA
        summary['cond_b_delta_hz'] = dB

    json_path = os.path.join(save_dir, "exp_i_results.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    for res in all_results:
        np.savez(os.path.join(save_dir, f"alpha_traces_run{res['run_id']}.npz"),
                 times=np.linspace(0, res['total_s'], len(res['alpha_d1_trace'])),
                 alpha_d1=res['alpha_d1_trace'],
                 alpha_d2=res['alpha_d2_trace'])

    # -------------------------------------------------------------------------
    # Terminal summary
    # -------------------------------------------------------------------------
    t_total = time.time() - t_total_start
    print("\n" + "=" * 72)
    print(f"  Total time: {_fmt_elapsed(t_total)}")
    print(f"  Results in: {save_dir}")
    print("=" * 72)

    print("\n  Summary table (DeltaRate = B1_Exp - B0_Ctrl during stim window):")
    hdr = f"  {'Run':<5} {'Label':<45} {'Stim@':<9} {'DeltaRate':<12} {'DeltaPeak':<12} {'Ratio':<8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for res in all_results:
        gm = res.get('gain_metrics', {}).get('gain', {})
        son = res.get('stim_onset_s')
        son_s = f"{son:.0f}s" if son is not None else "-"
        dr = gm.get('delta_rate_during_hz', float('nan'))
        dp = gm.get('delta_peak_hz', float('nan'))
        rt = gm.get('gain_ratio', float('nan'))
        dr_s = f"{dr:+.2f}" if not (isinstance(dr, float) and dr != dr) else "  n/a"
        dp_s = f"{dp:+.2f}" if not (isinstance(dp, float) and dp != dp) else "  n/a"
        rt_s = f"{rt:.3f}" if rt == rt else "  n/a"
        print(f"  {res['run_id']:<5} {res['run_label']:<45} {son_s:<9} "
              f"{dr_s:<12} {dp_s:<12} {rt_s:<8}")

    if 1 in run_map and 2 in run_map:
        dA = summary.get('cond_a_delta_hz', 0)
        dB = summary.get('cond_b_delta_hz', 0)
        pi = summary.get('priming_index_hz', 0)
        print("\n  Priming Index:")
        print(f"    Cond A (Stim -> DA) DeltaRate : {dA:+.2f} Hz")
        print(f"    Cond B (DA -> Stim) DeltaRate : {dB:+.2f} Hz")
        direction = "B > A (afterglow priming)" if pi > 0.3 else \
                    "A > B (consolidation effect)" if pi < -0.3 else \
                    "symmetric (no order effect)"
        print(f"    PRIMING INDEX = B - A         : {pi:+.2f} Hz  -> {direction}")
    print("=" * 72)


if __name__ == "__main__":
    main()
