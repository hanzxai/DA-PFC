#!/usr/bin/env python3
"""
Experiment G: Alpha-Function DA Spike — Cue-Evoked Phasic DA Release

Scientific Background (from FSCV paper):
  In a cue-triggered operant conditioning paradigm, DA release in PFC follows
  a stereotyped temporal pattern within each trial:

  1. CUE ONSET → first DA spike (anticipatory / predictive signal)
  2. FIXED DELAY (2s) + LEVER PRESS LATENCY (1.4–2.9s) + STIM DELAY (0.1–0.2s)
     → second DA spike (reward / outcome signal)
  3. INTER-TRIAL INTERVAL (ITI): 5–25s, normally distributed (variable timeout)

  Each DA spike follows an Alpha-function waveform:
    DA(t) = DA_base + A × (t/τ_rise) × exp(1 − t/τ_rise)
  Rise time ~0.5–1.0s, fall time ~1.0–1.5s.

Key Questions:
  - How do D1 (slow, τ_on≈31s) and D2 (fast, τ_on≈10s) receptors respond
    to this realistic phasic DA pattern?
  - Can D2 track individual spikes within a trial while D1 integrates
    across trials?
  - How does the random ITI affect receptor dynamics?

Trial Structure:
  |<-- cue spike -->|<-- intra-trial delay (3.5-5s) -->|<-- stim spike -->|<-- ITI (5-25s) -->|
  Each trial contains TWO alpha-function DA spikes.

Usage:
  # Default: paired spikes per trial (cue + stim, 3.5-5s apart)
  python -m experiments.exp_g_alpha_spike

  # Single-peak mode: one DA spike per trial (no cue/stim distinction)
  python -m experiments.exp_g_alpha_spike --mode single

  # Paired-wide mode: two DA spikes per trial with wider separation (10-15s)
  python -m experiments.exp_g_alpha_spike --mode paired-wide

  # Amplitude-scaling mode: DA spike amplitude scales with M (1,2,3,...)
  python -m experiments.exp_g_alpha_spike --mode amplitude --amp-levels 1 2 3
  python -m experiments.exp_g_alpha_spike --mode amplitude --amp-levels 1 2 3 --amp-repeats 3

  # Custom parameters:
  python -m experiments.exp_g_alpha_spike --n-trials 15 --gpu 0
  python -m experiments.exp_g_alpha_spike --tau-rise 0.8 --da-amp 15
  python -m experiments.exp_g_alpha_spike --intra-delay-min 3.5 --intra-delay-max 5.0
  python -m experiments.exp_g_alpha_spike --iti-mean 15 --iti-std 5
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
from matplotlib.patches import Patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.network import create_network_structure
from models.kernels import run_dynamic_d1_d2_kernel_da_schedule
from analysis.analyzer import PFCAnalyzer
from analysis.plotting import (plot_combined_raster,
                                plot_combined_rates_all,
                                plot_combined_rates_E,
                                plot_combined_rates_I)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment G: Alpha-Function DA Spike — Cue-Evoked Phasic DA Release",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Trial mode
    parser.add_argument("--mode", type=str, default="paired",
                        choices=["single", "paired", "paired-wide", "amplitude"],
                        help="Trial structure mode: "
                             "'single' = 1 DA spike per trial; "
                             "'paired' = 2 spikes per trial (cue+stim, 3.5-5s apart); "
                             "'paired-wide' = 2 spikes per trial with wider separation (10-15s); "
                             "'amplitude' = single spike per trial with amplitude scaling by M levels. "
                             "Default: paired")
    # Amplitude-scaling mode parameters
    parser.add_argument("--amp-levels", type=float, nargs='+', default=[1.0, 2.0, 3.0],
                        help="Amplitude multiplier levels M for 'amplitude' mode. "
                             "DA_transient = k * M * alpha(t). Default: [1, 2, 3]")
    parser.add_argument("--amp-repeats", type=int, default=5,
                        help="Number of trials per amplitude level in 'amplitude' mode. "
                             "Total trials = len(amp_levels) * amp_repeats. Default: 5")
    # DA waveform parameters
    parser.add_argument("--da-base", type=float, default=2.0,
                        help="Baseline DA concentration (nM), default 2.0")
    parser.add_argument("--da-amp", type=float, default=15.0,
                        help="DA spike peak amplitude above baseline (nM), default 15.0")
    parser.add_argument("--tau-rise", type=float, default=0.8,
                        help="Alpha function rise time constant (s), default 0.8. "
                             "Peak occurs at t=tau_rise. Typical range: 0.5–1.0s")

    # Trial structure parameters (from paper)
    parser.add_argument("--n-trials", type=int, default=15,
                        help="Number of trials (each trial = cue spike + stim spike), default 15")
    parser.add_argument("--intra-delay-min", type=float, default=3.5,
                        help="Min intra-trial delay between cue and stim spikes (s), default 3.5. "
                             "Paper: 2s fixed + 1.4s min latency + 0.1s stim delay = 3.5s")
    parser.add_argument("--intra-delay-max", type=float, default=5.0,
                        help="Max intra-trial delay between cue and stim spikes (s), default 5.0. "
                             "Paper: 2s fixed + 2.9s max latency + 0.1s stim delay = 5.0s")
    parser.add_argument("--iti-mean", type=float, default=15.0,
                        help="Mean inter-trial interval (s), default 15.0. "
                             "Paper: 5–25s, normally distributed")
    parser.add_argument("--iti-std", type=float, default=5.0,
                        help="Std of inter-trial interval (s), default 5.0")
    parser.add_argument("--iti-min", type=float, default=5.0,
                        help="Min inter-trial interval (s), default 5.0")
    parser.add_argument("--iti-max", type=float, default=25.0,
                        help="Max inter-trial interval (s), default 25.0")

    # Timing
    parser.add_argument("--pre-stim", type=float, default=30.0,
                        help="Pre-stimulus baseline duration (s), default 30")
    parser.add_argument("--post-stim", type=float, default=80.0,
                        help="Post-stimulus recovery duration (s), default 80")

    # System
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint path (auto-detected if not specified)")
    parser.add_argument("--skip-ckpt", action="store_true",
                        help="Skip checkpoint generation, reuse existing")
    parser.add_argument("--base-dur", type=float, default=500.0,
                        help="Baseline checkpoint duration in seconds (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for trial timing generation, default 42")
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
# Alpha-Function DA Waveform Generator
# ==============================================================================

def alpha_function(t, tau_rise):
    """
    Normalized Alpha function: peaks at t=tau_rise with value 1.0.

    f(t) = (t / tau_rise) * exp(1 - t / tau_rise)   for t >= 0
         = 0                                          for t < 0
    """
    f = np.zeros_like(t)
    pos = t >= 0
    t_pos = t[pos]
    f[pos] = (t_pos / tau_rise) * np.exp(1.0 - t_pos / tau_rise)
    return f


def generate_trial_schedule(args):
    """
    Generate the trial timing schedule.

    Supports four modes:
      - 'single':      1 DA spike per trial (no cue/stim distinction)
      - 'paired':      2 DA spikes per trial (cue + stim, intra-delay 3.5-5s)
      - 'paired-wide': 2 DA spikes per trial (wider intra-delay, 10-15s)
      - 'amplitude':   1 DA spike per trial, amplitude scales by M levels.
                       Trials are grouped by amplitude level, each repeated amp_repeats times.
                       n_trials is overridden to len(amp_levels) * amp_repeats.

    Returns:
        trial_info: list of dicts (each dict has 'amp_multiplier' key for amplitude mode)
        all_spike_onsets_s: flat list of all DA spike onset times
    """
    rng = np.random.default_rng(args.seed)
    mode = args.mode

    # Override intra-delay for paired-wide mode
    if mode == 'paired-wide':
        intra_min = 10.0
        intra_max = 15.0
    else:
        intra_min = args.intra_delay_min
        intra_max = args.intra_delay_max

    # For amplitude mode, build the M-level sequence and override n_trials
    if mode == 'amplitude':
        amp_levels = sorted(args.amp_levels)
        amp_repeats = args.amp_repeats
        # Build sequence: [M1]*repeats + [M2]*repeats + ... (ascending order)
        amp_sequence = []
        for m in amp_levels:
            amp_sequence.extend([m] * amp_repeats)
        n_trials = len(amp_sequence)
        args.n_trials = n_trials  # override
    else:
        amp_sequence = [1.0] * args.n_trials
        n_trials = args.n_trials

    trial_info = []
    all_spike_onsets_s = []
    current_time = args.pre_stim  # start after baseline

    for i in range(n_trials):
        spike_onset = current_time
        m = amp_sequence[i]

        if mode in ('single', 'amplitude'):
            # Single spike per trial
            trial_info.append({
                'trial_num': i + 1,
                'cue_onset_s': spike_onset,
                'stim_onset_s': None,
                'intra_delay_s': 0.0,
                'amp_multiplier': m,
            })
            all_spike_onsets_s.append(spike_onset)
            last_event = spike_onset
        else:
            # Paired modes: cue spike + stim spike
            cue_onset = spike_onset
            intra_delay = rng.uniform(intra_min, intra_max)
            stim_onset = cue_onset + intra_delay

            trial_info.append({
                'trial_num': i + 1,
                'cue_onset_s': cue_onset,
                'stim_onset_s': stim_onset,
                'intra_delay_s': intra_delay,
                'amp_multiplier': m,
            })
            all_spike_onsets_s.extend([cue_onset, stim_onset])
            last_event = stim_onset

        # Inter-trial interval
        if i < n_trials - 1:
            iti = rng.normal(args.iti_mean, args.iti_std)
            iti = np.clip(iti, args.iti_min, args.iti_max)
            current_time = last_event + iti
            trial_info[-1]['iti_s'] = iti
        else:
            trial_info[-1]['iti_s'] = None

    return trial_info, all_spike_onsets_s


def build_trial_da_schedule(args, trial_info, all_spike_onsets_s, dt: float):
    """
    Build DA schedule with Alpha-function spikes at all trial event times.

    In 'amplitude' mode, each spike's amplitude is scaled by the trial's M multiplier:
        DA_transient(t) = da_base + k * M * alpha(t)
    where k = da_amp (base amplitude for M=1).

    Returns:
        da_schedule: (steps, 2) numpy array — col 0 = Ctrl, col 1 = Exp
        total_s: total duration in seconds
    """
    tau_rise = args.tau_rise
    da_base = args.da_base
    da_amp = args.da_amp

    # Total duration: last event spike + decay window + post-stim recovery
    last_stim = trial_info[-1]['stim_onset_s'] or trial_info[-1]['cue_onset_s']
    total_s = last_stim + 8.0 * tau_rise + args.post_stim
    total_steps = int(total_s * 1000.0 / dt)

    t_ms = np.arange(total_steps) * dt
    t_s = t_ms / 1000.0

    da_ctrl = np.full(total_steps, da_base)
    da_exp = np.full(total_steps, da_base, dtype=np.float64)

    # Build onset-to-multiplier mapping for amplitude mode
    onset_to_amp = {}
    for ti in trial_info:
        m = ti.get('amp_multiplier', 1.0)
        onset_to_amp[ti['cue_onset_s']] = m
        if ti['stim_onset_s'] is not None:
            onset_to_amp[ti['stim_onset_s']] = m

    # Superimpose alpha spikes at all event times
    for onset in all_spike_onsets_s:
        t_relative = t_s - onset
        spike_waveform = alpha_function(t_relative, tau_rise)
        m = onset_to_amp.get(onset, 1.0)
        da_exp += da_amp * m * spike_waveform

    # Clamp DA >= 0.1 nM
    da_exp = np.clip(da_exp, 0.1, None)

    da_schedule = np.stack([da_ctrl, da_exp], axis=1)
    return da_schedule, total_s


# ==============================================================================
# Simulation Runner
# ==============================================================================

def run_trial_sim(ckpt_path: str, device: torch.device,
                  da_schedule: np.ndarray, total_s: float,
                  alpha_record_interval: int = 10):
    """
    Run the trial-based DA simulation from checkpoint.

    Returns:
        data: dict compatible with PFCAnalyzer (same format as main.py output)
    """
    dt = config.DT

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

    print(f"  ⚡ Running simulation ({total_s:.0f}s)...")
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

    # Build data dict compatible with PFCAnalyzer
    # da_onset = time of first cue spike (for PFCAnalyzer baseline/post-DA split)
    first_cue_onset_ms = float(da_schedule[:, 1].argmax()) * dt  # approximate
    # More precise: find first time DA > baseline + 0.5
    da_exp_arr = da_schedule[:, 1]
    above_bl = np.where(da_exp_arr > da_schedule[0, 0] + 0.5)[0]
    if len(above_bl) > 0:
        first_cue_onset_ms = float(above_bl[0]) * dt
    else:
        first_cue_onset_ms = 0.0

    data = {
        'config': {
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': total_s * 1000.0, 'dt': config.DT,
            'da_onset': first_cue_onset_ms,
            'da_level': float(da_schedule[:, 1].max()),
            'control_da': float(da_schedule[0, 0]),
            'mode': 'alpha_spike',
        },
        'masks': {'d1': mask_d1.cpu(), 'd2': mask_d2.cpu()},
        'groups_info': groups_info,
        'spikes': all_spikes.cpu(),
        'v_traces': v_traces.cpu(),
        'record_indices': record_indices.cpu(),
        'alpha_d1_trace': alpha_d1_trace.cpu().numpy(),
        'alpha_d2_trace': alpha_d2_trace.cpu().numpy(),
        'da_schedule': da_schedule,
        'total_s': total_s,
        'sim_time': elapsed,
    }
    return data


# ==============================================================================
# Supplementary Plots: Receptor Dynamics & Trial Analysis
# ==============================================================================

def plot_receptor_dynamics(data: dict, trial_info: list, all_spike_onsets_s: list,
                           args, save_dir: Path):
    """
    Plot supplementary receptor dynamics figures:
    1. DA waveform + D1/D2 alpha traces (full view)
    2. Zoomed view of 2-3 consecutive trials
    3. Cumulative buildup analysis across trials
    4. Post-session decay comparison
    """
    alpha_d1_exp = data['alpha_d1_trace'][:, 1]
    alpha_d2_exp = data['alpha_d2_trace'][:, 1]
    alpha_d1_ctrl = data['alpha_d1_trace'][:, 0]
    alpha_d2_ctrl = data['alpha_d2_trace'][:, 0]
    total_s = data['total_s']
    da_schedule = data['da_schedule']

    alpha_times = np.linspace(0, total_s, len(alpha_d1_exp))
    da_t = np.linspace(0, total_s, da_schedule.shape[0])

    # Baseline values (before first cue)
    first_cue = trial_info[0]['cue_onset_s']
    bl_mask = alpha_times < first_cue
    bl_d1 = float(np.mean(alpha_d1_exp[bl_mask])) if np.any(bl_mask) else alpha_d1_exp[0]
    bl_d2 = float(np.mean(alpha_d2_exp[bl_mask])) if np.any(bl_mask) else alpha_d2_exp[0]
    delta_d1 = alpha_d1_exp - bl_d1
    delta_d2 = alpha_d2_exp - bl_d2

    # ================================================================
    # Figure 1: Full view — DA waveform + Alpha dynamics
    # ================================================================
    fig = plt.figure(figsize=(28, 20))
    gs = GridSpec(4, 1, figure=fig, hspace=0.3, height_ratios=[1.2, 2, 2, 1.5])
    fig.suptitle(
        f"Experiment G: Cue-Evoked Phasic DA — Receptor Dynamics\n"
        f"{args.n_trials} trials, intra-delay={args.intra_delay_min}–{args.intra_delay_max}s, "
        f"ITI={args.iti_mean}±{args.iti_std}s [{args.iti_min}–{args.iti_max}s]  |  "
        f"τ_rise={args.tau_rise}s, DA amp={args.da_amp}nM",
        fontsize=14, fontweight='bold'
    )

    # Panel A: DA waveform with trial markers
    ax = fig.add_subplot(gs[0])
    ax.plot(da_t, da_schedule[:, 1], color='#E91E63', linewidth=1.5, label='DA (Exp)')
    ax.fill_between(da_t, args.da_base, da_schedule[:, 1], alpha=0.15, color='#E91E63')
    ax.plot(da_t, da_schedule[:, 0], '--', color='gray', linewidth=1, alpha=0.4,
            label='DA (Ctrl)')
    # Mark trial boundaries
    for ti in trial_info:
        ax.axvline(ti['cue_onset_s'], color='#2196F3', linestyle=':', linewidth=0.6, alpha=0.4)
        if ti['stim_onset_s'] is not None:
            ax.axvline(ti['stim_onset_s'], color='#FF9800', linestyle=':', linewidth=0.6, alpha=0.4)
    mode_label = args.mode.replace('-', ' ').title()
    spikes_per_trial = 1 if args.mode == 'single' else 2
    ax.set_ylabel('[DA] (nM)', fontsize=13)
    ax.set_title(f'A. DA Concentration — {mode_label} Mode ({spikes_per_trial} spike/trial)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, total_s)
    ax.grid(True, alpha=0.2)

    # Panel B: Alpha D1 and D2 (absolute values)
    ax = fig.add_subplot(gs[1])
    ax.plot(alpha_times, alpha_d1_exp, '-', color='#d62728', linewidth=2,
            label='α_D1 (Exp) — SLOW integrator')
    ax.plot(alpha_times, alpha_d2_exp, '-', color='#1f77b4', linewidth=2,
            label='α_D2 (Exp) — FAST tracker')
    ax.plot(alpha_times, alpha_d1_ctrl, '--', color='#d62728', linewidth=1, alpha=0.3)
    ax.plot(alpha_times, alpha_d2_ctrl, '--', color='#1f77b4', linewidth=1, alpha=0.3)
    # Shade trial session
    last_stim = trial_info[-1]['stim_onset_s'] or trial_info[-1]['cue_onset_s']
    ax.axvspan(first_cue, last_stim + 8 * args.tau_rise, alpha=0.04, color='green',
               label='Trial session')
    ax.set_ylabel('Receptor Activation (α)', fontsize=13)
    ax.set_title('B. D1/D2 Receptor Activation — Full Session',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, total_s)
    ax.grid(True, alpha=0.2)

    # Panel C: Δα × 1000 (deviation from baseline)
    ax = fig.add_subplot(gs[2])
    ax.plot(alpha_times, delta_d1 * 1000, '-', color='#d62728', linewidth=2,
            label='Δα_D1 × 1000 (cumulative ramp)')
    ax.plot(alpha_times, delta_d2 * 1000, '-', color='#1f77b4', linewidth=2,
            label='Δα_D2 × 1000 (phasic tracking)')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    for ti in trial_info:
        ax.axvline(ti['cue_onset_s'], color='#2196F3', linestyle=':', linewidth=0.5, alpha=0.3)
        if ti['stim_onset_s'] is not None:
            ax.axvline(ti['stim_onset_s'], color='#FF9800', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.set_ylabel('Δα × 1000', fontsize=13)
    ax.set_title('C. Deviation from Baseline — D1 Integrates Across Trials, D2 Tracks Events',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(0, total_s)
    ax.grid(True, alpha=0.2)

    # Panel D: Zoomed view of 2-3 consecutive trials in the middle
    ax = fig.add_subplot(gs[3])
    mid_idx = len(trial_info) // 2
    zoom_start = trial_info[max(0, mid_idx - 1)]['cue_onset_s'] - 2
    zoom_end_trial = min(len(trial_info) - 1, mid_idx + 1)
    zoom_end_event = trial_info[zoom_end_trial]['stim_onset_s'] or trial_info[zoom_end_trial]['cue_onset_s']
    zoom_end = zoom_end_event + 8

    zoom_mask = (alpha_times >= zoom_start) & (alpha_times <= zoom_end)
    ax.plot(alpha_times[zoom_mask], delta_d2[zoom_mask] * 1000, '-', color='#1f77b4',
            linewidth=2.5, label='Δα_D2 × 1000')
    ax.plot(alpha_times[zoom_mask], delta_d1[zoom_mask] * 1000, '-', color='#d62728',
            linewidth=2.5, label='Δα_D1 × 1000')

    # DA on twin axis
    ax_twin = ax.twinx()
    da_zoom_mask = (da_t >= zoom_start) & (da_t <= zoom_end)
    ax_twin.fill_between(da_t[da_zoom_mask], args.da_base, da_schedule[da_zoom_mask, 1],
                         alpha=0.12, color='#E91E63')
    ax_twin.set_ylabel('[DA] (nM)', fontsize=10, color='#E91E63', alpha=0.6)

    # Mark cue (blue) and stim (orange) within zoom
    first_in_zoom = True
    for ti in trial_info:
        if zoom_start <= ti['cue_onset_s'] <= zoom_end:
            spike_label = 'Spike' if args.mode == 'single' else 'Cue'
            ax.axvline(ti['cue_onset_s'], color='#2196F3', linestyle='--', linewidth=1.5,
                       alpha=0.6, label=spike_label if first_in_zoom else '')
            first_in_zoom = False
        if ti['stim_onset_s'] is not None and zoom_start <= ti['stim_onset_s'] <= zoom_end:
            ax.axvline(ti['stim_onset_s'], color='#FF9800', linestyle='--', linewidth=1.5,
                       alpha=0.6, label='Stim' if first_in_zoom or ti == trial_info[max(0, mid_idx-1)] else '')

    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Δα × 1000', fontsize=13)
    ax.set_title(f'D. Zoomed: Trials {max(1,mid_idx)}–{zoom_end_trial+1} — '
                 f'D2 Tracks Each Spike, D1 Drifts Slowly',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.2)

    plt.savefig(save_dir / "receptor_dynamics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: receptor_dynamics.png")

    # ================================================================
    # Figure 2: Per-trial cumulative analysis
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(26, 9))
    fig.suptitle(
        f"Trial-by-Trial Analysis — {args.n_trials} Trials\n"
        f"D1: temporal integrator (τ_on≈{config.TAU_ON_D1/1000:.0f}s)  |  "
        f"D2: event tracker (τ_on≈{config.TAU_ON_D2/1000:.0f}s)",
        fontsize=14, fontweight='bold'
    )

    # Panel A: Pre-cue baseline Δα across trials (cumulative buildup)
    d1_pre_cue = []
    d2_pre_cue = []
    for ti in trial_info:
        pre_mask = (alpha_times >= ti['cue_onset_s'] - 1.0) & (alpha_times < ti['cue_onset_s'])
        if np.any(pre_mask):
            d1_pre_cue.append(float(np.mean(delta_d1[pre_mask])) * 1000)
            d2_pre_cue.append(float(np.mean(delta_d2[pre_mask])) * 1000)
        else:
            d1_pre_cue.append(0.0)
            d2_pre_cue.append(0.0)

    ax = axes[0]
    trial_nums = np.arange(1, args.n_trials + 1)
    ax.plot(trial_nums, d1_pre_cue, 'o-', color='#d62728', linewidth=2.5, markersize=7,
            label='D1 (pre-cue baseline)')
    ax.plot(trial_nums, d2_pre_cue, 's-', color='#1f77b4', linewidth=2.5, markersize=7,
            label='D2 (pre-cue baseline)')
    ax.set_xlabel('Trial Number', fontsize=13)
    ax.set_ylabel('Δα × 1000 (pre-cue)', fontsize=13)
    ax.set_title('A. Cumulative Buildup Across Trials\nD1 ramps up, D2 saturates',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel B: D2 phasic modulation depth per trial (cue spike response)
    d2_cue_peaks = []
    for ti in trial_info:
        peak_mask = (alpha_times >= ti['cue_onset_s']) & \
                    (alpha_times <= ti['cue_onset_s'] + 3 * args.tau_rise)
        if np.any(peak_mask):
            d2_cue_peaks.append(float(np.max(delta_d2[peak_mask])) * 1000)
        else:
            d2_cue_peaks.append(0.0)

    d2_cue_mod = [p - b for p, b in zip(d2_cue_peaks, d2_pre_cue)]

    ax = axes[1]
    ax.bar(trial_nums, d2_cue_mod, color='#1f77b4', alpha=0.8, width=0.6)
    ax.set_xlabel('Trial Number', fontsize=13)
    ax.set_ylabel('D2 Cue Response (Δα × 1000)', fontsize=13)
    ax.set_title('B. D2 Phasic Response to Cue per Trial\n(peak − pre-cue baseline)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Post-session decay comparison
    ax = axes[2]
    last_event = trial_info[-1]['stim_onset_s'] or trial_info[-1]['cue_onset_s']
    session_end = last_event + 8 * args.tau_rise
    post_mask = alpha_times >= session_end
    if np.any(post_mask):
        post_times = alpha_times[post_mask] - session_end
        ax.plot(post_times, delta_d1[post_mask] * 1000, '-', color='#d62728',
                linewidth=2.5, label='D1 (slow decay → "memory")')
        ax.plot(post_times, delta_d2[post_mask] * 1000, '-', color='#1f77b4',
                linewidth=2.5, label='D2 (fast decay → reset)')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    ax.set_xlabel('Time After Session End (s)', fontsize=13)
    ax.set_ylabel('Δα × 1000', fontsize=13)
    ax.set_title('C. Post-Session Decay\nD1 retains "memory", D2 returns quickly',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "trial_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: trial_analysis.png")

    return {
        'd1_pre_cue': d1_pre_cue,
        'd2_pre_cue': d2_pre_cue,
        'd2_cue_mod': d2_cue_mod,
    }


def plot_amplitude_analysis(data: dict, trial_info: list, args, save_dir: Path):
    """
    Plot amplitude-scaling analysis for 'amplitude' mode.

    Shows how D1/D2 receptors respond differently to varying DA spike magnitudes (M levels).
    Key insight: D2 (fast) should track amplitude differences within each level,
    while D1 (slow) integrates across levels showing cumulative effects.

    Panels:
      A. DA waveform with M-level color coding
      B. D1/D2 peak response vs amplitude level M (dose-response)
      C. D1/D2 Δα time course with M-level shading
      D. D2/D1 response ratio vs M (sensitivity comparison)
    """
    alpha_d1_exp = data['alpha_d1_trace'][:, 1]
    alpha_d2_exp = data['alpha_d2_trace'][:, 1]
    total_s = data['total_s']
    da_schedule = data['da_schedule']

    alpha_times = np.linspace(0, total_s, len(alpha_d1_exp))
    da_t = np.linspace(0, total_s, da_schedule.shape[0])

    # Baseline values (before first spike)
    first_cue = trial_info[0]['cue_onset_s']
    bl_mask = alpha_times < first_cue
    bl_d1 = float(np.mean(alpha_d1_exp[bl_mask])) if np.any(bl_mask) else alpha_d1_exp[0]
    bl_d2 = float(np.mean(alpha_d2_exp[bl_mask])) if np.any(bl_mask) else alpha_d2_exp[0]
    delta_d1 = alpha_d1_exp - bl_d1
    delta_d2 = alpha_d2_exp - bl_d2

    # Group trials by amplitude level
    amp_levels = sorted(set(ti['amp_multiplier'] for ti in trial_info))
    level_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(amp_levels)))

    # Compute per-trial peak responses
    trial_d1_peaks = []
    trial_d2_peaks = []
    trial_d1_pre = []
    trial_d2_pre = []
    for ti in trial_info:
        onset = ti['cue_onset_s']
        # Pre-spike baseline (1s window before spike)
        pre_mask = (alpha_times >= onset - 1.0) & (alpha_times < onset)
        if np.any(pre_mask):
            pre_d1 = float(np.mean(delta_d1[pre_mask])) * 1000
            pre_d2 = float(np.mean(delta_d2[pre_mask])) * 1000
        else:
            pre_d1, pre_d2 = 0.0, 0.0
        trial_d1_pre.append(pre_d1)
        trial_d2_pre.append(pre_d2)

        # Peak response (within 3*tau_rise after spike)
        peak_mask = (alpha_times >= onset) & (alpha_times <= onset + 3 * args.tau_rise)
        if np.any(peak_mask):
            trial_d1_peaks.append(float(np.max(delta_d1[peak_mask])) * 1000)
            trial_d2_peaks.append(float(np.max(delta_d2[peak_mask])) * 1000)
        else:
            trial_d1_peaks.append(0.0)
            trial_d2_peaks.append(0.0)

    # Compute phasic response (peak - pre-spike baseline)
    trial_d1_phasic = [p - b for p, b in zip(trial_d1_peaks, trial_d1_pre)]
    trial_d2_phasic = [p - b for p, b in zip(trial_d2_peaks, trial_d2_pre)]

    # Group by M level
    level_d1_phasic = {}
    level_d2_phasic = {}
    level_d1_peak = {}
    level_d2_peak = {}
    for ti, d1p, d2p, d1pk, d2pk in zip(trial_info, trial_d1_phasic, trial_d2_phasic,
                                          trial_d1_peaks, trial_d2_peaks):
        m = ti['amp_multiplier']
        level_d1_phasic.setdefault(m, []).append(d1p)
        level_d2_phasic.setdefault(m, []).append(d2p)
        level_d1_peak.setdefault(m, []).append(d1pk)
        level_d2_peak.setdefault(m, []).append(d2pk)

    # ================================================================
    # Figure: Amplitude-Scaling Analysis (4 panels)
    # ================================================================
    fig = plt.figure(figsize=(28, 22))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[1.2, 2, 2, 1.5])
    fig.suptitle(
        f"Experiment G: Amplitude-Scaling Analysis — DA_transient = k × M × α(t)\n"
        f"M levels: {amp_levels}  |  {args.amp_repeats} repeats/level  |  "
        f"τ_rise={args.tau_rise}s, base DA amp={args.da_amp}nM",
        fontsize=14, fontweight='bold'
    )

    # Panel A: DA waveform with M-level color coding (full width)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(da_t, da_schedule[:, 0], '--', color='gray', linewidth=1, alpha=0.4,
            label='DA (Ctrl)')
    ax.plot(da_t, da_schedule[:, 1], color='#E91E63', linewidth=1.2, alpha=0.5,
            label='DA (Exp)')
    # Color-code trial regions by M level
    for ti in trial_info:
        m = ti['amp_multiplier']
        cidx = amp_levels.index(m)
        onset = ti['cue_onset_s']
        end = onset + 5 * args.tau_rise
        ax.axvspan(onset, end, alpha=0.08, color=level_colors[cidx])
        ax.axvline(onset, color=level_colors[cidx], linestyle=':', linewidth=0.8, alpha=0.5)
    # Add M-level legend patches
    legend_patches = [Patch(facecolor=level_colors[i], alpha=0.3,
                            label=f'M={m:.0f} (peak={args.da_base + args.da_amp*m:.0f}nM)')
                      for i, m in enumerate(amp_levels)]
    ax.legend(handles=legend_patches, fontsize=10, loc='upper right', ncol=len(amp_levels))
    ax.set_ylabel('[DA] (nM)', fontsize=13)
    ax.set_title('A. DA Concentration — Amplitude-Scaled Alpha Spikes',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, total_s)
    ax.grid(True, alpha=0.2)

    # Panel B: Dose-response curve — D1/D2 phasic response vs M (left)
    ax = fig.add_subplot(gs[1, 0])
    m_vals = np.array(amp_levels)
    d1_means = [np.mean(level_d1_phasic[m]) for m in amp_levels]
    d1_stds = [np.std(level_d1_phasic[m]) for m in amp_levels]
    d2_means = [np.mean(level_d2_phasic[m]) for m in amp_levels]
    d2_stds = [np.std(level_d2_phasic[m]) for m in amp_levels]

    ax.errorbar(m_vals, d1_means, yerr=d1_stds, fmt='o-', color='#d62728',
                linewidth=2.5, markersize=10, capsize=5, capthick=2,
                label='D1 phasic response')
    ax.errorbar(m_vals, d2_means, yerr=d2_stds, fmt='s-', color='#1f77b4',
                linewidth=2.5, markersize=10, capsize=5, capthick=2,
                label='D2 phasic response')
    ax.set_xlabel('Amplitude Multiplier M', fontsize=13)
    ax.set_ylabel('Phasic Response (Δα × 1000)', fontsize=13)
    ax.set_title('B. Dose-Response: Phasic Response vs M\n'
                 '(peak − pre-spike baseline per trial)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(m_vals)

    # Panel C: Absolute peak response vs M (right)
    ax = fig.add_subplot(gs[1, 1])
    d1_pk_means = [np.mean(level_d1_peak[m]) for m in amp_levels]
    d1_pk_stds = [np.std(level_d1_peak[m]) for m in amp_levels]
    d2_pk_means = [np.mean(level_d2_peak[m]) for m in amp_levels]
    d2_pk_stds = [np.std(level_d2_peak[m]) for m in amp_levels]

    ax.errorbar(m_vals, d1_pk_means, yerr=d1_pk_stds, fmt='o-', color='#d62728',
                linewidth=2.5, markersize=10, capsize=5, capthick=2,
                label='D1 absolute peak')
    ax.errorbar(m_vals, d2_pk_means, yerr=d2_pk_stds, fmt='s-', color='#1f77b4',
                linewidth=2.5, markersize=10, capsize=5, capthick=2,
                label='D2 absolute peak')
    ax.set_xlabel('Amplitude Multiplier M', fontsize=13)
    ax.set_ylabel('Peak Δα × 1000', fontsize=13)
    ax.set_title('C. Absolute Peak Response vs M\n'
                 '(includes cumulative buildup)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(m_vals)

    # Panel D: D1/D2 Δα time course with M-level shading (full width)
    ax = fig.add_subplot(gs[2, :])
    ax.plot(alpha_times, delta_d1 * 1000, '-', color='#d62728', linewidth=2,
            label='Δα_D1 × 1000 (slow integrator)')
    ax.plot(alpha_times, delta_d2 * 1000, '-', color='#1f77b4', linewidth=2,
            label='Δα_D2 × 1000 (fast tracker)')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    # Shade M-level regions
    for ti in trial_info:
        m = ti['amp_multiplier']
        cidx = amp_levels.index(m)
        onset = ti['cue_onset_s']
        end = onset + 5 * args.tau_rise
        ax.axvspan(onset, end, alpha=0.06, color=level_colors[cidx])
    # Add M-level annotations at top
    prev_m = None
    for ti in trial_info:
        m = ti['amp_multiplier']
        if m != prev_m:
            ax.annotate(f'M={m:.0f}', xy=(ti['cue_onset_s'], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0),
                        fontsize=10, fontweight='bold', color=level_colors[amp_levels.index(m)],
                        ha='left', va='bottom')
            prev_m = m
    ax.set_ylabel('Δα × 1000', fontsize=13)
    ax.set_title('D. Receptor Activation Time Course — D2 Tracks Amplitude, D1 Integrates',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(0, total_s)
    ax.grid(True, alpha=0.2)

    # Panel E: D2/D1 sensitivity ratio vs M (left) + bar chart per trial (right)
    ax = fig.add_subplot(gs[3, 0])
    # Ratio of phasic responses
    d2_d1_ratio = []
    for m in amp_levels:
        d1_m = np.mean(level_d1_phasic[m])
        d2_m = np.mean(level_d2_phasic[m])
        ratio = d2_m / d1_m if abs(d1_m) > 1e-6 else 0.0
        d2_d1_ratio.append(ratio)
    ax.bar(m_vals, d2_d1_ratio, color=[level_colors[i] for i in range(len(amp_levels))],
           alpha=0.8, width=0.4, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Amplitude Multiplier M', fontsize=13)
    ax.set_ylabel('D2/D1 Phasic Response Ratio', fontsize=13)
    ax.set_title('E. D2/D1 Sensitivity Ratio\n'
                 'Higher = D2 more responsive to this amplitude',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(m_vals)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel F: Per-trial phasic response bar chart
    ax = fig.add_subplot(gs[3, 1])
    trial_nums = np.arange(1, len(trial_info) + 1)
    bar_colors = [level_colors[amp_levels.index(ti['amp_multiplier'])] for ti in trial_info]
    width = 0.35
    ax.bar(trial_nums - width/2, trial_d2_phasic, width, color='#1f77b4', alpha=0.7,
           label='D2 phasic')
    ax.bar(trial_nums + width/2, trial_d1_phasic, width, color='#d62728', alpha=0.7,
           label='D1 phasic')
    # Add M-level background shading
    prev_m = None
    for i, ti in enumerate(trial_info):
        m = ti['amp_multiplier']
        if m != prev_m:
            # Find end of this M block
            end_idx = i
            while end_idx < len(trial_info) and trial_info[end_idx]['amp_multiplier'] == m:
                end_idx += 1
            cidx = amp_levels.index(m)
            ax.axvspan(i + 0.5, end_idx + 0.5, alpha=0.08, color=level_colors[cidx])
            ax.text((i + end_idx) / 2 + 0.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0,
                    f'M={m:.0f}', ha='center', fontsize=9, fontweight='bold',
                    color=level_colors[cidx])
            prev_m = m
    ax.set_xlabel('Trial Number', fontsize=13)
    ax.set_ylabel('Phasic Response (Δα × 1000)', fontsize=13)
    ax.set_title('F. Per-Trial Phasic Response\n'
                 'D2 scales with M, D1 accumulates',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.savefig(save_dir / "amplitude_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved: amplitude_analysis.png")

    return {
        'amp_levels': amp_levels,
        'd1_phasic_by_level': {m: level_d1_phasic[m] for m in amp_levels},
        'd2_phasic_by_level': {m: level_d2_phasic[m] for m in amp_levels},
        'd1_phasic_mean': {m: float(np.mean(level_d1_phasic[m])) for m in amp_levels},
        'd2_phasic_mean': {m: float(np.mean(level_d2_phasic[m])) for m in amp_levels},
        'd2_d1_ratio': {m: r for m, r in zip(amp_levels, d2_d1_ratio)},
    }


def print_trial_schedule(trial_info: list):
    """Print the trial timing schedule."""
    has_stim = any(ti['stim_onset_s'] is not None for ti in trial_info)
    w = 90
    print("\n" + "=" * w)
    print("  📋 Trial Schedule")
    print("=" * w)
    has_amp = any(ti.get('amp_multiplier', 1.0) != 1.0 for ti in trial_info)
    if has_stim:
        hdr = f"  {'Trial':<8} {'Cue (s)':>10} {'Stim (s)':>10} {'Intra-delay':>14} {'ITI (s)':>10}"
        if has_amp:
            hdr += f" {'M':>6}"
        print(hdr)
        sep = f"  {'─'*8} {'─'*10} {'─'*10} {'─'*14} {'─'*10}"
        if has_amp:
            sep += f" {'─'*6}"
        print(sep)
        for ti in trial_info:
            iti_str = f"{ti['iti_s']:.1f}" if ti['iti_s'] is not None else "—"
            stim_str = f"{ti['stim_onset_s']:>10.1f}" if ti['stim_onset_s'] is not None else f"{'—':>10}"
            line = (f"  {ti['trial_num']:<8} {ti['cue_onset_s']:>10.1f} {stim_str} "
                    f"{ti['intra_delay_s']:>14.2f} {iti_str:>10}")
            if has_amp:
                line += f" {ti.get('amp_multiplier', 1.0):>6.1f}"
            print(line)
    else:
        hdr = f"  {'Trial':<8} {'Spike (s)':>10} {'ITI (s)':>10}"
        if has_amp:
            hdr += f" {'M':>6} {'Peak DA':>10}"
        print(hdr)
        sep = f"  {'─'*8} {'─'*10} {'─'*10}"
        if has_amp:
            sep += f" {'─'*6} {'─'*10}"
        print(sep)
        for ti in trial_info:
            iti_str = f"{ti['iti_s']:.1f}" if ti['iti_s'] is not None else "—"
            line = f"  {ti['trial_num']:<8} {ti['cue_onset_s']:>10.1f} {iti_str:>10}"
            if has_amp:
                m = ti.get('amp_multiplier', 1.0)
                # Peak DA = da_base + da_amp * M (alpha function peaks at 1.0)
                # We don't have args here, so just show M
                line += f" {m:>6.1f}"
            print(line)
    print("=" * w)


def print_summary(data: dict, trial_info: list, trial_analysis: dict, args):
    """Print comprehensive experiment summary."""
    alpha_d1_exp = data['alpha_d1_trace'][:, 1]
    alpha_d2_exp = data['alpha_d2_trace'][:, 1]
    total_s = data['total_s']
    alpha_times = np.linspace(0, total_s, len(alpha_d1_exp))

    first_cue = trial_info[0]['cue_onset_s']
    bl_mask = alpha_times < first_cue
    bl_d1 = float(np.mean(alpha_d1_exp[bl_mask])) if np.any(bl_mask) else alpha_d1_exp[0]
    bl_d2 = float(np.mean(alpha_d2_exp[bl_mask])) if np.any(bl_mask) else alpha_d2_exp[0]

    last_event = trial_info[-1]['stim_onset_s'] or trial_info[-1]['cue_onset_s']
    session_end = last_event + 8 * args.tau_rise

    w = 85
    print("\n" + "=" * w)
    print("  📊 Experiment G: Summary")
    print("=" * w)
    print(f"  Trials: {args.n_trials}")
    print(f"  Intra-trial delay: {args.intra_delay_min}–{args.intra_delay_max}s")
    print(f"  ITI: {args.iti_mean}±{args.iti_std}s [{args.iti_min}–{args.iti_max}s]")
    print(f"  Alpha function: τ_rise={args.tau_rise}s, DA amp={args.da_amp}nM")
    spikes_per_trial = 1 if args.mode in ('single', 'amplitude') else 2
    total_spikes = spikes_per_trial * args.n_trials
    print(f"  Mode: {args.mode}")
    print(f"  Total spikes: {total_spikes} ({spikes_per_trial} per trial × {args.n_trials} trials)")
    if args.mode == 'amplitude':
        amp_levels = sorted(args.amp_levels)
        print(f"  Amplitude levels M: {amp_levels}")
        print(f"  Repeats per level: {args.amp_repeats}")
        for m in amp_levels:
            peak = args.da_base + args.da_amp * m
            print(f"    M={m:.1f}: peak DA = {args.da_base} + {args.da_amp}×{m:.1f} = {peak:.1f} nM")
    print(f"  Session duration: {session_end - first_cue:.1f}s")
    print(f"  Total sim duration: {total_s:.0f}s")

    print(f"\n  Cumulative buildup (Δα × 1000, pre-cue baseline):")
    print(f"  {'Trial':<8} {'D1':>10} {'D2':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10}")
    for i, (d1v, d2v) in enumerate(zip(trial_analysis['d1_pre_cue'],
                                        trial_analysis['d2_pre_cue'])):
        print(f"  {i+1:<8} {d1v:>10.3f} {d2v:>10.3f}")

    # Post-session retention
    delta_d1 = alpha_d1_exp - bl_d1
    delta_d2 = alpha_d2_exp - bl_d2
    for dt_s in [5, 20, 40]:
        post_mask = (alpha_times >= session_end + dt_s) & (alpha_times < session_end + dt_s + 1)
        if np.any(post_mask):
            d1_v = float(np.mean(delta_d1[post_mask])) * 1000
            d2_v = float(np.mean(delta_d2[post_mask])) * 1000
            if dt_s == 5:
                print(f"\n  Post-session retention (Δα × 1000):")
            print(f"    +{dt_s}s:  D1={d1_v:.3f}, D2={d2_v:.3f}")

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
    save_dir = PROJECT_ROOT / "outputs" / f"exp_g_alpha_spike_{args.mode}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint
    ckpt_path = find_or_create_checkpoint(args)

    # ================================================================
    # Generate trial schedule
    # ================================================================
    trial_info, all_spike_onsets_s = generate_trial_schedule(args)
    print_trial_schedule(trial_info)

    # Build DA schedule
    da_schedule, total_s = build_trial_da_schedule(args, trial_info, all_spike_onsets_s, config.DT)

    # Save experiment config
    exp_config = {
        'da_base': args.da_base,
        'da_amplitude': args.da_amp,
        'tau_rise_s': args.tau_rise,
        'n_trials': args.n_trials,
        'intra_delay_range_s': [args.intra_delay_min, args.intra_delay_max],
        'iti_mean_s': args.iti_mean,
        'iti_std_s': args.iti_std,
        'iti_range_s': [args.iti_min, args.iti_max],
        'pre_stim_s': args.pre_stim,
        'post_stim_s': args.post_stim,
        'total_duration_s': total_s,
        'total_da_spikes': len(all_spike_onsets_s),
        'checkpoint': ckpt_path,
        'device': str(device),
        'seed': args.seed,
        'D1_tau_on_ms': config.TAU_ON_D1,
        'D1_tau_off_ms': config.TAU_OFF_D1,
        'D2_tau_on_ms': config.TAU_ON_D2,
        'D2_tau_off_ms': config.TAU_OFF_D2,
        'D1_EC50': config.EC50_D1,
        'D2_EC50': config.EC50_D2,
        'alpha_function': f'f(t) = (t/τ) * exp(1 - t/τ), τ={args.tau_rise}s',
        'spike_peak_da': args.da_base + args.da_amp,
        'trial_schedule': trial_info,
    }
    with open(save_dir / "experiment_config.json", 'w') as f:
        json.dump(exp_config, f, indent=2, default=str)

    # Print experiment header
    print(f"\n{'='*70}")
    print(f"  🧪 Experiment G: Alpha-Function DA Spike — Cue-Evoked Phasic Release")
    print(f"{'='*70}")
    print(f"  Alpha function: f(t) = (t/τ) × exp(1 − t/τ), τ={args.tau_rise}s")
    print(f"  DA: baseline={args.da_base}nM, spike peak={args.da_base + args.da_amp}nM")
    spikes_per_trial = 1 if args.mode in ('single', 'amplitude') else 2
    print(f"  Mode: {args.mode}")
    print(f"  Trial structure: {args.n_trials} trials × {spikes_per_trial} spikes/trial = "
          f"{spikes_per_trial * args.n_trials} total DA spikes")
    if args.mode == 'single':
        print(f"  Single spike per trial (no cue/stim distinction)")
    elif args.mode == 'amplitude':
        amp_levels = sorted(args.amp_levels)
        print(f"  Amplitude-scaling mode: M = {amp_levels}, {args.amp_repeats} repeats each")
        for m in amp_levels:
            peak = args.da_base + args.da_amp * m
            print(f"    M={m:.1f}: DA_transient = {args.da_amp}×{m:.1f} = {args.da_amp*m:.1f} nM → peak = {peak:.1f} nM")
    elif args.mode == 'paired-wide':
        print(f"  Intra-trial delay: 10.0–15.0s (wide separation)")
    else:
        print(f"  Intra-trial delay: {args.intra_delay_min}–{args.intra_delay_max}s "
              f"(cue → stim)")
    print(f"  Inter-trial interval: {args.iti_mean}±{args.iti_std}s "
          f"[{args.iti_min}–{args.iti_max}s]")
    print(f"  D1: τ_on={config.TAU_ON_D1:.0f}ms ({config.TAU_ON_D1/1000:.0f}s), "
          f"τ_off={config.TAU_OFF_D1:.0f}ms ({config.TAU_OFF_D1/1000:.0f}s)")
    print(f"  D2: τ_on={config.TAU_ON_D2:.0f}ms ({config.TAU_ON_D2/1000:.0f}s), "
          f"τ_off={config.TAU_OFF_D2:.0f}ms ({config.TAU_OFF_D2/1000:.0f}s)")
    print(f"  Total duration: {total_s:.0f}s")
    print(f"  Output: {save_dir}")
    print(f"{'='*70}")

    # ================================================================
    # Run simulation
    # ================================================================
    data = run_trial_sim(ckpt_path, device, da_schedule, total_s)

    # ================================================================
    # Standard plots (same as main.py)
    # ================================================================
    print("\n🎨 Generating standard plots (same as main.py)...")
    analyzer = PFCAnalyzer(data)

    plot_combined_raster(analyzer, save_dir=save_dir)
    plot_combined_rates_all(analyzer, save_dir=save_dir)
    plot_combined_rates_E(analyzer, save_dir=save_dir)
    plot_combined_rates_I(analyzer, save_dir=save_dir)

    # Analysis report
    analyzer.save_report(os.path.join(save_dir, "analysis_report.txt"))

    # ================================================================
    # Supplementary plots: Receptor dynamics & trial analysis
    # ================================================================
    print("\n🎨 Generating supplementary receptor dynamics plots...")
    trial_analysis = plot_receptor_dynamics(data, trial_info, all_spike_onsets_s, args, save_dir)

    # Amplitude-scaling analysis (only for amplitude mode)
    amp_analysis = None
    if args.mode == 'amplitude':
        print("\n🎨 Generating amplitude-scaling analysis plots...")
        amp_analysis = plot_amplitude_analysis(data, trial_info, args, save_dir)

    # ================================================================
    # Save results
    # ================================================================
    summary = {
        'tau_rise_s': args.tau_rise,
        'da_base': args.da_base,
        'da_amplitude': args.da_amp,
        'n_trials': args.n_trials,
        'total_da_spikes': len(all_spike_onsets_s),
        'mode': args.mode,
        'trial_analysis': {
            'd1_pre_cue_buildup': trial_analysis['d1_pre_cue'],
            'd2_pre_cue_buildup': trial_analysis['d2_pre_cue'],
            'd2_cue_modulation': trial_analysis['d2_cue_mod'],
        },
    }
    if amp_analysis is not None:
        summary['amplitude_analysis'] = {
            'amp_levels': amp_analysis['amp_levels'],
            'd1_phasic_mean_by_M': amp_analysis['d1_phasic_mean'],
            'd2_phasic_mean_by_M': amp_analysis['d2_phasic_mean'],
            'd2_d1_ratio_by_M': amp_analysis['d2_d1_ratio'],
        }
    with open(save_dir / "exp_g_results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n📄 Results saved: {save_dir / 'exp_g_results.json'}")

    # Save alpha traces
    np.savez(save_dir / "alpha_traces.npz",
             times=np.linspace(0, data['total_s'],
                               len(data['alpha_d1_trace'][:, 0])),
             alpha_d1=data['alpha_d1_trace'],
             alpha_d2=data['alpha_d2_trace'])

    # Print summary
    print_trial_schedule(trial_info)
    print_summary(data, trial_info, trial_analysis, args)

    # Total time
    t_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"  ⏱️  Total time: {_fmt_elapsed(t_total)}")
    print(f"  📁 Results in: {save_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
