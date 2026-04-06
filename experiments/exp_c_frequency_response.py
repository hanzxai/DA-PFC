#!/usr/bin/env python3
"""
Experiment C: Frequency-Dependent DA Filtering

Scientific Question:
  Does the PFC network respond differently to DA fluctuations at different
  frequencies? Do D1 and D2 receptors act as low-pass filters with different
  cutoff frequencies?

Key Predictions:
  1. Ultra-slow DA oscillations (f << 1/τ_D1): Both D1 and D2 can follow
     → network response amplitude is maximal → same as static model
  2. Intermediate frequencies (1/τ_D1 < f < 1/τ_D2): D2 can follow but D1
     cannot → D1/D2 decouple → unique network state
  3. Fast DA oscillations (f >> 1/τ_D2): Neither can follow → network
     barely responds → DA fluctuations are filtered out

  Theoretical cutoff frequencies:
    D1: f_c ≈ 1/(2π·τ_eff_D1) ≈ 1/(2π·98s) ≈ 0.0016 Hz
    D2: f_c ≈ 1/(2π·τ_eff_D2) ≈ 1/(2π·49s) ≈ 0.0033 Hz

Protocol:
  For each frequency f:
    DA(t) = DA_base + amplitude * sin(2π * f * t)
    Run for at least 3 complete cycles
    Measure the amplitude of α_D1, α_D2, and firing rate oscillations

Usage:
  python -m experiments.exp_c_frequency_response
  python -m experiments.exp_c_frequency_response --frequencies 0.001 0.005 0.01 0.05
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from models.network import create_network_structure
from models.kernels import run_dynamic_d1_d2_kernel_sine
from analysis.analyzer import PFCAnalyzer

CHECKPOINT_PATH = "checkpoints/ckpt_DA2nM_500s.pkl"
RATE_GROUPS = ['E-D1', 'E-D2', 'E-Other', 'All-E', 'I-D1', 'I-D2', 'I-Other', 'All-I']

# Default frequencies to sweep (Hz)
DEFAULT_FREQUENCIES = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment C: Frequency-Dependent DA Filtering")
    parser.add_argument("--da-base", type=float, default=5.0,
                        help="Baseline DA (nM), default 5.0 (near D1 EC50)")
    parser.add_argument("--da-amplitude", type=float, default=4.0,
                        help="Sine wave amplitude (nM), default 4.0")
    parser.add_argument("--min-cycles", type=int, default=3,
                        help="Minimum number of complete cycles per frequency, default 3")
    parser.add_argument("--frequencies", type=float, nargs='+', default=None,
                        help="Custom frequency list (Hz)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    return parser.parse_args()


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:05.2f}s"
    return f"{int(m)}m {s:05.2f}s" if m > 0 else f"{s:.2f}s"


def compute_oscillation_amplitude(trace, times, freq_hz, n_skip_cycles=1):
    """
    Compute the amplitude of oscillation in a trace at a given frequency.

    Skips the first n_skip_cycles to avoid transient effects.
    Uses peak-to-trough measurement in the steady-state region.

    Returns:
        amplitude: Half of peak-to-trough range
        phase_lag: Phase lag relative to input (radians)
    """
    period = 1.0 / freq_hz  # seconds
    skip_time = n_skip_cycles * period

    # Select steady-state region
    mask = times >= skip_time
    if np.sum(mask) < 10:
        return 0.0, 0.0

    ss_trace = trace[mask]
    ss_times = times[mask]

    # Amplitude: half of peak-to-trough
    amplitude = (np.max(ss_trace) - np.min(ss_trace)) / 2.0

    # Phase lag: cross-correlate with reference sine
    ref_sine = np.sin(2 * np.pi * freq_hz * ss_times)
    if len(ref_sine) > 0 and np.std(ss_trace) > 1e-6:
        # Normalize
        trace_norm = (ss_trace - np.mean(ss_trace)) / (np.std(ss_trace) + 1e-10)
        ref_norm = ref_sine / (np.std(ref_sine) + 1e-10)
        correlation = np.correlate(trace_norm, ref_norm, mode='full')
        lag_idx = np.argmax(correlation) - len(ref_norm) + 1
        dt_s = np.mean(np.diff(ss_times)) if len(ss_times) > 1 else 1.0
        phase_lag = 2 * np.pi * freq_hz * lag_idx * dt_s
    else:
        phase_lag = 0.0

    return amplitude, phase_lag


def run_single_frequency(freq_hz, args, device, W_t, mask_d1, mask_d2, groups_info, init_state):
    """Run simulation for a single frequency."""
    period_s = 1.0 / freq_hz
    duration_s = max(args.min_cycles * period_s + 50.0, 100.0)  # At least 100s
    duration_ms = duration_s * 1000.0

    target_d1 = 0
    target_d2 = groups_info['e_d1_end']
    record_indices = torch.tensor([
        [0, target_d1], [1, target_d1],
        [0, target_d2], [1, target_d2],
    ], device=device, dtype=torch.long)

    alpha_record_interval = 100  # Every 100ms

    spikes, v_traces, final_state, alpha_d1_trace, alpha_d2_trace, da_trace = \
        run_dynamic_d1_d2_kernel_sine(
            W_t, mask_d1, mask_d2, init_state,
            float(args.da_base), float(args.da_amplitude), float(freq_hz),
            float(duration_ms), float(config.DT),
            record_indices, config.N_E,
            alpha_record_interval,
            config.build_kernel_params(device),
        )

    data = {
        'config': {
            'N_E': config.N_E, 'N_I': config.N_I,
            'duration': duration_ms, 'dt': config.DT,
            'da_onset': 0.0, 'da_level': args.da_base,
        },
        'masks': {'d1': mask_d1.cpu(), 'd2': mask_d2.cpu()},
        'groups_info': groups_info,
        'spikes': spikes.cpu(),
        'v_traces': v_traces.cpu(),
        'record_indices': record_indices.cpu(),
    }

    return data, alpha_d1_trace.cpu().numpy(), alpha_d2_trace.cpu().numpy(), \
           da_trace.cpu().numpy(), duration_s


def plot_bode_diagram(freq_results, args, save_dir):
    """
    Generate Bode-like diagram showing amplitude and phase vs frequency.
    """
    freqs = sorted(freq_results.keys())
    freq_arr = np.array(freqs)

    # ================================================================
    # Figure 1: Bode diagram — Amplitude response
    # ================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"Experiment C: Frequency Response (Bode Diagram)\n"
        f"DA(t) = {args.da_base} + {args.da_amplitude}·sin(2πft) nM",
        fontsize=15, fontweight='bold'
    )

    # Theoretical cutoff frequencies
    TAU_EFF_D1 = 98.0  # seconds (approximate)
    TAU_EFF_D2 = 49.0  # seconds (approximate)
    fc_d1 = 1.0 / (2 * np.pi * TAU_EFF_D1)
    fc_d2 = 1.0 / (2 * np.pi * TAU_EFF_D2)

    # --- Panel A: Alpha amplitude ---
    ax = axes[0]
    d1_amps = [freq_results[f]['alpha_d1_amplitude'] for f in freqs]
    d2_amps = [freq_results[f]['alpha_d2_amplitude'] for f in freqs]

    # Normalize to DC gain (lowest frequency)
    d1_amps_norm = np.array(d1_amps) / (d1_amps[0] + 1e-10)
    d2_amps_norm = np.array(d2_amps) / (d2_amps[0] + 1e-10)

    ax.semilogx(freq_arr, d1_amps_norm, 'o-', color='#d62728', linewidth=2.5, markersize=8,
                label='α_D1 (slow receptor)')
    ax.semilogx(freq_arr, d2_amps_norm, 's-', color='#1f77b4', linewidth=2.5, markersize=8,
                label='α_D2 (fast receptor)')

    # Theoretical first-order low-pass filter curves
    f_theory = np.logspace(np.log10(freq_arr[0] * 0.5), np.log10(freq_arr[-1] * 2), 100)
    lp_d1 = 1.0 / np.sqrt(1 + (f_theory / fc_d1)**2)
    lp_d2 = 1.0 / np.sqrt(1 + (f_theory / fc_d2)**2)
    ax.semilogx(f_theory, lp_d1, '--', color='#d62728', linewidth=1.5, alpha=0.5,
                label=f'D1 theory (fc={fc_d1:.4f} Hz)')
    ax.semilogx(f_theory, lp_d2, '--', color='#1f77b4', linewidth=1.5, alpha=0.5,
                label=f'D2 theory (fc={fc_d2:.4f} Hz)')

    ax.axvline(fc_d1, color='#d62728', linestyle=':', alpha=0.4, linewidth=1.5)
    ax.axvline(fc_d2, color='#1f77b4', linestyle=':', alpha=0.4, linewidth=1.5)
    ax.axhline(0.707, color='gray', linestyle=':', alpha=0.3, linewidth=1)  # -3dB line
    ax.text(freq_arr[-1] * 0.5, 0.72, '-3dB', fontsize=9, color='gray')

    ax.set_ylabel('Normalized Amplitude', fontsize=12)
    ax.set_title('A. Receptor Activation Amplitude vs Frequency', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(-0.05, 1.15)

    # --- Panel B: Firing rate amplitude ---
    ax = axes[1]
    for grp, color, marker, label in [
        ('All-E', '#e377c2', 'o', 'All-E'),
        ('All-I', '#17becf', 's', 'All-I'),
        ('E-D1', '#d62728', '^', 'E-D1'),
        ('E-D2', '#1f77b4', 'v', 'E-D2'),
    ]:
        amps = [freq_results[f].get(f'rate_{grp}_amplitude', 0) for f in freqs]
        if amps[0] > 0:
            amps_norm = np.array(amps) / (amps[0] + 1e-10)
        else:
            amps_norm = np.array(amps)
        ax.semilogx(freq_arr, amps_norm, f'{marker}-', color=color, linewidth=2, markersize=7,
                     label=label)

    ax.axvline(fc_d1, color='#d62728', linestyle=':', alpha=0.4, linewidth=1.5,
               label=f'D1 fc={fc_d1:.4f} Hz')
    ax.axvline(fc_d2, color='#1f77b4', linestyle=':', alpha=0.4, linewidth=1.5,
               label=f'D2 fc={fc_d2:.4f} Hz')

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Normalized Rate Amplitude', fontsize=12)
    ax.set_title('B. Firing Rate Oscillation Amplitude vs Frequency', fontsize=13)
    ax.legend(fontsize=10, ncol=3)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bode_diagram.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: bode_diagram.png")

    # ================================================================
    # Figure 2: Example time traces for 3 representative frequencies
    # ================================================================
    # Pick low, mid, high frequencies
    n_freq = len(freqs)
    if n_freq >= 3:
        example_freqs = [freqs[0], freqs[n_freq // 2], freqs[-1]]
    else:
        example_freqs = freqs

    fig, axes = plt.subplots(len(example_freqs), 2, figsize=(20, 5 * len(example_freqs)),
                             sharex=False)
    if len(example_freqs) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Example Time Traces at Different Frequencies",
                 fontsize=15, fontweight='bold')

    for row, freq in enumerate(example_freqs):
        fr = freq_results[freq]
        times = fr['alpha_times']
        da = fr['da_trace']

        # Left: Alpha traces
        ax = axes[row, 0]
        ax.plot(times, fr['alpha_d1_exp'], '-', color='#d62728', linewidth=2, label='α_D1')
        ax.plot(times, fr['alpha_d2_exp'], '-', color='#1f77b4', linewidth=2, label='α_D2')

        # DA input (secondary axis)
        ax2 = ax.twinx()
        ax2.plot(times, da, '-', color='green', linewidth=1, alpha=0.4, label='DA(t)')
        ax2.set_ylabel('DA (nM)', color='green', fontsize=10)

        ax.set_ylabel('α', fontsize=11)
        ax.set_title(f'f = {freq:.4f} Hz (T = {1/freq:.0f}s)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

        # Right: Firing rates
        ax = axes[row, 1]
        rate_times = fr.get('rate_All-E_times', np.array([]))
        rate_exp = fr.get('rate_All-E_exp', np.array([]))
        rate_ctrl = fr.get('rate_All-E_ctrl', np.array([]))

        if len(rate_times) > 0:
            ax.plot(rate_times, rate_exp, '-', color='#e377c2', linewidth=2, label='All-E (Exp)')
            ax.plot(rate_times, rate_ctrl, ':', color='gray', linewidth=1, alpha=0.5,
                    label='All-E (Ctrl)')

        ax2 = ax.twinx()
        ax2.plot(times, da, '-', color='green', linewidth=1, alpha=0.4)
        ax2.set_ylabel('DA (nM)', color='green', fontsize=10)

        ax.set_ylabel('Rate (Hz)', fontsize=11)
        ax.set_title(f'Firing Rate at f = {freq:.4f} Hz', fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('Time (s)', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "example_traces.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: example_traces.png")


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

    frequencies = args.frequencies or DEFAULT_FREQUENCIES
    frequencies = sorted(frequencies)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("outputs", f"exp_c_freq_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  🧪 Experiment C: Frequency-Dependent DA Filtering")
    print(f"{'='*70}")
    print(f"  DA(t) = {args.da_base} + {args.da_amplitude}·sin(2πft) nM")
    print(f"  Frequencies: {frequencies} Hz ({len(frequencies)} points)")
    print(f"  Min cycles: {args.min_cycles}")
    print(f"  Output: {save_dir}")
    print(f"{'='*70}\n")

    # Build network
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    W_t, mask_d1, mask_d2, groups_info = create_network_structure(config.N_E, config.N_I, device)

    # Load checkpoint
    print(f"📂 Loading checkpoint: {args.checkpoint}")
    with open(args.checkpoint, 'rb') as f:
        ckpt_data = pickle.load(f)
    init_state = ckpt_data['final_state'].to(device)

    # ---- Sweep frequencies ----
    freq_results = {}

    for i, freq in enumerate(frequencies):
        period_s = 1.0 / freq
        duration_s = max(args.min_cycles * period_s + 50.0, 100.0)

        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(frequencies)}] f = {freq:.4f} Hz  (T = {period_s:.0f}s, duration = {duration_s:.0f}s)")
        print(f"{'─'*60}")

        t0 = time.time()

        data, alpha_d1, alpha_d2, da_trace, dur_s = run_single_frequency(
            freq, args, device, W_t, mask_d1, mask_d2, groups_info, init_state)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sim_time = time.time() - t0
        print(f"  ⏱️  Sim time: {_fmt_elapsed(sim_time)}")

        # Compute alpha oscillation amplitudes
        alpha_times = np.linspace(0, dur_s, len(alpha_d1))

        d1_amp, d1_phase = compute_oscillation_amplitude(
            alpha_d1[:, 1], alpha_times, freq, n_skip_cycles=1)
        d2_amp, d2_phase = compute_oscillation_amplitude(
            alpha_d2[:, 1], alpha_times, freq, n_skip_cycles=1)

        print(f"  📊 α_D1 amplitude: {d1_amp:.4f}, α_D2 amplitude: {d2_amp:.4f}")

        # Compute firing rate oscillation amplitudes
        analyzer = PFCAnalyzer(data)
        result_entry = {
            'freq_hz': freq,
            'period_s': period_s,
            'duration_s': dur_s,
            'sim_time': sim_time,
            'alpha_d1_amplitude': float(d1_amp),
            'alpha_d2_amplitude': float(d2_amp),
            'alpha_d1_phase': float(d1_phase),
            'alpha_d2_phase': float(d2_phase),
            'alpha_times': alpha_times,
            'alpha_d1_exp': alpha_d1[:, 1],
            'alpha_d2_exp': alpha_d2[:, 1],
            'da_trace': da_trace,
        }

        for grp in ['All-E', 'All-I', 'E-D1', 'E-D2']:
            for batch_id, batch_label in [(0, 'ctrl'), (1, 'exp')]:
                centers, rate = analyzer.compute_group_rate(batch_id, grp, time_win=1000.0)
                if rate is not None:
                    rate_times = centers / 1000.0
                    result_entry[f'rate_{grp}_times'] = rate_times
                    result_entry[f'rate_{grp}_{batch_label}'] = rate

                    if batch_label == 'exp':
                        rate_amp, rate_phase = compute_oscillation_amplitude(
                            rate, rate_times, freq, n_skip_cycles=1)
                        result_entry[f'rate_{grp}_amplitude'] = float(rate_amp)
                        result_entry[f'rate_{grp}_phase'] = float(rate_phase)
                        print(f"  📊 {grp} rate amplitude: {rate_amp:.2f} Hz")

        freq_results[freq] = result_entry

        # Free memory
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Generate plots ----
    print(f"\n🎨 Generating Bode diagram and example traces...")
    plot_bode_diagram(freq_results, args, save_dir)

    # ---- Save results (JSON-serializable subset) ----
    json_results = {}
    for freq in frequencies:
        fr = freq_results[freq]
        json_results[str(freq)] = {
            'freq_hz': fr['freq_hz'],
            'period_s': fr['period_s'],
            'duration_s': fr['duration_s'],
            'sim_time': fr['sim_time'],
            'alpha_d1_amplitude': fr['alpha_d1_amplitude'],
            'alpha_d2_amplitude': fr['alpha_d2_amplitude'],
        }
        for grp in ['All-E', 'All-I', 'E-D1', 'E-D2']:
            key = f'rate_{grp}_amplitude'
            if key in fr:
                json_results[str(freq)][key] = fr[key]

    json_results['_meta'] = {
        'da_base': args.da_base,
        'da_amplitude': args.da_amplitude,
        'frequencies': frequencies,
        'min_cycles': args.min_cycles,
        'timestamp': timestamp,
    }

    json_path = os.path.join(save_dir, "exp_c_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"📄 Results saved: {json_path}")

    # ---- Print summary ----
    t_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"  📊 Experiment C — Frequency Response Summary")
    print(f"{'='*70}")
    print(f"  {'Freq (Hz)':<12} │ {'α_D1 Amp':>10} {'α_D2 Amp':>10} │ {'All-E Amp':>10} {'E-D1 Amp':>10} {'E-D2 Amp':>10}")
    print(f"  {'─'*12}─┼─{'─'*10}─{'─'*10}─┼─{'─'*10}─{'─'*10}─{'─'*10}")
    for freq in frequencies:
        fr = freq_results[freq]
        ae = fr.get('rate_All-E_amplitude', 0)
        ed1 = fr.get('rate_E-D1_amplitude', 0)
        ed2 = fr.get('rate_E-D2_amplitude', 0)
        print(f"  {freq:<12.4f} │ {fr['alpha_d1_amplitude']:>10.4f} {fr['alpha_d2_amplitude']:>10.4f} │"
              f" {ae:>10.2f} {ed1:>10.2f} {ed2:>10.2f}")
    print(f"{'='*70}")
    print(f"  ⏱️  Total time: {_fmt_elapsed(t_total)}")
    print(f"  📁 Results in: {save_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
