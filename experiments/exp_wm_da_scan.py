"""DA Slow-Scan WM Diagnostic Experiment.

Goal
----
In a single 80 s run, ramp DA from da_base up to da_peak and back down
*slowly* (each ramp >= τ_on_D1 of the slowest DA receptor channel, so the
receptor occupancy `alpha_D1` actually has time to track the input).
Inject a brief cue into Mem-A near the start, then watch:

    1. Whether the Mem-A attractor is sustained at low DA;
    2. Whether ramping DA up quenches / strengthens it;
    3. Whether ramping DA back down restores / loses it (hysteresis);
    4. The critical DA concentration at which attractor switches state.

Trapezoidal DA schedule (defaults; all in ms, run-relative):

    DA(t)
     ▲
     │            da_peak ┌──────────┐
     │                   ╱            ╲
     │                  ╱              ╲
     │ da_base ────────╱                ╲────────
     │                 ↑   ↑        ↑    ↑
     │           pulse_on  +up      off-down  pulse_off
     └─────────────────────────────────────────► t
            5s    35s          65s         80s

Cue (Mem-A) is injected at [5s, 6s) — the same as the standard WM-1 demo —
but with a long delay window (74 s) instead of 5 s.

Outputs
-------
    wm_da_scan_overview.png  — 5-row diagnostic figure
        row 1: DA(t)   (Batch-1 ramp + Batch-0 control flat line)
        row 2: alpha_D1(t)
        row 3: alpha_D2(t)
        row 4: Mem-A firing rate (Batch-1 vs Batch-0)
        row 5: Mem-B firing rate (Batch-1 vs Batch-0)
    plus all standard plot_combined_raster / rates / wm_overview figures.

Usage
-----
    python experiments/exp_wm_da_scan.py \
        --ckpt checkpoints/ckpt_DA2nM_bg160_60s.pkl \
        --da 2.0 --da-peak 12.0 \
        --ramp-up-ms 30000 --hold-ms 10000 --ramp-down-ms 30000 \
        --baseline-ms 5000 --cue-ms 1000 --delay-ms 74000 --probe-ms 0 \
        --tag da_scan
"""
import argparse
import os
import sys
import time

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from simulation.runners import run_wm_simulation_from_checkpoint
from simulation.utils import setup_experiment_folder, save_args, save_raw_data
from analysis.analyzer import PFCAnalyzer
from analysis.plotting import plot_combined_raster, plot_combined_rates_all
from analysis.wm_plotting import (plot_wm_overview, plot_wm_rates_pools,
                                  print_wm_report, register_wm_groups)


def parse_args():
    p = argparse.ArgumentParser(description="DA slow-scan WM diagnostic")
    p.add_argument('--ckpt', type=str, required=True,
                   help='Path to the DA-baseline checkpoint pkl')
    # ── DA trapezoid ──
    p.add_argument('--da', type=float, default=2.0,
                   help='Baseline DA tone (nM); Batch-0 stays here forever')
    p.add_argument('--da-peak', type=float, default=12.0,
                   help='Peak DA during the slow scan (nM)')
    p.add_argument('--ramp-up-ms', type=float, default=30000.0,
                   help='Linear ramp-up duration (ms). >= tau_on_D1 strongly recommended.')
    p.add_argument('--hold-ms', type=float, default=10000.0,
                   help='Hold-at-peak duration (ms)')
    p.add_argument('--ramp-down-ms', type=float, default=30000.0,
                   help='Linear ramp-down duration (ms)')
    # ── Cue / protocol windows ──
    p.add_argument('--cue-amp', type=float, default=None,
                   help='Cue-A amplitude (pA, default = WM_CUE_AMPLITUDE)')
    p.add_argument('--baseline-ms', type=float, default=5000.0,
                   help='Baseline (pre-cue) window (ms)')
    p.add_argument('--cue-ms', type=float, default=1000.0,
                   help='Cue window (ms)')
    p.add_argument('--probe-ms', type=float, default=0.0,
                   help='Probe window (ms, default 0 — DA scan IS the probe)')
    # ── Optional WM structural overrides (defaults from config) ──
    p.add_argument('--pool-size', type=int, default=None)
    p.add_argument('--intra-w', type=float, default=None)
    p.add_argument('--intra-prob', type=float, default=None)
    p.add_argument('--no-wta', action='store_true')
    p.add_argument('--iwm-size', type=int, default=None)
    p.add_argument('--e2i-prob', type=float, default=None)
    p.add_argument('--e2i-w', type=float, default=None)
    p.add_argument('--i2e-prob', type=float, default=None)
    p.add_argument('--i2e-w', type=float, default=None)
    p.add_argument('--mem-bg-offset', type=float, default=None)
    p.add_argument('--sfa-b', type=float, default=None)
    p.add_argument('--sfa-tau', type=float, default=None)
    p.add_argument('--alpha-gate', type=float, default=None)
    p.add_argument('--bg-mean', type=float, default=None)
    # ── Receptor pharmacology ──
    p.add_argument('--d1-block', action='store_true')
    p.add_argument('--d2-block', action='store_true')
    # ── Misc ──
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--tag', type=str, default='da_scan')
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # ── BG_MEAN must be overridden BEFORE any runner/network construction ──
    if args.bg_mean is not None:
        config.BG_MEAN = args.bg_mean
        v_inf = config.V_REST + config.R_BASE * args.bg_mean
        print(f"🔧 BG_MEAN overridden to {args.bg_mean} pA "
              f"(V_inf = {v_inf:.1f} mV vs V_th={config.V_TH} mV)")
    if args.alpha_gate is not None:
        config.NMDA_ALPHA_GATE = args.alpha_gate
        print(f"🔧 NMDA_ALPHA_GATE overridden to {args.alpha_gate:.3f}")

    # ── Resolve protocol windows ──
    config.WM_BASELINE_MS = args.baseline_ms
    config.WM_CUE_DURATION_MS = args.cue_ms
    config.WM_PROBE_MS = args.probe_ms
    # delay = ramp_up + hold + ramp_down  (DA scan happens entirely inside delay)
    delay_ms = args.ramp_up_ms + args.hold_ms + args.ramp_down_ms
    config.WM_DELAY_MS = delay_ms

    # Run-relative timestamps
    cue_on    = config.WM_BASELINE_MS                    # 5000
    cue_off   = cue_on + config.WM_CUE_DURATION_MS       # 6000
    delay_end = cue_off + config.WM_DELAY_MS             # 6000+70000=76000
    probe_end = delay_end + config.WM_PROBE_MS

    # DA trapezoid window: starts right after the cue, fills the entire delay
    da_pulse_on  = cue_off
    da_pulse_off = delay_end

    print("🧪 Protocol windows:")
    print(f"   baseline : [0, {cue_on:.0f}) ms")
    print(f"   cue (A)  : [{cue_on:.0f}, {cue_off:.0f}) ms  amp={args.cue_amp}")
    print(f"   delay    : [{cue_off:.0f}, {delay_end:.0f}) ms  ← DA SCAN HERE")
    if config.WM_PROBE_MS > 0:
        print(f"   probe    : [{delay_end:.0f}, {probe_end:.0f}) ms")
    print(f"🧪 DA trapezoid:")
    print(f"   ramp-up   : {args.ramp_up_ms/1000:.1f} s "
          f"({args.da:g} → {args.da_peak:g} nM)")
    print(f"   hold      : {args.hold_ms/1000:.1f} s @ {args.da_peak:g} nM")
    print(f"   ramp-down : {args.ramp_down_ms/1000:.1f} s "
          f"({args.da_peak:g} → {args.da:g} nM)")

    # ── Device ──
    if torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        if args.gpu >= 0:
            print(f"⚠️  GPU {args.gpu} not available, falling back to CPU.")
        device = torch.device("cpu")
    print(f"🔧 Using device: {device}")

    # ── Output folder ──
    exp_tag = f"{args.tag}_DA{args.da:g}to{args.da_peak:g}nM"
    save_dir = setup_experiment_folder(tag=exp_tag)

    cfg = {
        'kind': 'WM-DA-slow-scan diagnostic',
        'checkpoint': args.ckpt,
        'da_base': args.da,
        'da_peak': args.da_peak,
        'da_ramp_up_ms': args.ramp_up_ms,
        'da_hold_ms': args.hold_ms,
        'da_ramp_down_ms': args.ramp_down_ms,
        'da_pulse_onset': da_pulse_on,
        'da_pulse_offset': da_pulse_off,
        'cue_amplitude': (args.cue_amp if args.cue_amp is not None
                          else config.WM_CUE_AMPLITUDE),
        'baseline_ms': config.WM_BASELINE_MS,
        'cue_ms': config.WM_CUE_DURATION_MS,
        'delay_ms': config.WM_DELAY_MS,
        'probe_ms': config.WM_PROBE_MS,
        'bg_mean': config.BG_MEAN,
        'block_d1': bool(args.d1_block),
        'block_d2': bool(args.d2_block),
        'device': str(device),
        'note': ('DA slow-scan WM diagnostic — Batch0 = control (DA=da_base), '
                 'Batch1 = trapezoidal DA scan inside delay.'),
    }
    save_args(cfg, save_dir)

    # ── Run ──
    print(f"🚀 Starting DA slow-scan (resume from: {args.ckpt})")
    data = run_wm_simulation_from_checkpoint(
        checkpoint_path=args.ckpt,
        da_base=args.da,
        da_pulse=args.da_peak,
        da_pulse_onset=da_pulse_on,
        da_pulse_offset=da_pulse_off,
        da_ramp_up_ms=args.ramp_up_ms,
        da_ramp_down_ms=args.ramp_down_ms,
        cue_a_amplitude=args.cue_amp,
        pool_size=args.pool_size,
        intra_prob=args.intra_prob,
        intra_w=args.intra_w,
        use_wta=(False if args.no_wta else None),
        iwm_size=args.iwm_size,
        e2i_prob=args.e2i_prob,
        e2i_w=args.e2i_w,
        i2e_prob=args.i2e_prob,
        i2e_w=args.i2e_w,
        mem_bg_offset=args.mem_bg_offset,
        sfa_b=args.sfa_b,
        sfa_tau=args.sfa_tau,
        block_d1=args.d1_block,
        block_d2=args.d2_block,
        device=device,
    )

    save_raw_data(data, save_dir)

    # ── Plots ──
    print("\n🎨 Generating plots ...")
    analyzer = PFCAnalyzer(data)
    register_wm_groups(analyzer)

    plot_combined_raster(analyzer, save_dir=save_dir)
    plot_combined_rates_all(analyzer, save_dir=save_dir)
    plot_wm_overview(analyzer, save_dir=save_dir)
    plot_wm_rates_pools(analyzer, save_dir=save_dir)
    plot_da_scan_diagnostic(analyzer, args, save_dir=save_dir)

    print_wm_report(analyzer,
                    save_path=os.path.join(str(save_dir), 'wm_report.txt'))
    analyzer.save_report(os.path.join(str(save_dir), 'analysis_report.txt'))

    elapsed = time.time() - t0
    print(f"\n✅ DA slow-scan finished in {elapsed:.1f}s. Outputs: {save_dir}")


# ----------------------------------------------------------------------
#  DA-scan diagnostic figure (5 rows × 1 col)
# ----------------------------------------------------------------------
def _reconstruct_da_trace(args, t_ms):
    """Recompute DA(t) on the experiment-relative time grid t_ms (ms array)."""
    da_base, da_peak = float(args.da), float(args.da_peak)
    pulse_on  = float(args.baseline_ms + args.cue_ms)
    pulse_off = pulse_on + float(args.ramp_up_ms + args.hold_ms + args.ramp_down_ms)
    ramp_up_end   = pulse_on  + float(args.ramp_up_ms)
    ramp_down_beg = pulse_off - float(args.ramp_down_ms)

    da = np.full_like(t_ms, da_base, dtype=np.float64)
    in_pulse = (t_ms >= pulse_on) & (t_ms < pulse_off)
    rising  = in_pulse & (t_ms < ramp_up_end)
    holding = in_pulse & (t_ms >= ramp_up_end) & (t_ms < ramp_down_beg)
    falling = in_pulse & (t_ms >= ramp_down_beg)

    if args.ramp_up_ms > 0:
        frac_up = (t_ms[rising] - pulse_on) / args.ramp_up_ms
        da[rising] = da_base + (da_peak - da_base) * frac_up
    da[holding] = da_peak
    if args.ramp_down_ms > 0:
        frac_dn = (pulse_off - t_ms[falling]) / args.ramp_down_ms
        da[falling] = da_base + (da_peak - da_base) * frac_dn
    return da


def plot_da_scan_diagnostic(analyzer, args, save_dir=None):
    """5-row diagnostic figure: DA, αD1, αD2, Mem-A rate, Mem-B rate."""
    duration = analyzer.data['config']['duration']    # ms
    dt       = analyzer.data['config']['dt']           # ms
    t_ms = np.arange(0, duration, dt)
    t_s  = t_ms / 1000.0

    # 1) DA(t) — analytical reconstruction
    da_exp  = _reconstruct_da_trace(args, t_ms)
    da_ctrl = np.full_like(t_ms, args.da, dtype=np.float64)

    # 2) alpha traces (recorded by the kernel)
    alpha_d1_trace = analyzer.data.get('alpha_d1_trace', None)  # (n_rec, batch)
    alpha_d2_trace = analyzer.data.get('alpha_d2_trace', None)
    alpha_record_interval = analyzer.data['config'].get('alpha_record_interval', 1)
    if alpha_d1_trace is not None:
        if hasattr(alpha_d1_trace, 'cpu'):
            alpha_d1_trace = alpha_d1_trace.cpu().numpy()
            alpha_d2_trace = alpha_d2_trace.cpu().numpy()
        n_rec = alpha_d1_trace.shape[0]
        t_alpha_s = np.arange(n_rec) * alpha_record_interval * dt / 1000.0
    else:
        t_alpha_s = None

    # 3) firing-rate traces of Mem-A and Mem-B (one trace per batch)
    bin_ms = 50.0
    centers_ms, rate_A0 = analyzer.compute_group_rate(0, 'Mem-A', time_win=bin_ms)
    _,          rate_A1 = analyzer.compute_group_rate(1, 'Mem-A', time_win=bin_ms)
    _,          rate_B0 = analyzer.compute_group_rate(0, 'Mem-B', time_win=bin_ms)
    _,          rate_B1 = analyzer.compute_group_rate(1, 'Mem-B', time_win=bin_ms)
    t_rate_s = centers_ms / 1000.0

    # ── Figure ──
    fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=True)
    fig.suptitle(f"DA Slow-Scan Diagnostic  "
                 f"(DA: {args.da:g} → {args.da_peak:g} nM, "
                 f"ramps {args.ramp_up_ms/1000:.0f}+{args.hold_ms/1000:.0f}+"
                 f"{args.ramp_down_ms/1000:.0f} s)",
                 fontsize=18, fontweight='bold')

    # Shaded protocol bands
    cue_on  = args.baseline_ms / 1000.0
    cue_off = (args.baseline_ms + args.cue_ms) / 1000.0
    pulse_on  = cue_off
    pulse_off = pulse_on + (args.ramp_up_ms + args.hold_ms + args.ramp_down_ms) / 1000.0
    ramp_up_end   = pulse_on + args.ramp_up_ms / 1000.0
    ramp_down_beg = pulse_off - args.ramp_down_ms / 1000.0

    def _shade(ax):
        ax.axvspan(cue_on, cue_off, color='gold', alpha=0.25, label='Cue-A')
        ax.axvspan(pulse_on, ramp_up_end, color='lightcoral',
                   alpha=0.15, label='ramp-up')
        ax.axvspan(ramp_up_end, ramp_down_beg, color='salmon',
                   alpha=0.20, label='hold')
        ax.axvspan(ramp_down_beg, pulse_off, color='lightcoral',
                   alpha=0.15, label='ramp-down')

    # Row 1: DA
    ax = axes[0]
    _shade(ax)
    ax.plot(t_s, da_ctrl, color='gray', lw=1.8, label='Batch-0 (Control)')
    ax.plot(t_s, da_exp,  color='crimson', lw=2.2, label='Batch-1 (DA scan)')
    ax.set_ylabel('DA (nM)')
    ax.set_title('① DA concentration schedule')
    ax.legend(loc='upper right', ncol=2, fontsize=10)
    ax.grid(alpha=0.3)

    # Row 2: alpha_D1
    ax = axes[1]
    _shade(ax)
    if t_alpha_s is not None:
        ax.plot(t_alpha_s, alpha_d1_trace[:, 0], color='gray', lw=1.6,
                label='Batch-0')
        ax.plot(t_alpha_s, alpha_d1_trace[:, 1], color='crimson', lw=2.0,
                label='Batch-1')
    ax.set_ylabel(r'$\alpha_{D1}$')
    ax.set_title('② D1 receptor occupancy (Langmuir)')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)

    # Row 3: alpha_D2
    ax = axes[2]
    _shade(ax)
    if t_alpha_s is not None:
        ax.plot(t_alpha_s, alpha_d2_trace[:, 0], color='gray', lw=1.6,
                label='Batch-0')
        ax.plot(t_alpha_s, alpha_d2_trace[:, 1], color='crimson', lw=2.0,
                label='Batch-1')
    ax.set_ylabel(r'$\alpha_{D2}$')
    ax.set_title('③ D2 receptor occupancy (Langmuir)')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)

    # Row 4: Mem-A rate
    ax = axes[3]
    _shade(ax)
    ax.plot(t_rate_s, rate_A0, color='gray',    lw=1.6, label='Batch-0 (Control)')
    ax.plot(t_rate_s, rate_A1, color='crimson', lw=2.0, label='Batch-1 (DA scan)')
    ax.set_ylabel('Mem-A rate (Hz)')
    ax.set_title('④ Mem-A population firing rate')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)

    # Row 5: Mem-B rate
    ax = axes[4]
    _shade(ax)
    ax.plot(t_rate_s, rate_B0, color='gray',    lw=1.6, label='Batch-0 (Control)')
    ax.plot(t_rate_s, rate_B1, color='steelblue', lw=2.0,
            label='Batch-1 (DA scan)')
    ax.set_ylabel('Mem-B rate (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title('⑤ Mem-B population firing rate')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_dir is not None:
        out = os.path.join(str(save_dir), 'da_scan_diagnostic.png')
        fig.savefig(out, dpi=150)
        print(f"   📊 Saved: {out}")
    plt.close(fig)


if __name__ == '__main__':
    main()
