"""WM-1 Demo: Single-item Working-Memory maintenance task.

Pipeline
--------
1.  Load a DA-baseline checkpoint (default: 2 nM steady state).
2.  Build a structured network (`create_wm_network`): two strongly
    self-recurrent memory pools Mem-A and Mem-B carved out of E-Other.
3.  Run the WM kernel:
        baseline   [0,    5000) ms   — let the network settle on the
                                       new (WM-augmented) connectivity.
        cue (A)    [5000, 6000) ms   — inject 250 pA into Mem-A.
        delay      [6000,11000) ms   — no input.  Mem-A should *persist*.
        probe      [11000,11500) ms  — read-out window.
4.  Save raw_data.pkl and produce 4 plots:
        combined_raster.png       (project-standard 3×2 raster)
        combined_rates_all.png    (project-standard 3×2 firing-rate)
        wm_overview.png           (raster + rates with shaded protocol bands)
        wm_rates_pools.png        (Mem-A vs Mem-B vs E-BG firing rates)

Usage
-----
    # 1) generate the 2 nM baseline checkpoint (only needs to be done once)
    python main.py --da 2.0 --duration 100 --save-ckpt

    # 2) run the WM demo, pointing at the produced ckpt
    python experiments/exp_wm_demo.py \
        --ckpt checkpoints/ckpt_DA2nM_bg200_100s.pkl

    # Optional flags:
    #   --da 2.0                base DA tone (nM, must match ckpt)
    #   --cue-amp 250           cue current amplitude (pA)
    #   --pool-size 100         neurons per memory pool
    #   --intra-w 6.0           intra-pool weight (pA)
    #   --intra-prob 0.30       intra-pool connection probability
    #   --gpu 0                 GPU id (-1 / invalid → CPU)
    #   --tag wm_demo           experiment-folder suffix
"""
import argparse
import os
import sys
import time

# Allow running from anywhere
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch

import config
from simulation.runners import run_wm_simulation_from_checkpoint
from simulation.utils import setup_experiment_folder, save_args, save_raw_data
from analysis.analyzer import PFCAnalyzer
from analysis.plotting import plot_combined_raster, plot_combined_rates_all
from analysis.wm_plotting import (plot_wm_overview, plot_wm_rates_pools,
                                  print_wm_report, register_wm_groups)


def parse_args():
    p = argparse.ArgumentParser(description="WM-1 single-item maintenance demo")
    p.add_argument('--ckpt', type=str, required=True,
                   help='Path to the DA-baseline checkpoint pkl '
                        '(e.g. checkpoints/ckpt_DA2nM_bg200_100s.pkl)')
    p.add_argument('--da', type=float, default=None,
                   help='Base DA tone (nM). Default = config.DA_BASELINE')
    p.add_argument('--cue-amp', type=float, default=None,
                   help='Cue-A injection amplitude (pA). '
                        'Default = config.WM_CUE_AMPLITUDE')
    p.add_argument('--pool-size', type=int, default=None,
                   help='Neurons per memory pool (default = WM_POOL_SIZE)')
    p.add_argument('--intra-w', type=float, default=None,
                   help='Intra-pool weight (pA, default = WM_INTRA_W)')
    p.add_argument('--intra-prob', type=float, default=None,
                   help='Intra-pool connection probability '
                        '(default = WM_INTRA_PROB)')
    p.add_argument('--baseline-ms', type=float, default=None,
                   help='Baseline window length (ms, default = WM_BASELINE_MS)')
    p.add_argument('--cue-ms', type=float, default=None,
                   help='Cue duration (ms, default = WM_CUE_DURATION_MS)')
    p.add_argument('--delay-ms', type=float, default=None,
                   help='Delay window length (ms, default = WM_DELAY_MS)')
    p.add_argument('--probe-ms', type=float, default=None,
                   help='Probe window length (ms, default = WM_PROBE_MS)')
    # ── WTA (winner-take-all) shared-inhibition controls ──
    p.add_argument('--no-wta', action='store_true',
                   help='Disable the shared-inhibition WTA loop (Mem-{A,B} <-> I-WM).')
    p.add_argument('--iwm-size', type=int, default=None,
                   help='Neurons in the I-WM sub-pool (default = WM_IWM_SIZE)')
    p.add_argument('--e2i-prob', type=float, default=None,
                   help='Mem-{A,B} -> I-WM connection probability (default = WM_E2I_PROB)')
    p.add_argument('--e2i-w', type=float, default=None,
                   help='Mem-{A,B} -> I-WM weight (pA, default = WM_E2I_W)')
    p.add_argument('--i2e-prob', type=float, default=None,
                   help='I-WM -> Mem-{A,B} connection probability (default = WM_I2E_PROB)')
    p.add_argument('--i2e-w', type=float, default=None,
                   help='I-WM -> Mem-{A,B} weight (pA, NEGATIVE; default = WM_I2E_W)')
    p.add_argument('--mem-bg-offset', type=float, default=None,
                   help='DC offset (pA, typ. NEGATIVE) added only to Mem-A∪Mem-B '
                        'background current. Scheme-B knob: -25..-35 puts the '
                        'memory pool in the LOW state at baseline so the cue can '
                        'flip it into a clean HIGH attractor state. '
                        'Default = config.WM_MEM_BG_OFFSET (=0).')
    p.add_argument('--sfa-b', type=float, default=None,
                   help='Spike-frequency adaptation increment per spike (pA). '
                        'Adds a slow hyperpolarising aftercurrent to Mem-pool '
                        'neurons, causing natural decay of persistent activity. '
                        'Default = config.WM_SFA_B (=5.0). Set 0 to disable.')
    p.add_argument('--sfa-tau', type=float, default=None,
                   help='SFA decay time constant (ms). '
                        'Default = config.WM_SFA_TAU (=1500).')
    p.add_argument('--alpha-gate', type=float, default=None,
                   help='NMDA saturation gain (Wang 2002). Lower values make '
                        'NMDA harder to saturate at low rates, widening the '
                        'bistable region. Default = config.NMDA_ALPHA_GATE (=0.5).')
    p.add_argument('--bg-mean', type=float, default=None,
                   help='Override BG_MEAN (pA). MUST match the value embedded '
                        'in the checkpoint fingerprint (e.g. 160 for the low-BG '
                        'sub-threshold WM regime). Default = config.BG_MEAN=200')
    # ── DA neuromodulation (Batch 1 only; Batch 0 = Control held at da_base) ──
    p.add_argument('--da-pulse', type=float, default=None,
                   help='Experimental DA concentration (nM) applied to Batch 1 '
                        'inside the [--da-pulse-onset, --da-pulse-offset) window. '
                        'Batch 0 stays at --da throughout (Control). '
                        'If unset, equals --da → no DA modulation (pure WM).')
    p.add_argument('--da-window', type=str, default=None,
                   help="DA pulse window preset. One of: 'cue' (= cue-A window), "
                        "'delay' (= delay window), 'cue+delay' (cue onset → delay end), "
                        "'whole' (entire run), or '<start_ms>:<end_ms>' explicit range. "
                        'Overrides --da-pulse-onset / --da-pulse-offset when set.')
    p.add_argument('--da-pulse-onset', type=float, default=None,
                   help='DA pulse onset (ms, run-relative). Default = cue onset.')
    p.add_argument('--da-pulse-offset', type=float, default=None,
                   help='DA pulse offset (ms, run-relative). Default = cue offset.')
    # ── D1 / D2 pathway pharmacological block (receptor antagonism) ──
    p.add_argument('--d1-block', action='store_true',
                   help='Block D1 receptor pathway (set EPS_D1=BIAS_D1=LAM_D1=0). '
                        'Equivalent to a saturating D1 antagonist; isolates D2 effects.')
    p.add_argument('--d2-block', action='store_true',
                   help='Block D2 receptor pathway (set EPS_D2=BIAS_D2=LAM_D2=0). '
                        'Equivalent to a saturating D2 antagonist; isolates D1 effects.')
    p.add_argument('--gpu', type=int, default=0,
                   help='GPU id; invalid id falls back to CPU')
    p.add_argument('--tag', type=str, default='wm_demo',
                   help='Suffix for the output folder name')
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # ── BG_MEAN must be overridden BEFORE any runner/network construction
    #     because it gates V_inf and is checked by the checkpoint fingerprint.
    if args.bg_mean is not None:
        config.BG_MEAN = args.bg_mean
        v_inf = config.V_REST + config.R_BASE * args.bg_mean
        print(f"🔧 BG_MEAN overridden to {args.bg_mean} pA "
              f"(V_inf = {v_inf:.1f} mV vs V_th={config.V_TH} mV)")

    # ── NMDA alpha_gate override ──
    if args.alpha_gate is not None:
        config.NMDA_ALPHA_GATE = args.alpha_gate
        print(f"🔧 NMDA_ALPHA_GATE overridden to {args.alpha_gate:.3f}")

    # ── Optional protocol overrides ──
    if args.baseline_ms is not None:
        config.WM_BASELINE_MS = args.baseline_ms
    if args.cue_ms is not None:
        config.WM_CUE_DURATION_MS = args.cue_ms
    if args.delay_ms is not None:
        config.WM_DELAY_MS = args.delay_ms
    if args.probe_ms is not None:
        config.WM_PROBE_MS = args.probe_ms

    # ── Device selection ──
    if torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        if args.gpu >= 0:
            print(f"⚠️  Requested GPU {args.gpu} not available — falling back to CPU.")
        device = torch.device("cpu")
    print(f"🔧 Using device: {device}")

    # ── Resolve DA-pulse window (Batch-1 experimental DA tone) ──
    #   Default = NO modulation (da_pulse = da_base, zero-length window).
    #   The kernel and runner already implement the full Langmuir D1/D2
    #   dynamics + mod_R / I_mod / scale_syn modulation paths; we only need
    #   to define when DA is elevated above the resting tone in Batch 1.
    cue_on  = config.WM_BASELINE_MS
    cue_off = cue_on + config.WM_CUE_DURATION_MS
    delay_end = cue_off + config.WM_DELAY_MS
    probe_end = delay_end + config.WM_PROBE_MS

    da_base_val = args.da if args.da is not None else config.DA_BASELINE
    if args.da_pulse is None:
        da_pulse_val   = da_base_val   # no pulse
        da_pulse_on    = 0.0
        da_pulse_off   = 0.0           # zero-length window ⇒ never fires
    else:
        da_pulse_val = args.da_pulse
        if args.da_window is not None:
            preset = args.da_window.strip().lower()
            if preset == 'cue':
                da_pulse_on, da_pulse_off = cue_on, cue_off
            elif preset == 'delay':
                da_pulse_on, da_pulse_off = cue_off, delay_end
            elif preset in ('cue+delay', 'cuedelay'):
                da_pulse_on, da_pulse_off = cue_on, delay_end
            elif preset == 'whole':
                da_pulse_on, da_pulse_off = 0.0, probe_end
            elif ':' in preset:
                a, b = preset.split(':', 1)
                da_pulse_on, da_pulse_off = float(a), float(b)
            else:
                raise ValueError(
                    f"Unknown --da-window preset: {args.da_window!r}. "
                    "Use cue / delay / cue+delay / whole / <start>:<end>.")
        else:
            da_pulse_on  = (args.da_pulse_onset  if args.da_pulse_onset  is not None
                            else cue_on)
            da_pulse_off = (args.da_pulse_offset if args.da_pulse_offset is not None
                            else cue_off)

    da_modulated = (da_pulse_val != da_base_val) and (da_pulse_off > da_pulse_on)
    print(f"🧪 DA schedule: Batch0=Control (DA={da_base_val:g} nM throughout)")
    if da_modulated:
        print(f"               Batch1=Exp     (DA: {da_base_val:g} → {da_pulse_val:g} nM "
              f"in [{da_pulse_on:.0f}, {da_pulse_off:.0f}) ms)")
    else:
        print(f"               Batch1=Exp     (DA={da_base_val:g} nM throughout — "
              f"NO pulse, identical to Control modulo random noise)")

    # ── Output folder ──
    da_label = da_base_val
    if da_modulated:
        exp_tag = f"{args.tag}_DA{da_base_val:g}to{da_pulse_val:g}nM"
    else:
        exp_tag = f"{args.tag}_DA{da_label:g}nM"
    save_dir = setup_experiment_folder(tag=exp_tag)

    # ── Persist the experiment configuration ──
    cfg = {
        'kind': 'WM-1 single-item maintenance',
        'checkpoint': args.ckpt,
        'da_base': da_label,
        'cue_amplitude': (args.cue_amp if args.cue_amp is not None
                          else config.WM_CUE_AMPLITUDE),
        'pool_size': (args.pool_size if args.pool_size is not None
                      else config.WM_POOL_SIZE),
        'intra_prob': (args.intra_prob if args.intra_prob is not None
                       else config.WM_INTRA_PROB),
        'intra_w':    (args.intra_w if args.intra_w is not None
                       else config.WM_INTRA_W),
        'use_wta':    (not args.no_wta) and config.WM_USE_WTA,
        'iwm_size':   (args.iwm_size if args.iwm_size is not None
                       else config.WM_IWM_SIZE),
        'e2i_prob':   (args.e2i_prob if args.e2i_prob is not None
                       else config.WM_E2I_PROB),
        'e2i_w':      (args.e2i_w if args.e2i_w is not None
                       else config.WM_E2I_W),
        'i2e_prob':   (args.i2e_prob if args.i2e_prob is not None
                       else config.WM_I2E_PROB),
        'i2e_w':      (args.i2e_w if args.i2e_w is not None
                       else config.WM_I2E_W),
        'mem_bg_offset': (args.mem_bg_offset if args.mem_bg_offset is not None
                          else config.WM_MEM_BG_OFFSET),
        'sfa_b':       (args.sfa_b if args.sfa_b is not None
                        else config.WM_SFA_B),
        'sfa_tau':     (args.sfa_tau if args.sfa_tau is not None
                        else config.WM_SFA_TAU),
        'block_d1':    bool(args.d1_block),
        'block_d2':    bool(args.d2_block),
        'baseline_ms': config.WM_BASELINE_MS,
        'cue_ms':      config.WM_CUE_DURATION_MS,
        'delay_ms':    config.WM_DELAY_MS,
        'probe_ms':    config.WM_PROBE_MS,
        'bg_mean':     config.BG_MEAN,
        'v_inf':       config.V_REST + config.R_BASE * config.BG_MEAN,
        # DA modulation schedule (Batch1=Exp; Batch0 stays at da_base)
        'da_base':         da_base_val,
        'da_pulse':        da_pulse_val,
        'da_pulse_onset':  da_pulse_on,
        'da_pulse_offset': da_pulse_off,
        'da_window_preset': args.da_window,
        'da_modulated':    bool(da_modulated),
        'device': str(device),
        'note': ('Working-memory maintenance — Scheme A (structured pools)'
                 + (' + DA modulation' if da_modulated else '')),
    }
    save_args(cfg, save_dir)

    # ── Run WM simulation ──
    print(f"🚀 Starting WM-1 demo (resume from: {args.ckpt})")
    data = run_wm_simulation_from_checkpoint(
        checkpoint_path=args.ckpt,
        da_base=da_base_val,
        da_pulse=da_pulse_val,
        da_pulse_onset=da_pulse_on,
        da_pulse_offset=da_pulse_off,
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

    # ── Save raw data ──
    save_raw_data(data, save_dir)

    # ── Analyze & plot ──
    print("\n🎨 Generating plots ...")
    analyzer = PFCAnalyzer(data)
    register_wm_groups(analyzer)

    # Project-standard combined plots (raster + firing-rate)
    plot_combined_raster(analyzer, save_dir=save_dir)
    plot_combined_rates_all(analyzer, save_dir=save_dir)

    # WM-specific overview & pool-rate figures
    plot_wm_overview(analyzer, save_dir=save_dir)
    plot_wm_rates_pools(analyzer, save_dir=save_dir)

    # WM persistence report
    print_wm_report(analyzer,
                    save_path=os.path.join(str(save_dir), 'wm_report.txt'))

    # Standard analyzer report (firing-rate / FFT tables)
    analyzer.save_report(os.path.join(str(save_dir), 'analysis_report.txt'))

    elapsed = time.time() - t0
    print(f"\n✅ WM-1 demo finished in {elapsed:.1f}s.  Outputs: {save_dir}")


if __name__ == '__main__':
    main()
