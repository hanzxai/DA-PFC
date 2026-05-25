#!/usr/bin/env python3
"""WM Inverted-U Dose-Response Sweep.

For each DA concentration in DA_CONCENTRATIONS, run N_SEEDS independent
WM simulations, compute the 6 WM metrics, and plot the inverted-U curves.

Usage:
    cd DA-PFC
    python -m wm_eval.run_inverted_u \
        --ckpt checkpoints/ckpt_DA2nM_bg200_100s.pkl \
        --n-seeds 3
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

# ---- Project path setup -----------------------------------------------------
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import config
from simulation.runners import run_wm_simulation_from_checkpoint
from analysis.analyzer import PFCAnalyzer

from wm_eval.config_eval import (DA_CONCENTRATIONS, DEFAULT_N_SEEDS,
                                 WM_PROTOCOL, DEFAULT_CHECKPOINT,
                                 build_wm_kwargs)
from wm_eval.metrics import compute_all_metrics
from wm_eval.plotting import (plot_inverted_u, plot_inverted_u_with_alpha,
                              plot_representative_traces,
                              plot_metric_correlations)


def parse_args():
    p = argparse.ArgumentParser(description='WM Inverted-U Dose-Response Sweep')
    p.add_argument('--ckpt', type=str, default=DEFAULT_CHECKPOINT,
                   help='Path to baseline checkpoint')
    p.add_argument('--da-list', type=float, nargs='+', default=None,
                   help='Custom DA concentrations (nM)')
    p.add_argument('--n-seeds', type=int, default=DEFAULT_N_SEEDS,
                   help='Number of independent seeds per DA point')
    p.add_argument('--intra-w', type=float, default=None,
                   help='Override Mem-A/B intra-pool weight (pA). Lower => '
                        'attractor closer to bistability boundary => DA '
                        'modulation more visible. Default uses WM_SCHEME_A.')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--tag', type=str, default='inverted_u',
                   help='Output folder tag')
    return p.parse_args()


def _seed_after_network_creation(seed: int):
    """Returns a context manager that re-seeds torch RNG immediately after
    create_wm_network is called inside the runner.

    Why we need this: simulation/runners.py:673 calls
        torch.manual_seed(config.RANDOM_SEED)
    so the WM network topology stays fingerprint-compatible. After this
    call, the runner uses torch.randn(...) for background noise during the
    WM trial. If we re-seed torch right after create_wm_network() returns,
    we get different noise per seed without breaking ckpt compatibility.

    Implementation note: create_wm_network is imported *inside* the runner
    function (`from models.network import create_wm_network`), so we must
    patch it on the source module `models.network`, not on
    `simulation.runners`.
    """
    from contextlib import contextmanager
    import models.network as _net
    _orig = _net.create_wm_network

    @contextmanager
    def _ctx():
        def _wrapped(*a, **kw):
            ret = _orig(*a, **kw)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            return ret
        _net.create_wm_network = _wrapped
        try:
            yield
        finally:
            _net.create_wm_network = _orig
    return _ctx()


def run_one_simulation(ckpt: str, da: float, seed: int, device,
                       overrides: dict = None) -> dict:
    """Run a single WM simulation and compute all WM metrics.

    NB1: We deliberately do NOT modify config.RANDOM_SEED -- it is part of
    the checkpoint fingerprint and changing it would fail
    verify_checkpoint_fingerprint.

    NB2: The runner internally re-seeds torch with config.RANDOM_SEED to
    keep the network topology fingerprint-stable. We use a context manager
    to inject our own seed AFTER network creation, so background noise
    during the WM trial actually varies across seeds.
    """
    kwargs = build_wm_kwargs(da, WM_PROTOCOL, overrides=overrides)
    with _seed_after_network_creation(seed):
        data = run_wm_simulation_from_checkpoint(
            checkpoint_path=ckpt,
            device=device,
            **kwargs,
        )

    analyzer = PFCAnalyzer(data)
    metrics = compute_all_metrics(analyzer, batch_idx=1)

    # Steady-state alpha (for the right-Y-axis figure)
    alpha_d1_trace = data.get('alpha_d1_trace')
    alpha_d2_trace = data.get('alpha_d2_trace')
    if alpha_d1_trace is not None and alpha_d2_trace is not None:
        if hasattr(alpha_d1_trace, 'cpu'):
            alpha_d1_trace = alpha_d1_trace.cpu().numpy()
        if hasattr(alpha_d2_trace, 'cpu'):
            alpha_d2_trace = alpha_d2_trace.cpu().numpy()
        try:
            n = len(alpha_d1_trace)
            tail = max(n // 10, 1)
            # Use exp batch (index 1) if available; else batch 0
            bidx = 1 if alpha_d1_trace.ndim > 1 and alpha_d1_trace.shape[1] > 1 else 0
            metrics['alpha_d1_steady'] = float(np.mean(alpha_d1_trace[-tail:, bidx]))
            metrics['alpha_d2_steady'] = float(np.mean(alpha_d2_trace[-tail:, bidx]))
        except Exception:
            metrics['alpha_d1_steady'] = float('nan')
            metrics['alpha_d2_steady'] = float('nan')
    else:
        metrics['alpha_d1_steady'] = float('nan')
        metrics['alpha_d2_steady'] = float('nan')

    # Save the firing rate traces for representative plotting
    try:
        centers_a, rate_a = analyzer.compute_group_rate(1, 'Mem-A',
                                                         time_win=20.0, sigma=2.0)
        centers_b, rate_b = analyzer.compute_group_rate(1, 'Mem-B',
                                                         time_win=20.0, sigma=2.0)
        metrics['_traces'] = {
            'time_s': np.asarray(centers_a) / 1000.0,
            'mem_a_rate': np.asarray(rate_a),
            'mem_b_rate': np.asarray(rate_b),
            'da': da,
            'protocol': analyzer.cfg.get('wm_protocol', {}),
        }
    except Exception:
        metrics['_traces'] = None

    return metrics


def aggregate_seeds(per_seed_metrics: list) -> dict:
    """Compute mean and std across multiple seeds."""
    keys = [k for k in per_seed_metrics[0].keys()
            if k not in ('_pass', '_overall_pass', '_traces')
            and isinstance(per_seed_metrics[0][k], (int, float))]
    mean = {k: float(np.nanmean([m[k] for m in per_seed_metrics])) for k in keys}
    std  = {k: float(np.nanstd([m[k]  for m in per_seed_metrics])) for k in keys}
    return {
        'mean': mean,
        'std':  std,
        'all_seeds': per_seed_metrics,
        'alpha_d1_steady': mean.get('alpha_d1_steady', float('nan')),
        'alpha_d2_steady': mean.get('alpha_d2_steady', float('nan')),
    }


def _serializable(metrics: dict) -> dict:
    """Strip non-JSON-friendly fields."""
    out = {}
    for k, v in metrics.items():
        if k in ('_traces', 'all_seeds'):
            continue
        if isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items()
                      if not kk.startswith('_') and isinstance(vv, (int, float, bool, str))}
        elif isinstance(v, (int, float, bool, str)):
            out[k] = v
    return out


def main():
    args = parse_args()
    t_start = time.time()

    # Device
    if torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'[device] {device}')

    da_list = sorted(args.da_list or DA_CONCENTRATIONS)
    n_seeds = args.n_seeds

    # Build optional WM_SCHEME_A overrides from CLI flags
    overrides = {}
    if args.intra_w is not None:
        overrides['intra_w'] = args.intra_w

    # Output folder (include intra_w in tag if overridden)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    extra_tag = ''
    if 'intra_w' in overrides:
        extra_tag += f"_iw{overrides['intra_w']:g}"
    save_dir = os.path.join(PROJ_ROOT, 'wm_eval', 'outputs',
                            f'{args.tag}{extra_tag}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f'[output] {save_dir}')

    print(f'\n{"=" * 70}')
    print(f'  WM Inverted-U Sweep')
    print(f'{"=" * 70}')
    print(f'  DA points : {da_list} nM ({len(da_list)} pts)')
    print(f'  Seeds/pt  : {n_seeds}')
    print(f'  Total runs: {len(da_list) * n_seeds}')
    print(f'  Checkpoint: {args.ckpt}')
    if overrides:
        print(f'  Overrides : {overrides}')
    print(f'{"=" * 70}\n')

    # ----------------------------------------------------------------
    #  Run sweep
    # ----------------------------------------------------------------
    sweep_results = {}
    representative_data = {'low': None, 'opt': None, 'high': None}

    for i, da in enumerate(da_list):
        per_seed = []
        for s in range(n_seeds):
            seed = 42 + s * 1000
            print(f'\n[{i+1}/{len(da_list)}] DA={da:g} nM, seed={seed} '
                  f'({s+1}/{n_seeds})')
            t0 = time.time()
            m = run_one_simulation(args.ckpt, da, seed, device,
                                   overrides=overrides)
            print(f'   t={time.time() - t0:.1f}s | '
                  f'P={m["persistence"]:+.2f}Hz '
                  f'SI={m["selectivity"]:+.3f} '
                  f'PR={m["decay_ratio"]:+.3f} '
                  f'd\'={m["d_prime"]:+.2f}')
            per_seed.append(m)

        agg = aggregate_seeds(per_seed)
        agg['_traces'] = per_seed[0].get('_traces')
        sweep_results[da] = agg

    # ----------------------------------------------------------------
    #  Identify low / opt / high DA for representative figure
    # ----------------------------------------------------------------
    persistence_arr = np.array([sweep_results[d]['mean']['persistence']
                                for d in da_list])
    if np.any(np.isfinite(persistence_arr)):
        opt_idx = int(np.nanargmax(persistence_arr))
    else:
        opt_idx = len(da_list) // 2
    representative_data['low']  = sweep_results[da_list[0]].get('_traces')
    representative_data['opt']  = sweep_results[da_list[opt_idx]].get('_traces')
    representative_data['high'] = sweep_results[da_list[-1]].get('_traces')

    # ----------------------------------------------------------------
    #  Plot
    # ----------------------------------------------------------------
    print('\n[plot] Generating figures...')
    plot_inverted_u(sweep_results,
                    os.path.join(save_dir, 'inverted_u_curves.png'),
                    title_suffix=f' (n_seeds={n_seeds})')
    plot_inverted_u_with_alpha(sweep_results,
                               os.path.join(save_dir, 'inverted_u_with_alpha.png'))
    plot_representative_traces(representative_data['low'],
                               representative_data['opt'],
                               representative_data['high'],
                               os.path.join(save_dir, 'representative_traces.png'),
                               labels=[f"DA too LOW ({da_list[0]:g} nM)",
                                       f"OPTIMAL ({da_list[opt_idx]:g} nM)",
                                       f"DA too HIGH ({da_list[-1]:g} nM)"])
    plot_metric_correlations(sweep_results,
                             os.path.join(save_dir, 'metric_correlations.png'))

    # ----------------------------------------------------------------
    #  Save JSON summary
    # ----------------------------------------------------------------
    summary = {
        'da_list': da_list,
        'n_seeds': n_seeds,
        'checkpoint': args.ckpt,
        'optimal_da': da_list[opt_idx],
        'sweep_results': {str(d): _serializable(sweep_results[d])
                          for d in da_list},
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[save] {os.path.join(save_dir, 'summary.json')}")

    # ----------------------------------------------------------------
    #  Console summary table
    # ----------------------------------------------------------------
    print(f'\n{"=" * 100}')
    print(f'  Summary (mean across {n_seeds} seeds)')
    print(f'{"=" * 100}')
    print(f"  {'DA(nM)':>7}  {'Pers(Hz)':>10}  {'SelIdx':>8}  {'DecayR':>8}  "
          f"{'d-prime':>8}  {'Acc':>5}  {'Comp':>6}  {'PASS':>6}")
    print('  ' + '-' * 88)
    for d in da_list:
        m = sweep_results[d]['mean']
        n_pass = sum(int(s.get('_overall_pass', False))
                     for s in sweep_results[d]['all_seeds'])
        flag = 'PASS' if n_pass >= n_seeds // 2 + 1 else 'FAIL'
        print(f'  {d:>7.2f}  {m["persistence"]:>+10.2f}  '
              f'{m["selectivity"]:>+8.3f}  {m["decay_ratio"]:>+8.3f}  '
              f'{m["d_prime"]:>+8.2f}  {m["accuracy"]*100:>4.0f}%  '
              f'{m["composite_score"]:>6.3f}  {flag:>6}')
    print('  ' + '-' * 88)
    print(f'  Optimal DA: {da_list[opt_idx]:g} nM '
          f'(Persistence = {persistence_arr[opt_idx]:+.2f} Hz)')
    print(f'{"=" * 100}')
    print(f'\nTotal time: {time.time() - t_start:.1f}s')
    print(f'Outputs:    {save_dir}')


if __name__ == '__main__':
    main()
