#!/usr/bin/env python3
"""Pharmacological batch comparison -- multiple inverted-U curves overlaid.

Runs the inverted-U sweep under multiple conditions:
    1. Vehicle (normal D1 + D2)
    2. D1 antagonist (block D1)
    3. D2 antagonist (block D2)
    4. Both blocked

Usage:
    python -m wm_eval.run_batch_eval --ckpt <ckpt> --n-seeds 2
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import config
from simulation.runners import run_wm_simulation_from_checkpoint
from analysis.analyzer import PFCAnalyzer

from wm_eval.config_eval import (DA_CONCENTRATIONS, WM_PROTOCOL,
                                 PHARMA_CONDITIONS, DEFAULT_CHECKPOINT,
                                 build_wm_kwargs)
from wm_eval.metrics import compute_all_metrics
from wm_eval.plotting import plot_pharma_comparison


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument('--da-list', type=float, nargs='+', default=None)
    p.add_argument('--n-seeds', type=int, default=2)
    p.add_argument('--conditions', type=str, nargs='+',
                   default=list(PHARMA_CONDITIONS.keys()),
                   help='Subset of conditions: vehicle d1_block d2_block both_block')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--tag', type=str, default='pharma_compare')
    return p.parse_args()


def run_one(ckpt, da, seed, device, block_d1, block_d2):
    # NB: don't modify config.RANDOM_SEED (it's part of the checkpoint
    # fingerprint). Only advance torch/numpy RNGs.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    kwargs = build_wm_kwargs(da, WM_PROTOCOL,
                             block_d1=block_d1, block_d2=block_d2)
    data = run_wm_simulation_from_checkpoint(
        checkpoint_path=ckpt,
        device=device,
        **kwargs,
    )
    return compute_all_metrics(PFCAnalyzer(data), batch_idx=1)


def main():
    args = parse_args()
    t_start = time.time()

    if torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'[device] {device}')

    da_list = sorted(args.da_list or DA_CONCENTRATIONS)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(PROJ_ROOT, 'wm_eval', 'outputs',
                            f'{args.tag}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f'[output] {save_dir}')

    n_total = len(args.conditions) * len(da_list) * args.n_seeds
    print(f'\n{"=" * 60}')
    print(f'  Pharma comparison: {len(args.conditions)} cond x '
          f'{len(da_list)} DA x {args.n_seeds} seeds = {n_total} runs')
    print(f'{"=" * 60}')

    condition_results = {}
    for cond_name in args.conditions:
        if cond_name not in PHARMA_CONDITIONS:
            print(f'[warn] Unknown condition: {cond_name}, skipping')
            continue
        cfg = PHARMA_CONDITIONS[cond_name]
        print(f'\n{"=" * 60}')
        print(f'  Condition: {cond_name}  '
              f'(D1_block={cfg["block_d1"]}, D2_block={cfg["block_d2"]})')
        print(f'{"=" * 60}')

        sweep = {}
        for i, da in enumerate(da_list):
            per_seed = []
            for s in range(args.n_seeds):
                seed = 42 + s * 1000
                t0 = time.time()
                m = run_one(args.ckpt, da, seed, device,
                            cfg['block_d1'], cfg['block_d2'])
                print(f'  [{i+1}/{len(da_list)}] DA={da:g}nM seed={seed} '
                      f't={time.time()-t0:.1f}s | '
                      f'P={m["persistence"]:+.2f}Hz '
                      f'SI={m["selectivity"]:+.3f}')
                per_seed.append(m)
            keys = [k for k in per_seed[0].keys()
                    if k not in ('_pass', '_overall_pass', '_traces')
                    and isinstance(per_seed[0][k], (int, float))]
            sweep[da] = {
                'mean': {k: float(np.nanmean([m[k] for m in per_seed])) for k in keys},
                'std':  {k: float(np.nanstd([m[k]  for m in per_seed])) for k in keys},
            }
        condition_results[cond_name] = {
            'sweep_results': sweep,
            'color': cfg['color'],
        }

    # Plot for several metrics
    for metric in ['persistence', 'selectivity', 'decay_ratio', 'composite_score']:
        plot_pharma_comparison(
            condition_results,
            os.path.join(save_dir, f'pharma_compare_{metric}.png'),
            metric_key=metric,
        )

    # JSON summary
    summary = {
        'da_list': da_list,
        'n_seeds': args.n_seeds,
        'conditions': {
            cn: {'sweep_results': {str(d): cr['sweep_results'][d]
                                   for d in da_list}}
            for cn, cr in condition_results.items()
        },
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[save] {os.path.join(save_dir, 'summary.json')}")

    print(f'\nTotal: {time.time() - t_start:.1f}s | Output: {save_dir}')


if __name__ == '__main__':
    main()
