#!/usr/bin/env python3
"""Single-condition WM evaluation -- deep diagnostic for one DA level.

Usage:
    python -m wm_eval.run_single_eval --ckpt <ckpt> --da 8.0
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import config
from simulation.runners import run_wm_simulation_from_checkpoint
from analysis.analyzer import PFCAnalyzer
from analysis.plotting import plot_combined_raster, plot_combined_rates_all
from analysis.wm_plotting import (plot_wm_overview, plot_wm_rates_pools,
                                  print_wm_report)
from simulation.utils import save_raw_data

from wm_eval.config_eval import (WM_PROTOCOL, DEFAULT_CHECKPOINT,
                                 build_wm_kwargs)
from wm_eval.metrics import compute_all_metrics, format_metrics_report


def parse_args():
    p = argparse.ArgumentParser(description='Single-condition WM evaluation')
    p.add_argument('--ckpt', type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument('--da', type=float, default=8.0,
                   help='DA concentration (nM)')
    p.add_argument('--block-d1', action='store_true',
                   help='Block D1 receptors (D1 antagonist)')
    p.add_argument('--block-d2', action='store_true',
                   help='Block D2 receptors (D2 antagonist)')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--tag', type=str, default='single_eval')
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    if torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f'[device] {device}')

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cond_tag = ''
    if args.block_d1:
        cond_tag += '_d1block'
    if args.block_d2:
        cond_tag += '_d2block'
    save_dir = os.path.join(PROJ_ROOT, 'wm_eval', 'outputs',
                            f'{args.tag}_DA{args.da:g}nM{cond_tag}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    save_dir_p = Path(save_dir)

    print(f'\nSingle WM evaluation @ DA={args.da:g} nM '
          f'(block_d1={args.block_d1}, block_d2={args.block_d2})')
    print(f'[output] {save_dir}')

    p = WM_PROTOCOL
    kwargs = build_wm_kwargs(args.da, p,
                             block_d1=args.block_d1,
                             block_d2=args.block_d2)
    data = run_wm_simulation_from_checkpoint(
        checkpoint_path=args.ckpt,
        device=device,
        **kwargs,
    )

    save_raw_data(data, save_dir_p)

    analyzer = PFCAnalyzer(data)
    metrics = compute_all_metrics(analyzer, batch_idx=1)

    # Standard plots (reuse existing project plots)
    try:
        plot_combined_raster(analyzer, save_dir=save_dir)
        plot_combined_rates_all(analyzer, save_dir=save_dir)
    except Exception as e:
        print(f'[warn] standard plots failed: {e}')
    try:
        plot_wm_overview(analyzer, save_dir=save_dir)
        plot_wm_rates_pools(analyzer, save_dir=save_dir)
        print_wm_report(analyzer,
                        save_path=os.path.join(save_dir, 'wm_report.txt'))
    except Exception as e:
        print(f'[warn] WM plots failed: {e}')

    # WM metrics report
    label = f"DA={args.da:g} nM"
    if args.block_d1:
        label += " | D1 blocked"
    if args.block_d2:
        label += " | D2 blocked"
    report = format_metrics_report(metrics, condition_label=label)
    print(report)
    with open(os.path.join(save_dir, 'wm_metrics.txt'), 'w') as f:
        f.write(report)

    print(f'\nDone in {time.time() - t0:.1f}s | Output: {save_dir}')


if __name__ == '__main__':
    main()
