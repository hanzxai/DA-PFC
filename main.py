# main.py
"""
DA-PFC 仿真主入口 (SNN Simulation)

使用方法示例 (Usage Examples):
  1. 使用默认参数运行 (GPU 0, 100秒, 3.0 nM DA):
     python main.py

  2. 指定 DA 浓度为 4.0 nM:
     python main.py --da 4.0

  3. 指定自定义仿真时间 (如 50000 ms) 和 GPU 卡号 (如使用 GPU 1):
     python main.py --duration 50000 --da 3.0 --gpu 1

  4. 如果因为显卡被占用或没有显卡，想用或被迫回退到 CPU:
     python main.py --gpu 999  # 无效 GPU 号会自动回退到 CPU 进行计算

支持的主要命令行参数:
  --duration : 仿真总时长 (ms)，默认 100000
  --da       : 目标给药浓度 (nM)，默认 3.0
  --batch    : 绘图 Batch ID (0=Control, 1=Exp)，默认 1
  --gpu      : 运行的 GPU 卡号，默认 0
"""
import argparse
import time
import torch
import config
from simulation.runners import run_simulation_d1_d2_kinetics
from analysis.analyzer import PFCAnalyzer
from analysis.plotting import (plot_population_rates, plot_raster_figure,
                                plot_excitatory_rates, plot_inhibitory_rates,
                                plot_population_rates_zoom, plot_excitatory_rates_zoom,
                                plot_inhibitory_rates_zoom, plot_raster_figure_zoom)
from utils import setup_experiment_folder, save_args, save_raw_data


def _fmt_elapsed(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS.ss"""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:05.2f}s"
    if m > 0:
        return f"{m}m {s:05.2f}s"
    return f"{s:.2f}s"


def parse_args():
    parser = argparse.ArgumentParser(description="DA-PFC SNN Simulation")
    parser.add_argument("--duration", type=float, default=100000.0,
                        help="仿真总时长 (ms), 默认 100000 (100s)")
    parser.add_argument("--da", type=float, default=3.0,
                        help="给药浓度 (nM), 默认 3.0")
    parser.add_argument("--batch", type=int, default=1,
                        help="绘图 Batch ID (0=Control, 1=Exp), 默认 1")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU卡号, 默认 0. 无GPU或输入无效将回退到CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    t_total_start = time.time()

    # 决定设备
    if torch.cuda.is_available():
        if args.gpu >= 0 and args.gpu < torch.cuda.device_count():
            device = torch.device(f"cuda:{args.gpu}")
        else:
            print(f"⚠️ 指定的 GPU {args.gpu} 无效，将回退到 CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"🔧 Using device: {device}")

    # 1. 准备实验文件夹
    save_dir = setup_experiment_folder()

    # 2. 实验配置
    exp_config = {
        "duration": args.duration,
        "target_da": args.da,
        "dt": config.DT,
        "device": str(device),
        "note": f"D1+D2 受体动力学 (D1 Tau_on={config.TAU_ON_D1}ms, D2 Tau_on={config.TAU_ON_D2}ms, DA={args.da}nM)",
    }
    save_args(exp_config, save_dir)

    # 3. 运行仿真
    print(f"🚀 Starting Experiment: {exp_config['note']}")
    t_sim_start = time.time()
    data = run_simulation_d1_d2_kinetics(
        duration=args.duration,
        target_da=args.da,
        device=device,
    )
    t_sim_elapsed = time.time() - t_sim_start

    # 4. 保存原始数据
    t_save_start = time.time()
    save_raw_data(data, save_dir)
    t_save_elapsed = time.time() - t_save_start

    # 5. 分析与绘图
    print("🎨 Generating Plots...")
    t_plot_start = time.time()
    import numpy as np
    analyzer = PFCAnalyzer(data)

    # Pre-compute unified zoom y-axis limit across ALL batches and ALL subgroups
    # so that batch_0 and batch_1 zoom plots share the same y scale
    _t_start, _t_end = 9500.0, 13000.0
    _zoom_ymax = 0.0
    for _bid in [0, 1]:
        for _grp in ['E-D1', 'E-D2', 'E-Other', 'I-D1', 'I-D2', 'I-Other']:
            if _grp not in analyzer.groups:
                continue
            _centers, _rate = analyzer.compute_group_rate(_bid, _grp, time_win=5.0)
            if _rate is None or len(_rate) == 0:
                continue
            _mask = (_centers >= _t_start) & (_centers <= _t_end)
            if np.any(_mask):
                _zoom_ymax = max(_zoom_ymax, float(np.nanmax(_rate[_mask])))
    _zoom_ylim = _zoom_ymax * 1.15 if _zoom_ymax > 0 else None

    # 同时绘制 Batch 0 (Control) 和 Batch 1 (Exp) 的图
    for batch_id in [0, 1]:
        print(f"\n--- Batch {batch_id} ({'Control' if batch_id == 0 else 'Exp'}) ---")
        plot_population_rates(analyzer, save_dir=save_dir, batch_idx=batch_id)
        plot_excitatory_rates(analyzer, save_dir=save_dir, batch_idx=batch_id)
        plot_inhibitory_rates(analyzer, save_dir=save_dir, batch_idx=batch_id)
        plot_raster_figure(analyzer, save_dir=save_dir, batch_idx=batch_id)
        # Short time-scale zoom-in plots — use shared ylim for cross-batch consistency
        plot_population_rates_zoom(analyzer, save_dir=save_dir, batch_idx=batch_id, ylim=_zoom_ylim)
        plot_excitatory_rates_zoom(analyzer, save_dir=save_dir, batch_idx=batch_id, ylim=_zoom_ylim)
        plot_inhibitory_rates_zoom(analyzer, save_dir=save_dir, batch_idx=batch_id, ylim=_zoom_ylim)
        plot_raster_figure_zoom(analyzer, save_dir=save_dir, batch_idx=batch_id)

    # Frequency analysis reports — save to file
    import os
    analyzer.save_report(os.path.join(save_dir, "analysis_report.txt"))

    t_plot_elapsed = time.time() - t_plot_start

    # 6. 总计时报告
    t_total_elapsed = time.time() - t_total_start
    print("\n" + "=" * 50)
    print("⏱️  计时报告")
    print("=" * 50)
    print(f"  仿真计算:  {_fmt_elapsed(t_sim_elapsed)}")
    print(f"  数据保存:  {_fmt_elapsed(t_save_elapsed)}")
    print(f"  分析绘图:  {_fmt_elapsed(t_plot_elapsed)}")
    print(f"  ────────────────────────")
    print(f"  总计:      {_fmt_elapsed(t_total_elapsed)}")
    print("=" * 50)
    print(f"\n✅ Experiment Complete. Results saved in: {save_dir}")


if __name__ == "__main__":
    main()
