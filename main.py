# main.py
"""
DA-PFC 仿真主入口
支持命令行参数:
  python main.py --duration 100000 --da 3.0
  python main.py                          # 使用默认值
"""
import argparse
import time
import config
from simulation.runners import run_simulation_d1_d2_kinetics
from analysis.analyzer import PFCAnalyzer
from analysis.plotting import plot_population_rates, plot_raster_figure, plot_excitatory_rates, plot_inhibitory_rates
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
    return parser.parse_args()


def main():
    args = parse_args()
    t_total_start = time.time()

    # 1. 准备实验文件夹
    save_dir = setup_experiment_folder()

    # 2. 实验配置
    exp_config = {
        "duration": args.duration,
        "target_da": args.da,
        "dt": config.DT,
        "note": f"D1+D2 受体动力学 (D1 Tau_on={config.TAU_ON_D1}ms, D2 Tau_on={config.TAU_ON_D2}ms, DA={args.da}nM)",
    }
    save_args(exp_config, save_dir)

    # 3. 运行仿真
    print(f"🚀 Starting Experiment: {exp_config['note']}")
    t_sim_start = time.time()
    data = run_simulation_d1_d2_kinetics(
        duration=args.duration,
        target_da=args.da,
    )
    t_sim_elapsed = time.time() - t_sim_start

    # 4. 保存原始数据
    t_save_start = time.time()
    save_raw_data(data, save_dir)
    t_save_elapsed = time.time() - t_save_start

    # 5. 分析与绘图
    print("🎨 Generating Plots...")
    t_plot_start = time.time()
    analyzer = PFCAnalyzer(data)
    plot_population_rates(analyzer, save_dir=save_dir, batch_idx=args.batch)
    plot_excitatory_rates(analyzer, save_dir=save_dir, batch_idx=args.batch)
    plot_inhibitory_rates(analyzer, save_dir=save_dir, batch_idx=args.batch)
    plot_raster_figure(analyzer, save_dir=save_dir, batch_idx=args.batch)
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
