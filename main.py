# main.py
"""
DA-PFC 仿真主入口
支持命令行参数:
  python main.py --duration 100000 --da 3.0
  python main.py                          # 使用默认值
"""
import argparse
import config
from simulation.runners import run_simulation_d1_kinetics
from analysis.analyzer import PFCAnalyzer
from analysis.plotting import plot_population_rates
from utils import setup_experiment_folder, save_args, save_raw_data


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

    # 1. 准备实验文件夹
    save_dir = setup_experiment_folder()

    # 2. 实验配置
    exp_config = {
        "duration": args.duration,
        "target_da": args.da,
        "dt": config.DT,
        "note": f"D1 受体动力学 (Tau_on={config.TAU_ON_D1}ms, DA={args.da}nM)",
    }
    save_args(exp_config, save_dir)

    # 3. 运行仿真
    print(f"🚀 Starting Experiment: {exp_config['note']}")
    data = run_simulation_d1_kinetics(
        duration=args.duration,
        target_da=args.da,
    )

    # 4. 保存原始数据
    save_raw_data(data, save_dir)

    # 5. 分析与绘图
    print("🎨 Generating Plots...")
    analyzer = PFCAnalyzer(data)
    plot_population_rates(analyzer, save_dir=save_dir, batch_idx=args.batch)

    print(f"\n✅ Experiment Complete. Results saved in: {save_dir}")


if __name__ == "__main__":
    main()
