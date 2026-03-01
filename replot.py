# replot.py
"""
从已保存的 raw_data.pkl 重新绘图。
用法:
  python replot.py outputs/exp_2026-01-12_02-45-24/raw_data.pkl
  python replot.py outputs/exp_2026-01-12_02-45-24/raw_data.pkl --batch 1
"""
import argparse
import pickle
from pathlib import Path

from analysis.analyzer import PFCAnalyzer
from analysis.plotting import plot_population_rates


def parse_args():
    parser = argparse.ArgumentParser(description="从已有数据重新绘图")
    parser.add_argument("data_path", type=str,
                        help="raw_data.pkl 文件路径")
    parser.add_argument("--batch", type=int, default=0,
                        help="Batch ID (0=Control, 1=Exp), 默认 0")
    parser.add_argument("--time_win", type=float, default=100.0,
                        help="时间窗宽度 (ms), 默认 100")
    return parser.parse_args()


def main():
    args = parse_args()
    file_path = Path(args.data_path)

    if not file_path.exists():
        print(f"❌ 找不到文件: {file_path}")
        return

    print(f"📂 Loading data from {file_path} ...")
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    analyzer = PFCAnalyzer(data)
    save_dir = file_path.parent

    print("🎨 Plotting...")
    plot_population_rates(analyzer, save_dir=save_dir,
                          batch_idx=args.batch, time_win=args.time_win)
    print(f"✅ Done! Check the plot in: {save_dir}")


if __name__ == "__main__":
    main()
