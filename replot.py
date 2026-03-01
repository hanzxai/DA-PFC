# replot.py
import pickle
import os
import sys
from pathlib import Path

# 确保能找到 analysis 模块
sys.path.append(os.getcwd())

from analysis.analyzer import PFCAnalyzer
from analysis.plotting import plot_population_rates

# 👇 这里填你之前跑成功的那个数据文件的路径
# (就是你刚才用 inspect_data.py 检查过的那个文件)
# DATA_PATH = "/home/zhanghongye/code/PFC_SNN_Project/outputs/exp_2025-12-29_03-37-53/raw_data.pkl"
DATA_PATH="/home/zhanghongye/code/PFC_SNN_Project/outputs/exp_2026-01-12_02-45-24/raw_data.pkl"
def main():
    file_path = Path(DATA_PATH)
    if not file_path.exists():
        print(f"❌ 找不到文件: {file_path}")
        return

    print(f"📂 Loading data from {file_path} ...")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    # 1. 初始化分析器
    analyzer = PFCAnalyzer(data)
    print("✅ Analyzer initialized.")
    
    # 2. 确定保存图片的目录 (就保存在数据旁边)
    save_dir = file_path.parent
    
    # 3. 调用画图函数 (注意这里是正确的调用方式)
    print("🎨 Plotting...")
    
    # 时间窗设为 1000ms (1秒)，因为总时长是 100秒，平滑一点好看
    plot_population_rates(analyzer, save_dir=save_dir, batch_idx=0)
    
    print(f"✅ Done! Check the plot in: {save_dir}")

if __name__ == "__main__":
    main()