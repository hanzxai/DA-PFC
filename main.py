# main.py
import argparse
import matplotlib.pyplot as plt
import config
from simulation.runners import run_simulation_d1_kinetics
from analysis.analyzer import PFCAnalyzer
from analysis.plotting import plot_population_rates
from utils import setup_experiment_folder, save_args, save_raw_data

def main():
    # 1. 准备实验文件夹
    save_dir = setup_experiment_folder()
    
    # 2. 实验参数配置
    # 注意：为了验证 Tau ~ 30s 的动力学，我们需要跑很久
    # 推荐跑 100,000 ms (100秒)，这样能看到 alpha 从 0 爬升到 0.95 的全过程
    exp_config = {
        "duration": 100000.0,  # 100秒
        "target_da": 3.0,     # 给药浓度 20 nM (保证 S 接近 1.0)
        "dt": config.DT,
        "note": "验证 D1 受体动力学 (Tau_on=30.9s)"
    }
    
    # 保存配置
    save_args(exp_config, save_dir)
    
    # 3. 运行仿真 (使用新的动力学 Runner)
    print(f"🚀 Starting Experiment: {exp_config['note']}")
    data = run_simulation_d1_kinetics(
        duration=exp_config["duration"], 
        target_da=exp_config["target_da"]
    )
    
    # 4. 保存原始数据
    save_raw_data(data, save_dir)
    
    # 5. 分析与绘图
    print("🎨 Generating Plots...")
    analyzer = PFCAnalyzer(data)
    
    # 画出 Batch 1 (实验组) 的结果
    # 注意：因为跑了 100秒，生成的图横坐标会很长，建议放大看
    plot_population_rates(analyzer, save_dir=save_dir, batch_idx=1)
    
    print(f"\n✅ Experiment Complete. Check results in: {save_dir}")

if __name__ == "__main__":
    main()