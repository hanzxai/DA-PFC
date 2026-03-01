# analysis/plotting.py
import matplotlib
# 强制使用非交互式后端，防止在服务器上报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

def plot_population_rates(analyzer, save_dir=None, batch_idx=1, time_win=100.0):
    """
    画群体发放率曲线
    :param analyzer: PFCAnalyzer 实例
    :param save_dir: 保存路径 (Path object)
    :param batch_idx: Batch ID (0=Control, 1=Exp)
    :param time_win: 平滑窗口 (ms)。对于 100s 的长仿真，建议设大一点 (e.g. 100ms - 500ms)
    """
    print(f"🎨 Plotting logic started for Batch {batch_idx}...")
    
    # 1. 准备画布
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    # 2. 定义要画的亚群
    # 确保这些名字在 analyzer.groups 里存在
    target_groups = ['E-D1', 'E-D2', 'I-D1']
    
    # 颜色映射
    colors = {
        'E-D1': '#d62728', # Red
        'E-D2': '#1f77b4', # Blue
        'E-Other': 'gray',
        'I-D1': '#ff7f0e', # Orange
        'I-D2': '#9467bd'  # Purple
    }
    
    lines_drawn = 0
    
    # 3. 循环画线 (真正的逻辑在这里！)
    for grp_name in target_groups:
        if grp_name not in analyzer.groups:
            print(f"   ⚠️ Group {grp_name} not found in masks, skipping.")
            continue
            
        # 调用分析器计算数据
        print(f"   - Computing rate for {grp_name}...")
        centers, rate = analyzer.compute_group_rate(batch_idx, grp_name, time_win=time_win)
        
        # 检查数据有效性
        if rate is None or len(rate) == 0:
            print(f"   ⚠️ No data for {grp_name}")
            continue
            
        # --- [优化] 自动单位转换 (ms -> s) ---
        # 如果总时长超过 10,000ms (10s)，把 X 轴换成 秒(s)
        if centers[-1] > 10000:
            x_data = centers / 1000.0
            x_label = "Time (s)"
            # 标记给药时间 (从 ms 转 s)
            onset_line = analyzer.da_onset / 1000.0
        else:
            x_data = centers
            x_label = "Time (ms)"
            onset_line = analyzer.da_onset
            
        # 正式画图
        mean_rate = np.mean(rate)
        print(f"   - Drawing {grp_name}: Mean Rate = {mean_rate:.2f} Hz")
        ax.plot(x_data, rate, color=colors.get(grp_name, 'k'), label=f"{grp_name}", lw=1.5, alpha=0.9)
        lines_drawn += 1

    # 4. 如果一条线都没画出来，打印警告
    if lines_drawn == 0:
        print("❌ ERROR: No lines were drawn! Check batch_idx or group names.")
        plt.close(fig)
        return

    # 5. 美化图表
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Firing Rate (Hz)", fontsize=12)
    ax.set_title(f"Population Activities (Batch {batch_idx})", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    # 画 DA Onset 竖线
    if onset_line > 0:
        ax.axvline(onset_line, color='black', linestyle='--', alpha=0.6)
        ax.text(onset_line, ax.get_ylim()[1]*0.95, " DA Onset", fontsize=10, va='top')

    # 6. 保存
    if save_dir:
        filename = f"firing_rates_batch_{batch_idx}.png"
        save_path = save_dir / filename
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
        plt.close(fig)
    else:
        # 本地调试用
        plt.show()