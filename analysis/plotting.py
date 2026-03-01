# analysis/plotting.py
"""
独立绘图函数: 接收 PFCAnalyzer 实例, 生成并保存图表。
使用 Agg 后端以兼容无头服务器。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analysis.analyzer import PFCAnalyzer


def plot_population_rates(analyzer: PFCAnalyzer, save_dir=None,
                          batch_idx: int = 1, time_win: float = 100.0):
    """
    绘制群体发放率曲线。

    Args:
        analyzer:   PFCAnalyzer 实例
        save_dir:   保存路径 (Path), None 则 plt.show()
        batch_idx:  Batch ID (0=Control, 1=Exp)
        time_win:   时间窗宽度 (ms), 长仿真建议 100~500
    """
    print(f"🎨 Plotting firing rates for Batch {batch_idx}...")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    target_groups = ['E-D1', 'E-D2', 'I-D1']
    lines_drawn = 0

    for grp_name in target_groups:
        if grp_name not in analyzer.groups:
            print(f"   ⚠️ Group '{grp_name}' not found, skipping.")
            continue

        centers, rate = analyzer.compute_group_rate(batch_idx, grp_name, time_win=time_win)
        if rate is None or len(rate) == 0:
            continue

        # 自动 ms → s 转换
        if centers[-1] > 10000:
            x_data = centers / 1000.0
            x_label = "Time (s)"
            onset_line = analyzer.da_onset / 1000.0
        else:
            x_data = centers
            x_label = "Time (ms)"
            onset_line = analyzer.da_onset

        color = PFCAnalyzer.COLORS.get(grp_name, 'k')
        ax.plot(x_data, rate, color=color, label=grp_name, lw=1.5, alpha=0.9)
        lines_drawn += 1

    if lines_drawn == 0:
        print("❌ ERROR: No lines were drawn! Check batch_idx or group names.")
        plt.close(fig)
        return

    # 美化
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Firing Rate (Hz)", fontsize=12)
    batch_label = "Control" if batch_idx == 0 else f"Exp ({analyzer.da_level} nM)"
    ax.set_title(f"Population Activities — {batch_label}", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    # DA Onset 竖线
    if onset_line > 0:
        ax.axvline(onset_line, color='black', linestyle='--', alpha=0.6)
        ax.text(onset_line, ax.get_ylim()[1] * 0.95, " DA Onset", fontsize=10, va='top')

    # 保存或显示
    if save_dir:
        save_path = save_dir / f"firing_rates_batch_{batch_idx}.png"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()
