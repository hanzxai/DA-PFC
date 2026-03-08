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

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    target_groups = ['E-D1', 'E-D2', 'E-Other', 'I-D1', 'I-D2', 'I-Other']
    # 线型区分 E / I 亚群
    line_styles = {
        'E-D1': '-', 'E-D2': '-', 'E-Other': '-',
        'I-D1': '--', 'I-D2': '--', 'I-Other': '--',
    }
    lines_drawn = 0
    x_label = "Time (ms)"
    onset_line = analyzer.da_onset

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
        ls = line_styles.get(grp_name, '-')
        # E 亚群实线稍粗，I 亚群虚线稍细，透明度适当降低避免重叠遮挡
        lw = 1.8 if grp_name.startswith('E') else 1.4
        alpha = 0.85 if grp_name.startswith('E') else 0.70
        ax.plot(x_data, rate, color=color, label=grp_name,
                lw=lw, alpha=alpha, linestyle=ls)
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


def plot_raster_figure(analyzer: PFCAnalyzer, save_dir=None,
                       batch_idx: int = 1,
                       target_groups=None,
                       max_spikes_per_group: int = 50000):
    """
    绘制 Raster Plot (散点图), 每个点代表一次神经元发放。
    默认绘制全部 6 个亚群 (neuron ID 0-999)。

    Args:
        analyzer:              PFCAnalyzer 实例
        save_dir:              保存路径 (Path), None 则 plt.show()
        batch_idx:             Batch ID (0=Control, 1=Exp)
        target_groups:         要绘制的亚群列表, None 则绘制全部 6 个亚群
        max_spikes_per_group:  每个亚群最多绘制的 spike 数 (随机下采样上限)
    """
    if target_groups is None:
        # 全部 6 个亚群，覆盖 neuron ID 0-999
        target_groups = ['E-D1', 'E-D2', 'E-Other', 'I-D1', 'I-D2', 'I-Other']

    print(f"🎨 Plotting raster for Batch {batch_idx}...")

    # --- 获取 spike 数据 ---
    all_s = analyzer.data['spikes']
    mask_batch = all_s[:, 1] == batch_idx
    spikes_batch = all_s[mask_batch][:, [0, 2]].numpy()   # [time_step, neuron_id]

    if len(spikes_batch) == 0:
        print("❌ No spikes found for this batch.")
        return

    ts_ms = spikes_batch[:, 0] * analyzer.dt   # ms
    neuron_ids = spikes_batch[:, 1]

    # --- 自动 ms → s 转换 ---
    if analyzer.duration > 10000:
        x_data = ts_ms / 1000.0
        x_label = "Time (s)"
        onset_line = analyzer.da_onset / 1000.0
        x_max = analyzer.duration / 1000.0
    else:
        x_data = ts_ms
        x_label = "Time (ms)"
        onset_line = analyzer.da_onset
        x_max = analyzer.duration

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)

    rng = np.random.default_rng(42)
    any_drawn = False

    for grp_name in target_groups:
        if grp_name not in analyzer.groups:
            print(f"   ⚠️ Group '{grp_name}' not found, skipping.")
            continue

        valid_neurons = np.where(analyzer.groups[grp_name])[0]
        mask_grp = np.isin(neuron_ids, valid_neurons)
        grp_x = x_data[mask_grp]
        grp_y = neuron_ids[mask_grp]

        if len(grp_x) == 0:
            continue

        # 下采样（上限调高，尽量保留全部 spike）
        if len(grp_x) > max_spikes_per_group:
            idx = rng.choice(len(grp_x), size=max_spikes_per_group, replace=False)
            grp_x = grp_x[idx]
            grp_y = grp_y[idx]

        color = PFCAnalyzer.COLORS.get(grp_name, 'black')
        # 点大小 s=3，alpha=0.8，让每个 spike 清晰可见
        ax.scatter(grp_x, grp_y, s=3, color=color,
                   label=f"{grp_name} ({int(np.sum(mask_grp))} spikes)",
                   alpha=0.8, linewidths=0, rasterized=True)
        any_drawn = True

    if not any_drawn:
        print("❌ ERROR: No spikes drawn! Check batch_idx or group names.")
        plt.close(fig)
        return

    # --- E/I 分界线 ---
    ax.axhline(analyzer.N_E - 0.5, color='gray', linestyle='-',
               linewidth=1.0, alpha=0.6)
    ax.text(x_max * 0.01, analyzer.N_E - 0.5 + 5,
            "← I neurons", fontsize=9, color='gray', va='bottom')
    ax.text(x_max * 0.01, analyzer.N_E - 0.5 - 5,
            "E neurons →", fontsize=9, color='gray', va='top')

    # --- DA Onset 竖线 ---
    if onset_line > 0 and batch_idx == 1:
        ax.axvline(onset_line, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.text(onset_line, analyzer.N * 0.98,
                " DA Onset", fontsize=10, va='top', ha='left', color='black')

    # --- 美化 ---
    ax.set_xlim(0, x_max)
    ax.set_ylim(-1, analyzer.N)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel("Neuron ID", fontsize=13)
    batch_label = "Control" if batch_idx == 0 else f"Exp ({analyzer.da_level} nM)"
    ax.set_title(f"Raster Plot — {batch_label}  (all {analyzer.N} neurons)", fontsize=14)
    ax.grid(True, axis='x', linestyle='--', alpha=0.2)
    legend = ax.legend(loc='upper right', fontsize=9,
                       markerscale=4, framealpha=0.85)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
        handle._sizes = [30]

    plt.tight_layout()

    # --- 保存或显示 ---
    if save_dir:
        save_path = save_dir / f"raster_batch_{batch_idx}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"📊 Raster saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def _plot_group_rates(analyzer: PFCAnalyzer, group_names: list, title_prefix: str,
                      save_dir=None, batch_idx: int = 1, time_win: float = 100.0,
                      filename: str = "rates.png"):
    """
    内部通用函数: 绘制指定亚群列表的发放率曲线。
    """
    print(f"🎨 Plotting {title_prefix} firing rates for Batch {batch_idx}...")

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    x_label = "Time (ms)"
    onset_line = analyzer.da_onset
    lines_drawn = 0

    for grp_name in group_names:
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
        ax.plot(x_data, rate, color=color, label=grp_name, lw=1.8, alpha=0.85)
        lines_drawn += 1

    if lines_drawn == 0:
        print(f"❌ ERROR: No lines drawn for {title_prefix}!")
        plt.close(fig)
        return

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Firing Rate (Hz)", fontsize=12)
    batch_label = "Control" if batch_idx == 0 else f"Exp ({analyzer.da_level} nM)"
    ax.set_title(f"{title_prefix} Population Activities — {batch_label}", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    if onset_line > 0:
        ax.axvline(onset_line, color='black', linestyle='--', alpha=0.6)
        ax.text(onset_line, ax.get_ylim()[1] * 0.95, " DA Onset", fontsize=10, va='top')

    if save_dir:
        save_path = save_dir / filename
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_excitatory_rates(analyzer: PFCAnalyzer, save_dir=None,
                          batch_idx: int = 1, time_win: float = 100.0):
    """
    单独绘制所有 E（兴奋性）亚群的发放率曲线。
    输出文件: firing_rates_E_batch_{batch_idx}.png
    """
    _plot_group_rates(
        analyzer,
        group_names=['E-D1', 'E-D2', 'E-Other'],
        title_prefix="Excitatory (E)",
        save_dir=save_dir,
        batch_idx=batch_idx,
        time_win=time_win,
        filename=f"firing_rates_E_batch_{batch_idx}.png",
    )


def plot_inhibitory_rates(analyzer: PFCAnalyzer, save_dir=None,
                          batch_idx: int = 1, time_win: float = 100.0):
    """
    单独绘制所有 I（抑制性）亚群的发放率曲线。
    输出文件: firing_rates_I_batch_{batch_idx}.png
    """
    _plot_group_rates(
        analyzer,
        group_names=['I-D1', 'I-D2', 'I-Other'],
        title_prefix="Inhibitory (I)",
        save_dir=save_dir,
        batch_idx=batch_idx,
        time_win=time_win,
        filename=f"firing_rates_I_batch_{batch_idx}.png",
    )
