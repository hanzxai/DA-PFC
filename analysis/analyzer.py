# analysis/analyzer.py
"""
PFC 网络仿真数据分析器
提供: 亚群发放率计算、FFT 频率分析、Raster 绘图
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch


class PFCAnalyzer:
    """解析仿真数据包, 提供群体发放率、频谱分析等功能。"""

    # 亚群默认颜色
    COLORS = {
        'E-D1': '#d62728',   # Red
        'E-D2': '#1f77b4',   # Blue
        'E-Other': 'gray',
        'I-D1': '#ff7f0e',   # Orange
        'I-D2': '#9467bd',   # Purple
        'I-Other': '#2ca02c', # Green
    }

    def __init__(self, data_source: dict):
        """
        Args:
            data_source: 由 simulation/runners 返回的数据包 (dict)
        """
        self.data = data_source
        self.cfg = data_source['config']

        # 基本参数
        self.dt = self.cfg['dt']
        self.duration = self.cfg['duration']
        self.N_E = self.cfg['N_E']
        self.N_I = self.cfg['N_I']
        self.N = self.N_E + self.N_I
        self.da_onset = self.cfg.get('da_onset', 1000.0)
        self.da_level = self.cfg.get('da_level', 10.0)

        # 构建亚群 Mask
        self._build_group_masks()
        print("📊 Analyzer initialized.")

    # ------------------------------------------------------------------
    #  内部方法
    # ------------------------------------------------------------------

    def _build_group_masks(self):
        """根据 D1/D2 mask 与 E/I 边界, 构建 6 个亚群的布尔索引。"""
        m_d1 = self.data['masks']['d1'].numpy()
        m_d2 = self.data['masks']['d2'].numpy()
        ids = np.arange(self.N)
        is_exc = ids < self.N_E
        is_inh = ~is_exc

        self.groups = {
            'E-D1':    is_exc & m_d1,
            'E-D2':    is_exc & m_d2,
            'E-Other': is_exc & (~m_d1) & (~m_d2),
            'I-D1':    is_inh & m_d1,
            'I-D2':    is_inh & m_d2,
            'I-Other': is_inh & (~m_d1) & (~m_d2),
        }

    def _get_spikes_for_batch(self, batch_idx: int) -> np.ndarray:
        """获取指定 Batch 的脉冲数据, 返回 (N_spikes, 2) -> [time_step, neuron_id]"""
        all_s = self.data['spikes']
        mask = all_s[:, 1] == batch_idx
        return all_s[mask][:, [0, 2]].numpy()

    # ------------------------------------------------------------------
    #  核心计算: 群体发放率
    # ------------------------------------------------------------------

    def compute_group_rate(self, batch_idx: int, group_name: str,
                           time_win: float = 5.0, sigma: float = 1.5):
        """
        计算指定亚群的平滑放电率曲线。

        Args:
            batch_idx:  Batch ID (0=Control, 1=Exp)
            group_name: 亚群名称 (e.g. 'E-D1')
            time_win:   时间窗宽度 (ms)
            sigma:      高斯平滑系数

        Returns:
            (centers, rate_smooth) — 时间中心点 (ms) 与平滑后的发放率 (Hz)
        """
        if group_name not in self.groups:
            print(f"⚠️ Warning: Group '{group_name}' not found.")
            return None, None

        mask_ids = self.groups[group_name]
        count = int(np.sum(mask_ids))
        bins = np.arange(0, self.duration + time_win, time_win)
        centers = (bins[:-1] + bins[1:]) / 2

        if count == 0:
            return centers, np.zeros_like(centers)

        spikes = self._get_spikes_for_batch(batch_idx)
        if len(spikes) == 0:
            return centers, np.zeros_like(centers)

        ts = spikes[:, 0] * self.dt  # ms
        ids = spikes[:, 1]

        valid_neurons = np.where(mask_ids)[0]
        mask_spikes = np.isin(ids, valid_neurons)

        h, _ = np.histogram(ts[mask_spikes], bins=bins)
        rate = h / (time_win / 1000.0 * count)  # Hz

        return centers, gaussian_filter1d(rate, sigma=sigma)

    # ------------------------------------------------------------------
    #  核心计算: FFT 频率分析
    # ------------------------------------------------------------------

    def calculate_frequency_in_window(self, rate_curve: np.ndarray,
                                      time_win: float, start_ms: float, end_ms: float) -> float:
        """对 rate_curve 的指定时间片段进行 Welch FFT, 返回主频 (Hz)。"""
        idx_start = max(0, int(start_ms / time_win))
        idx_end = min(len(rate_curve), int(end_ms / time_win))

        segment = rate_curve[idx_start:idx_end]
        if len(segment) < 10:
            return 0.0

        sig = segment - np.mean(segment)
        fs = 1000.0 / time_win
        freqs, power = welch(sig, fs=fs, nperseg=len(sig), nfft=4096)

        mask_roi = freqs > 1.0
        if np.any(mask_roi):
            return float(freqs[mask_roi][np.argmax(power[mask_roi])])
        return 0.0

    # ------------------------------------------------------------------
    #  高级分析: 频率报告
    # ------------------------------------------------------------------

    def print_frequency_report(self, target_group: str = 'E-D1', time_win: float = 5.0):
        """打印 Control vs Experiment 的分阶段频率对比报告。"""
        _, r_ctrl = self.compute_group_rate(0, target_group, time_win)
        _, r_exp = self.compute_group_rate(1, target_group, time_win)
        if r_ctrl is None or r_exp is None:
            return

        onset = int(self.da_onset)
        dur = int(self.duration)

        phases = []
        if onset > 200:
            phases.append(("Baseline", 0, onset))
        if dur - onset > 500:
            phases.append(("DA Phase", onset, min(onset + 1000, dur)))
        if dur > onset + 1000:
            phases.append(("Late DA", onset + 1000, dur))

        print("\n" + "=" * 50)
        print(f" 📊 频率分析报告 (Target: {target_group})")
        print("=" * 50)
        print(f"{'Phase':<12} | {'Time (ms)':<15} | {'Ctrl (Hz)':<10} | {'Exp (Hz)':<10} | {'Diff':<10}")
        print("-" * 68)

        for name, start, end in phases:
            f_ctrl = self.calculate_frequency_in_window(r_ctrl, time_win, start, end)
            f_exp = self.calculate_frequency_in_window(r_exp, time_win, start, end)
            print(f"{name:<12} | {f'{start}-{end}':<15} | {f_ctrl:<10.2f} | {f_exp:<10.2f} | {f_exp - f_ctrl:+.2f}")

        print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    #  绘图: Raster
    # ------------------------------------------------------------------

    def plot_raster(self, batch_idx: int = 1, target_groups=None,
                    ax=None, title: str = None):
        """绘制 Raster Plot。"""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        spikes = self._get_spikes_for_batch(batch_idx)
        if len(spikes) == 0:
            return

        ts = spikes[:, 0] * self.dt
        ids = spikes[:, 1]

        if target_groups is None:
            target_groups = list(self.groups.keys())

        for grp_name in target_groups:
            if grp_name not in self.groups:
                continue
            valid_neurons = np.where(self.groups[grp_name])[0]
            mask_spikes = np.isin(ids, valid_neurons)
            if np.any(mask_spikes):
                ax.scatter(ts[mask_spikes], ids[mask_spikes], s=1.0,
                           color=self.COLORS.get(grp_name, 'black'),
                           label=grp_name, alpha=0.6)

        if self.da_onset > 0 and batch_idx == 1:
            ax.axvline(self.da_onset, color='k', linestyle='--')

        batch_label = 'Control' if batch_idx == 0 else f'Exp ({self.da_level} nM)'
        ax.set_title(title or f"Raster Plot - {batch_label}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron ID")
        ax.set_xlim(0, self.duration)
