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
        'All-E': '#e377c2',  # Pink
        'All-I': '#17becf',  # Cyan
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
            'All-E':   is_exc,
            'All-I':   is_inh,
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
        nfft = max(4096, len(sig))
        freqs, power = welch(sig, fs=fs, nperseg=len(sig), nfft=nfft)

        mask_roi = freqs > 1.0
        if np.any(mask_roi):
            return float(freqs[mask_roi][np.argmax(power[mask_roi])])
        return 0.0

    # ------------------------------------------------------------------
    #  高级分析: 频率报告
    # ------------------------------------------------------------------

    def print_frequency_report(self, target_group: str = 'E-D1', time_win: float = 5.0) -> str:
        """打印并返回 Control vs Experiment 的分阶段频率对比报告。"""
        _, r_ctrl = self.compute_group_rate(0, target_group, time_win)
        _, r_exp = self.compute_group_rate(1, target_group, time_win)
        if r_ctrl is None or r_exp is None:
            return ""

        onset = int(self.da_onset)
        dur = int(self.duration)

        phases = []
        if onset > 200:
            phases.append(("Baseline", 0, onset))
        if dur - onset > 500:
            phases.append(("DA Phase", onset, min(onset + 1000, dur)))
        if dur > onset + 1000:
            phases.append(("Late DA", onset + 1000, dur))

        lines = []
        lines.append("\n" + "=" * 50)
        lines.append(f" 📊 频率分析报告 (Target: {target_group})")
        lines.append("=" * 50)
        lines.append(f"{'Phase':<12} | {'Time (ms)':<15} | {'Ctrl (Hz)':<10} | {'Exp (Hz)':<10} | {'Diff':<10}")
        lines.append("-" * 68)

        for name, start, end in phases:
            f_ctrl = self.calculate_frequency_in_window(r_ctrl, time_win, start, end)
            f_exp = self.calculate_frequency_in_window(r_exp, time_win, start, end)
            diff = f_exp - f_ctrl
            # Mark baseline consistency check
            if name == "Baseline":
                status = " ✅" if abs(diff) < 0.5 else " ⚠️ MISMATCH"
            else:
                status = ""
            lines.append(f"{name:<12} | {f'{start}-{end}':<15} | {f_ctrl:<10.2f} | {f_exp:<10.2f} | {diff:+.2f}{status}")

        lines.append("=" * 50 + "\n")

        text = "\n".join(lines)
        print(text)
        return text

    def print_fft_comparison_report(self, time_win: float = 5.0) -> str:
        """Print and return a side-by-side FFT frequency comparison table for Control (B0) vs Exp (B1).

        Analyses 8 groups: E-D1, E-D2, E-Other, All-E, I-D1, I-D2, I-Other, All-I.
        Each group is split into Baseline [0, da_onset] and Post-DA [da_onset, duration].
        """
        fft_groups = ['E-D1', 'E-D2', 'E-Other', 'All-E',
                      'I-D1', 'I-D2', 'I-Other', 'All-I']
        da_onset = self.da_onset
        dur = self.duration

        # Collect FFT results: {group: {batch: {phase: freq}}}
        fft_results = {}
        for grp_name in fft_groups:
            fft_results[grp_name] = {}
            for batch_id in [0, 1]:
                centers, rate = self.compute_group_rate(batch_id, grp_name, time_win=time_win)
                if rate is None or len(rate) == 0:
                    fft_results[grp_name][batch_id] = {'baseline': 0.0, 'post_da': 0.0}
                    continue
                freq_bl = self.calculate_frequency_in_window(rate, time_win, 0.0, da_onset)
                freq_pd = self.calculate_frequency_in_window(rate, time_win, da_onset, dur)
                fft_results[grp_name][batch_id] = {'baseline': freq_bl, 'post_da': freq_pd}

        # Build table lines
        w = 100
        lines = []
        lines.append("\n" + "=" * w)
        lines.append("  📐 FFT Frequency Analysis — Control (B0) vs Exp (B1)")
        lines.append("=" * w)
        header = (f"  {'Group':<10} │ {'B0 Baseline':>12} {'B0 Post-DA':>12} │"
                  f" {'B1 Baseline':>12} {'B1 Post-DA':>12} │"
                  f" {'BL Diff':>10}  {'Status':>6}")
        lines.append(header)
        lines.append(f"  {'─'*10}─┼─{'─'*12}─{'─'*12}─┼─{'─'*12}─{'─'*12}─┼─{'─'*12}─{'─'*6}")
        for grp_name in fft_groups:
            r0 = fft_results[grp_name][0]
            r1 = fft_results[grp_name][1]
            bl_diff = r1['baseline'] - r0['baseline']
            bl_status = "✅" if abs(bl_diff) < 0.5 else "⚠️"
            line = (f"  {grp_name:<10} │ {r0['baseline']:>11.2f}  {r0['post_da']:>11.2f}  │"
                    f" {r1['baseline']:>11.2f}  {r1['post_da']:>11.2f}  │"
                    f" {bl_diff:>+10.2f}  {bl_status:>6}")
            lines.append(line)
            # Print a separator after All-E to visually split E and I groups
            if grp_name == 'All-E':
                lines.append(f"  {'─'*10}─┼─{'─'*12}─{'─'*12}─┼─{'─'*12}─{'─'*12}─┼─{'─'*12}─{'─'*6}")
        lines.append("=" * w)

        # Summary: Baseline Consistency Check
        lines.append("")
        lines.append("  🔍 Baseline Consistency Check (B0 vs B1, threshold < 0.5 Hz):")
        all_ok = True
        for grp_name in fft_groups:
            bl_diff = fft_results[grp_name][1]['baseline'] - fft_results[grp_name][0]['baseline']
            if abs(bl_diff) >= 0.5:
                lines.append(f"     ⚠️  {grp_name:<10}: Diff = {bl_diff:+.2f} Hz")
                all_ok = False
        if all_ok:
            lines.append("     ✅  All groups: Baseline MATCHED (diff < 0.5 Hz)")
        lines.append("")

        # DA Effect Summary
        lines.append("  💊 DA Effect Summary (Post-DA: Exp - Ctrl):")
        for grp_name in fft_groups:
            r0 = fft_results[grp_name][0]
            r1 = fft_results[grp_name][1]
            da_effect = r1['post_da'] - r0['post_da']
            arrow = "↑" if da_effect > 0 else "↓" if da_effect < 0 else "→"
            lines.append(f"     {grp_name:<10}: {r0['post_da']:>8.2f} → {r1['post_da']:>8.2f}  ({da_effect:+.2f} Hz {arrow})")
        lines.append("=" * w)

        text = "\n".join(lines)
        print(text)
        return text

    def save_report(self, filepath: str, time_win: float = 5.0):
        """Generate all analysis reports and save to a text file.

        Includes:
        - FFT comparison table (overview first for quick reading)
        - Per-group frequency reports for all 8 groups
        """
        import os
        from datetime import datetime
        report_groups = ['E-D1', 'E-D2', 'E-Other', 'All-E',
                         'I-D1', 'I-D2', 'I-Other', 'All-I']
        sections = []

        # Header with experiment metadata
        sections.append("=" * 100)
        sections.append(f"  📋 PFC Network Simulation — Analysis Report")
        sections.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sections.append(f"  Duration: {self.duration:.0f} ms | DA Onset: {self.da_onset:.0f} ms | DA Level: {self.da_level} nM")
        sections.append(f"  N_E: {self.N_E} | N_I: {self.N_I} | N_total: {self.N}")
        sections.append("=" * 100)

        # FFT comparison table — put overview FIRST for quick reading
        sections.append(self.print_fft_comparison_report(time_win=time_win))

        # Per-group frequency reports (detailed)
        sections.append("\n" + "~" * 100)
        sections.append("  📊 Detailed Per-Group Phase Analysis")
        sections.append("~" * 100)
        for grp in report_groups:
            text = self.print_frequency_report(target_group=grp, time_win=time_win)
            if text:
                sections.append(text)

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(sections))
        print(f"\n📄 Report saved to: {filepath}")

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
