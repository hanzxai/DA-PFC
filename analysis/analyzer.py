# analysis/analyzer.py
import numpy as np
from scipy.signal import welch

# 画图与分析工具
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch

class PFCAnalyzer:
    def __init__(self, data_source):
        """
        初始化：接收内存数据包
        :param data_source: dict, 由 run_simulation_in_memory 返回的数据包
        """
        self.data = data_source
        
        # 解包配置
        self.cfg = self.data['config']
        self.dt = self.cfg['dt']
        self.duration = self.cfg['duration']
        self.N_E = self.cfg['N_E']
        self.N_I = self.cfg['N_I']
        self.N = self.N_E + self.N_I
        self.da_onset = self.cfg.get('da_onset', 1000.0) # 默认为1000ms
        # [新增] 读取浓度配置
        self.da_level = self.cfg.get('da_level', 10.0)
        # 预处理：构建亚群的 Boolean Masks
        self._build_group_masks()
        
        # 颜色配置
        self.colors = {
            'E-D1': 'red', 'E-D2': 'blue', 'E-Other': 'gray',
            'I-D1': 'orange', 'I-D2': 'purple', 'I-Other': 'green'
        }
        print("📊 Analyzer initialized. Frequency analysis tools ready.")

    def _build_group_masks(self):
        """内部方法：自动生成所有亚群的索引"""
        m_d1 = self.data['masks']['d1'].numpy()
        m_d2 = self.data['masks']['d2'].numpy()
        all_ids = np.arange(self.N)
        
        self.groups = {}
        self.groups['E-D1']    = (all_ids < self.N_E) & m_d1
        self.groups['E-D2']    = (all_ids < self.N_E) & m_d2
        self.groups['E-Other'] = (all_ids < self.N_E) & (~m_d1) & (~m_d2)
        self.groups['I-D1']    = (all_ids >= self.N_E) & m_d1
        self.groups['I-D2']    = (all_ids >= self.N_E) & m_d2
        self.groups['I-Other'] = (all_ids >= self.N_E) & (~m_d1) & (~m_d2)

    def _get_spikes_for_batch(self, batch_idx):
        """获取指定 Batch 的脉冲数据"""
        all_s = self.data['spikes']
        mask = all_s[:, 1] == batch_idx
        return all_s[mask][:, [0, 2]].numpy()

    # =========================================================
    #  核心计算模块 (基于你的 SignalProcessor)
    # =========================================================
    def compute_group_rate(self, batch_idx, group_name, time_win=5.0, sigma=1.5):
        """
        计算指定群体的平滑放电率曲线
        :param time_win: 时间窗宽度 (ms)
        :param sigma: 高斯平滑系数
        :return: (centers, rate_smooth)
        """
        if group_name not in self.groups:
            print(f"⚠️ Warning: Group {group_name} not found.")
            return None, None

        spikes = self._get_spikes_for_batch(batch_idx)
        if len(spikes) == 0:
            centers = np.arange(0, self.duration, time_win)
            return centers, np.zeros_like(centers)

        ts = spikes[:, 0] * self.dt # ms
        ids = spikes[:, 1]
        
        mask_ids = self.groups[group_name]
        count = np.sum(mask_ids)
        
        if count == 0:
            centers = np.arange(0, self.duration, time_win)
            return centers, np.zeros_like(centers)

        # 筛选属于该群体的脉冲
        # 优化：对于 N=1000，使用 isin 足够快
        valid_neurons = np.where(mask_ids)[0]
        mask_spikes = np.isin(ids, valid_neurons)
        
        # 创建时间 Bin
        bins = np.arange(0, self.duration + time_win, time_win)
        centers = (bins[:-1] + bins[1:]) / 2
        
        # --- [关键逻辑] 直方图统计 ---
        h, _ = np.histogram(ts[mask_spikes], bins=bins)
        
        # --- [关键逻辑] 归一化: Hz = spikes / (seconds * neurons) ---
        time_win_s = time_win / 1000.0
        rate = h / (time_win_s * count)
        
        # 高斯平滑
        rate_smooth = gaussian_filter1d(rate, sigma=sigma)
        
        return centers, rate_smooth

    def calculate_frequency_in_window(self, rate_curve, time_win, start_ms, end_ms):
        """
        对 rate_curve 的指定片段进行高分辨率 FFT 分析
        """
        # 将 ms 转换为数组索引
        idx_start = int(start_ms / time_win)
        idx_end = int(end_ms / time_win)
        
        # 边界检查
        if idx_start < 0: idx_start = 0
        if idx_end > len(rate_curve): idx_end = len(rate_curve)
        
        segment = rate_curve[idx_start:idx_end]
        if len(segment) < 10: return 0.0 # 数据太少
        
        # 去除直流分量
        sig = segment - np.mean(segment)
        fs = 1000.0 / time_win # 采样率 (Hz)
        
        # Welch 方法
        freqs, power = welch(sig, fs=fs, nperseg=len(sig), nfft=4096)
        
        # 寻找主频 (只关注 > 1.0 Hz 以排除慢漂移)
        mask_roi = freqs > 1.0
        if np.any(mask_roi):
            peak_idx = np.argmax(power[mask_roi])
            peak_freq = freqs[mask_roi][peak_idx]
            return peak_freq
        return 0.0

    # =========================================================
    #  高级分析报告 (Report)
    # =========================================================
    def print_frequency_report(self, target_group='E-D1', time_win=5.0):
        """
        打印 Control vs Experiment 的分阶段频率对比报告
        """
        # 1. 获取两条曲线
        _, r_ctrl = self.compute_group_rate(0, target_group, time_win)
        _, r_exp = self.compute_group_rate(1, target_group, time_win)
        
        if r_ctrl is None or r_exp is None: return

        print("\n" + "="*50)
        print(f" 📊 深度频率分析报告 (Target: {target_group})")
        print("="*50)
        
        # 定义阶段：根据 da_onset 自动调整
        onset = int(self.da_onset)
        dur = int(self.duration)
        
        # 自动划分：Baseline (前段), Dopamine (中段), Late (后段)
        # 确保窗口有意义
        phases = []
        if onset > 200:
            phases.append(("Baseline", 0, onset))
        if dur - onset > 500:
            phases.append(("DA Phase", onset, min(onset+1000, dur)))
        if dur > onset + 1000:
            phases.append(("Late DA", onset+1000, dur))
            
        print(f"{'Phase':<12} | {'Time (ms)':<15} | {'Ctrl (Hz)':<10} | {'Exp (Hz)':<10} | {'Diff':<10}")
        print("-" * 68)
        
        for name, start, end in phases:
            freq_ctrl = self.calculate_frequency_in_window(r_ctrl, time_win, start, end)
            freq_exp = self.calculate_frequency_in_window(r_exp, time_win, start, end)
            diff = freq_exp - freq_ctrl
            
            time_str = f"{start}-{end}"
            print(f"{name:<12} | {time_str:<15} | {freq_ctrl:<10.2f} | {freq_exp:<10.2f} | {diff:+.2f}")
        print("="*50 + "\n")

    # =========================================================
    #  绘图功能 (调用上面的计算核心)
    # =========================================================
    def plot_population_rates(self, batch_idx=1, target_groups=None, ax=None, time_win=5.0, smooth_sigma=2.0):
        if ax is None: fig, ax = plt.subplots(figsize=(12, 4))
        
        if target_groups is None: target_groups = ['E-D1', 'E-D2', 'E-Other']
        # 自动生成更具信息的标题
        batch_name = "Control" if batch_idx == 0 else f"Experiment ({self.da_level} nM)"
        # 获取 x 轴 (只需计算一次)
        bins = np.arange(0, self.duration + time_win, time_win)
        centers = (bins[:-1] + bins[1:]) / 2
        
        for grp_name in target_groups:
            # 调用核心计算函数
            _, rate_smooth = self.compute_group_rate(batch_idx, grp_name, time_win, smooth_sigma)
            
            if rate_smooth is not None and np.max(rate_smooth) > 0.1:
                ax.plot(centers, rate_smooth, color=self.colors.get(grp_name, 'black'), 
                        label=grp_name, lw=2)

        # 画 DA 竖线
        if self.da_onset > 0 and batch_idx == 1:
            ax.axvline(self.da_onset, color='k', linestyle='--', alpha=0.7)
            ax.text(self.da_onset + 50, ax.get_ylim()[1]*0.95, "DA Onset", fontsize=10)

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_title(f"Population Rates - {batch_name}")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    def plot_raster(self, batch_idx=1, target_groups=None, ax=None, title=None):
        # ... (保持之前的 Raster 代码不变，或者根据需要复制过来) ...
        # 为了完整性，这里放一个简版
        if ax is None: fig, ax = plt.subplots(figsize=(10, 6))
        spikes = self._get_spikes_for_batch(batch_idx)
        if len(spikes) == 0: return
        ts = spikes[:, 0] * self.dt
        ids = spikes[:, 1]
        
        if target_groups is None: target_groups = list(self.groups.keys())
        
        for grp_name in target_groups:
            if grp_name not in self.groups: continue
            mask_ids = self.groups[grp_name]
            valid_neurons = np.where(mask_ids)[0]
            mask_spikes = np.isin(ids, valid_neurons)
            if np.sum(mask_spikes) > 0:
                ax.scatter(ts[mask_spikes], ids[mask_spikes], s=1.0, 
                           color=self.colors.get(grp_name, 'black'), label=grp_name, alpha=0.6)
        
        if self.da_onset > 0 and batch_idx == 1:
            ax.axvline(self.da_onset, color='k', linestyle='--')
            
        # 默认标题也带上浓度
        default_title = f"Raster Plot - {'Control' if batch_idx==0 else f'Exp ({self.da_level} nM)'}"
        ax.set_title(title if title else default_title)
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Neuron ID")
        ax.set_xlim(0, self.duration)