# analysis/plotting.py
"""
Combined plotting functions for DA-PFC simulation.

Each plot type produces ONE figure with a 3×2 layout:
  - Row 0: Full time-range
  - Row 1: Zoom — Before DA (baseline segment)
  - Row 2: Zoom — After DA (steady-state segment)
  - Col 0: Batch 0 (Control)
  - Col 1: Batch 1 (Experiment)

Y-axes are unified across all panels for fair comparison.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Global font settings — make everything larger for readability
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 15,
})

from analysis.analyzer import PFCAnalyzer


# ---------------------------------------------------------------------------
# Helper: draw DA onset vertical line(s)
# ---------------------------------------------------------------------------
def _draw_onset_line(ax, onset_x, ylim_top, label=" DA", color='black'):
    ax.axvline(onset_x, color=color, linestyle='--', alpha=0.6, linewidth=1.8)
    ax.text(onset_x, ylim_top * 0.97, label, fontsize=14, va='top', color=color,
            fontweight='bold')


def _draw_two_stage_lines(ax, analyzer, use_seconds=True):
    """Draw two vertical lines for two-stage DA dosing mode."""
    cfg = analyzer.cfg
    if cfg.get('mode') != 'dynamic_d1_d2_two_stage':
        return False

    phase1_onset = cfg.get('phase1_da_onset')  # when resting DA starts
    phase2_onset = cfg.get('phase2_onset')      # when DA challenge starts
    da1 = cfg.get('da_level_1', 0)
    da2 = cfg.get('da_level_2', 0)

    if phase1_onset is None or phase2_onset is None:
        return False

    ylim_top = ax.get_ylim()[1]
    if use_seconds:
        x1 = phase1_onset / 1000.0
        x2 = phase2_onset / 1000.0
    else:
        x1 = phase1_onset
        x2 = phase2_onset

    _draw_onset_line(ax, x1, ylim_top, label=f" DA={da1}nM", color='#2196F3')
    _draw_onset_line(ax, x2, ylim_top, label=f" DA={da2}nM", color='#F44336')
    return True


# ---------------------------------------------------------------------------
# 1. Combined Raster Plot  (2×2)
# ---------------------------------------------------------------------------
def plot_combined_raster(analyzer: PFCAnalyzer, save_dir=None,
                         max_spikes_per_group: int = 80000,
                         zoom_window: float = 2000.0):
    """
    Produce a single 3×2 raster figure:
      [full-Control]        [full-Exp ]
      [before-DA-Control]   [before-DA-Exp ]
      [after-DA-Control]    [after-DA-Exp ]
    """
    print("🎨 Plotting combined raster (3×2)...")

    target_groups = ['E-D1', 'E-D2', 'E-Other', 'I-D1', 'I-D2', 'I-Other']
    da_onset = analyzer.da_onset

    # Before-DA window: a segment from baseline period
    before_end = da_onset
    before_start = max(0.0, before_end - zoom_window)

    # After-DA window: pick a segment well after DA onset so alpha is near steady-state
    # Use the last `zoom_window` ms of the simulation, or midpoint if duration is very long
    da_duration = analyzer.duration - da_onset
    if da_duration > zoom_window * 3:
        # Pick a window starting at 2/3 of the DA period
        after_start = da_onset + da_duration * 0.6
        after_end = after_start + zoom_window
        if after_end > analyzer.duration:
            after_end = analyzer.duration
            after_start = after_end - zoom_window
    else:
        after_end = analyzer.duration
        after_start = max(da_onset, after_end - zoom_window)

    fig, axes = plt.subplots(3, 2, figsize=(32, 30), dpi=200)

    for col, batch_idx in enumerate([0, 1]):
        # --- get spike data for this batch ---
        all_s = analyzer.data['spikes']
        mask_batch = all_s[:, 1] == batch_idx
        spikes_batch = all_s[mask_batch][:, [0, 2]].numpy()
        if len(spikes_batch) == 0:
            continue

        ts_ms = spikes_batch[:, 0] * analyzer.dt
        neuron_ids = spikes_batch[:, 1]

        # Auto ms → s for full plot
        if analyzer.duration > 10000:
            x_full = ts_ms / 1000.0
            x_label_full = "Time (s)"
            onset_full = da_onset / 1000.0
            x_max_full = analyzer.duration / 1000.0
        else:
            x_full = ts_ms
            x_label_full = "Time (ms)"
            onset_full = da_onset
            x_max_full = analyzer.duration

        rng = np.random.default_rng(42)

        # Row 0: Full, Row 1: Before DA, Row 2: After DA
        zoom_configs = [
            {'zoom': False, 'start': None, 'end': None, 'label': 'Full'},
            {'zoom': True, 'start': before_start, 'end': before_end,
             'label': f'Before DA [{before_start:.0f}–{before_end:.0f} ms]'},
            {'zoom': True, 'start': after_start, 'end': after_end,
             'label': f'After DA [{after_start:.0f}–{after_end:.0f} ms]'},
        ]

        for row, cfg in enumerate(zoom_configs):
            ax = axes[row, col]

            if cfg['zoom']:
                mask_time = (ts_ms >= cfg['start']) & (ts_ms <= cfg['end'])
                x_data = ts_ms[mask_time]
                n_ids = neuron_ids[mask_time]
            else:
                x_data = x_full
                n_ids = neuron_ids

            for grp_name in target_groups:
                if grp_name not in analyzer.groups:
                    continue
                valid_neurons = np.where(analyzer.groups[grp_name])[0]
                mask_grp = np.isin(n_ids, valid_neurons)
                gx, gy = x_data[mask_grp], n_ids[mask_grp]
                if len(gx) == 0:
                    continue
                if len(gx) > max_spikes_per_group:
                    idx = rng.choice(len(gx), size=max_spikes_per_group, replace=False)
                    gx, gy = gx[idx], gy[idx]
                color = PFCAnalyzer.COLORS.get(grp_name, 'black')

                # Adjust marker size & alpha for readability
                if cfg['zoom']:
                    # Zoom panels: larger, more opaque dots
                    ax.scatter(gx, gy, s=8, color=color, alpha=0.85,
                               linewidths=0, rasterized=True)
                else:
                    # Full panel: moderate size, slightly transparent
                    ax.scatter(gx, gy, s=4, color=color, alpha=0.55,
                               linewidths=0, rasterized=True)

            # E/I boundary
            ax.axhline(analyzer.N_E - 0.5, color='gray', linestyle='-',
                       linewidth=0.8, alpha=0.5)
            ax.set_ylim(-1, analyzer.N)
            ax.grid(True, axis='x', linestyle='--', alpha=0.2)

            if cfg['zoom']:
                ax.set_xlim(cfg['start'], cfg['end'])
                ax.set_xlabel("Time (ms)")
            else:
                ax.set_xlim(0, x_max_full)
                ax.set_xlabel(x_label_full)
                # Draw vertical lines: two-stage mode draws 2 lines, single-stage draws 1
                if batch_idx == 1:
                    use_sec = analyzer.duration > 10000
                    if not _draw_two_stage_lines(ax, analyzer, use_seconds=use_sec):
                        if onset_full > 0:
                            _draw_onset_line(ax, onset_full, analyzer.N)

            if col == 0:
                ax.set_ylabel("Neuron ID")

            ax.tick_params(axis='both', which='major', labelsize=14)

            # Title
            batch_label = f"Control ({analyzer.control_da} nM)" if batch_idx == 0 else f"Exp ({analyzer.da_level} nM)"
            ax.set_title(f"Raster — {batch_label} ({cfg['label']})")

    plt.tight_layout()
    if save_dir:
        save_path = save_dir / "combined_raster.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        print(f"📊 Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Combined Firing Rate Plot (generic helper)
# ---------------------------------------------------------------------------
def _plot_combined_rates(analyzer: PFCAnalyzer, group_names: list,
                         title_prefix: str, filename: str,
                         save_dir=None,
                         time_win_full: float = 100.0,
                         time_win_zoom: float = 5.0,
                         zoom_window: float = 2000.0):
    """
    Generic 3×2 firing-rate figure.
    Row 0 = full time-range, Row 1 = Before DA zoom, Row 2 = After DA zoom.
    Col 0 = Control, Col 1 = Exp.
    Y-axes are unified across all panels.
    """
    print(f"🎨 Plotting combined {title_prefix} rates (3×2)...")

    da_onset = analyzer.da_onset

    # Before-DA window
    before_end = da_onset
    before_start = max(0.0, before_end - zoom_window)

    # After-DA window: well after onset so alpha is near steady-state
    da_duration = analyzer.duration - da_onset
    if da_duration > zoom_window * 3:
        after_start = da_onset + da_duration * 0.6
        after_end = after_start + zoom_window
        if after_end > analyzer.duration:
            after_end = analyzer.duration
            after_start = after_end - zoom_window
    else:
        after_end = analyzer.duration
        after_start = max(da_onset, after_end - zoom_window)

    line_styles = {
        'E-D1': '-', 'E-D2': '-', 'E-Other': '-',
        'I-D1': '-', 'I-D2': '-', 'I-Other': '-',
    }

    fig, axes = plt.subplots(3, 2, figsize=(32, 28), dpi=200)

    # Define row configs: (is_zoom, zoom_start, zoom_end, label)
    row_configs = [
        (False, None, None, 'Full'),
        (True, before_start, before_end,
         f'Before DA [{before_start:.0f}–{before_end:.0f} ms]'),
        (True, after_start, after_end,
         f'After DA [{after_start:.0f}–{after_end:.0f} ms]'),
    ]

    # ---- first pass: draw all curves, collect y-ranges ----
    y_max_full = -np.inf
    y_min_full = np.inf
    y_max_zoom = -np.inf
    y_min_zoom = np.inf

    for col, batch_idx in enumerate([0, 1]):
        for row, (is_zoom, z_start, z_end, _label) in enumerate(row_configs):
            ax = axes[row, col]
            tw = time_win_zoom if is_zoom else time_win_full

            for grp_name in group_names:
                if grp_name not in analyzer.groups:
                    continue
                centers, rate = analyzer.compute_group_rate(batch_idx, grp_name, time_win=tw)
                if rate is None or len(rate) == 0:
                    continue

                if is_zoom:
                    mask = (centers >= z_start) & (centers <= z_end)
                    if not np.any(mask):
                        continue
                    x_data = centers[mask]
                    y_data = rate[mask]
                else:
                    if centers[-1] > 10000:
                        x_data = centers / 1000.0
                    else:
                        x_data = centers
                    y_data = rate

                color = PFCAnalyzer.COLORS.get(grp_name, 'k')
                ls = line_styles.get(grp_name, '-')
                lw = 2.5 if grp_name.startswith('E') else 2.0
                alpha = 0.85 if grp_name.startswith('E') else 0.70
                ax.plot(x_data, y_data, color=color, label=grp_name,
                        lw=lw, alpha=alpha, linestyle=ls)

                cur_max = float(np.nanmax(y_data))
                cur_min = float(np.nanmin(y_data))
                if is_zoom:
                    y_max_zoom = max(y_max_zoom, cur_max)
                    y_min_zoom = min(y_min_zoom, cur_min)
                else:
                    y_max_full = max(y_max_full, cur_max)
                    y_min_full = min(y_min_full, cur_min)

    # ---- second pass: unify y-axes and add decorations ----
    # Adaptive ylim: use data range with 10% margin on each side
    if y_max_full > -np.inf and y_min_full < np.inf:
        y_range_full = y_max_full - y_min_full
        margin_full = y_range_full * 0.10 if y_range_full > 0 else 1.0
        ylim_full = (y_min_full - margin_full, y_max_full + margin_full)
    else:
        ylim_full = None

    if y_max_zoom > -np.inf and y_min_zoom < np.inf:
        y_range_zoom = y_max_zoom - y_min_zoom
        margin_zoom = y_range_zoom * 0.10 if y_range_zoom > 0 else 1.0
        ylim_zoom = (y_min_zoom - margin_zoom, y_max_zoom + margin_zoom)
    else:
        ylim_zoom = None

    for col, batch_idx in enumerate([0, 1]):
        for row, (is_zoom, z_start, z_end, row_label) in enumerate(row_configs):
            ax = axes[row, col]

            if is_zoom:
                ax.set_xlim(z_start, z_end)
                ax.set_xlabel("Time (ms)")
                if ylim_zoom:
                    ax.set_ylim(ylim_zoom[0], ylim_zoom[1])
            else:
                if analyzer.duration > 10000:
                    ax.set_xlim(0, analyzer.duration / 1000.0)
                    ax.set_xlabel("Time (s)")
                    onset_x = da_onset / 1000.0
                else:
                    ax.set_xlim(0, analyzer.duration)
                    ax.set_xlabel("Time (ms)")
                    onset_x = da_onset
                if ylim_full:
                    ax.set_ylim(ylim_full[0], ylim_full[1])
                # Draw vertical lines: two-stage mode draws 2 lines, single-stage draws 1
                use_sec = analyzer.duration > 10000
                if not _draw_two_stage_lines(ax, analyzer, use_seconds=use_sec):
                    if onset_x > 0:
                        _draw_onset_line(ax, onset_x, ax.get_ylim()[1])

            if col == 0:
                ax.set_ylabel("Firing Rate (Hz)")

            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(True, linestyle='--', alpha=0.3)

            # Title
            cfg_mode = analyzer.cfg.get('mode', '')
            if cfg_mode == 'dynamic_d1_d2_two_stage':
                da1 = analyzer.cfg.get('da_level_1', 0)
                da2 = analyzer.cfg.get('da_level_2', 0)
                batch_label = f"Control ({da1} nM)" if batch_idx == 0 else f"Exp ({da1}→{da2} nM)"
            else:
                batch_label = f"Control ({analyzer.control_da} nM)" if batch_idx == 0 else f"Exp ({analyzer.da_level} nM)"
            ax.set_title(f"{title_prefix} — {batch_label} ({row_label})")

            # Legend only on top-left panel to save space
            if row == 0 and col == 0:
                ax.legend(loc='upper left')

    plt.tight_layout()
    if save_dir:
        save_path = save_dir / filename
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public API: 3 combined rate figures
# ---------------------------------------------------------------------------
def plot_combined_rates_all(analyzer: PFCAnalyzer, save_dir=None):
    """All 6 subgroups: 2×2 combined figure."""
    _plot_combined_rates(
        analyzer,
        group_names=['E-D1', 'E-D2', 'E-Other', 'I-D1', 'I-D2', 'I-Other'],
        title_prefix="All Population",
        filename="combined_rates_all.png",
        save_dir=save_dir,
    )


def plot_combined_rates_E(analyzer: PFCAnalyzer, save_dir=None):
    """Excitatory subgroups only: 2×2 combined figure."""
    _plot_combined_rates(
        analyzer,
        group_names=['E-D1', 'E-D2', 'E-Other'],
        title_prefix="Excitatory (E)",
        filename="combined_rates_E.png",
        save_dir=save_dir,
    )


def plot_combined_rates_I(analyzer: PFCAnalyzer, save_dir=None):
    """Inhibitory subgroups only: 2×2 combined figure."""
    _plot_combined_rates(
        analyzer,
        group_names=['I-D1', 'I-D2', 'I-Other'],
        title_prefix="Inhibitory (I)",
        filename="combined_rates_I.png",
        save_dir=save_dir,
    )
