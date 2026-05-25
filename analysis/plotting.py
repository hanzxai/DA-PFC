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
# 1. Combined Raster Plot  (3×N where N=1 if Ctrl/Exp share DA, else 2)
# ---------------------------------------------------------------------------
def _is_da_modulated(analyzer: PFCAnalyzer) -> bool:
    """Return True iff Ctrl (batch 0) and Exp (batch 1) DA schedules differ.

    For WM demos this is encoded as cfg['da_modulated'].  For other
    experiment modes we conservatively assume the two batches differ.
    """
    cfg = analyzer.cfg
    if 'da_modulated' in cfg:
        return bool(cfg['da_modulated'])
    # Fall back: if there's no explicit flag, treat any non-trivial DA onset
    # as modulated (legacy paths).
    return analyzer.da_onset > 0 and analyzer.da_level != analyzer.control_da


def plot_combined_raster(analyzer: PFCAnalyzer, save_dir=None,
                         max_spikes_per_group: int = 80000,
                         zoom_window: float = 2000.0):
    """
    Produce a 3-row raster figure with WM-overview-style colouring.

    Layout:
      Row 0: Full time-range (with red dashed boxes marking the zoom windows)
      Row 1: Zoom-in of the pre-cue / before-DA segment
      Row 2: Zoom-in of the post-DA / late-delay segment

    Columns:
      • If DA is *not* modulated between Ctrl and Exp (e.g. a DA=2 nM demo
        run), the two batches are bit-exact identical at the network level
        and we draw a SINGLE column (1 panel per row).
      • Otherwise, we draw two columns (Control | Experiment).

    Group colouring matches `wm_overview.png` exactly: it uses the
    `register_wm_groups` palette (Mem-A red, Mem-B blue, E-BG gray,
    I-D1 orange, I-D2 purple, I-Other green, etc.), so the raster legend
    is consistent with the firing-rate panel below.
    """
    print("🎨 Plotting combined raster ...")

    # ---- Decide the group palette ---------------------------------------
    # Prefer WM-augmented groups when present so the colours match
    # wm_overview.png exactly; fall back to the standard 6 receptor groups.
    if 'Mem-A' in analyzer.groups:
        # WM run — use the same labels & colours as wm_overview.
        target_groups = ['Mem-A', 'Mem-B', 'E-BG', 'E-D2',
                         'I-D1', 'I-D2', 'I-Other']
    else:
        target_groups = ['E-D1', 'E-D2', 'E-Other', 'I-D1', 'I-D2', 'I-Other']

    # ---- Single- vs two-column layout -----------------------------------
    da_modulated = _is_da_modulated(analyzer)
    batch_indices = [0, 1] if da_modulated else [0]
    n_cols = len(batch_indices)

    da_onset = analyzer.da_onset

    # ---- Pick zoom windows ----------------------------------------------
    # For WM runs use cue/delay metadata; otherwise fall back to the
    # legacy Before-DA / After-DA windows.
    protocol = analyzer.cfg.get('wm_protocol', {})
    if protocol:
        cue_on  = float(protocol.get('cue_a_onset', 0.0))
        cue_off = float(protocol.get('cue_a_offset', cue_on))
        delay_end = cue_off + float(protocol.get('delay_ms', 0.0))
        # Pre-cue window: last `zoom_window` ms of baseline ending at cue_on
        pre_end   = cue_on
        pre_start = max(0.0, pre_end - zoom_window)
        # Late-delay window: last `zoom_window` ms of the delay phase
        post_end   = delay_end
        post_start = max(cue_off, post_end - zoom_window)
        zoom_labels = ('Zoom-in: Pre-cue', 'Zoom-in: Late-delay')
    else:
        pre_end = da_onset
        pre_start = max(0.0, pre_end - zoom_window)
        da_duration = analyzer.duration - da_onset
        if da_duration > zoom_window * 3:
            post_start = da_onset + da_duration * 0.6
            post_end   = post_start + zoom_window
            if post_end > analyzer.duration:
                post_end   = analyzer.duration
                post_start = post_end - zoom_window
        else:
            post_end   = analyzer.duration
            post_start = max(da_onset, post_end - zoom_window)
        zoom_labels = ('Zoom-in: Before DA', 'Zoom-in: After DA')

    # ---- Figure layout ---------------------------------------------------
    fig_w = 16 * n_cols
    fig, axes = plt.subplots(3, n_cols, figsize=(fig_w, 30), dpi=200,
                             squeeze=False)

    rng = np.random.default_rng(42)

    for col, batch_idx in enumerate(batch_indices):
        # --- get spike data for this batch ---
        all_s = analyzer.data['spikes']
        mask_batch = all_s[:, 1] == batch_idx
        spikes_batch = all_s[mask_batch][:, [0, 2]].numpy()
        if len(spikes_batch) == 0:
            continue

        ts_ms = spikes_batch[:, 0] * analyzer.dt
        neuron_ids = spikes_batch[:, 1]

        # Auto ms → s for the full panel
        if analyzer.duration > 10000:
            x_full = ts_ms / 1000.0
            x_label_full = "Time (s)"
            x_max_full   = analyzer.duration / 1000.0
            sec_scale    = 1000.0  # divide ms-windows by this for the box
        else:
            x_full = ts_ms
            x_label_full = "Time (ms)"
            x_max_full   = analyzer.duration
            sec_scale    = 1.0

        # Row 0: Full, Row 1: Pre, Row 2: Post
        zoom_configs = [
            {'zoom': False, 'start': None, 'end': None,
             'label': 'Full',                   'is_zoom': False},
            {'zoom': True,  'start': pre_start,  'end': pre_end,
             'label': zoom_labels[0],           'is_zoom': True},
            {'zoom': True,  'start': post_start, 'end': post_end,
             'label': zoom_labels[1],           'is_zoom': True},
        ]

        for row, zcfg in enumerate(zoom_configs):
            ax = axes[row, col]

            if zcfg['zoom']:
                mask_time = (ts_ms >= zcfg['start']) & (ts_ms <= zcfg['end'])
                x_data = ts_ms[mask_time]
                n_ids = neuron_ids[mask_time]
            else:
                x_data = x_full
                n_ids = neuron_ids

            # ---- scatter each group with its registered colour ----------
            for grp_name in target_groups:
                if grp_name not in analyzer.groups:
                    continue
                valid_neurons = np.where(analyzer.groups[grp_name])[0]
                mask_grp = np.isin(n_ids, valid_neurons)
                gx, gy = x_data[mask_grp], n_ids[mask_grp]
                if len(gx) == 0:
                    continue
                if len(gx) > max_spikes_per_group:
                    idx = rng.choice(len(gx), size=max_spikes_per_group,
                                     replace=False)
                    gx, gy = gx[idx], gy[idx]
                color = analyzer.COLORS.get(grp_name, 'black')

                if zcfg['is_zoom']:
                    ax.scatter(gx, gy, s=8, color=color, alpha=0.85,
                               linewidths=0, rasterized=True, label=grp_name)
                else:
                    ax.scatter(gx, gy, s=4, color=color, alpha=0.65,
                               linewidths=0, rasterized=True, label=grp_name)

            # E/I boundary
            ax.axhline(analyzer.N_E - 0.5, color='gray', linestyle='-',
                       linewidth=0.8, alpha=0.5)
            ax.set_ylim(-1, analyzer.N)
            ax.grid(True, axis='x', linestyle='--', alpha=0.2)

            if zcfg['is_zoom']:
                ax.set_xlim(zcfg['start'], zcfg['end'])
                ax.set_xlabel("Time (ms)")
            else:
                ax.set_xlim(0, x_max_full)
                ax.set_xlabel(x_label_full)

                # Draw red dashed rectangles marking the two zoom windows.
                # Convert ms-windows into the panel's x-axis units.
                box_specs = [
                    (pre_start  / sec_scale, pre_end  / sec_scale,
                     '#d62728', zoom_labels[0]),
                    (post_start / sec_scale, post_end / sec_scale,
                     '#d62728', zoom_labels[1]),
                ]
                for bx0, bx1, bc, blabel in box_specs:
                    if bx1 <= bx0:
                        continue
                    ax.axvspan(bx0, bx1, ymin=0.0, ymax=1.0,
                               facecolor='none', edgecolor=bc,
                               linestyle='--', linewidth=2.0, alpha=0.85,
                               zorder=5)
                    ax.text((bx0 + bx1) / 2.0, analyzer.N * 0.985, blabel,
                            ha='center', va='top', fontsize=11,
                            color=bc, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.8,
                                      edgecolor='none', pad=1.5),
                            zorder=6)

                # DA onset / offset markers — only when DA actually changes
                if da_modulated and batch_idx == 1:
                    use_sec = analyzer.duration > 10000
                    if not _draw_two_stage_lines(ax, analyzer,
                                                 use_seconds=use_sec):
                        if da_onset > 0:
                            onset_x = da_onset / sec_scale
                            _draw_onset_line(ax, onset_x, analyzer.N)
                        da_offset_ms = analyzer.cfg.get('da_offset', None)
                        if (da_offset_ms is not None and
                                analyzer.cfg.get('mode') == 'pulse_response'):
                            offset_x = da_offset_ms / sec_scale
                            _draw_onset_line(ax, offset_x, analyzer.N,
                                             label=" DA off", color='#F44336')

            if col == 0:
                ax.set_ylabel("Neuron ID")

            ax.tick_params(axis='both', which='major', labelsize=14)

            # Title
            if da_modulated:
                if batch_idx == 0:
                    batch_label = f"Control ({analyzer.control_da} nM)"
                else:
                    batch_label = f"Exp ({analyzer.da_level} nM)"
            else:
                batch_label = f"DA = {analyzer.control_da:g} nM"
            ax.set_title(f"Raster — {batch_label} ({zcfg['label']})")

            # Legend on the top-right of every panel; one entry per group
            # encountered.  Use markerscale to enlarge the dots in the legend.
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                # de-duplicate while preserving order
                seen = set(); uniq_h = []; uniq_l = []
                for h, l in zip(handles, labels):
                    if l in seen:
                        continue
                    seen.add(l); uniq_h.append(h); uniq_l.append(l)
                ax.legend(uniq_h, uniq_l,
                          loc='upper right',
                          ncol=2,
                          markerscale=4,
                          fontsize=12,
                          framealpha=0.85,
                          handletextpad=0.4,
                          columnspacing=0.8,
                          borderpad=0.4)

    plt.tight_layout()
    if save_dir:
        save_path = save_dir / "combined_raster.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        print(f"📊 Saved: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helper: draw DA concentration timeline for a single batch
# ---------------------------------------------------------------------------
def _draw_da_timeline(ax, analyzer: PFCAnalyzer, batch_idx: int):
    """
    Draw [DA] vs time for the given batch.
    If data contains a pre-computed 'da_schedule' array, use it directly.
    Otherwise, constructs the DA schedule from analyzer.cfg based on simulation mode.
    """
    cfg = analyzer.cfg
    mode = cfg.get('mode', '')
    duration = analyzer.duration
    da_onset = analyzer.da_onset
    use_seconds = duration > 10000
    scale = 1000.0 if use_seconds else 1.0
    x_label = "Time (s)" if use_seconds else "Time (ms)"
    dt = analyzer.dt

    # Check if pre-computed DA schedule is available (from waveform experiments)
    da_schedule = analyzer.data.get('da_schedule', None)
    if da_schedule is not None:
        # da_schedule shape: (steps, 2) — col 0 = Ctrl, col 1 = Exp
        t_all = np.arange(da_schedule.shape[0]) * dt
        d0 = da_schedule[:, 0]
        d1 = da_schedule[:, 1]
        d_plot = d0 if batch_idx == 0 else d1
        # Subsample for plotting efficiency (max 2000 points)
        max_pts = 2000
        if len(d_plot) > max_pts:
            step = len(d_plot) // max_pts
            t_plot = t_all[::step]
            d_plot_sub = d_plot[::step]
            d0_sub = d0[::step]
            d1_sub = d1[::step]
        else:
            t_plot = t_all
            d_plot_sub = d_plot
            d0_sub = d0
            d1_sub = d1
        ax.plot(t_plot / scale, d_plot_sub, color='#E91E63', linewidth=2.0, alpha=0.9)
        ax.fill_between(t_plot / scale, 0, d_plot_sub, color='#E91E63', alpha=0.15)
        ax.set_xlabel(x_label)
        ax.set_ylabel("[DA] (nM)")
        ax.set_xlim(0, duration / scale)
        all_da = np.concatenate([d0_sub, d1_sub])
        da_max = float(np.max(all_da))
        da_min = float(np.min(all_da))
        margin = max((da_max - da_min) * 0.15, 0.5)
        ax.set_ylim(max(0, da_min - margin), da_max + margin)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        return

    if mode == 'pulse_response':
        # Pulse mode: baseline → pulse → baseline (square pulse)
        ctrl_da = cfg.get('control_da', 2.0)
        target_da = cfg.get('da_level', 15.0)
        da_offset = cfg.get('da_offset', None)
        # Batch 0: constant control DA
        t0 = np.array([0.0, duration])
        d0 = np.array([ctrl_da, ctrl_da])
        # Batch 1: baseline → pulse → baseline
        if da_offset is not None and da_offset < duration:
            t1 = np.array([0.0, da_onset, da_onset, da_offset, da_offset, duration])
            d1 = np.array([ctrl_da, ctrl_da, target_da, target_da, ctrl_da, ctrl_da])
        else:
            # Fallback: no offset info, draw as step function
            t1 = np.array([0.0, da_onset, da_onset, duration])
            d1 = np.array([ctrl_da, ctrl_da, target_da, target_da])
    elif mode == 'dynamic_d1_d2_two_stage':
        phase1_onset = cfg.get('phase1_da_onset', da_onset)
        phase2_onset = cfg.get('phase2_onset', da_onset)
        da1 = cfg.get('da_level_1', 2.0)
        da2 = cfg.get('da_level_2', 15.0)
        ctrl_da = cfg.get('control_da', 0.0)
        # Batch 0: constant control DA
        t0 = np.array([0.0, duration])
        d0 = np.array([ctrl_da, ctrl_da])
        # Batch 1: 3-phase schedule
        t1 = np.array([0.0, phase1_onset, phase1_onset, phase2_onset, phase2_onset, duration])
        d1 = np.array([ctrl_da, ctrl_da, da1, da1, da2, da2])
    elif mode == 'resume_from_checkpoint':
        baseline_da = cfg.get('control_da', 2.0)
        new_da = cfg.get('da_level', 15.0)
        # Batch 0: constant baseline DA
        t0 = np.array([0.0, duration])
        d0 = np.array([baseline_da, baseline_da])
        # Batch 1: baseline → new DA at onset
        t1 = np.array([0.0, da_onset, da_onset, duration])
        d1 = np.array([baseline_da, baseline_da, new_da, new_da])
    else:
        # Default: kinetics / ckpt mode
        ctrl_da = cfg.get('control_da', 0.0)
        target_da = cfg.get('da_level', 10.0)
        # Batch 0: constant control DA
        t0 = np.array([0.0, duration])
        d0 = np.array([ctrl_da, ctrl_da])
        # Batch 1: 0 → target DA at onset
        t1 = np.array([0.0, da_onset, da_onset, duration])
        d1 = np.array([ctrl_da, ctrl_da, target_da, target_da])

    # Select which batch to draw
    if batch_idx == 0:
        t_plot, d_plot = t0, d0
    else:
        t_plot, d_plot = t1, d1

    ax.plot(t_plot / scale, d_plot, color='#E91E63', linewidth=2.5, alpha=0.9)
    ax.fill_between(t_plot / scale, 0, d_plot, color='#E91E63', alpha=0.15)
    ax.set_xlabel(x_label)
    ax.set_ylabel("[DA] (nM)")
    ax.set_xlim(0, duration / scale)
    # Y-axis: add margin above max DA
    all_da = np.concatenate([d0, d1])
    da_max = float(np.max(all_da))
    da_min = float(np.min(all_da))
    margin = max((da_max - da_min) * 0.15, 0.5)
    ax.set_ylim(max(0, da_min - margin), da_max + margin)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=14)


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
    Generic 4×2 firing-rate figure.
    Row 0 = DA concentration timeline (new!)
    Row 1 = full time-range, Row 2 = Before DA zoom, Row 3 = After DA zoom.
    Col 0 = Control, Col 1 = Exp.
    Y-axes are unified across all rate panels.
    """
    print(f"🎨 Plotting combined {title_prefix} rates (4×2)...")

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

    fig, axes = plt.subplots(4, 2, figsize=(32, 36), dpi=200,
                              gridspec_kw={'height_ratios': [1, 3, 3, 3]})

    # ---- Row 0: DA concentration timeline ----
    for col, batch_idx in enumerate([0, 1]):
        ax_da = axes[0, col]
        _draw_da_timeline(ax_da, analyzer, batch_idx)
        cfg_mode = analyzer.cfg.get('mode', '')
        if cfg_mode == 'dynamic_d1_d2_two_stage':
            da1 = analyzer.cfg.get('da_level_1', 0)
            da2 = analyzer.cfg.get('da_level_2', 0)
            batch_label = f"Control ({da1} nM)" if batch_idx == 0 else f"Exp ({da1}→{da2} nM)"
        else:
            batch_label = f"Control ({analyzer.control_da} nM)" if batch_idx == 0 else f"Exp ({analyzer.da_level} nM)"
        ax_da.set_title(f"DA Concentration — {batch_label}")

    # Define row configs for rate panels (rows 1-3): (is_zoom, zoom_start, zoom_end, label)
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
        for row_offset, (is_zoom, z_start, z_end, _label) in enumerate(row_configs):
            ax = axes[row_offset + 1, col]  # +1 because row 0 is DA timeline
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
        for row_offset, (is_zoom, z_start, z_end, row_label) in enumerate(row_configs):
            ax = axes[row_offset + 1, col]  # +1 because row 0 is DA timeline

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
                    # For pulse_response mode, also draw DA offset line
                    da_offset_ms = analyzer.cfg.get('da_offset', None)
                    if da_offset_ms is not None and analyzer.cfg.get('mode') == 'pulse_response':
                        offset_x = da_offset_ms / 1000.0 if use_sec else da_offset_ms
                        _draw_onset_line(ax, offset_x, ax.get_ylim()[1], label=" DA off", color='#F44336')

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

            # Legend only on first rate row, left panel to save space
            if row_offset == 0 and col == 0:
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


# ---------------------------------------------------------------------------
# 4. Simplified 2×2 Firing Rate Plot (DA timeline + Full rates only)
# ---------------------------------------------------------------------------
def _plot_rates_simple(analyzer: PFCAnalyzer, group_names: list,
                       title_prefix: str, filename: str,
                       save_dir=None,
                       time_win_full: float = 100.0):
    """
    Simplified 2×2 firing-rate figure (no zoom rows).
    Row 0 = DA concentration timeline
    Row 1 = Full time-range firing rates
    Col 0 = Control, Col 1 = Exp.
    """
    print(f"🎨 Plotting simplified {title_prefix} rates (2×2)...")

    da_onset = analyzer.da_onset

    line_styles = {
        'E-D1': '-', 'E-D2': '-', 'E-Other': '-',
        'I-D1': '-', 'I-D2': '-', 'I-Other': '-',
    }

    fig, axes = plt.subplots(2, 2, figsize=(32, 18), dpi=200,
                              gridspec_kw={'height_ratios': [1, 3]})

    # ---- Row 0: DA concentration timeline ----
    for col, batch_idx in enumerate([0, 1]):
        ax_da = axes[0, col]
        _draw_da_timeline(ax_da, analyzer, batch_idx)
        cfg_mode = analyzer.cfg.get('mode', '')
        if cfg_mode == 'dynamic_d1_d2_two_stage':
            da1 = analyzer.cfg.get('da_level_1', 0)
            da2 = analyzer.cfg.get('da_level_2', 0)
            batch_label = f"Control ({da1} nM)" if batch_idx == 0 else f"Exp ({da1}→{da2} nM)"
        else:
            batch_label = f"Control ({analyzer.control_da} nM)" if batch_idx == 0 else f"Exp ({analyzer.da_level} nM)"
        ax_da.set_title(f"DA Concentration — {batch_label}")

    # ---- Row 1: Full time-range firing rates ----
    y_max = -np.inf
    y_min = np.inf

    for col, batch_idx in enumerate([0, 1]):
        ax = axes[1, col]
        for grp_name in group_names:
            if grp_name not in analyzer.groups:
                continue
            centers, rate = analyzer.compute_group_rate(batch_idx, grp_name, time_win=time_win_full)
            if rate is None or len(rate) == 0:
                continue

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
            y_max = max(y_max, cur_max)
            y_min = min(y_min, cur_min)

    # Unify y-axes and add decorations
    if y_max > -np.inf and y_min < np.inf:
        y_range = y_max - y_min
        margin = y_range * 0.10 if y_range > 0 else 1.0
        ylim = (y_min - margin, y_max + margin)
    else:
        ylim = None

    for col, batch_idx in enumerate([0, 1]):
        ax = axes[1, col]
        if analyzer.duration > 10000:
            ax.set_xlim(0, analyzer.duration / 1000.0)
            ax.set_xlabel("Time (s)")
            onset_x = da_onset / 1000.0
        else:
            ax.set_xlim(0, analyzer.duration)
            ax.set_xlabel("Time (ms)")
            onset_x = da_onset
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])

        # Draw DA onset/offset lines
        use_sec = analyzer.duration > 10000
        if not _draw_two_stage_lines(ax, analyzer, use_seconds=use_sec):
            if onset_x > 0:
                _draw_onset_line(ax, onset_x, ax.get_ylim()[1])
            da_offset_ms = analyzer.cfg.get('da_offset', None)
            if da_offset_ms is not None and analyzer.cfg.get('mode') == 'pulse_response':
                offset_x = da_offset_ms / 1000.0 if use_sec else da_offset_ms
                _draw_onset_line(ax, offset_x, ax.get_ylim()[1], label=" DA off", color='#F44336')

        if col == 0:
            ax.set_ylabel("Firing Rate (Hz)")
        ax.legend(fontsize=10, loc='lower left', ncol=3, framealpha=0.7)

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)

        cfg_mode = analyzer.cfg.get('mode', '')
        if cfg_mode == 'dynamic_d1_d2_two_stage':
            da1 = analyzer.cfg.get('da_level_1', 0)
            da2 = analyzer.cfg.get('da_level_2', 0)
            batch_label = f"Control ({da1} nM)" if batch_idx == 0 else f"Exp ({da1}→{da2} nM)"
        else:
            batch_label = f"Control ({analyzer.control_da} nM)" if batch_idx == 0 else f"Exp ({analyzer.da_level} nM)"
        ax.set_title(f"{title_prefix} — {batch_label}")

        # ---- Annotate D1/D2 ΔRate on Exp panel ----
        if col == 1 and da_onset > 0:
            delta_groups = [g for g in ['E-D1', 'E-D2', 'I-D1', 'I-D2'] if g in group_names and g in analyzer.groups]
            if delta_groups:
                delta_lines = []
                for grp_name in delta_groups:
                    centers, rate = analyzer.compute_group_rate(batch_idx, grp_name, time_win=time_win_full)
                    if rate is None or len(rate) == 0:
                        continue
                    pre_mask = centers < da_onset
                    post_mask = centers >= da_onset
                    if np.any(pre_mask) and np.any(post_mask):
                        pre_mean = float(np.mean(rate[pre_mask]))
                        post_mean = float(np.mean(rate[post_mask]))
                        delta = post_mean - pre_mean
                        arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '→')
                        delta_lines.append(f"{grp_name}: {pre_mean:.2f}→{post_mean:.2f} ({delta:+.2f} {arrow})")
                if delta_lines:
                    text_str = "ΔRate (Pre→Post DA)\n" + "\n".join(delta_lines)
                    ax.text(0.98, 0.97, text_str, transform=ax.transAxes,
                            fontsize=11, verticalalignment='top', horizontalalignment='right',
                            fontfamily='monospace',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                                      edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    if save_dir:
        save_path = save_dir / filename
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")
    plt.close(fig)


def plot_rates_simple_all(analyzer: PFCAnalyzer, save_dir=None):
    """Simplified 2×2: All 6 subgroups, DA timeline + full rates only."""
    _plot_rates_simple(
        analyzer,
        group_names=['E-D1', 'E-D2', 'E-Other', 'I-D1', 'I-D2', 'I-Other'],
        title_prefix="All Population",
        filename="combined_rates_all.png",
        save_dir=save_dir,
    )


def plot_rates_simple_E(analyzer: PFCAnalyzer, save_dir=None):
    """Simplified 2×2: Excitatory subgroups only."""
    _plot_rates_simple(
        analyzer,
        group_names=['E-D1', 'E-D2', 'E-Other'],
        title_prefix="Excitatory (E)",
        filename="combined_rates_E.png",
        save_dir=save_dir,
    )


def plot_rates_simple_I(analyzer: PFCAnalyzer, save_dir=None):
    """Simplified 2×2: Inhibitory subgroups only."""
    _plot_rates_simple(
        analyzer,
        group_names=['I-D1', 'I-D2', 'I-Other'],
        title_prefix="Inhibitory (I)",
        filename="combined_rates_I.png",
        save_dir=save_dir,
    )
