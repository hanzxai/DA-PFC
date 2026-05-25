"""Visualization for WM evaluation — Inverted-U curves and friends.

Five figures:
    1. plot_inverted_u            -- 6-panel WM performance vs DA
    2. plot_inverted_u_with_alpha -- WM curve + receptor occupancy overlay
    3. plot_representative_traces -- Mem-A / Mem-B traces at low/opt/high DA
    4. plot_pharma_comparison     -- overlay multiple inverted-U curves
    5. plot_metric_correlations   -- inter-metric scatter matrix
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import make_interp_spline
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

import config
from wm_eval.config_eval import PASS_THRESHOLDS

# Aesthetic defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})


# ----------------------------------------------------------------------
#  Helper: smooth curve via cubic spline
# ----------------------------------------------------------------------
def _smooth(x, y, n_pts=200):
    """Cubic spline interpolation. Falls back to raw points if too few."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 4 or np.any(~np.isfinite(y)) or not _HAS_SCIPY:
        return x, y
    x_smooth = np.linspace(x.min(), x.max(), n_pts)
    try:
        spl = make_interp_spline(x, y, k=3)
        return x_smooth, spl(x_smooth)
    except Exception:
        return x, y


def _decorate_da_axis(ax, da_values, show_ec50=True):
    """Standard X-axis decoration: log scale, EC50 lines, grid."""
    ax.set_xscale('log')
    ax.set_xlabel('DA Concentration (nM)', fontsize=13)
    if show_ec50:
        ax.axvline(config.EC50_D1, color='#d62728', ls=':', lw=1.4, alpha=0.6,
                   label=f'D1 EC50={config.EC50_D1:g} nM')
        ax.axvline(config.EC50_D2, color='#1f77b4', ls=':', lw=1.4, alpha=0.6,
                   label=f'D2 EC50={config.EC50_D2:g} nM')
    ax.grid(True, which='both', alpha=0.3)


# ----------------------------------------------------------------------
#  Per-metric plot config
# ----------------------------------------------------------------------
METRIC_PLOT_CONFIG = {
    'persistence': {
        'label': 'Persistence dR_late (Hz)',
        'threshold': PASS_THRESHOLDS['persistence_hz'],
        'threshold_label': 'WM PASS (5 Hz)',
        'color': '#d62728',
    },
    'selectivity': {
        'label': 'Selectivity Index',
        'threshold': PASS_THRESHOLDS['selectivity'],
        'threshold_label': 'PASS (0.3)',
        'color': '#1f77b4',
    },
    'decay_ratio': {
        'label': 'Decay Ratio (delay/cue)',
        'threshold': PASS_THRESHOLDS['decay_ratio'],
        'threshold_label': 'PASS (0.3)',
        'color': '#9467bd',
    },
    'd_prime': {
        'label': "d-prime (SNR)",
        'threshold': PASS_THRESHOLDS['d_prime'],
        'threshold_label': 'PASS (2.0)',
        'color': '#ff7f0e',
    },
    'accuracy': {
        'label': 'Accuracy (%)',
        'threshold': PASS_THRESHOLDS['accuracy_pct'],
        'threshold_label': 'chance level (50%)',
        'color': '#2ca02c',
    },
    'composite_score': {
        'label': 'Composite WM Score',
        'threshold': PASS_THRESHOLDS['composite_score'],
        'threshold_label': 'PASS (0.5)',
        'color': '#17becf',
    },
}


# ----------------------------------------------------------------------
#  Figure 1: 6-panel inverted-U curves
# ----------------------------------------------------------------------
def plot_inverted_u(sweep_results: dict, save_path: str,
                    title_suffix: str = ''):
    """Generate the main 6-panel inverted-U figure.

    Args:
        sweep_results: {da_value: {'mean': dict, 'std': dict, ...}}
        save_path: output path (.png)
        title_suffix: additional info appended to figure title
    """
    da_values = sorted(sweep_results.keys())
    da_arr = np.array(da_values)

    metric_keys = list(METRIC_PLOT_CONFIG.keys())
    n_metrics = len(metric_keys)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows))
    axes = np.array(axes).flatten()

    fig.suptitle(f'WM Inverted-U Dose-Response{title_suffix}',
                 fontsize=17, fontweight='bold', y=1.00)

    for i, key in enumerate(metric_keys):
        ax = axes[i]
        cfg = METRIC_PLOT_CONFIG[key]

        means = np.array([sweep_results[d]['mean'].get(key, np.nan)
                          for d in da_values])
        stds  = np.array([sweep_results[d]['std'].get(key, 0.0)
                          for d in da_values])
        if key == 'accuracy':
            means = means * 100.0
            stds  = stds  * 100.0

        # Scatter with error bars
        ax.errorbar(da_arr, means, yerr=stds, fmt='o', color=cfg['color'],
                    markersize=8, capsize=4, lw=1.5, alpha=0.85,
                    markeredgecolor='black', markeredgewidth=0.8,
                    label='Simulation (mean +/- SD)', zorder=5)

        # Smooth fit
        x_s, y_s = _smooth(da_arr, means)
        ax.plot(x_s, y_s, '-', color=cfg['color'], lw=2.2, alpha=0.6,
                label='Cubic spline fit')

        # PASS threshold line and shaded region
        th = cfg['threshold']
        ax.axhline(th, color='green', ls='--', lw=1.4, alpha=0.7,
                   label=cfg['threshold_label'])
        y_arr = np.asarray(y_s)
        if y_arr.shape == np.asarray(x_s).shape:
            ax.fill_between(x_s, th, y_arr,
                            where=(y_arr >= th),
                            alpha=0.12, color='green',
                            interpolate=True)

        # Mark optimal DA
        if np.any(np.isfinite(means)):
            opt_idx = int(np.nanargmax(means))
            ax.axvline(da_arr[opt_idx], color='gold', ls='-', lw=1.5, alpha=0.5)
            ax.annotate(f'Opt: {da_arr[opt_idx]:g} nM',
                        xy=(da_arr[opt_idx], means[opt_idx]),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        color='black',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='gold', alpha=0.6, edgecolor='none'))

        ax.set_ylabel(cfg['label'], fontsize=12)
        ax.set_title(key.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        _decorate_da_axis(ax, da_arr)
        ax.legend(fontsize=8, loc='best', framealpha=0.85)

    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


# ----------------------------------------------------------------------
#  Figure 2: Inverted-U + receptor occupancy overlay
# ----------------------------------------------------------------------
def plot_inverted_u_with_alpha(sweep_results: dict, save_path: str):
    """Persistence curve + alpha_D1/alpha_D2 occupancy on twin Y-axis."""
    da_values = sorted(sweep_results.keys())
    da_arr = np.array(da_values)
    persistence = np.array([sweep_results[d]['mean'].get('persistence', np.nan)
                            for d in da_values])
    persistence_std = np.array([sweep_results[d]['std'].get('persistence', 0.0)
                                for d in da_values])
    alpha_d1 = np.array([sweep_results[d].get('alpha_d1_steady', np.nan)
                         for d in da_values])
    alpha_d2 = np.array([sweep_results[d].get('alpha_d2_steady', np.nan)
                         for d in da_values])

    fig, ax1 = plt.subplots(figsize=(11, 7))

    # Left axis: persistence
    ax1.errorbar(da_arr, persistence, yerr=persistence_std,
                 fmt='o-', color='#d62728', markersize=9, lw=2.0, capsize=4,
                 markeredgecolor='black', markeredgewidth=0.8,
                 label='Persistence dR_late (Hz)', zorder=5)
    x_s, y_s = _smooth(da_arr, persistence)
    ax1.plot(x_s, y_s, '-', color='#d62728', lw=2.0, alpha=0.4)

    th = PASS_THRESHOLDS['persistence_hz']
    ax1.axhline(th, color='green', ls='--', lw=1.4, alpha=0.7,
                label=f'WM PASS ({th:g} Hz)')

    ax1.set_xlabel('DA Concentration (nM)', fontsize=14)
    ax1.set_ylabel('Persistence dR_late (Hz)', fontsize=14, color='#d62728')
    ax1.tick_params(axis='y', labelcolor='#d62728')

    # Right axis: receptor occupancy
    ax2 = ax1.twinx()
    if not np.all(np.isnan(alpha_d1)):
        ax2.plot(da_arr, alpha_d1, 's--', color='#8b0000', lw=1.6,
                 alpha=0.7, label='alpha_D1 (occupancy)')
    if not np.all(np.isnan(alpha_d2)):
        ax2.plot(da_arr, alpha_d2, '^--', color='#00008b', lw=1.6,
                 alpha=0.7, label='alpha_D2 (occupancy)')
    ax2.set_ylabel('Receptor occupancy alpha', fontsize=14, color='gray')
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis='y', labelcolor='gray')

    _decorate_da_axis(ax1, da_arr)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               fontsize=10, framealpha=0.9)

    ax1.set_title('WM Performance vs Receptor Occupancy '
                  '(Mechanism of Inverted-U)',
                  fontsize=15, fontweight='bold')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


# ----------------------------------------------------------------------
#  Figure 3: Representative Mem-A traces at low / opt / high DA
# ----------------------------------------------------------------------
def plot_representative_traces(low_data, opt_data, high_data, save_path: str,
                               labels=None):
    """3-panel comparison of Mem-A firing rates at three key DA points.

    Each *_data should be {'time_s': array, 'mem_a_rate': array,
                           'mem_b_rate': array, 'da': float, 'protocol': dict}
    """
    if labels is None:
        labels = ['DA too LOW', 'DA OPTIMAL', 'DA too HIGH']
    datasets = [low_data, opt_data, high_data]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle('Representative Mem-A / Mem-B Traces at Three DA Levels',
                 fontsize=16, fontweight='bold')

    box_colors = ['#ffcccc', '#ccffcc', '#ffcccc']

    for ax, ds, lab, bc in zip(axes, datasets, labels, box_colors):
        if ds is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='gray')
            continue
        t = ds['time_s']
        ax.plot(t, ds['mem_a_rate'], '-', color='#d62728', lw=2.2,
                label='Mem-A (target)')
        ax.plot(t, ds['mem_b_rate'], '-', color='#1f77b4', lw=1.6, alpha=0.7,
                label='Mem-B (distractor)')

        # Protocol shading
        p = ds.get('protocol', {})
        if p:
            cue_on  = p.get('cue_a_onset',  0) / 1000.0
            cue_off = p.get('cue_a_offset', 0) / 1000.0
            delay_end = cue_off + p.get('delay_ms', 0) / 1000.0
            probe_end = delay_end + p.get('probe_ms', 0) / 1000.0
            ax.axvspan(cue_on,  cue_off,    color='gold',     alpha=0.30, label='Cue')
            ax.axvspan(cue_off, delay_end,  color='#fff2b3',  alpha=0.40, label='Delay')
            ax.axvspan(delay_end, probe_end, color='#cce5ff', alpha=0.40, label='Probe')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_title(f"{lab}  (DA={ds['da']:g} nM)", fontsize=13, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor=bc, alpha=0.5))
        ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Firing Rate (Hz)', fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


# ----------------------------------------------------------------------
#  Figure 4: Pharma comparison (overlay multiple inverted-U curves)
# ----------------------------------------------------------------------
def plot_pharma_comparison(condition_results: dict, save_path: str,
                           metric_key: str = 'persistence'):
    """Overlay inverted-U curves from multiple pharmacology conditions.

    Args:
        condition_results: {condition_name: {'sweep_results': ..., 'color': ...}}
        metric_key: which WM metric to plot
    """
    cfg = METRIC_PLOT_CONFIG[metric_key]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get DA range from first condition
    first_cond = next(iter(condition_results.values()))
    da_values = sorted(first_cond['sweep_results'].keys())
    da_arr = np.array(da_values)

    for cond_name, cond_data in condition_results.items():
        sr = cond_data['sweep_results']
        color = cond_data.get('color', None)
        means = np.array([sr[d]['mean'].get(metric_key, np.nan) for d in da_values])
        stds  = np.array([sr[d]['std'].get(metric_key, 0.0) for d in da_values])
        if metric_key == 'accuracy':
            means = means * 100.0
            stds  = stds  * 100.0

        ax.errorbar(da_arr, means, yerr=stds, fmt='o-', color=color,
                    markersize=8, lw=2.0, capsize=4,
                    markeredgecolor='black', markeredgewidth=0.7,
                    label=cond_name, alpha=0.85)
        x_s, y_s = _smooth(da_arr, means)
        ax.plot(x_s, y_s, '-', color=color, lw=1.5, alpha=0.4)

    th = cfg['threshold']
    ax.axhline(th, color='green', ls='--', lw=1.4, alpha=0.7,
               label=cfg['threshold_label'])

    ax.set_ylabel(cfg['label'], fontsize=14)
    ax.set_title(f'Pharmacology Comparison '
                 f'({metric_key.replace("_", " ").title()})',
                 fontsize=16, fontweight='bold')
    _decorate_da_axis(ax, da_arr)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")


# ----------------------------------------------------------------------
#  Figure 5: Inter-metric correlation matrix
# ----------------------------------------------------------------------
def plot_metric_correlations(sweep_results: dict, save_path: str):
    """Pairwise scatter of all metrics across DA levels (sanity-check)."""
    da_values = sorted(sweep_results.keys())
    metric_keys = ['persistence', 'selectivity', 'decay_ratio',
                   'd_prime', 'accuracy', 'composite_score']
    n = len(metric_keys)

    data = {k: np.array([sweep_results[d]['mean'].get(k, np.nan)
                         for d in da_values])
            for k in metric_keys}
    da_arr = np.array(da_values)

    fig, axes = plt.subplots(n, n, figsize=(3.0 * n, 3.0 * n))
    fig.suptitle('Inter-Metric Correlations (color = DA concentration)',
                 fontsize=15, fontweight='bold', y=1.00)

    for i, ki in enumerate(metric_keys):
        for j, kj in enumerate(metric_keys):
            ax = axes[i, j]
            if i == j:
                vals = data[ki][np.isfinite(data[ki])]
                if len(vals) > 0:
                    ax.hist(vals, bins=10, color='steelblue',
                            edgecolor='black', alpha=0.7)
                ax.set_title(ki, fontsize=10)
            else:
                ax.scatter(data[kj], data[ki], c=da_arr, cmap='viridis',
                           s=40, edgecolors='black', linewidths=0.5)
                xi = data[ki]; xj = data[kj]
                mask = np.isfinite(xi) & np.isfinite(xj)
                if mask.sum() >= 3 and np.std(xi[mask]) > 1e-3 and np.std(xj[mask]) > 1e-3:
                    r = float(np.corrcoef(xi[mask], xj[mask])[0, 1])
                    ax.text(0.05, 0.95, f'r={r:+.2f}',
                            transform=ax.transAxes,
                            fontsize=9, fontweight='bold',
                            verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='white', alpha=0.8))
            if i == n - 1:
                ax.set_xlabel(kj, fontsize=9)
            if j == 0:
                ax.set_ylabel(ki, fontsize=9)
            ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved: {save_path}")
