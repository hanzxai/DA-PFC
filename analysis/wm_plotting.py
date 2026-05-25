"""WM-specific plotting helpers.

We want the WM demo to *also* reuse the project's standard combined raster
and combined firing-rate figures (raster.png + rates_all.png style), AND to
add a dedicated WM-overview plot that highlights:
    - cue / delay / probe windows (shaded background bands)
    - Mem-A vs Mem-B vs E-Other firing rate traces
    - Mem-A raster restricted to neurons in the WM pool

Because PFCAnalyzer groups its sub-populations from the D1/D2 receptor masks
only, the Mem-A / Mem-B pools are NOT part of the standard 6 groups.  We
register them here as additional groups on the analyzer instance (in-place
patch) so existing plotting code can pick them up.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.analyzer import PFCAnalyzer

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
})


# ----------------------------------------------------------------------
#  Helper: register WM pool groups onto the analyzer
# ----------------------------------------------------------------------
def register_wm_groups(analyzer: PFCAnalyzer):
    """Add Mem-A and Mem-B as analysis groups using groups_info indices.

    Also carves "background" sub-groups out of whichever standard E-segment
    the memory pools live in, so plotting can show pure-receptor traces
    (D1-BG / E-BG) that exclude the memory pools.
    """
    gi = analyzer.data['groups_info']
    if 'mem_a_start' not in gi:
        return  # not a WM run
    N = analyzer.N
    mem_a = np.zeros(N, dtype=bool)
    mem_b = np.zeros(N, dtype=bool)
    mem_a[gi['mem_a_start']:gi['mem_a_end']] = True
    mem_b[gi['mem_b_start']:gi['mem_b_end']] = True
    analyzer.groups['Mem-A'] = mem_a
    analyzer.groups['Mem-B'] = mem_b

    # Carve background groups out of *every* standard E-segment, removing
    # whatever fraction of Mem-A/B happens to live there.  In practice the
    # pools are entirely inside one segment (E-D1 in the homogeneous-D1
    # layout, or E-Other in the legacy layout), but doing it generically
    # keeps the code robust to future layout changes.
    for src_name, dst_name in [('E-D1',   'D1-BG'),
                               ('E-D2',   'D2-BG'),
                               ('E-Other','E-BG')]:
        if src_name not in analyzer.groups:
            continue
        bg = analyzer.groups[src_name].copy()
        bg[mem_a] = False
        bg[mem_b] = False
        analyzer.groups[dst_name] = bg

    # Custom colors
    analyzer.COLORS['Mem-A'] = '#e41a1c'   # bright red
    analyzer.COLORS['Mem-B'] = '#377eb8'   # bright blue
    analyzer.COLORS['E-BG']  = '#999999'   # gray
    analyzer.COLORS['D1-BG'] = '#7f3b08'   # dark brown (D1 BG)
    analyzer.COLORS['D2-BG'] = '#1b7837'   # dark green (D2 BG)


# ----------------------------------------------------------------------
#  Helper: shade WM protocol windows on a matplotlib axes
# ----------------------------------------------------------------------
def _shade_wm_protocol(ax, protocol: dict, x_in_seconds: bool):
    """Shade cue/delay/probe windows on a time-axis."""
    factor = 1.0 / 1000.0 if x_in_seconds else 1.0
    cue_a_on  = protocol['cue_a_onset']  * factor
    cue_a_off = protocol['cue_a_offset'] * factor
    delay_end = cue_a_off + protocol['delay_ms'] * factor
    probe_end = delay_end + protocol['probe_ms'] * factor

    ylim = ax.get_ylim()
    ax.axvspan(cue_a_on,  cue_a_off,  color='#ffcccc', alpha=0.55, zorder=0,
               label='Cue (A)')
    ax.axvspan(cue_a_off, delay_end,  color='#fff2b3', alpha=0.45, zorder=0,
               label='Delay')
    ax.axvspan(delay_end, probe_end,  color='#cce5ff', alpha=0.45, zorder=0,
               label='Probe')

    # Optional cue B (distractor)
    if protocol.get('cue_b_amplitude', 0.0) > 0.0:
        b_on  = protocol['cue_b_onset']  * factor
        b_off = protocol['cue_b_offset'] * factor
        ax.axvspan(b_on, b_off, color='#cccccc', alpha=0.40, zorder=0,
                   label='Cue (B)')
    ax.set_ylim(ylim)


# ----------------------------------------------------------------------
#  Helper: draw the DA(t) schedule for a given WM batch
# ----------------------------------------------------------------------
def _draw_wm_da_timeline(ax, protocol: dict, batch_idx: int,
                         duration: float, use_seconds: bool):
    """Draw [DA] vs time for a WM batch.

    Batch 0 (Control): DA constant at da_base throughout.
    Batch 1 (Experiment): DA = da_base, except [da_pulse_onset, da_pulse_offset)
                          where DA = da_pulse (square pulse).
    """
    factor = 1.0 / 1000.0 if use_seconds else 1.0
    da_base   = float(protocol.get('da_base',   0.0))
    da_pulse  = float(protocol.get('da_pulse',  da_base))
    pulse_on  = float(protocol.get('da_pulse_onset',  0.0))
    pulse_off = float(protocol.get('da_pulse_offset', 0.0))
    has_pulse = (pulse_off > pulse_on) and (abs(da_pulse - da_base) > 1e-9)

    if batch_idx == 0 or not has_pulse:
        # Flat DA = da_base
        t_arr = np.array([0.0, duration]) * factor
        d_arr = np.array([da_base, da_base])
    else:
        # Square pulse: da_base -> da_pulse -> da_base
        t_arr = np.array([0.0, pulse_on, pulse_on,
                          pulse_off, pulse_off, duration]) * factor
        d_arr = np.array([da_base, da_base, da_pulse,
                          da_pulse, da_base, da_base])

    ax.plot(t_arr, d_arr, color='#E91E63', linewidth=2.6, alpha=0.95,
            zorder=3)
    ax.fill_between(t_arr, 0, d_arr, color='#E91E63', alpha=0.18, zorder=2)
    ax.set_xlim(0, duration * factor)
    ax.set_xlabel("Time (s)" if use_seconds else "Time (ms)")
    ax.set_ylabel("[DA] (nM)")
    # Y range with margin
    da_max = max(da_base, da_pulse)
    da_min = min(da_base, da_pulse)
    margin = max((da_max - da_min) * 0.20, 0.5)
    ax.set_ylim(max(0.0, da_min - margin), da_max + margin)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)


# ----------------------------------------------------------------------
#  WM Overview Figure: rasters + firing-rate traces with protocol bands
# ----------------------------------------------------------------------
def plot_wm_overview(analyzer: PFCAnalyzer, save_dir=None,
                     filename: str = 'wm_overview.png'):
    """
    A 3-row × 2-col figure tailored for WM demos:
        Row 0 = DA(t) timeline (Control flat | Experiment with optional pulse)
        Row 1 = Raster (Mem-A / Mem-B / background) per batch
        Row 2 = Firing-rate traces per batch
    Cue / delay / probe windows are shaded on rows 1 & 2.
    """
    print("🎨 Plotting WM overview figure...")
    register_wm_groups(analyzer)
    protocol = analyzer.cfg.get('wm_protocol', {})
    if not protocol:
        print("⚠️  Skipping wm_overview: no wm_protocol metadata.")
        return

    duration = analyzer.duration
    use_seconds = duration > 10000

    da_base  = float(protocol.get('da_base',  analyzer.cfg.get('da_level', 0.0)))
    da_pulse = float(protocol.get('da_pulse', da_base))
    pulse_on  = float(protocol.get('da_pulse_onset',  0.0))
    pulse_off = float(protocol.get('da_pulse_offset', 0.0))
    has_pulse = (pulse_off > pulse_on) and (abs(da_pulse - da_base) > 1e-9)

    fig, axes = plt.subplots(3, 2, figsize=(22, 16), dpi=180,
                             gridspec_kw={'height_ratios': [1, 3, 3]})

    # ---- Row 0: DA(t) timeline ----
    for col, batch_idx in enumerate([0, 1]):
        ax = axes[0, col]
        _draw_wm_da_timeline(ax, protocol, batch_idx, duration, use_seconds)
        if batch_idx == 0:
            title = f"DA Concentration — Control (constant {da_base:g} nM)"
        else:
            if has_pulse:
                title = (f"DA Concentration — Experiment "
                         f"({da_base:g}→{da_pulse:g} nM in "
                         f"[{pulse_on:.0f},{pulse_off:.0f}) ms)")
            else:
                title = (f"DA Concentration — Experiment "
                         f"(constant {da_base:g} nM, no pulse)")
        ax.set_title(title)
        # Shade protocol windows on DA timeline too (helps line up cue/delay)
        _shade_wm_protocol(ax, protocol, use_seconds)

    # ---- Row 1: rasters of both batches ----
    for col, batch_idx in enumerate([0, 1]):
        ax = axes[1, col]
        spikes = analyzer._get_spikes_for_batch(batch_idx)
        if len(spikes) > 0:
            ts_ms = spikes[:, 0] * analyzer.dt
            ids   = spikes[:, 1]
            x = ts_ms / 1000.0 if use_seconds else ts_ms
            for grp in ['Mem-A', 'Mem-B', 'E-BG', 'I-D1', 'I-D2', 'I-Other',
                        'E-D1', 'E-D2']:
                if grp not in analyzer.groups:
                    continue
                valid = np.where(analyzer.groups[grp])[0]
                m = np.isin(ids, valid)
                if m.any():
                    ax.scatter(x[m], ids[m], s=1.4,
                               c=analyzer.COLORS.get(grp, 'k'),
                               alpha=0.7, label=grp)
        batch_label = 'Control' if batch_idx == 0 else 'Experiment'
        if batch_idx == 0:
            da_str = f"DA={da_base:g} nM"
        else:
            da_str = (f"DA={da_base:g}→{da_pulse:g} nM"
                      if has_pulse else f"DA={da_base:g} nM")
        ax.set_title(f"Raster — {batch_label} ({da_str})")
        ax.set_xlabel("Time (s)" if use_seconds else "Time (ms)")
        ax.set_ylabel("Neuron ID")
        ax.set_xlim(0, duration / 1000.0 if use_seconds else duration)
        ax.set_ylim(0, analyzer.N)
        ax.legend(markerscale=4, loc='upper right', ncol=2, fontsize=10,
                  framealpha=0.85)
        _shade_wm_protocol(ax, protocol, use_seconds)

    # ---- Row 2: firing-rate traces ----
    rate_groups_top = ['Mem-A', 'Mem-B', 'E-BG']
    rate_groups_bot = ['I-D1', 'I-D2', 'I-Other']

    for col, batch_idx in enumerate([0, 1]):
        ax = axes[2, col]
        for grp in rate_groups_top + rate_groups_bot:
            if grp not in analyzer.groups:
                continue
            centers, rate = analyzer.compute_group_rate(batch_idx, grp,
                                                       time_win=20.0, sigma=2.0)
            if rate is None:
                continue
            x = centers / 1000.0 if use_seconds else centers
            lw = 2.2 if grp in rate_groups_top else 1.3
            ls = '-' if grp in rate_groups_top else '--'
            ax.plot(x, rate, lw=lw, ls=ls,
                    color=analyzer.COLORS.get(grp, 'k'), label=grp)
        batch_label = 'Control' if batch_idx == 0 else 'Experiment'
        ax.set_title(f"Firing Rate — {batch_label}")
        ax.set_xlabel("Time (s)" if use_seconds else "Time (ms)")
        ax.set_ylabel("Rate (Hz)")
        ax.set_xlim(0, duration / 1000.0 if use_seconds else duration)
        ax.legend(loc='upper right', ncol=2, fontsize=11, framealpha=0.85)
        _shade_wm_protocol(ax, protocol, use_seconds)

    fig.suptitle("Working-Memory Demo (Scheme A: structured selective sub-pools)",
                 fontsize=20, y=1.00)
    fig.tight_layout()

    if save_dir is not None:
        out = os.path.join(str(save_dir), filename)
        fig.savefig(out, dpi=160, bbox_inches='tight')
        print(f"  💾 Saved: {out}")
    plt.close(fig)


# ----------------------------------------------------------------------
#  WM-specific firing-rate-only figure (mirrors combined_rates_all.png style)
#  Uses the standard E and I groups + Mem-A/Mem-B traces overlaid.
# ----------------------------------------------------------------------
def plot_wm_rates_pools(analyzer: PFCAnalyzer, save_dir=None,
                        filename: str = 'wm_rates_pools.png'):
    """
    A 1-row × 2-col figure (Control | Exp) showing Mem-A vs Mem-B vs E-BG
    firing rate, with protocol shading.  This is the cleanest indicator
    that working memory has been established (Mem-A persists, Mem-B does not).
    """
    print("🎨 Plotting WM pool firing-rate figure...")
    register_wm_groups(analyzer)
    protocol = analyzer.cfg.get('wm_protocol', {})
    if not protocol:
        return

    duration = analyzer.duration
    use_seconds = duration > 10000

    fig, axes = plt.subplots(1, 2, figsize=(22, 7), dpi=180, sharey=True)
    ymax = 0.0
    line_data = {0: {}, 1: {}}
    for batch_idx in [0, 1]:
        for grp in ['Mem-A', 'Mem-B', 'E-BG']:
            centers, rate = analyzer.compute_group_rate(batch_idx, grp,
                                                       time_win=20.0, sigma=2.0)
            line_data[batch_idx][grp] = (centers, rate)
            if rate is not None and len(rate) > 0:
                ymax = max(ymax, float(np.max(rate)))
    ymax = max(5.0, ymax * 1.1)

    for col, batch_idx in enumerate([0, 1]):
        ax = axes[col]
        for grp in ['Mem-A', 'Mem-B', 'E-BG']:
            centers, rate = line_data[batch_idx][grp]
            if rate is None:
                continue
            x = centers / 1000.0 if use_seconds else centers
            ax.plot(x, rate, lw=2.5, color=analyzer.COLORS.get(grp, 'k'),
                    label=grp)
        batch_label = 'Control' if batch_idx == 0 else 'Experiment'
        da_label = analyzer.cfg.get('da_level', 0.0)
        ax.set_title(f"{batch_label} (DA={da_label} nM)")
        ax.set_xlabel("Time (s)" if use_seconds else "Time (ms)")
        if col == 0:
            ax.set_ylabel("Firing Rate (Hz)")
        ax.set_xlim(0, duration / 1000.0 if use_seconds else duration)
        ax.set_ylim(0, ymax)
        ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
        _shade_wm_protocol(ax, protocol, use_seconds)

    fig.suptitle("Working-Memory Pool Activity (Mem-A vs Mem-B vs Background-E)",
                 fontsize=20, y=1.02)
    fig.tight_layout()

    if save_dir is not None:
        out = os.path.join(str(save_dir), filename)
        fig.savefig(out, dpi=160, bbox_inches='tight')
        print(f"  💾 Saved: {out}")
    plt.close(fig)


# ----------------------------------------------------------------------
#  WM persistence metrics (Hz averages within cue/delay/probe windows)
# ----------------------------------------------------------------------
def compute_wm_metrics(analyzer: PFCAnalyzer) -> dict:
    """Compute mean Mem-A / Mem-B firing rate inside each protocol window."""
    register_wm_groups(analyzer)
    protocol = analyzer.cfg.get('wm_protocol', {})
    if not protocol:
        return {}

    cue_on  = protocol['cue_a_onset']
    cue_off = protocol['cue_a_offset']
    delay_end = cue_off + protocol['delay_ms']
    probe_end = delay_end + protocol['probe_ms']

    # Split delay into early (first 1/3) and late (last 1/3) so we can
    # distinguish a true attractor (late ≈ early) from a slow decay
    # (late ≪ early).  This is the key WM diagnostic.
    delay_dur = delay_end - cue_off
    delay_early_end = cue_off + delay_dur / 3.0
    delay_late_start = delay_end - delay_dur / 3.0

    windows = {
        'baseline':    (0.0, cue_on),
        'cue':         (cue_on, cue_off),
        'delay':       (cue_off, delay_end),
        'delay_early': (cue_off, delay_early_end),
        'delay_late':  (delay_late_start, delay_end),
        'probe':       (delay_end, probe_end),
    }

    results = {}
    for batch_idx in [0, 1]:
        results[batch_idx] = {}
        for grp in ['Mem-A', 'Mem-B', 'E-BG', 'I-D1', 'I-D2']:
            centers, rate = analyzer.compute_group_rate(batch_idx, grp,
                                                       time_win=20.0, sigma=2.0)
            if rate is None:
                continue
            results[batch_idx][grp] = {}
            for wname, (lo, hi) in windows.items():
                m = (centers >= lo) & (centers < hi)
                results[batch_idx][grp][wname] = (
                    float(np.mean(rate[m])) if m.any() else 0.0)
    return results


def print_wm_report(analyzer: PFCAnalyzer, save_path: str = None) -> str:
    """Pretty-print a WM persistence summary table."""
    metrics = compute_wm_metrics(analyzer)
    if not metrics:
        return ""

    lines = []
    lines.append("\n" + "=" * 110)
    lines.append("  \U0001f9e0 Working-Memory Persistence Report  (Wang-2002 NMDA attractor)")
    lines.append("=" * 110)
    protocol = analyzer.cfg.get('wm_protocol', {})
    da_base = protocol.get('da_base', None)
    da_pulse = protocol.get('da_pulse', None)
    da_on = protocol.get('da_pulse_onset', 0.0)
    da_off = protocol.get('da_pulse_offset', 0.0)
    if da_base is not None:
        if da_pulse is not None and da_pulse != da_base and da_off > da_on:
            lines.append(f"  DA schedule: Batch0={da_base:g} nM (Control)  |  "
                         f"Batch1={da_base:g}->{da_pulse:g} nM in "
                         f"[{da_on:.0f},{da_off:.0f}) ms (Exp)")
        else:
            lines.append(f"  DA schedule: both batches held at {da_base:g} nM (no pulse)")
    delta_label = "\u0394(Late-BL)"
    header = (f"  {'Group':<10}{'Batch':<8}"
              f"{'Baseline':>10}{'Cue':>10}{'D-Early':>10}{'D-Late':>10}{'Probe':>10}"
              f"{delta_label:>14}")
    lines.append(header)
    lines.append("  " + "\u2500" * 108)
    for batch_idx in [0, 1]:
        bl = 'Control' if batch_idx == 0 else 'Exp'
        for grp in ['Mem-A', 'Mem-B', 'E-BG', 'I-D1', 'I-D2']:
            if grp not in metrics[batch_idx]:
                continue
            r = metrics[batch_idx][grp]
            d = r['delay_late'] - r['baseline']
            lines.append(
                f"  {grp:<10}{bl:<8}"
                f"{r['baseline']:>9.2f} {r['cue']:>9.2f} "
                f"{r['delay_early']:>9.2f} {r['delay_late']:>9.2f} "
                f"{r['probe']:>9.2f} {d:>+13.2f}"
            )
        lines.append("  " + "\u2500" * 108)

    # ---- Automatic WM-PASS judgement (Exp batch, Mem-A vs Mem-B) ----
    if 'Mem-A' in metrics[1] and 'Mem-B' in metrics[1]:
        a = metrics[1]['Mem-A']
        b = metrics[1]['Mem-B']
        d_late_a = a['delay_late'] - a['baseline']
        d_late_b = b['delay_late'] - b['baseline']
        decay_ratio = (a['delay_late'] - a['baseline']) / max(
            (a['cue'] - a['baseline']), 1e-3)
        lines.append(f"  WM judgement (Mem-A, Exp batch):")
        lines.append(f"    \u0394(Late-BL) Mem-A   = {d_late_a:+.2f} Hz    "
                     f"(>= +5 Hz required for WM)")
        lines.append(f"    \u0394(Late-BL) Mem-B   = {d_late_b:+.2f} Hz    "
                     f"(should be \u2248 0 for selectivity)")
        lines.append(f"    Persistence ratio    = {decay_ratio*100:5.1f}% of cue elevation "
                     f"retained in late delay")
        passes_persistence = d_late_a >= 5.0
        passes_selectivity = abs(d_late_b) < 2.0
        passes_persratio = decay_ratio >= 0.30
        verdict = ("\u2705 WM ESTABLISHED" if (passes_persistence and passes_selectivity
                                              and passes_persratio)
                   else "\u274c WM NOT established")
        lines.append(f"    Verdict: {verdict}")
    lines.append("=" * 110)

    text = "\n".join(lines)
    print(text)

    if save_path is not None:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"  \U0001f4be WM report saved: {save_path}")
    return text
