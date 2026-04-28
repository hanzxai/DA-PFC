#!/usr/bin/env python3
"""
Alpha Dynamics Visualizer — Langmuir Receptor Binding Kinetics

Visualize how α_D1 and α_D2 evolve over time under various DA protocols,
WITHOUT running the full SNN simulation. Pure mathematical computation.

Key curves plotted:
  1. α(t)    : Langmuir kinetics trajectory (Euler integration)
  2. S(DA)   : Sigmoid target = 1 / (1 + exp(-β·(DA - EC50)))
  3. α_ss    : Langmuir steady-state = k_on·S / (k_on·S + k_off)

Usage:
  # Single DA step (default: 2 nM → 15 nM)
  python -m models.alpha_visualizer

  # Custom DA levels comparison
  python -m models.alpha_visualizer --da-levels 5 10 15 20

  # Two-stage protocol
  python -m models.alpha_visualizer --mode two-stage --da1 2 --da2 15

  # Pulse protocol
  python -m models.alpha_visualizer --mode pulse --da-base 2 --da-pulse 15 --pulse-dur 30
"""
import os
import sys
import math
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ======================================================================
# Core computation: Langmuir kinetics parameters (same as kernels.py)
# ======================================================================

def _get_kinetics_params():
    """
    Derive Langmuir k_on / k_off from config, matching kernels.py exactly.

    Returns:
        dict with k_on_d1, k_off_d1, k_on_d2, k_off_d2, ec50_d1, ec50_d2, beta
    """
    return {
        'k_on_d1':  1.0 / (config.TAU_ON_D1 - 3000),
        'k_off_d1': 1.0 / (config.TAU_OFF_D1 + 3000),
        'k_on_d2':  1.0 / config.TAU_ON_D2,
        'k_off_d2': 1.0 / config.TAU_OFF_D2,
        'ec50_d1':  config.EC50_D1,
        'ec50_d2':  config.EC50_D2,
        'beta':     config.BETA,
        'da_baseline': config.DA_BASELINE,
    }


def sigmoid(da, ec50, beta):
    """Sigmoid activation: S(DA) = 1 / (1 + exp(-β·(DA - EC50)))"""
    return 1.0 / (1.0 + np.exp(-beta * (da - ec50)))


def langmuir_steady_state(s_val, k_on, k_off):
    """Langmuir steady-state: α_ss = k_on·S / (k_on·S + k_off)"""
    return k_on * s_val / (k_on * s_val + k_off)


# ======================================================================
# Simulation: Euler integration of Langmuir ODE
# ======================================================================

def simulate_alpha_dynamics(
    da_schedule_fn,
    duration_s: float = 300.0,
    dt_ms: float = 1.0,
    alpha_d1_init: float = 0.0,
    alpha_d2_init: float = 0.0,
):
    """
    Simulate α_D1 and α_D2 evolution using Langmuir kinetics (Euler method).

    Args:
        da_schedule_fn : callable(t_ms) -> float, DA concentration at time t (ms)
        duration_s     : total simulation duration (seconds)
        dt_ms          : time step (ms), default 1.0 (matches config.DT)
        alpha_d1_init  : initial α_D1 value
        alpha_d2_init  : initial α_D2 value

    Returns:
        dict with keys:
          'time_s'   : (N,) time array in seconds
          'da'       : (N,) DA concentration at each step (nM)
          'alpha_d1' : (N,) α_D1 trajectory
          'alpha_d2' : (N,) α_D2 trajectory
          's_d1'     : (N,) Sigmoid target S_D1(DA)
          's_d2'     : (N,) Sigmoid target S_D2(DA)
          'alpha_ss_d1' : (N,) Langmuir steady-state α_ss for D1
          'alpha_ss_d2' : (N,) Langmuir steady-state α_ss for D2
    """
    p = _get_kinetics_params()
    duration_ms = duration_s * 1000.0
    steps = int(duration_ms / dt_ms)

    # Pre-allocate arrays
    time_s = np.zeros(steps)
    da_arr = np.zeros(steps)
    alpha_d1 = np.zeros(steps)
    alpha_d2 = np.zeros(steps)
    s_d1_arr = np.zeros(steps)
    s_d2_arr = np.zeros(steps)
    alpha_ss_d1 = np.zeros(steps)
    alpha_ss_d2 = np.zeros(steps)

    # Initial conditions
    a_d1 = alpha_d1_init
    a_d2 = alpha_d2_init

    for i in range(steps):
        t_ms = i * dt_ms
        t_s = t_ms / 1000.0
        da = da_schedule_fn(t_ms)

        # Sigmoid targets
        s1 = sigmoid(da, p['ec50_d1'], p['beta'])
        s2 = sigmoid(da, p['ec50_d2'], p['beta'])

        # Langmuir steady-state
        ss1 = langmuir_steady_state(s1, p['k_on_d1'], p['k_off_d1'])
        ss2 = langmuir_steady_state(s2, p['k_on_d2'], p['k_off_d2'])

        # Record
        time_s[i] = t_s
        da_arr[i] = da
        alpha_d1[i] = a_d1
        alpha_d2[i] = a_d2
        s_d1_arr[i] = s1
        s_d2_arr[i] = s2
        alpha_ss_d1[i] = ss1
        alpha_ss_d2[i] = ss2

        # Euler step: dα/dt = k_on · S · (1 - α) - k_off · α
        d_a_d1 = p['k_on_d1'] * s1 * (1.0 - a_d1) - p['k_off_d1'] * a_d1
        d_a_d2 = p['k_on_d2'] * s2 * (1.0 - a_d2) - p['k_off_d2'] * a_d2

        a_d1 = np.clip(a_d1 + d_a_d1 * dt_ms, 0.0, 1.0)
        a_d2 = np.clip(a_d2 + d_a_d2 * dt_ms, 0.0, 1.0)

    return {
        'time_s': time_s,
        'da': da_arr,
        'alpha_d1': alpha_d1,
        'alpha_d2': alpha_d2,
        's_d1': s_d1_arr,
        's_d2': s_d2_arr,
        'alpha_ss_d1': alpha_ss_d1,
        'alpha_ss_d2': alpha_ss_d2,
    }


# ======================================================================
# DA schedule factories
# ======================================================================

def make_step_schedule(da_before: float, da_after: float, onset_s: float):
    """Step function: da_before → da_after at onset_s."""
    onset_ms = onset_s * 1000.0
    def schedule(t_ms):
        return da_after if t_ms >= onset_ms else da_before
    return schedule


def make_two_stage_schedule(da_baseline: float, da1: float, da2: float,
                            onset1_s: float, onset2_s: float):
    """Three-phase: baseline → da1 → da2."""
    onset1_ms = onset1_s * 1000.0
    onset2_ms = onset2_s * 1000.0
    def schedule(t_ms):
        if t_ms >= onset2_ms:
            return da2
        elif t_ms >= onset1_ms:
            return da1
        else:
            return da_baseline
    return schedule


def make_pulse_schedule(da_base: float, da_pulse: float,
                        pulse_onset_s: float, pulse_offset_s: float):
    """Pulse: da_base → da_pulse → da_base."""
    onset_ms = pulse_onset_s * 1000.0
    offset_ms = pulse_offset_s * 1000.0
    def schedule(t_ms):
        if onset_ms <= t_ms < offset_ms:
            return da_pulse
        return da_base
    return schedule


# ======================================================================
# Plotting functions
# ======================================================================

def plot_alpha_step(da_before: float = None, da_after: float = 15.0,
                    onset_s: float = 10.0, duration_s: float = 300.0,
                    save_path: str = None):
    """
    Plot α_D1 and α_D2 dynamics for a single DA step.

    Shows:
      - α(t) trajectories (solid lines)
      - Sigmoid target S(DA) (dashed horizontal lines)
      - Langmuir steady-state α_ss (dotted horizontal lines)
      - DA concentration timeline (top panel)

    Args:
        da_before  : DA before step (nM). Default = config.DA_BASELINE
        da_after   : DA after step (nM). Default = 15.0
        onset_s    : step onset time (s). Default = 10.0
        duration_s : total duration (s). Default = 300.0
        save_path  : file path to save figure. None = show interactively.
    """
    if da_before is None:
        da_before = config.DA_BASELINE

    p = _get_kinetics_params()

    # Compute initial alpha at baseline DA steady-state
    s1_bl = sigmoid(da_before, p['ec50_d1'], p['beta'])
    s2_bl = sigmoid(da_before, p['ec50_d2'], p['beta'])
    a_d1_init = langmuir_steady_state(s1_bl, p['k_on_d1'], p['k_off_d1'])
    a_d2_init = langmuir_steady_state(s2_bl, p['k_on_d2'], p['k_off_d2'])

    schedule = make_step_schedule(da_before, da_after, onset_s)
    data = simulate_alpha_dynamics(schedule, duration_s=duration_s,
                                   alpha_d1_init=a_d1_init,
                                   alpha_d2_init=a_d2_init)

    # Target values at da_after
    s1_target = sigmoid(da_after, p['ec50_d1'], p['beta'])
    s2_target = sigmoid(da_after, p['ec50_d2'], p['beta'])
    ss1_target = langmuir_steady_state(s1_target, p['k_on_d1'], p['k_off_d1'])
    ss2_target = langmuir_steady_state(s2_target, p['k_on_d2'], p['k_off_d2'])

    # Kd values for annotation
    kd_d1 = p['k_off_d1'] / p['k_on_d1']
    kd_d2 = p['k_off_d2'] / p['k_on_d2']

    # ---- Figure: 3 panels ----
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True,
                              gridspec_kw={'height_ratios': [1, 3, 3]})
    fig.suptitle(
        f"Langmuir Receptor Binding Kinetics — DA Step: {da_before} → {da_after} nM\n"
        f"D1: k_on={p['k_on_d1']:.2e} ms⁻¹, k_off={p['k_off_d1']:.2e} ms⁻¹, Kd={kd_d1:.4f}  |  "
        f"D2: k_on={p['k_on_d2']:.2e} ms⁻¹, k_off={p['k_off_d2']:.2e} ms⁻¹, Kd={kd_d2:.4f}",
        fontsize=13, fontweight='bold',
    )

    t = data['time_s']

    # --- Panel 0: DA concentration ---
    ax = axes[0]
    ax.fill_between(t, 0, data['da'], color='#4CAF50', alpha=0.3)
    ax.plot(t, data['da'], color='#4CAF50', linewidth=2)
    ax.set_ylabel('[DA] (nM)', fontsize=12)
    ax.set_title('DA Concentration', fontsize=13)
    ax.set_ylim(0, max(da_before, da_after) * 1.2)
    ax.grid(True, alpha=0.3)

    # --- Panel 1: α_D1 ---
    ax = axes[1]
    ax.plot(t, data['alpha_d1'], '-', color='#d62728', linewidth=2.5,
            label=f'α_D1(t) — Langmuir trajectory')
    ax.axhline(s1_target, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.6,
               label=f'S_D1(DA={da_after}) = {s1_target:.4f} (Sigmoid target)')
    ax.axhline(ss1_target, color='#d62728', linestyle=':', linewidth=1.5, alpha=0.8,
               label=f'α_ss_D1 = {ss1_target:.4f} (Langmuir steady-state)')
    # Baseline reference
    ax.axhline(a_d1_init, color='gray', linestyle='-.', linewidth=1, alpha=0.4,
               label=f'α_D1 baseline (DA={da_before}) = {a_d1_init:.4f}')
    ax.axvline(onset_s, color='green', linestyle=':', linewidth=1.5, alpha=0.5)

    ax.set_ylabel('α_D1', fontsize=12)
    ax.set_title(f'D1 Receptor Activation  (EC50={p["ec50_d1"]} nM, '
                 f'τ_on={config.TAU_ON_D1/1000:.1f}s, τ_off={config.TAU_OFF_D1/1000:.1f}s)',
                 fontsize=13)
    ax.legend(fontsize=10, loc='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, max(s1_target, ss1_target, data['alpha_d1'].max()) * 1.15 + 0.02)

    # --- Panel 2: α_D2 ---
    ax = axes[2]
    ax.plot(t, data['alpha_d2'], '-', color='#1f77b4', linewidth=2.5,
            label=f'α_D2(t) — Langmuir trajectory')
    ax.axhline(s2_target, color='#1f77b4', linestyle='--', linewidth=1.5, alpha=0.6,
               label=f'S_D2(DA={da_after}) = {s2_target:.4f} (Sigmoid target)')
    ax.axhline(ss2_target, color='#1f77b4', linestyle=':', linewidth=1.5, alpha=0.8,
               label=f'α_ss_D2 = {ss2_target:.4f} (Langmuir steady-state)')
    ax.axhline(a_d2_init, color='gray', linestyle='-.', linewidth=1, alpha=0.4,
               label=f'α_D2 baseline (DA={da_before}) = {a_d2_init:.4f}')
    ax.axvline(onset_s, color='green', linestyle=':', linewidth=1.5, alpha=0.5)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('α_D2', fontsize=12)
    ax.set_title(f'D2 Receptor Activation  (EC50={p["ec50_d2"]} nM, '
                 f'τ_on={config.TAU_ON_D2/1000:.1f}s, τ_off={config.TAU_OFF_D2/1000:.1f}s)',
                 fontsize=13)
    ax.legend(fontsize=10, loc='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, max(s2_target, ss2_target, data['alpha_d2'].max()) * 1.15 + 0.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")
    else:
        plt.savefig("outputs/alpha_step.png", dpi=150, bbox_inches='tight')
        print(f"📊 Saved: outputs/alpha_step.png")
    plt.close(fig)

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Alpha Dynamics Summary — DA Step: {da_before} → {da_after} nM")
    print(f"{'='*70}")
    print(f"  {'':15s} {'D1':>12s} {'D2':>12s}")
    print(f"  {'─'*15}─{'─'*12}─{'─'*12}")
    print(f"  {'S(DA) target':<15s} {s1_target:>12.4f} {s2_target:>12.4f}")
    print(f"  {'α_ss Langmuir':<15s} {ss1_target:>12.4f} {ss2_target:>12.4f}")
    print(f"  {'α_ss / S(DA)':<15s} {ss1_target/s1_target:>12.4f} {ss2_target/s2_target:>12.4f}")
    print(f"  {'Kd':<15s} {kd_d1:>12.4f} {kd_d2:>12.4f}")
    print(f"  {'α baseline':<15s} {a_d1_init:>12.4f} {a_d2_init:>12.4f}")
    print(f"  {'α final':<15s} {data['alpha_d1'][-1]:>12.4f} {data['alpha_d2'][-1]:>12.4f}")
    print(f"{'='*70}")

    return data


def plot_alpha_multi_da(da_levels: list = None, onset_s: float = 10.0,
                        duration_s: float = 300.0, save_path: str = None):
    """
    Compare α dynamics across multiple DA concentrations (overlay plot).

    Args:
        da_levels  : list of DA concentrations (nM). Default = [3, 5, 10, 15, 20]
        onset_s    : step onset time (s)
        duration_s : total duration (s)
        save_path  : file path to save figure
    """
    if da_levels is None:
        da_levels = [3.0, 5.0, 10.0, 15.0, 20.0]

    p = _get_kinetics_params()
    da_before = config.DA_BASELINE

    # Baseline initial conditions
    s1_bl = sigmoid(da_before, p['ec50_d1'], p['beta'])
    s2_bl = sigmoid(da_before, p['ec50_d2'], p['beta'])
    a_d1_init = langmuir_steady_state(s1_bl, p['k_on_d1'], p['k_off_d1'])
    a_d2_init = langmuir_steady_state(s2_bl, p['k_on_d2'], p['k_off_d2'])

    cmap = plt.cm.viridis(np.linspace(0.15, 0.95, len(da_levels)))

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(
        f"Langmuir α Dynamics — Multi-DA Comparison (Baseline={da_before} nM)\n"
        f"D1: EC50={p['ec50_d1']} nM  |  D2: EC50={p['ec50_d2']} nM",
        fontsize=14, fontweight='bold',
    )

    all_data = {}
    for idx, da in enumerate(da_levels):
        schedule = make_step_schedule(da_before, da, onset_s)
        data = simulate_alpha_dynamics(schedule, duration_s=duration_s,
                                       alpha_d1_init=a_d1_init,
                                       alpha_d2_init=a_d2_init)
        all_data[da] = data
        color = cmap[idx]
        t = data['time_s']

        # Top-left: α_D1 trajectories
        axes[0, 0].plot(t, data['alpha_d1'], '-', color=color, linewidth=2,
                        label=f'DA={da} nM')
        # Top-right: α_D2 trajectories
        axes[0, 1].plot(t, data['alpha_d2'], '-', color=color, linewidth=2,
                        label=f'DA={da} nM')

    # Bottom-left: Sigmoid S(DA) and α_ss vs DA (dose-response curve)
    da_range = np.linspace(0, max(da_levels) * 1.3, 200)
    s1_curve = sigmoid(da_range, p['ec50_d1'], p['beta'])
    s2_curve = sigmoid(da_range, p['ec50_d2'], p['beta'])
    ss1_curve = langmuir_steady_state(s1_curve, p['k_on_d1'], p['k_off_d1'])
    ss2_curve = langmuir_steady_state(s2_curve, p['k_on_d2'], p['k_off_d2'])

    ax = axes[1, 0]
    ax.plot(da_range, s1_curve, '--', color='#d62728', linewidth=2, label='S_D1 (Sigmoid)')
    ax.plot(da_range, ss1_curve, '-', color='#d62728', linewidth=2.5, label='α_ss_D1 (Langmuir)')
    ax.plot(da_range, s2_curve, '--', color='#1f77b4', linewidth=2, label='S_D2 (Sigmoid)')
    ax.plot(da_range, ss2_curve, '-', color='#1f77b4', linewidth=2.5, label='α_ss_D2 (Langmuir)')
    # Mark tested DA levels
    for idx, da in enumerate(da_levels):
        s1 = sigmoid(da, p['ec50_d1'], p['beta'])
        ss1 = langmuir_steady_state(s1, p['k_on_d1'], p['k_off_d1'])
        s2 = sigmoid(da, p['ec50_d2'], p['beta'])
        ss2 = langmuir_steady_state(s2, p['k_on_d2'], p['k_off_d2'])
        ax.plot(da, ss1, 'o', color='#d62728', markersize=8, zorder=5)
        ax.plot(da, ss2, 's', color='#1f77b4', markersize=8, zorder=5)
    ax.axvline(p['ec50_d1'], color='#d62728', linestyle=':', alpha=0.3, linewidth=1)
    ax.axvline(p['ec50_d2'], color='#1f77b4', linestyle=':', alpha=0.3, linewidth=1)
    ax.set_xlabel('[DA] (nM)', fontsize=12)
    ax.set_ylabel('Activation', fontsize=12)
    ax.set_title('Dose-Response: Sigmoid Target vs Langmuir Steady-State', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Bottom-right: D1 vs D2 overlay for a representative DA level
    mid_da = da_levels[len(da_levels) // 2]
    data_mid = all_data[mid_da]
    ax = axes[1, 1]
    ax.plot(data_mid['time_s'], data_mid['alpha_d1'], '-', color='#d62728',
            linewidth=2.5, label=f'α_D1 (DA={mid_da} nM)')
    ax.plot(data_mid['time_s'], data_mid['alpha_d2'], '-', color='#1f77b4',
            linewidth=2.5, label=f'α_D2 (DA={mid_da} nM)')
    # Shade D1 afterglow region (where D1 > D2 after both start rising)
    d1 = data_mid['alpha_d1']
    d2 = data_mid['alpha_d2']
    ax.fill_between(data_mid['time_s'], d1, d2,
                    where=(d1 > d2), alpha=0.15, color='red', label='D1 > D2')
    ax.fill_between(data_mid['time_s'], d1, d2,
                    where=(d2 > d1), alpha=0.15, color='blue', label='D2 > D1')
    ax.axvline(onset_s, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('α', fontsize=12)
    ax.set_title(f'D1 vs D2 Temporal Segregation (DA={mid_da} nM)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Decorate top panels
    for i, (ax, label) in enumerate(zip([axes[0, 0], axes[0, 1]], ['α_D1', 'α_D2'])):
        ax.axvline(onset_s, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} Trajectories at Different DA Levels', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")
    else:
        plt.savefig("outputs/alpha_multi_da.png", dpi=150, bbox_inches='tight')
        print(f"📊 Saved: outputs/alpha_multi_da.png")
    plt.close(fig)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  Steady-State Summary — Sigmoid S(DA) vs Langmuir α_ss")
    print(f"{'='*80}")
    print(f"  {'DA (nM)':<10} │ {'S_D1':>8} {'α_ss_D1':>10} {'ratio':>8} │ "
          f"{'S_D2':>8} {'α_ss_D2':>10} {'ratio':>8}")
    print(f"  {'─'*10}─┼─{'─'*8}─{'─'*10}─{'─'*8}─┼─{'─'*8}─{'─'*10}─{'─'*8}")
    for da in da_levels:
        s1 = sigmoid(da, p['ec50_d1'], p['beta'])
        ss1 = langmuir_steady_state(s1, p['k_on_d1'], p['k_off_d1'])
        s2 = sigmoid(da, p['ec50_d2'], p['beta'])
        ss2 = langmuir_steady_state(s2, p['k_on_d2'], p['k_off_d2'])
        r1 = ss1 / s1 if s1 > 1e-6 else 0
        r2 = ss2 / s2 if s2 > 1e-6 else 0
        print(f"  {da:<10.1f} │ {s1:>8.4f} {ss1:>10.4f} {r1:>8.4f} │ "
              f"{s2:>8.4f} {ss2:>10.4f} {r2:>8.4f}")
    print(f"{'='*80}")

    return all_data


def plot_alpha_pulse(da_base: float = None, da_pulse: float = 15.0,
                     pulse_onset_s: float = 20.0, pulse_duration_s: float = 30.0,
                     post_pulse_s: float = 250.0, save_path: str = None):
    """
    Plot α dynamics for a DA pulse protocol (onset → pulse → withdrawal).
    Highlights the D1 afterglow window after DA withdrawal.

    Args:
        da_base        : baseline DA (nM). Default = config.DA_BASELINE
        da_pulse       : pulse DA (nM). Default = 15.0
        pulse_onset_s  : pulse start time (s). Default = 20.0
        pulse_duration_s: pulse duration (s). Default = 30.0
        post_pulse_s   : post-pulse observation (s). Default = 250.0
        save_path      : file path to save figure
    """
    if da_base is None:
        da_base = config.DA_BASELINE

    p = _get_kinetics_params()
    pulse_offset_s = pulse_onset_s + pulse_duration_s
    duration_s = pulse_onset_s + pulse_duration_s + post_pulse_s

    # Initial conditions at baseline steady-state
    s1_bl = sigmoid(da_base, p['ec50_d1'], p['beta'])
    s2_bl = sigmoid(da_base, p['ec50_d2'], p['beta'])
    a_d1_init = langmuir_steady_state(s1_bl, p['k_on_d1'], p['k_off_d1'])
    a_d2_init = langmuir_steady_state(s2_bl, p['k_on_d2'], p['k_off_d2'])

    schedule = make_pulse_schedule(da_base, da_pulse, pulse_onset_s, pulse_offset_s)
    data = simulate_alpha_dynamics(schedule, duration_s=duration_s,
                                   alpha_d1_init=a_d1_init,
                                   alpha_d2_init=a_d2_init)

    t = data['time_s']
    d1 = data['alpha_d1']
    d2 = data['alpha_d2']

    fig, axes = plt.subplots(3, 1, figsize=(18, 15), sharex=True,
                              gridspec_kw={'height_ratios': [1, 3, 2]})
    fig.suptitle(
        f"DA Pulse Protocol — α Dynamics & D1 Afterglow Window\n"
        f"DA: {da_base} → {da_pulse} → {da_base} nM  |  "
        f"Pulse: [{pulse_onset_s:.0f}s, {pulse_offset_s:.0f}s)",
        fontsize=14, fontweight='bold',
    )

    # --- Panel 0: DA protocol ---
    ax = axes[0]
    ax.fill_between(t, 0, data['da'], color='#4CAF50', alpha=0.3)
    ax.plot(t, data['da'], color='#4CAF50', linewidth=2)
    ax.set_ylabel('[DA] (nM)', fontsize=12)
    ax.set_title('DA Concentration Protocol', fontsize=13)
    ax.grid(True, alpha=0.3)

    # --- Panel 1: α_D1 and α_D2 ---
    ax = axes[1]
    ax.plot(t, d1, '-', color='#d62728', linewidth=2.5, label='α_D1 (slow)')
    ax.plot(t, d2, '-', color='#1f77b4', linewidth=2.5, label='α_D2 (fast)')

    # Sigmoid targets during pulse
    s1_pulse = sigmoid(da_pulse, p['ec50_d1'], p['beta'])
    s2_pulse = sigmoid(da_pulse, p['ec50_d2'], p['beta'])
    ss1_pulse = langmuir_steady_state(s1_pulse, p['k_on_d1'], p['k_off_d1'])
    ss2_pulse = langmuir_steady_state(s2_pulse, p['k_on_d2'], p['k_off_d2'])

    ax.axhline(s1_pulse, color='#d62728', linestyle='--', linewidth=1, alpha=0.4,
               label=f'S_D1(DA={da_pulse}) = {s1_pulse:.4f}')
    ax.axhline(s2_pulse, color='#1f77b4', linestyle='--', linewidth=1, alpha=0.4,
               label=f'S_D2(DA={da_pulse}) = {s2_pulse:.4f}')

    # Shade pulse window
    ax.axvspan(pulse_onset_s, pulse_offset_s, alpha=0.08, color='green', label='DA pulse')

    # Shade D1 afterglow (post-pulse, D1 > D2)
    post_mask = t >= pulse_offset_s
    afterglow = post_mask & (d1 > d2 * 1.05)
    if np.any(afterglow):
        ag_start = t[afterglow][0]
        ag_end = t[afterglow][-1]
        ax.axvspan(ag_start, ag_end, alpha=0.12, color='orange',
                   label=f'D1 afterglow [{ag_start:.0f}s–{ag_end:.0f}s]')

    ax.set_ylabel('α', fontsize=12)
    ax.set_title('D1/D2 Receptor Activation Dynamics', fontsize=13)
    ax.legend(fontsize=10, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Δα = α_D1 - α_D2 ---
    ax = axes[2]
    delta = d1 - d2
    ax.plot(t, delta, '-', color='purple', linewidth=2.5, label='Δα = α_D1 − α_D2')
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.fill_between(t, 0, delta, where=(delta > 0), alpha=0.25, color='red', label='D1 > D2')
    ax.fill_between(t, 0, delta, where=(delta < 0), alpha=0.25, color='blue', label='D2 > D1')
    ax.axvspan(pulse_onset_s, pulse_offset_s, alpha=0.08, color='green')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Δα (D1 − D2)', fontsize=12)
    ax.set_title('Temporal Segregation: D1 − D2', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")
    else:
        plt.savefig("outputs/alpha_pulse.png", dpi=150, bbox_inches='tight')
        print(f"📊 Saved: outputs/alpha_pulse.png")
    plt.close(fig)

    # Print afterglow info
    if np.any(afterglow):
        print(f"\n🌅 D1 Afterglow Window: {ag_start:.0f}s – {ag_end:.0f}s "
              f"(duration: {ag_end - ag_start:.0f}s)")
    else:
        print(f"\n⚠️  No D1 afterglow window detected.")

    return data


# ======================================================================
# CLI entry point
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize α_D1/α_D2 Langmuir receptor binding dynamics")
    parser.add_argument("--mode", type=str, default="step",
                        choices=["step", "multi", "pulse"],
                        help="Visualization mode: step | multi | pulse")
    parser.add_argument("--da-before", type=float, default=None,
                        help="DA before step (nM). Default = DA_BASELINE")
    parser.add_argument("--da-after", type=float, default=15.0,
                        help="DA after step (nM). Default = 15.0")
    parser.add_argument("--da-levels", type=float, nargs='+', default=None,
                        help="DA levels for multi-DA comparison (nM)")
    parser.add_argument("--da-base", type=float, default=None,
                        help="Baseline DA for pulse mode (nM)")
    parser.add_argument("--da-pulse", type=float, default=15.0,
                        help="Pulse DA (nM). Default = 15.0")
    parser.add_argument("--pulse-dur", type=float, default=30.0,
                        help="Pulse duration (s). Default = 30")
    parser.add_argument("--onset", type=float, default=10.0,
                        help="DA onset time (s). Default = 10")
    parser.add_argument("--duration", type=float, default=300.0,
                        help="Total duration (s). Default = 300")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: outputs/alpha_*.png)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🔬 Alpha Dynamics Visualizer")
    print(f"{'='*60}")
    print(f"  Config parameters:")
    print(f"    D1: EC50={config.EC50_D1} nM, τ_on={config.TAU_ON_D1/1000:.1f}s, "
          f"τ_off={config.TAU_OFF_D1/1000:.1f}s")
    print(f"    D2: EC50={config.EC50_D2} nM, τ_on={config.TAU_ON_D2/1000:.1f}s, "
          f"τ_off={config.TAU_OFF_D2/1000:.1f}s")
    print(f"    β={config.BETA}, DA_BASELINE={config.DA_BASELINE} nM")
    print(f"{'='*60}\n")

    if args.mode == "step":
        plot_alpha_step(
            da_before=args.da_before,
            da_after=args.da_after,
            onset_s=args.onset,
            duration_s=args.duration,
            save_path=args.output,
        )
    elif args.mode == "multi":
        plot_alpha_multi_da(
            da_levels=args.da_levels,
            onset_s=args.onset,
            duration_s=args.duration,
            save_path=args.output,
        )
    elif args.mode == "pulse":
        plot_alpha_pulse(
            da_base=args.da_base,
            da_pulse=args.da_pulse,
            pulse_onset_s=args.onset,
            pulse_duration_s=args.pulse_dur,
            post_pulse_s=args.duration - args.onset - args.pulse_dur,
            save_path=args.output,
        )


if __name__ == "__main__":
    main()
