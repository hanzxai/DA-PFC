"""WM evaluation metrics — six core indicators.

All metrics are computed from a PFCAnalyzer instance with WM groups registered.
Each metric is a single function so they can be used independently.
"""
import numpy as np

from analysis.analyzer import PFCAnalyzer
from analysis.wm_plotting import register_wm_groups
from wm_eval.config_eval import PASS_THRESHOLDS, COMPOSITE_WEIGHTS


# ----------------------------------------------------------------------
#  Window helpers
# ----------------------------------------------------------------------
def _get_protocol_windows(analyzer: PFCAnalyzer) -> dict:
    """Extract cue/delay/probe windows (ms) from analyzer config."""
    p = analyzer.cfg.get('wm_protocol', {})
    if not p:
        return {}
    cue_on = p.get('cue_a_onset', 0.0)
    cue_off = p.get('cue_a_offset', cue_on)
    delay_dur = p.get('delay_ms', 0.0)
    probe_dur = p.get('probe_ms', 0.0)
    delay_end = cue_off + delay_dur
    probe_end = delay_end + probe_dur

    return {
        'baseline':    (0.0, cue_on),
        'cue':         (cue_on, cue_off),
        'delay':       (cue_off, delay_end),
        'delay_early': (cue_off, cue_off + delay_dur / 3.0),
        'delay_mid':   (cue_off + delay_dur / 3.0, cue_off + 2.0 * delay_dur / 3.0),
        'delay_late':  (delay_end - delay_dur / 3.0, delay_end),
        'probe':       (delay_end, probe_end),
    }


def _window_mean(centers: np.ndarray, rate: np.ndarray,
                 lo: float, hi: float) -> float:
    if rate is None or len(rate) == 0 or hi <= lo:
        return 0.0
    centers = np.asarray(centers)
    rate = np.asarray(rate)
    m = (centers >= lo) & (centers < hi)
    return float(np.mean(rate[m])) if m.any() else 0.0


def _window_std(centers: np.ndarray, rate: np.ndarray,
                lo: float, hi: float) -> float:
    if rate is None or len(rate) == 0 or hi <= lo:
        return 0.0
    centers = np.asarray(centers)
    rate = np.asarray(rate)
    m = (centers >= lo) & (centers < hi)
    return float(np.std(rate[m])) if m.any() else 0.0


def _safe_group_rate(analyzer: PFCAnalyzer, batch_idx: int, group_name: str,
                     time_win: float = 20.0, sigma: float = 2.0):
    """Wrapper around analyzer.compute_group_rate that returns (None, None)
    if the group does not exist (e.g., for purely E-only configurations)."""
    try:
        return analyzer.compute_group_rate(batch_idx, group_name,
                                            time_win=time_win, sigma=sigma)
    except Exception:
        return None, None


# ----------------------------------------------------------------------
#  Single-metric calculators
# ----------------------------------------------------------------------
def compute_persistence(analyzer: PFCAnalyzer, batch_idx: int = 1) -> float:
    """ΔR_late = R(Mem-A, delay_late) - R(Mem-A, baseline). Unit: Hz."""
    register_wm_groups(analyzer)
    windows = _get_protocol_windows(analyzer)
    if not windows:
        return 0.0
    centers, rate = _safe_group_rate(analyzer, batch_idx, 'Mem-A')
    if rate is None:
        return 0.0
    bl   = _window_mean(centers, rate, *windows['baseline'])
    late = _window_mean(centers, rate, *windows['delay_late'])
    return late - bl


def compute_selectivity(analyzer: PFCAnalyzer, batch_idx: int = 1) -> float:
    """SI = (R_A - R_B)/(R_A + R_B) in delay window. Range [-1, 1]."""
    register_wm_groups(analyzer)
    windows = _get_protocol_windows(analyzer)
    if not windows:
        return 0.0
    cA, rA = _safe_group_rate(analyzer, batch_idx, 'Mem-A')
    cB, rB = _safe_group_rate(analyzer, batch_idx, 'Mem-B')
    if rA is None or rB is None:
        return 0.0
    R_A = _window_mean(cA, rA, *windows['delay'])
    R_B = _window_mean(cB, rB, *windows['delay'])
    denom = R_A + R_B
    return (R_A - R_B) / denom if denom > 1e-3 else 0.0


def compute_decay_ratio(analyzer: PFCAnalyzer, batch_idx: int = 1) -> float:
    """PR = (R_delay_late - R_baseline)/(R_cue - R_baseline). Range typically [0,1]."""
    register_wm_groups(analyzer)
    windows = _get_protocol_windows(analyzer)
    if not windows:
        return 0.0
    centers, rate = _safe_group_rate(analyzer, batch_idx, 'Mem-A')
    if rate is None:
        return 0.0
    bl   = _window_mean(centers, rate, *windows['baseline'])
    cue  = _window_mean(centers, rate, *windows['cue'])
    late = _window_mean(centers, rate, *windows['delay_late'])
    delta_cue = cue - bl
    if delta_cue < 1e-3:
        return 0.0
    return (late - bl) / delta_cue


def compute_d_prime(analyzer: PFCAnalyzer, batch_idx: int = 1) -> float:
    """d' = (R_target - μ_BG) / σ_BG, computed in delay window.

    Falls back to All-E if E-BG group is unavailable.
    """
    register_wm_groups(analyzer)
    windows = _get_protocol_windows(analyzer)
    if not windows:
        return 0.0
    cA, rA = _safe_group_rate(analyzer, batch_idx, 'Mem-A')
    if rA is None:
        return 0.0
    cBG, rBG = _safe_group_rate(analyzer, batch_idx, 'E-BG')
    if rBG is None:
        cBG, rBG = _safe_group_rate(analyzer, batch_idx, 'All-E')
    if rBG is None:
        return 0.0
    R_A = _window_mean(cA, rA, *windows['delay'])
    mu_BG = _window_mean(cBG, rBG, *windows['delay'])
    sigma_BG = _window_std(cBG, rBG, *windows['delay'])
    if sigma_BG < 1e-3:
        return 0.0
    return (R_A - mu_BG) / sigma_BG


def compute_accuracy(analyzer: PFCAnalyzer, batch_idx: int = 1) -> float:
    """Probe-phase decision: argmax(R_A, R_B). Returns 1.0 if Mem-A wins, else 0.0.

    NB: For a single trial, accuracy is binary. To get a meaningful
    percentage, aggregate over multiple seeds via the run_inverted_u driver.
    """
    register_wm_groups(analyzer)
    windows = _get_protocol_windows(analyzer)
    if not windows:
        return 0.0
    cA, rA = _safe_group_rate(analyzer, batch_idx, 'Mem-A')
    cB, rB = _safe_group_rate(analyzer, batch_idx, 'Mem-B')
    if rA is None or rB is None:
        return 0.0
    # Use probe window if it has positive duration, else delay_late
    win = 'probe' if windows['probe'][1] > windows['probe'][0] else 'delay_late'
    R_A = _window_mean(cA, rA, *windows[win])
    R_B = _window_mean(cB, rB, *windows[win])
    return 1.0 if R_A > R_B else 0.0


def compute_stability(analyzer: PFCAnalyzer, batch_idx: int = 1) -> float:
    """CV of Mem-A firing rate during delay (lower = more stable attractor)."""
    register_wm_groups(analyzer)
    windows = _get_protocol_windows(analyzer)
    if not windows:
        return float('inf')
    centers, rate = _safe_group_rate(analyzer, batch_idx, 'Mem-A')
    if rate is None:
        return float('inf')
    mu = _window_mean(centers, rate, *windows['delay'])
    sd = _window_std(centers, rate, *windows['delay'])
    return sd / mu if mu > 1e-3 else float('inf')


# ----------------------------------------------------------------------
#  Composite score
# ----------------------------------------------------------------------
def _normalize_metric(value: float, key: str) -> float:
    """Map a raw metric to [0, 1] using its PASS threshold as the midpoint
    of a logistic squash (so threshold value -> 0.5)."""
    threshold_map = {
        'persistence':  PASS_THRESHOLDS['persistence_hz'],
        'selectivity':  PASS_THRESHOLDS['selectivity'],
        'decay_ratio':  PASS_THRESHOLDS['decay_ratio'],
        'd_prime':      PASS_THRESHOLDS['d_prime'],
        'accuracy':     PASS_THRESHOLDS['accuracy_pct'] / 100.0,
    }
    th = threshold_map[key]
    if th <= 0:
        return 0.0
    x = value / th
    # Sigmoid centered at threshold (x=1 -> 0.5)
    return float(1.0 / (1.0 + np.exp(-2.0 * (x - 1.0))))


def compute_composite_score(metrics_dict: dict) -> float:
    """Weighted sum of normalized metrics. Range [0, 1]."""
    score = 0.0
    for key, w in COMPOSITE_WEIGHTS.items():
        if key in metrics_dict:
            score += w * _normalize_metric(metrics_dict[key], key)
    return score


# ----------------------------------------------------------------------
#  Main entry point: compute ALL metrics for one analyzer
# ----------------------------------------------------------------------
def compute_all_metrics(analyzer: PFCAnalyzer, batch_idx: int = 1) -> dict:
    """Compute all six WM metrics for one batch.

    Returns a dict containing both raw values, PASS booleans, and a composite
    score.
    """
    metrics = {
        'persistence':  compute_persistence(analyzer, batch_idx),
        'selectivity':  compute_selectivity(analyzer, batch_idx),
        'decay_ratio':  compute_decay_ratio(analyzer, batch_idx),
        'd_prime':      compute_d_prime(analyzer, batch_idx),
        'accuracy':     compute_accuracy(analyzer, batch_idx),
        'stability':    compute_stability(analyzer, batch_idx),
    }
    metrics['composite_score'] = compute_composite_score(metrics)

    # PASS flags
    metrics['_pass'] = {
        'persistence': metrics['persistence'] >= PASS_THRESHOLDS['persistence_hz'],
        'selectivity': metrics['selectivity'] >= PASS_THRESHOLDS['selectivity'],
        'decay_ratio': metrics['decay_ratio'] >= PASS_THRESHOLDS['decay_ratio'],
        'd_prime':     metrics['d_prime']     >= PASS_THRESHOLDS['d_prime'],
        'accuracy':    metrics['accuracy']    >= PASS_THRESHOLDS['accuracy_pct'] / 100.0,
        'composite':   metrics['composite_score'] >= PASS_THRESHOLDS['composite_score'],
    }
    metrics['_overall_pass'] = (
        metrics['_pass']['persistence']
        and metrics['_pass']['selectivity']
        and metrics['_pass']['decay_ratio']
    )
    return metrics


def format_metrics_report(metrics: dict, condition_label: str = "") -> str:
    """Pretty-print a metrics dict (returns a multi-line string)."""
    def mark(ok: bool) -> str:
        return "[PASS]" if ok else "[FAIL]"

    lines = []
    sep = "-" * 72
    lines.append(sep)
    if condition_label:
        lines.append(f" Condition: {condition_label}")
        lines.append(sep)
    lines.append(f"  Persistence (dR_late) : {metrics['persistence']:>+7.2f} Hz   "
                 f"{mark(metrics['_pass']['persistence'])}")
    lines.append(f"  Selectivity Index     : {metrics['selectivity']:>+7.3f}      "
                 f"{mark(metrics['_pass']['selectivity'])}")
    lines.append(f"  Decay Ratio           : {metrics['decay_ratio']:>+7.3f}      "
                 f"{mark(metrics['_pass']['decay_ratio'])}")
    lines.append(f"  d-prime (SNR)         : {metrics['d_prime']:>+7.2f}      "
                 f"{mark(metrics['_pass']['d_prime'])}")
    lines.append(f"  Accuracy (binary)     : {metrics['accuracy']:>+7.2f}      "
                 f"{mark(metrics['_pass']['accuracy'])}")
    stab = metrics['stability']
    stab_str = f"{stab:>+7.3f}" if np.isfinite(stab) else "    inf"
    lines.append(f"  Stability (CV)        : {stab_str}")
    lines.append(f"  Composite Score       : {metrics['composite_score']:>+7.3f}      "
                 f"{mark(metrics['_pass']['composite'])}")
    lines.append(sep)
    verdict = "WM ESTABLISHED" if metrics['_overall_pass'] else "WM FAILED"
    lines.append(f"  Verdict: [{verdict}]")
    lines.append(sep)
    return "\n".join(lines)
