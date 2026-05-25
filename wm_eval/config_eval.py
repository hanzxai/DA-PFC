"""Configuration for WM evaluation experiments.

All evaluation-specific knobs live here (DA sweep grid, protocol durations,
PASS thresholds, composite-score weights, pharma conditions, etc.).
"""

# DA concentration sweep (nM) -- log-spaced from sub-EC50 to super-EC50
DA_CONCENTRATIONS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0,
                     10.0, 12.0, 15.0, 20.0, 30.0]

# Number of independent seeds per DA point (for error bars)
DEFAULT_N_SEEDS = 3

# Standard WM protocol (ms)
WM_PROTOCOL = {
    'baseline_ms': 5000.0,
    'cue_ms':      1500.0,
    'delay_ms':    5000.0,
    'probe_ms':    500.0,
    'cue_amp_pA':  300.0,    # Scheme-A canonical value (was 350 default)
}

# ---------------------------------------------------------------------------
#  Scheme-A network parameters (REQUIRED for the inverted-U to appear).
#
#  These mirror the successful demo at outputs/exp_2026-05-12_18-42-17_*
#  which produced:
#      Mem-A: baseline=0.21 Hz -> cue=59 Hz -> delay-late=29 Hz  (WM PASS)
#
#  Without these parameters the network sits in a HIGH baseline state
#  (~19 Hz) where no bistable attractor exists -- the inverted-U is
#  mathematically impossible.
# ---------------------------------------------------------------------------
WM_SCHEME_A = {
    # --- Memory-pool structure ---
    'pool_size':       60,        # neurons per Mem-A / Mem-B pool
    'intra_prob':      0.6,       # within-pool connection probability
    'intra_w':         3.5,       # within-pool synaptic weight (pA)
                                  #   was 5.5 (deep HIGH attractor; insensitive
                                  #   to DA -> no inverted-U). 3.5 puts the
                                  #   network near the bistability boundary so
                                  #   DA modulation can switch HIGH on/off.

    # --- WTA (shared-inhibition) loop ---
    'use_wta':         True,
    'iwm_size':        60,        # I-WM pool size
    'e2i_prob':        0.2,
    'e2i_w':           3.0,       # Mem-{A,B} -> I-WM weight (pA)
    'i2e_prob':        0.15,
    'i2e_w':          -3.5,       # I-WM -> Mem-{A,B} weight (pA, negative)

    # --- Working-point offset for Mem pools ---
    # NEGATIVE pA added only to Mem-A & Mem-B background current.  This
    # pushes the memory pools into the LOW state at baseline so the cue
    # can switch them up to the HIGH state.  Without this knob, baseline
    # sits at ~19 Hz and there is no LOW fixed point at all.
    'mem_bg_offset':  -18.0,

    # --- Spike-frequency adaptation (Scheme-A keeps SFA OFF) ---
    'sfa_b':           0.0,       # 0 disables SFA
    'sfa_tau':         2000.0,    # ms (only used if sfa_b > 0)
}

# PASS thresholds for each metric
PASS_THRESHOLDS = {
    'persistence_hz':  5.0,
    'selectivity':     0.3,
    'decay_ratio':     0.3,
    'd_prime':         2.0,
    'accuracy_pct':    75.0,
    'composite_score': 0.5,
}

# Composite score weights (must sum to 1.0)
COMPOSITE_WEIGHTS = {
    'persistence':  0.25,
    'selectivity':  0.25,
    'decay_ratio':  0.20,
    'd_prime':      0.15,
    'accuracy':     0.15,
}

# Pharmacology conditions for batch comparison
PHARMA_CONDITIONS = {
    'vehicle':    {'block_d1': False, 'block_d2': False, 'color': '#2ca02c'},
    'd1_block':   {'block_d1': True,  'block_d2': False, 'color': '#d62728'},
    'd2_block':   {'block_d1': False, 'block_d2': True,  'color': '#1f77b4'},
    'both_block': {'block_d1': True,  'block_d2': True,  'color': '#7f7f7f'},
}

# Default checkpoint (relative to project root)
DEFAULT_CHECKPOINT = "checkpoints/ckpt_DA2nM_bg200_100s.pkl"


def build_wm_kwargs(da: float, protocol: dict = None,
                    block_d1: bool = False, block_d2: bool = False,
                    overrides: dict = None) -> dict:
    """Build the full kwargs dict for run_wm_simulation_from_checkpoint.

    Centralises the Scheme-A parameters so all three drivers stay in sync.

    Args:
        overrides: optional dict of WM_SCHEME_A overrides
                   (e.g. {'intra_w': 3.0}). Useful for parameter scans.
    """
    p = protocol or WM_PROTOCOL
    cue_on  = p['baseline_ms']
    cue_off = cue_on + p['cue_ms']

    kw = {
        'da_base':        da,
        'da_pulse':       da,                     # constant DA (no DA pulse)
        'da_pulse_onset':  cue_off,               # nominal (pulse==base anyway)
        'da_pulse_offset': cue_off + p['delay_ms'],
        'cue_a_onset':    cue_on,
        'cue_a_offset':   cue_off,
        'cue_a_amplitude': p['cue_amp_pA'],
        'block_d1':       block_d1,
        'block_d2':       block_d2,
    }
    # Inject all Scheme-A network parameters (with optional overrides)
    scheme = dict(WM_SCHEME_A)
    if overrides:
        scheme.update(overrides)
    kw.update(scheme)
    return kw
