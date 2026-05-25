# WM Evaluation Suite

Independent evaluation suite for Working Memory of the DA-PFC model.

## Layout

```
wm_eval/
├── config_eval.py     # DA sweep grid, protocol, PASS thresholds
├── metrics.py         # 6 core WM metrics
├── plotting.py        # Inverted-U curves and friends
├── run_inverted_u.py  # Main: DA dose-response sweep
├── run_single_eval.py # Deep diagnostic for one DA level
├── run_batch_eval.py  # Pharmacological comparison
└── outputs/           # auto-generated time-stamped subdirs
```

## Quick Start

```bash
cd DA-PFC

# 1) Inverted-U dose-response (main experiment)
python -m wm_eval.run_inverted_u \
    --ckpt checkpoints/ckpt_DA2nM_bg200_100s.pkl --n-seeds 3

# Smoke test (1 seed/point, ~10 min)
python -m wm_eval.run_inverted_u \
    --ckpt checkpoints/ckpt_DA2nM_bg200_100s.pkl --n-seeds 1

# 2) Single-condition deep evaluation
python -m wm_eval.run_single_eval \
    --ckpt checkpoints/ckpt_DA2nM_bg200_100s.pkl --da 8.0

# 3) Pharma comparison (vehicle vs D1/D2 antagonists)
python -m wm_eval.run_batch_eval \
    --ckpt checkpoints/ckpt_DA2nM_bg200_100s.pkl --n-seeds 2
```

## Six Core WM Metrics

| Metric | Formula | PASS |
|---|---|---|
| Persistence  | dR_late = R(delay_late) - R(baseline)              | >= 5 Hz |
| Selectivity  | SI = (R_A - R_B)/(R_A + R_B) over delay            | >= 0.3  |
| Decay Ratio  | (R_late - R_base)/(R_cue - R_base)                 | >= 0.3  |
| d-prime      | (R_target - mu_BG)/sigma_BG over delay             | >= 2.0  |
| Accuracy     | argmax(R_A, R_B) at probe (binary per trial)       | >= 75%  |
| Composite    | weighted avg of normalized metrics                 | >= 0.5  |

## Inverted-U Figure Axes

- **X axis**: DA concentration (nM), log scale, 13 points 0.5..30 nM
  - Vertical dashed lines: D1 EC50=4 nM, D2 EC50=8 nM
- **Y axis**: any of the 6 metrics above (one panel each)
  - Scatter (mean +/- SD across seeds) + cubic spline fit
  - Green dashed line: PASS threshold; green shading: PASS region
  - Gold vertical line + label: optimal DA

## Output Files

`outputs/<tag>_<timestamp>/`:
- `inverted_u_curves.png` — 6-panel main figure
- `inverted_u_with_alpha.png` — twin-axis with receptor occupancy
- `representative_traces.png` — Mem-A/B traces at low/opt/high DA
- `metric_correlations.png` — pairwise scatter matrix
- `summary.json` — all numerical results

## Tweak Points

- DA grid:        `config_eval.DA_CONCENTRATIONS`
- Protocol:       `config_eval.WM_PROTOCOL`
- Thresholds:     `config_eval.PASS_THRESHOLDS`
- Composite wts:  `config_eval.COMPOSITE_WEIGHTS`
- Pharma cond:    `config_eval.PHARMA_CONDITIONS`
