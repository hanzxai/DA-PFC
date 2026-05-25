#!/usr/bin/env bash
# ==============================================================================
#  Working-Memory Demo — Winner-Take-All (WTA) edition
#
#  Single-source-of-truth principle:
#    - All WM/WTA parameters live in config.py (WM_INTRA_W, WM_E2I_W,
#      WM_I2E_W, WM_IWM_SIZE, ...).
#    - This launcher does NOT bake parameter values into CLI flags.  It
#      only forwards a flag to exp_wm_demo.py when the user *explicitly*
#      sets the corresponding environment variable.  Otherwise the python
#      side falls back to config.py.
#
#  Why: hardcoding `INTRA_W=1.5` etc. in the shell silently overrides any
#  edit to config.py, which previously made the experiment run with stale
#  values.
#
#  Usage (defaults from config.py — recommended):
#     bash scripts/run_wm_demo_wta.sh
#
#  Override individual knobs (only the ones you set are forwarded):
#     INTRA_W=2.0  I2E_W=-3.5  bash scripts/run_wm_demo_wta.sh
#
#  Frequently-used env vars:
#     DA_BASE=2.0  CKPT_DURATION_S=100  BG_MEAN=200.0  GPU_ID=0  TAG=wm_wta
#     INTRA_W      INTRA_PROB
#     IWM_SIZE     E2I_W       E2I_PROB     I2E_W      I2E_PROB
#     CUE_AMP      CUE_MS      DELAY_MS     BASELINE_MS  PROBE_MS
#
#  Outputs land in:
#     outputs/exp_<timestamp>_${TAG}_DA${DA_BASE}nM/
#       ├── combined_raster.png         (project-standard)
#       ├── combined_rates_all.png      (project-standard)
#       ├── wm_overview.png             (raster + rates with protocol bands)
#       ├── wm_rates_pools.png          (Mem-A vs Mem-B vs E-BG)
#       ├── wm_report.txt
#       └── analysis_report.txt
# ==============================================================================
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

# ── Baseline (checkpoint) settings ────────────────────────────────────────────
DA_BASE="${DA_BASE:-2.0}"
CKPT_DURATION_S="${CKPT_DURATION_S:-100}"
BG_MEAN="${BG_MEAN:-200.0}"        # standard critical-point baseline
GPU_ID="${GPU_ID:-0}"
TAG="${TAG:-wm_wta}"

CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

# ── Locate or build the BG=200 / DA=2 nM baseline checkpoint ────────────────
BG_TAG="$(printf '%.0f' "${BG_MEAN}")"   # e.g. "200"
PATTERN1="$CKPT_DIR/ckpt_DA${DA_BASE%.*}nM_bg${BG_TAG}_${CKPT_DURATION_S}s.pkl"
PATTERN2="$CKPT_DIR/ckpt_DA${DA_BASE}nM_bg${BG_TAG}_${CKPT_DURATION_S}s.pkl"

CKPT="$(ls -t $PATTERN1 2>/dev/null | head -n1 || true)"
if [[ -z "$CKPT" ]]; then
  CKPT="$(ls -t $PATTERN2 2>/dev/null | head -n1 || true)"
fi

if [[ -z "$CKPT" ]]; then
  echo "──────────────────────────────────────────────────────────────"
  echo "  Step 1/2: generating DA=${DA_BASE} nM, BG=${BG_MEAN} pA, ${CKPT_DURATION_S}s checkpoint"
  echo "──────────────────────────────────────────────────────────────"
  python main.py --da "${DA_BASE}" \
                 --duration "${CKPT_DURATION_S}" \
                 --bg-mean "${BG_MEAN}" \
                 --gpu "${GPU_ID}" \
                 --save-ckpt
  CKPT="$(ls -t $PATTERN1 2>/dev/null | head -n1 || true)"
  if [[ -z "$CKPT" ]]; then
    CKPT="$(ls -t $PATTERN2 2>/dev/null | head -n1 || true)"
  fi
  if [[ -z "$CKPT" ]]; then
    echo "❌ Failed to locate the generated checkpoint." >&2
    exit 1
  fi
else
  echo "✅ Reusing existing checkpoint: ${CKPT}"
fi

# ── Build CLI override list ONLY for env vars the user actually set ──────────
# We use ${VAR+--flag value} which expands to nothing when VAR is unset, so
# the python side falls back to config.py.  When VAR IS set, the flag is
# forwarded with its value.
#
# Note: we use printf %s as a delimiter-safe way to splice into the cmd.
EXTRA=()
[[ -n "${INTRA_W+x}"     ]] && EXTRA+=( --intra-w     "${INTRA_W}"     )
[[ -n "${INTRA_PROB+x}"  ]] && EXTRA+=( --intra-prob  "${INTRA_PROB}"  )
[[ -n "${IWM_SIZE+x}"    ]] && EXTRA+=( --iwm-size    "${IWM_SIZE}"    )
[[ -n "${E2I_W+x}"       ]] && EXTRA+=( --e2i-w       "${E2I_W}"       )
[[ -n "${E2I_PROB+x}"    ]] && EXTRA+=( --e2i-prob    "${E2I_PROB}"    )
[[ -n "${I2E_W+x}"       ]] && EXTRA+=( --i2e-w       "${I2E_W}"       )
[[ -n "${I2E_PROB+x}"    ]] && EXTRA+=( --i2e-prob    "${I2E_PROB}"    )
[[ -n "${CUE_AMP+x}"     ]] && EXTRA+=( --cue-amp     "${CUE_AMP}"     )
[[ -n "${CUE_MS+x}"      ]] && EXTRA+=( --cue-ms      "${CUE_MS}"      )
[[ -n "${DELAY_MS+x}"    ]] && EXTRA+=( --delay-ms    "${DELAY_MS}"    )
[[ -n "${BASELINE_MS+x}" ]] && EXTRA+=( --baseline-ms "${BASELINE_MS}" )
[[ -n "${PROBE_MS+x}"    ]] && EXTRA+=( --probe-ms    "${PROBE_MS}"    )

# ── Run the WTA-augmented WM demo ────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  Step 2/2: running WM-1 demo (config.py is the source of truth)"
echo "  ckpt = ${CKPT}"
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  echo "  user overrides: ${EXTRA[*]}"
else
  echo "  user overrides: (none — using config.py defaults)"
fi
echo "──────────────────────────────────────────────────────────────"

python experiments/exp_wm_demo.py \
       --ckpt "${CKPT}" \
       --da "${DA_BASE}" \
       --bg-mean "${BG_MEAN}" \
       --gpu "${GPU_ID}" \
       --tag "${TAG}" \
       "${EXTRA[@]}"

echo ""
echo "✅ Done.  Latest output folder under outputs/exp_*_${TAG}_DA${DA_BASE}nM/"
