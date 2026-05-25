#!/usr/bin/env bash
# ==============================================================================
#  Working-Memory Demo — one-click launcher
#
#  Step 1 (auto-skipped if checkpoint already exists):
#     Generate the DA=2 nM baseline checkpoint by running a 100s simulation.
#
#  Step 2:
#     Run the WM-1 single-item-maintenance demo on the produced checkpoint.
#
#  Outputs land in:
#     outputs/exp_<timestamp>_wm_demo_DA2nM/
#       ├── combined_raster.png
#       ├── combined_rates_all.png
#       ├── wm_overview.png
#       ├── wm_rates_pools.png
#       ├── wm_report.txt
#       └── analysis_report.txt
# ==============================================================================
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

DA_BASE="${DA_BASE:-2.0}"
BASELINE_DURATION_S="${BASELINE_DURATION_S:-100}"
GPU_ID="${GPU_ID:-0}"
TAG="${TAG:-wm_demo}"

# We don't know exactly which BG_MEAN was active when the checkpoint was
# created (the filename embeds it), so we discover the latest matching one.
CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

CKPT="$(ls -t "$CKPT_DIR"/ckpt_DA${DA_BASE%.*}nM_bg*_${BASELINE_DURATION_S}s.pkl 2>/dev/null | head -n1 || true)"
# also try literal "2.0" form
if [[ -z "$CKPT" ]]; then
  CKPT="$(ls -t "$CKPT_DIR"/ckpt_DA${DA_BASE}nM_bg*_${BASELINE_DURATION_S}s.pkl 2>/dev/null | head -n1 || true)"
fi

if [[ -z "$CKPT" ]]; then
  echo "──────────────────────────────────────────────────────────────"
  echo "  Step 1/2: generating DA=${DA_BASE} nM baseline checkpoint"
  echo "            (duration = ${BASELINE_DURATION_S}s, GPU=${GPU_ID})"
  echo "──────────────────────────────────────────────────────────────"
  python main.py --da "${DA_BASE}" \
                 --duration "${BASELINE_DURATION_S}" \
                 --gpu "${GPU_ID}" \
                 --save-ckpt
  CKPT="$(ls -t "$CKPT_DIR"/ckpt_DA${DA_BASE%.*}nM_bg*_${BASELINE_DURATION_S}s.pkl 2>/dev/null | head -n1 || true)"
  if [[ -z "$CKPT" ]]; then
    CKPT="$(ls -t "$CKPT_DIR"/ckpt_DA${DA_BASE}nM_bg*_${BASELINE_DURATION_S}s.pkl 2>/dev/null | head -n1 || true)"
  fi
  if [[ -z "$CKPT" ]]; then
    echo "❌ Failed to locate generated checkpoint under ${CKPT_DIR}/" >&2
    exit 1
  fi
else
  echo "✅ Reusing existing checkpoint: ${CKPT}"
fi

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  Step 2/2: running WM-1 single-item maintenance demo"
echo "            ckpt = ${CKPT}"
echo "──────────────────────────────────────────────────────────────"
python experiments/exp_wm_demo.py \
       --ckpt "${CKPT}" \
       --da "${DA_BASE}" \
       --gpu "${GPU_ID}" \
       --tag "${TAG}"

echo ""
echo "✅ Done.  Browse the latest folder under outputs/ for results."
