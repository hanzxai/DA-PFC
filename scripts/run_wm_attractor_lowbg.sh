#!/usr/bin/env bash
# ==============================================================================
#  WM Attractor Demo — LOW-BG variant (sub-threshold baseline)
#
#  Compared to run_wm_attractor.sh, this version:
#    - Uses BG_MEAN=160 (V_inf = -54 mV, 4 mV below V_th)
#      -> baseline E rate ~1-3 Hz, much lower than the BG=200 critical regime
#    - Uses milder NMDA gain (intra_w=2.0) so baseline can't self-sustain
#      but cue-driven 40Hz can
#  This is the regime where Wang-2002 attractor bistability actually exists.
# ==============================================================================
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

DA_BASE="${DA_BASE:-2.0}"
BASELINE_DURATION_S="${BASELINE_DURATION_S:-100}"
GPU_ID="${GPU_ID:-0}"
TAG="${TAG:-wm_attr_lowbg}"
BG_MEAN="${BG_MEAN:-160}"

INTRA_W="${INTRA_W:-2.0}"
INTRA_PROB="${INTRA_PROB:-0.5}"
POOL_SIZE="${POOL_SIZE:-100}"
CUE_AMP="${CUE_AMP:-300.0}"
CUE_MS="${CUE_MS:-1500.0}"
DELAY_MS="${DELAY_MS:-5000.0}"
BASELINE_MS="${BASELINE_MS:-5000.0}"
PROBE_MS="${PROBE_MS:-500.0}"

USE_WTA_FLAG=""
IWM_SIZE="${IWM_SIZE:-60}"
E2I_W="${E2I_W:-3.0}"
E2I_PROB="${E2I_PROB:-0.20}"
I2E_W="${I2E_W:--3.5}"
I2E_PROB="${I2E_PROB:-0.15}"

CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

CKPT="${CKPT_DIR}/ckpt_DA${DA_BASE%.*}nM_bg${BG_MEAN}_${BASELINE_DURATION_S}s.pkl"
if [[ ! -f "$CKPT" ]]; then
  CKPT_ALT="${CKPT_DIR}/ckpt_DA${DA_BASE}nM_bg${BG_MEAN}_${BASELINE_DURATION_S}s.pkl"
  if [[ -f "$CKPT_ALT" ]]; then
    CKPT="$CKPT_ALT"
  fi
fi

if [[ ! -f "$CKPT" ]]; then
  echo "──────────────────────────────────────────────────────────────"
  echo "  Step 1/2: generating DA=${DA_BASE} nM BG=${BG_MEAN} pA baseline ckpt"
  echo "──────────────────────────────────────────────────────────────"
  python main.py --da "${DA_BASE}" \
                 --duration "${BASELINE_DURATION_S}" \
                 --bg-mean "${BG_MEAN}" \
                 --gpu "${GPU_ID}" \
                 --save-ckpt
fi

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  Step 2/2: running LOW-BG WM attractor demo"
echo "            ckpt        = ${CKPT}"
echo "            BG_MEAN     = ${BG_MEAN} pA  (V_inf ~= -54 mV)"
echo "            intra_w     = ${INTRA_W}   intra_prob = ${INTRA_PROB}"
echo "            cue         = ${CUE_AMP} pA × ${CUE_MS} ms"
echo "──────────────────────────────────────────────────────────────"

python experiments/exp_wm_demo.py \
       --ckpt "${CKPT}" \
       --da "${DA_BASE}" \
       --bg-mean "${BG_MEAN}" \
       --gpu "${GPU_ID}" \
       --tag "${TAG}" \
       --pool-size "${POOL_SIZE}" \
       --intra-w "${INTRA_W}" \
       --intra-prob "${INTRA_PROB}" \
       --cue-amp "${CUE_AMP}" \
       --cue-ms "${CUE_MS}" \
       --delay-ms "${DELAY_MS}" \
       --baseline-ms "${BASELINE_MS}" \
       --probe-ms "${PROBE_MS}" \
       --iwm-size "${IWM_SIZE}" \
       --e2i-w "${E2I_W}" \
       --e2i-prob "${E2I_PROB}" \
       --i2e-w "${I2E_W}" \
       --i2e-prob "${I2E_PROB}" \
       ${USE_WTA_FLAG}

echo ""
echo "✅ Done. Browse the latest folder under outputs/ for results."
echo "   Key file: wm_report.txt — look at Δ(Late-BL) for Mem-A."
