#!/usr/bin/env bash
# ==============================================================================
#  Working-Memory Demo (LOW-BG sub-threshold regime) — one-click launcher
#
#  Why low BG?
#     The standard config (BG_MEAN=200 pA → V_inf = V_th) puts every neuron
#     at the critical-point fluctuation-driven regime. As soon as the WM
#     pool's recurrent connections turn on, the pool latches into a ~120 Hz
#     attractor at t=0 and the cue becomes invisible.
#     Setting BG_MEAN = 160 pA pushes V_inf to ~-54 mV (sub-threshold) so the
#     pool sits at low activity by default. The cue then transiently lifts
#     the pool into the high-rate attractor; recurrence keeps it there during
#     the delay → real working-memory dynamics.
#
#  Pipeline
#     Step 1 (auto-skipped if checkpoint already exists):
#        Generate the DA = ${DA_BASE} nM, BG = ${BG_MEAN} pA baseline checkpoint.
#     Step 2:
#        Run the WM-1 maintenance demo on the produced checkpoint.
#
#  Tunables (env vars):
#     BG_MEAN              default 160        — background drive in pA
#     DA_BASE              default 2.0        — baseline DA tone in nM
#     BASELINE_DURATION_S  default 100        — ckpt duration in seconds
#     INTRA_W              default 2.5        — intra-pool synaptic weight (pA)
#     INTRA_PROB           default 0.20       — intra-pool connection probability
#     POOL_SIZE            default 100        — neurons per Mem-A / Mem-B pool
#     CUE_AMP              default 200        — cue-A current in pA
#     GPU_ID               default 0
#     TAG                  default wm_lowbg
# ==============================================================================
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

BG_MEAN="${BG_MEAN:-160}"
DA_BASE="${DA_BASE:-2.0}"
BASELINE_DURATION_S="${BASELINE_DURATION_S:-100}"
INTRA_W="${INTRA_W:-2.5}"
INTRA_PROB="${INTRA_PROB:-0.20}"
POOL_SIZE="${POOL_SIZE:-100}"
CUE_AMP="${CUE_AMP:-200}"
GPU_ID="${GPU_ID:-0}"
TAG="${TAG:-wm_lowbg}"

CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

# BG fragment used in the ckpt filename (matches simulation/utils.save_checkpoint)
BG_FRAG="$(python -c "print(f'{${BG_MEAN}:g}')")"
DA_FRAG="$(python -c "v=${DA_BASE}; s=f'{v}'.rstrip('0').rstrip('.'); print(s)")"
DUR_FRAG="${BASELINE_DURATION_S}"

CKPT="${CKPT_DIR}/ckpt_DA${DA_FRAG}nM_bg${BG_FRAG}_${DUR_FRAG}s.pkl"

echo "──────────────────────────────────────────────────────────────"
echo "  WM-Demo (low-BG)  ::  BG=${BG_MEAN}pA  DA=${DA_BASE}nM"
echo "                       intra_w=${INTRA_W}pA  intra_prob=${INTRA_PROB}"
echo "                       pool_size=${POOL_SIZE}  cue_amp=${CUE_AMP}pA"
echo "  Expected ckpt path : ${CKPT}"
echo "──────────────────────────────────────────────────────────────"

if [[ ! -f "$CKPT" ]]; then
  echo ""
  echo "  Step 1/2 — generating baseline checkpoint (BG=${BG_MEAN}, ${BASELINE_DURATION_S}s)"
  echo ""
  python main.py --da "${DA_BASE}" \
                 --duration "${BASELINE_DURATION_S}" \
                 --bg-mean "${BG_MEAN}" \
                 --gpu "${GPU_ID}" \
                 --save-ckpt
  if [[ ! -f "$CKPT" ]]; then
    echo "❌ Failed to locate generated checkpoint at ${CKPT}" >&2
    echo "   Available checkpoints:" >&2
    ls -lh "$CKPT_DIR" >&2 || true
    exit 1
  fi
else
  echo ""
  echo "  ✅ Reusing existing checkpoint: ${CKPT}"
fi

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  Step 2/2 — running WM-1 single-item maintenance demo"
echo "──────────────────────────────────────────────────────────────"
python experiments/exp_wm_demo.py \
       --ckpt "${CKPT}" \
       --da "${DA_BASE}" \
       --bg-mean "${BG_MEAN}" \
       --pool-size "${POOL_SIZE}" \
       --intra-w "${INTRA_W}" \
       --intra-prob "${INTRA_PROB}" \
       --cue-amp "${CUE_AMP}" \
       --gpu "${GPU_ID}" \
       --tag "${TAG}"

echo ""
echo "✅ Done.  Browse the latest folder under outputs/ for results."
