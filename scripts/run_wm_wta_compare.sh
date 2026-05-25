#!/usr/bin/env bash
# ==============================================================================
#  WM Demo — WTA on/off comparison (sweet spot G=8 working point)
#
#  Why this script?
#     The previous G=8 run (BG=200, intra_w=2.0, intra_prob=0.40) already
#     showed the network reaching a high-rate attractor, but Mem-A and
#     Mem-B were UNCOUPLED, so both pools climbed in lock-step and there
#     was no winner-take-all selectivity (Mem-A ≈ Mem-B during the delay).
#
#     The fix is to add a shared-inhibition WTA loop between Mem-{A,B} and
#     a dedicated I-WM sub-pool.  This script runs the SAME working point
#     twice — once with WTA disabled, once with WTA enabled — so you can
#     compare wm_rates_pools.png side-by-side.
#
#  Tunables (env vars):
#     DA_BASE              default 2.0
#     BG_MEAN              default 200       (matches existing ckpt)
#     POOL_SIZE            default 100
#     INTRA_W              default 2.0
#     INTRA_PROB           default 0.40
#     CUE_AMP              default 80
#     IWM_SIZE             default 40
#     E2I_PROB             default 0.30
#     E2I_W                default 5.0
#     I2E_PROB             default 0.50
#     I2E_W                default -25.0
#     BASELINE_DURATION_S  default 100
#     GPU_ID               default 0
#     TAG_PREFIX           default wm_wta
# ==============================================================================
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

DA_BASE="${DA_BASE:-2.0}"
BG_MEAN="${BG_MEAN:-200}"
POOL_SIZE="${POOL_SIZE:-100}"
INTRA_W="${INTRA_W:-2.0}"
INTRA_PROB="${INTRA_PROB:-0.40}"
CUE_AMP="${CUE_AMP:-80}"
IWM_SIZE="${IWM_SIZE:-40}"
E2I_PROB="${E2I_PROB:-0.30}"
E2I_W="${E2I_W:-5.0}"
I2E_PROB="${I2E_PROB:-0.50}"
I2E_W="${I2E_W:--25.0}"
BASELINE_DURATION_S="${BASELINE_DURATION_S:-100}"
GPU_ID="${GPU_ID:-0}"
TAG_PREFIX="${TAG_PREFIX:-wm_wta}"

CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"
BG_FRAG="$(python -c "print(f'{${BG_MEAN}:g}')")"
DA_FRAG="$(python -c "v=${DA_BASE}; s=f'{v}'.rstrip('0').rstrip('.'); print(s)")"
CKPT="${CKPT_DIR}/ckpt_DA${DA_FRAG}nM_bg${BG_FRAG}_${BASELINE_DURATION_S}s.pkl"

echo "──────────────────────────────────────────────────────────────"
echo "  WM-Demo WTA comparison @ G≈${INTRA_W}*${INTRA_PROB}*${POOL_SIZE} working point"
echo "  ckpt     : ${CKPT}"
echo "  pool     : ${POOL_SIZE}, intra_w=${INTRA_W}pA, intra_prob=${INTRA_PROB}"
echo "  cue      : ${CUE_AMP}pA"
echo "  WTA cfg  : I-WM=${IWM_SIZE}, E->I p=${E2I_PROB}/w=${E2I_W}"
echo "                            I->E p=${I2E_PROB}/w=${I2E_W}"
echo "──────────────────────────────────────────────────────────────"

if [[ ! -f "$CKPT" ]]; then
  echo "  Generating baseline checkpoint ..."
  python main.py --da "${DA_BASE}" \
                 --duration "${BASELINE_DURATION_S}" \
                 --bg-mean "${BG_MEAN}" \
                 --gpu "${GPU_ID}" \
                 --save-ckpt
fi

# ---- Run 1: WTA OFF (baseline reproducing the old G=8 result) ----
echo ""
echo "▶︎ Run 1/2 — NO WTA (control)"
python experiments/exp_wm_demo.py \
       --ckpt "${CKPT}" \
       --da "${DA_BASE}" \
       --bg-mean "${BG_MEAN}" \
       --pool-size "${POOL_SIZE}" \
       --intra-w "${INTRA_W}" \
       --intra-prob "${INTRA_PROB}" \
       --cue-amp "${CUE_AMP}" \
       --no-wta \
       --gpu "${GPU_ID}" \
       --tag "${TAG_PREFIX}_off"

# ---- Run 2: WTA ON ----
echo ""
echo "▶︎ Run 2/2 — WTA ENABLED"
python experiments/exp_wm_demo.py \
       --ckpt "${CKPT}" \
       --da "${DA_BASE}" \
       --bg-mean "${BG_MEAN}" \
       --pool-size "${POOL_SIZE}" \
       --intra-w "${INTRA_W}" \
       --intra-prob "${INTRA_PROB}" \
       --cue-amp "${CUE_AMP}" \
       --iwm-size "${IWM_SIZE}" \
       --e2i-prob "${E2I_PROB}" \
       --e2i-w "${E2I_W}" \
       --i2e-prob "${I2E_PROB}" \
       --i2e-w "${I2E_W}" \
       --gpu "${GPU_ID}" \
       --tag "${TAG_PREFIX}_on"

echo ""
echo "✅ Both runs done.  Compare wm_rates_pools.png in outputs/ for the WTA effect."
