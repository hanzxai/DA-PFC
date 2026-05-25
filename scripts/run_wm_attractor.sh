#!/usr/bin/env bash
# ==============================================================================
#  Working-Memory Attractor Demo — Wang-2002 NMDA-saturating attractor
#
#  Critical changes vs. the old run_wm_demo.sh:
#    1. Uses run_wm_kernel_dual (now the default in runners.py).
#    2. NMDA channel is saturating:
#         ds/dt = -s/tau + alpha*(1-s)*spike     with alpha=0.5
#       -> bistable f-I curve, supports a true high-rate fixed point.
#    3. Larger intra-pool gain:
#         intra_w   = 4.0 (vs 3.5)
#         intra_prob= 0.5 (vs 0.25)
#       Mean-field condition: I_NMDA(20Hz) = N*p*J*s_inf ≈ 100 pA  ->  ΔV +10 mV.
#    4. Slightly stronger cue (300 pA, 1.5 s) so the cue can fully saturate the
#       NMDA gates (s -> ~1) before delay starts.
#
#  Two-step pipeline (same shape as run_wm_demo.sh):
#    1) Auto-generate / reuse a 2 nM 100 s baseline checkpoint.
#    2) Run the WM attractor experiment on that checkpoint.
# ==============================================================================
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

DA_BASE="${DA_BASE:-2.0}"
BASELINE_DURATION_S="${BASELINE_DURATION_S:-100}"
GPU_ID="${GPU_ID:-0}"
TAG="${TAG:-wm_attractor}"

# Wang-2002-style attractor parameters (overridable via env)
INTRA_W="${INTRA_W:-4.0}"
INTRA_PROB="${INTRA_PROB:-0.5}"
POOL_SIZE="${POOL_SIZE:-100}"
CUE_AMP="${CUE_AMP:-300.0}"
CUE_MS="${CUE_MS:-1500.0}"
DELAY_MS="${DELAY_MS:-5000.0}"
BASELINE_MS="${BASELINE_MS:-5000.0}"
PROBE_MS="${PROBE_MS:-500.0}"

# WTA shared inhibition (helps suppress Mem-B, doesn't drive Mem-A)
USE_WTA_FLAG=""           # set to "--no-wta" to disable
IWM_SIZE="${IWM_SIZE:-60}"
E2I_W="${E2I_W:-3.0}"
E2I_PROB="${E2I_PROB:-0.20}"
I2E_W="${I2E_W:--3.5}"
I2E_PROB="${I2E_PROB:-0.15}"

CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

CKPT="$(ls -t "$CKPT_DIR"/ckpt_DA${DA_BASE%.*}nM_bg*_${BASELINE_DURATION_S}s.pkl 2>/dev/null | head -n1 || true)"
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
echo "  Step 2/2: running WM ATTRACTOR demo (Wang-2002 NMDA gating)"
echo "            ckpt        = ${CKPT}"
echo "            intra_w     = ${INTRA_W}   pool_size = ${POOL_SIZE}"
echo "            intra_prob  = ${INTRA_PROB}"
echo "            cue         = ${CUE_AMP} pA × ${CUE_MS} ms"
echo "            delay/probe = ${DELAY_MS}/${PROBE_MS} ms"
echo "──────────────────────────────────────────────────────────────"

python experiments/exp_wm_demo.py \
       --ckpt "${CKPT}" \
       --da "${DA_BASE}" \
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
