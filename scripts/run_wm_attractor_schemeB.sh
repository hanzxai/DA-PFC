#!/usr/bin/env bash
# ==============================================================================
#  WM Attractor Demo — SCHEME B: per-pool baseline-bias offset
#
#  Strategy:
#    - Use the standard BG=200 baseline ckpt (no need to regenerate).
#    - Apply an EXTRA constant DC offset (-30 pA by default) to ONLY the
#      Mem-A∪Mem-B pool neurons, so:
#         * E-BG / I / D1 / D2 keep V_inf = -50 mV (their original 200pA)
#         * Mem-A / Mem-B see V_inf = -53 mV (170 pA effective)
#       => Mem pools sit silently in baseline (~0-2 Hz) instead of latching
#          into the high attractor state at t=0.
#    - Cue (300 pA, 1.5 s) on Mem-A overrides the offset and pushes r→40 Hz.
#    - During those 1.5 s NMDA gates saturate (s ≈ 1).
#    - After cue offset, recurrent NMDA EPSC alone (≈ 50 * 4 * 1 = 200 pA
#      with intra_w=4, p=0.5) compensates for the -30 pA offset and keeps
#      Mem-A above V_th in a clean HIGH attractor state.
#  Result: clear Mem-A persistence in the delay window; Mem-B stays silent.
# ==============================================================================
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

DA_BASE="${DA_BASE:-2.0}"
BASELINE_DURATION_S="${BASELINE_DURATION_S:-100}"
GPU_ID="${GPU_ID:-0}"
TAG="${TAG:-wm_attr_schemeB}"

INTRA_W="${INTRA_W:-4.0}"
INTRA_PROB="${INTRA_PROB:-0.5}"
POOL_SIZE="${POOL_SIZE:-100}"
CUE_AMP="${CUE_AMP:-300.0}"
CUE_MS="${CUE_MS:-1500.0}"
DELAY_MS="${DELAY_MS:-5000.0}"
BASELINE_MS="${BASELINE_MS:-5000.0}"
PROBE_MS="${PROBE_MS:-500.0}"

# Scheme-B knob: NEGATIVE pA, applied only to Mem pools.
MEM_BG_OFFSET="${MEM_BG_OFFSET:--15.0}"

# Spike-frequency adaptation: natural decay of persistent activity.
SFA_B="${SFA_B:-0.8}"
SFA_TAU="${SFA_TAU:-2000.0}"

USE_WTA_FLAG=""
IWM_SIZE="${IWM_SIZE:-60}"
E2I_W="${E2I_W:-3.0}"
E2I_PROB="${E2I_PROB:-0.20}"
I2E_W="${I2E_W:--3.5}"
I2E_PROB="${I2E_PROB:-0.15}"

CKPT_DIR="checkpoints"
mkdir -p "$CKPT_DIR"

# Use the standard BG=200 ckpt (no need for low-bg variant).
CKPT="${CKPT_DIR}/ckpt_DA${DA_BASE%.*}nM_bg200_${BASELINE_DURATION_S}s.pkl"
if [[ ! -f "$CKPT" ]]; then
  CKPT="${CKPT_DIR}/ckpt_DA${DA_BASE}nM_bg200_${BASELINE_DURATION_S}s.pkl"
fi

if [[ ! -f "$CKPT" ]]; then
  echo "──────────────────────────────────────────────────────────────"
  echo "  Step 1/2: generating DA=${DA_BASE} nM BG=200 baseline ckpt"
  echo "──────────────────────────────────────────────────────────────"
  python main.py --da "${DA_BASE}" \
                 --duration "${BASELINE_DURATION_S}" \
                 --gpu "${GPU_ID}" \
                 --save-ckpt
  CKPT="${CKPT_DIR}/ckpt_DA${DA_BASE%.*}nM_bg200_${BASELINE_DURATION_S}s.pkl"
  if [[ ! -f "$CKPT" ]]; then
    CKPT="${CKPT_DIR}/ckpt_DA${DA_BASE}nM_bg200_${BASELINE_DURATION_S}s.pkl"
  fi
fi

if [[ ! -f "$CKPT" ]]; then
  echo "❌ Failed to locate BG=200 checkpoint." >&2
  exit 1
fi

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  Step 2/2: SCHEME-B WM attractor (per-pool BG offset)"
echo "            ckpt          = ${CKPT}"
echo "            mem_bg_offset = ${MEM_BG_OFFSET} pA"
echo "            sfa_b         = ${SFA_B} pA/spike, sfa_tau = ${SFA_TAU} ms"
echo "            intra_w       = ${INTRA_W}   intra_prob = ${INTRA_PROB}"
echo "            cue           = ${CUE_AMP} pA × ${CUE_MS} ms"
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
       --mem-bg-offset "${MEM_BG_OFFSET}" \
       --sfa-b "${SFA_B}" \
       --sfa-tau "${SFA_TAU}" \
       --iwm-size "${IWM_SIZE}" \
       --e2i-w "${E2I_W}" \
       --e2i-prob "${E2I_PROB}" \
       --i2e-w "${I2E_W}" \
       --i2e-prob "${I2E_PROB}" \
       ${USE_WTA_FLAG}

echo ""
echo "✅ Done. Browse the latest folder under outputs/ for results."
echo "   Key file: wm_report.txt — look at Δ(Late-BL) for Mem-A."
