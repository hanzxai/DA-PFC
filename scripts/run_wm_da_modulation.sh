#!/usr/bin/env bash
set -euo pipefail
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

DA_BASE="${DA_BASE:-2.0}"
DA_PULSE="${DA_PULSE:-15.0}"
DA_WIN="${DA_WIN:-cue+delay}"
GPU_ID="${GPU_ID:-0}"
TAG="${TAG:-wm_da}"
BASELINE_DURATION_S="${BASELINE_DURATION_S:-100}"
POOL_SIZE="${POOL_SIZE:-60}"
INTRA_W="${INTRA_W:-4.0}"
INTRA_PROB="${INTRA_PROB:-0.5}"
CUE_AMP="${CUE_AMP:-300.0}"
CUE_MS="${CUE_MS:-1500.0}"
DELAY_MS="${DELAY_MS:-5000.0}"
BASELINE_MS="${BASELINE_MS:-5000.0}"
PROBE_MS="${PROBE_MS:-500.0}"
MEM_BG_OFFSET="${MEM_BG_OFFSET:--10}"
SFA_B="${SFA_B:-0.8}"
SFA_TAU="${SFA_TAU:-2000}"
ALPHA_GATE="${ALPHA_GATE:-0.15}"
IWM_SIZE="${IWM_SIZE:-60}"
E2I_W="${E2I_W:-3.0}"
E2I_PROB="${E2I_PROB:-0.20}"
I2E_W="${I2E_W:--3.5}"
I2E_PROB="${I2E_PROB:-0.15}"

# Match save_checkpoint() naming: drops trailing zeros and trailing dot.
# 2.0 -> "2", 2.5 -> "2.5", 15.0 -> "15"
DA_BASE_TAG="$(python3 -c "v='${DA_BASE}'; s=v.rstrip('0').rstrip('.') if '.' in v else v; print(s)")"
CKPT="checkpoints/ckpt_DA${DA_BASE_TAG}nM_bg200_${BASELINE_DURATION_S}s.pkl"
mkdir -p checkpoints
if [[ ! -f "$CKPT" ]]; then
  echo "[1/2] Generating baseline checkpoint at DA=${DA_BASE} nM (${BASELINE_DURATION_S} s)"
  python main.py --da "$DA_BASE" --duration "$BASELINE_DURATION_S" --save-ckpt
  if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: expected checkpoint not found at $CKPT after main.py."
    echo "Files actually present in checkpoints/:"
    ls -1 checkpoints/ || true
    exit 1
  fi
else
  echo "[1/2] Reusing existing checkpoint: $CKPT"
fi

python experiments/exp_wm_demo.py \
  --ckpt "$CKPT" --da "$DA_BASE" \
  --da-pulse "$DA_PULSE" --da-window "$DA_WIN" \
  --gpu "$GPU_ID" --tag "$TAG" \
  --pool-size "$POOL_SIZE" --intra-w "$INTRA_W" --intra-prob "$INTRA_PROB" \
  --cue-amp "$CUE_AMP" --cue-ms "$CUE_MS" \
  --baseline-ms "$BASELINE_MS" --delay-ms "$DELAY_MS" --probe-ms "$PROBE_MS" \
  --mem-bg-offset "$MEM_BG_OFFSET" \
  --sfa-b "$SFA_B" --sfa-tau "$SFA_TAU" --alpha-gate "$ALPHA_GATE" \
  --iwm-size "$IWM_SIZE" \
  --e2i-w "$E2I_W" --e2i-prob "$E2I_PROB" \
  --i2e-w "$I2E_W" --i2e-prob "$I2E_PROB"
