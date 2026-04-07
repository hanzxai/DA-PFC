#!/usr/bin/env bash
# ==============================================================================
# quick_test.sh
# Quick test after parameter changes — short simulation to verify neurons alive
#
# Runs a short baseline + resume to quickly check if parameter changes
# (e.g. BIAS_D1, BIAS_D2) produce reasonable firing rates.
#
# Usage:
#   bash scripts/quick_test.sh              # default: baseline 50s, resume 20s
#   bash scripts/quick_test.sh --da 9       # test with DA=9nM
# ==============================================================================

set -euo pipefail

# ── Defaults (short durations for quick feedback) ──
BASELINE_DA="${BASELINE_DA:-2}"
BASELINE_DUR="${BASELINE_DUR:-50}"
RESUME_DA="${RESUME_DA:-15}"
RESUME_DUR="${RESUME_DUR:-20}"
GPU="${GPU:-0}"

# ── Parse CLI args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --da)         RESUME_DA="$2";       shift 2 ;;
        --dur)        RESUME_DUR="$2";      shift 2 ;;
        --base-dur)   BASELINE_DUR="$2";    shift 2 ;;
        --gpu)        GPU="$2";             shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/quick_test.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --da <nM>        Resume DA concentration (default: 15)"
            echo "  --dur <s>        Resume duration (default: 20)"
            echo "  --base-dur <s>   Baseline duration (default: 50)"
            echo "  --gpu <id>       GPU card number (default: 0)"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            exit 1
            ;;
    esac
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DA-PFC: Quick Parameter Test                               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Baseline: DA=${BASELINE_DA}nM × ${BASELINE_DUR}s (short)   "
echo "║  Resume:   DA=${RESUME_DA}nM × ${RESUME_DUR}s (short)       "
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

echo "📋 Current config.py key parameters:"
python3 -c "
import config
print(f'   BIAS_D1={config.BIAS_D1}, BIAS_D2={config.BIAS_D2}')
print(f'   BG_MEAN={config.BG_MEAN}, BG_STD={config.BG_STD}')
print(f'   EPS_D1={config.EPS_D1}, EPS_D2={config.EPS_D2}')
"
echo ""

# Step 1: Short baseline + save checkpoint
echo "📦 Step 1: Short baseline (${BASELINE_DUR}s)..."
python main.py --da "${BASELINE_DA}" --duration "${BASELINE_DUR}" --gpu "${GPU}" --save-ckpt

# Find the checkpoint file
BG_MEAN=$(python3 -c "import config; print(f'{config.BG_MEAN:g}')")
CKPT_FILE="checkpoints/ckpt_DA${BASELINE_DA}nM_bg${BG_MEAN}_${BASELINE_DUR}s.pkl"

if [ ! -f "${CKPT_FILE}" ]; then
    echo "❌ Checkpoint not found: ${CKPT_FILE}"
    exit 1
fi

echo ""

# Step 2: Short resume
echo "🚀 Step 2: Quick resume DA=${RESUME_DA}nM (${RESUME_DUR}s)..."
python main.py --resume "${CKPT_FILE}" --da "${RESUME_DA}" --duration "${RESUME_DUR}" --gpu "${GPU}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Quick test done! Check outputs/ for results.            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
