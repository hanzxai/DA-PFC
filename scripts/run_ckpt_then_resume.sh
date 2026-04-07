#!/usr/bin/env bash
# ==============================================================================
# run_ckpt_then_resume.sh
# One-shot script: Generate baseline checkpoint → Resume with new DA
#
# This script combines two steps that are normally run separately:
#   Step 1: Run baseline simulation (DA=2nM, 500s) and save checkpoint
#   Step 2: Resume from that checkpoint with a new DA concentration
#
# Usage:
#   bash scripts/run_ckpt_then_resume.sh                    # defaults: DA=15nM, 100s
#   bash scripts/run_ckpt_then_resume.sh --da 9             # resume DA=9nM
#   bash scripts/run_ckpt_then_resume.sh --da 15 --dur 200  # resume DA=15nM, 200s
#   bash scripts/run_ckpt_then_resume.sh --skip-ckpt        # skip step 1, reuse existing ckpt
#
# Configurable via environment variables or CLI args:
#   BASELINE_DA   : Baseline DA concentration (nM), default=2
#   BASELINE_DUR  : Baseline simulation duration (s), default=500
#   RESUME_DA     : Resume DA concentration (nM), default=15
#   RESUME_DUR    : Resume simulation duration (s), default=100
#   GPU           : GPU card number, default=0
# ==============================================================================

set -euo pipefail

# ── Defaults ──
BASELINE_DA="${BASELINE_DA:-2}"
BASELINE_DUR="${BASELINE_DUR:-500}"
RESUME_DA="${RESUME_DA:-15}"
RESUME_DUR="${RESUME_DUR:-100}"
GPU="${GPU:-0}"
SKIP_CKPT=false

# ── Parse CLI args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --da)         RESUME_DA="$2";       shift 2 ;;
        --dur)        RESUME_DUR="$2";      shift 2 ;;
        --base-da)    BASELINE_DA="$2";     shift 2 ;;
        --base-dur)   BASELINE_DUR="$2";    shift 2 ;;
        --gpu)        GPU="$2";             shift 2 ;;
        --skip-ckpt)  SKIP_CKPT=true;       shift   ;;
        -h|--help)
            echo "Usage: bash scripts/run_ckpt_then_resume.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --da <nM>        Resume DA concentration (default: 15)"
            echo "  --dur <s>        Resume duration in seconds (default: 100)"
            echo "  --base-da <nM>   Baseline DA concentration (default: 2)"
            echo "  --base-dur <s>   Baseline duration in seconds (default: 500)"
            echo "  --gpu <id>       GPU card number (default: 0)"
            echo "  --skip-ckpt      Skip checkpoint generation, reuse existing"
            echo "  -h, --help       Show this help"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Derived paths ──
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BG_MEAN=$(python3 -c "import sys; sys.path.insert(0, '${PROJECT_DIR}'); import config; print(f'{config.BG_MEAN:g}')")
CKPT_FILE="${PROJECT_DIR}/checkpoints/ckpt_DA${BASELINE_DA}nM_bg${BG_MEAN}_${BASELINE_DUR}s.pkl"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          DA-PFC: Checkpoint → Resume Pipeline               ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Baseline:  DA=${BASELINE_DA}nM, Duration=${BASELINE_DUR}s"
echo "║  Resume:    DA=${RESUME_DA}nM, Duration=${RESUME_DUR}s"
echo "║  GPU:       ${GPU}"
echo "║  Checkpoint: ${CKPT_FILE}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd "${PROJECT_DIR}"

# ── Step 1: Generate checkpoint (if not skipping) ──
if [ "$SKIP_CKPT" = true ]; then
    echo "⏭️  Skipping checkpoint generation (--skip-ckpt)"
    if [ ! -f "${CKPT_FILE}" ]; then
        echo "❌ Checkpoint file not found: ${CKPT_FILE}"
        echo "   Remove --skip-ckpt to generate it first."
        exit 1
    fi
    echo "✅ Using existing checkpoint: ${CKPT_FILE}"
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Step 1/2: Generating baseline checkpoint..."
    echo "   Command: python main.py --da ${BASELINE_DA} --duration ${BASELINE_DUR} --gpu ${GPU} --save-ckpt"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python main.py --da "${BASELINE_DA}" --duration "${BASELINE_DUR}" --gpu "${GPU}" --save-ckpt

    if [ ! -f "${CKPT_FILE}" ]; then
        echo "❌ Checkpoint was not created at expected path: ${CKPT_FILE}"
        echo "   Check the output above for errors."
        exit 1
    fi
    echo ""
    echo "✅ Checkpoint saved: ${CKPT_FILE}"
fi

echo ""

# ── Step 2: Resume from checkpoint ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Step 2/2: Resuming simulation with DA=${RESUME_DA}nM..."
echo "   Command: python main.py --resume ${CKPT_FILE} --da ${RESUME_DA} --duration ${RESUME_DUR} --gpu ${GPU}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python main.py --resume "${CKPT_FILE}" --da "${RESUME_DA}" --duration "${RESUME_DUR}" --gpu "${GPU}"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Pipeline Complete!                                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
