#!/usr/bin/env bash
# ==============================================================================
# sweep_resume_da.sh
# Sweep multiple DA concentrations from the same baseline checkpoint
#
# First generates (or reuses) a baseline checkpoint, then runs resume
# experiments for each DA level in the list.
#
# Usage:
#   bash scripts/sweep_resume_da.sh                          # default: 5,9,15,30 nM
#   bash scripts/sweep_resume_da.sh --da-list "3,5,9,15,30"  # custom list
#   bash scripts/sweep_resume_da.sh --skip-ckpt              # reuse existing ckpt
# ==============================================================================

set -euo pipefail

# ── Defaults ──
BASELINE_DA="${BASELINE_DA:-2}"
BASELINE_DUR="${BASELINE_DUR:-500}"
DA_LIST="${DA_LIST:-5,9,15,30}"
RESUME_DUR="${RESUME_DUR:-100}"
GPU="${GPU:-0}"
SKIP_CKPT=false

# ── Parse CLI args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --da-list)    DA_LIST="$2";         shift 2 ;;
        --dur)        RESUME_DUR="$2";      shift 2 ;;
        --base-da)    BASELINE_DA="$2";     shift 2 ;;
        --base-dur)   BASELINE_DUR="$2";    shift 2 ;;
        --gpu)        GPU="$2";             shift 2 ;;
        --skip-ckpt)  SKIP_CKPT=true;       shift   ;;
        -h|--help)
            echo "Usage: bash scripts/sweep_resume_da.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --da-list <list>   Comma-separated DA levels (default: 5,9,15,30)"
            echo "  --dur <s>          Resume duration per level (default: 100)"
            echo "  --base-da <nM>     Baseline DA (default: 2)"
            echo "  --base-dur <s>     Baseline duration (default: 500)"
            echo "  --gpu <id>         GPU card number (default: 0)"
            echo "  --skip-ckpt        Skip checkpoint generation"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            exit 1
            ;;
    esac
done

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BG_MEAN=$(python3 -c "import sys; sys.path.insert(0, '${PROJECT_DIR}'); import config; print(f'{config.BG_MEAN:g}')")
CKPT_FILE="${PROJECT_DIR}/checkpoints/ckpt_DA${BASELINE_DA}nM_bg${BG_MEAN}_${BASELINE_DUR}s.pkl"

# Convert comma-separated list to array
IFS=',' read -ra DA_ARRAY <<< "${DA_LIST}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          DA-PFC: DA Concentration Sweep                     ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Baseline:   DA=${BASELINE_DA}nM, Duration=${BASELINE_DUR}s"
echo "║  DA levels:  ${DA_LIST}"
echo "║  Resume dur: ${RESUME_DUR}s each"
echo "║  GPU:        ${GPU}"
echo "║  Total runs: ${#DA_ARRAY[@]}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd "${PROJECT_DIR}"

# ── Step 1: Generate checkpoint ──
if [ "$SKIP_CKPT" = true ]; then
    echo "⏭️  Skipping checkpoint generation"
    if [ ! -f "${CKPT_FILE}" ]; then
        echo "❌ Checkpoint not found: ${CKPT_FILE}"
        exit 1
    fi
else
    echo "📦 Generating baseline checkpoint..."
    python main.py --da "${BASELINE_DA}" --duration "${BASELINE_DUR}" --gpu "${GPU}" --save-ckpt
    echo ""
fi

# ── Step 2: Sweep DA levels ──
TOTAL=${#DA_ARRAY[@]}
for i in "${!DA_ARRAY[@]}"; do
    DA_VAL="${DA_ARRAY[$i]}"
    RUN_NUM=$((i + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 Run ${RUN_NUM}/${TOTAL}: DA=${DA_VAL}nM, Duration=${RESUME_DUR}s"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python main.py --resume "${CKPT_FILE}" --da "${DA_VAL}" --duration "${RESUME_DUR}" --gpu "${GPU}"
    echo ""
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Sweep Complete! ${TOTAL} DA levels tested.               "
echo "╚══════════════════════════════════════════════════════════════╝"
