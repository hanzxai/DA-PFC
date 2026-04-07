#!/usr/bin/env bash
# ==============================================================================
# run_two_stage.sh
# Two-stage DA dosing experiment (no checkpoint needed)
#
# Runs a single simulation where DA switches from level 1 to level 2
# at a specified onset time.
#
# Usage:
#   bash scripts/run_two_stage.sh                              # DA: 2→15 nM
#   bash scripts/run_two_stage.sh --da1 2 --da2 9              # DA: 2→9 nM
#   bash scripts/run_two_stage.sh --da1 2 --da2 15 --dur 200   # 200s total
# ==============================================================================

set -euo pipefail

# ── Defaults ──
DA1="${DA1:-2}"
DA2="${DA2:-15}"
DUR="${DUR:-100}"
ONSET="${ONSET:-}"
GPU="${GPU:-0}"

# ── Parse CLI args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --da1)    DA1="$2";    shift 2 ;;
        --da2)    DA2="$2";    shift 2 ;;
        --dur)    DUR="$2";    shift 2 ;;
        --onset)  ONSET="$2";  shift 2 ;;
        --gpu)    GPU="$2";    shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/run_two_stage.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --da1 <nM>     Phase 1 DA concentration (default: 2)"
            echo "  --da2 <nM>     Phase 2 DA concentration (default: 15)"
            echo "  --dur <s>      Total duration in seconds (default: 100)"
            echo "  --onset <s>    Phase 2 onset time in seconds (default: auto)"
            echo "  --gpu <id>     GPU card number (default: 0)"
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
echo "║  DA-PFC: Two-Stage DA Experiment                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Phase 1: DA=${DA1}nM"
echo "║  Phase 2: DA=${DA2}nM"
echo "║  Duration: ${DUR}s"
echo "║  Onset:   ${ONSET:-auto}"
echo "║  GPU:     ${GPU}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

CMD="python main.py --da ${DA1} --da2 ${DA2} --duration ${DUR} --gpu ${GPU}"
if [ -n "${ONSET}" ]; then
    CMD="${CMD} --phase2-onset ${ONSET}"
fi

echo "🚀 Running: ${CMD}"
eval "${CMD}"

echo ""
echo "✅ Two-stage experiment complete!"
