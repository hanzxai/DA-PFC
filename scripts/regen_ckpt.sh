#!/usr/bin/env bash
# ==============================================================================
# regen_ckpt.sh
# Regenerate baseline checkpoint after config.py parameter changes
#
# When you modify parameters in config.py (e.g. BIAS_D1, BIAS_D2, BG_MEAN...),
# existing checkpoints become stale and the fingerprint check will reject them.
# This script regenerates the checkpoint with current parameters.
#
# Usage:
#   bash scripts/regen_ckpt.sh                     # defaults: DA=2nM, 500s
#   bash scripts/regen_ckpt.sh --da 2 --dur 500    # explicit
#   bash scripts/regen_ckpt.sh --da 2 --dur 300    # shorter baseline
# ==============================================================================

set -euo pipefail

# ── Defaults ──
DA="${DA:-2}"
DUR="${DUR:-500}"
GPU="${GPU:-0}"

# ── Parse CLI args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --da)   DA="$2";   shift 2 ;;
        --dur)  DUR="$2";  shift 2 ;;
        --gpu)  GPU="$2";  shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/regen_ckpt.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --da <nM>    Baseline DA concentration (default: 2)"
            echo "  --dur <s>    Simulation duration (default: 500)"
            echo "  --gpu <id>   GPU card number (default: 0)"
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
echo "║  Regenerating Checkpoint (DA=${DA}nM, ${DUR}s)              "
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Current config.py key parameters:"
python3 -c "
import config
print(f'   BIAS_D1  = {config.BIAS_D1}')
print(f'   BIAS_D2  = {config.BIAS_D2}')
print(f'   BG_MEAN  = {config.BG_MEAN}')
print(f'   BG_STD   = {config.BG_STD}')
print(f'   EPS_D1   = {config.EPS_D1}')
print(f'   EPS_D2   = {config.EPS_D2}')
print(f'   V_REST   = {config.V_REST}')
print(f'   V_TH     = {config.V_TH}')
print(f'   R_BASE   = {config.R_BASE}')
"
echo ""

python main.py --da "${DA}" --duration "${DUR}" --gpu "${GPU}" --save-ckpt

echo ""
echo "✅ Checkpoint regenerated with current config.py parameters."
echo "   You can now run resume experiments."
