#!/usr/bin/env bash
# ==============================================================================
# run_da_waveform.sh
# Run DA waveform experiments from checkpoint
#
# Usage:
#   bash scripts/run_da_waveform.sh                     # run ALL modes
#   bash scripts/run_da_waveform.sh --mode spike        # single mode
#   bash scripts/run_da_waveform.sh --mode sine --freq 0.1 --amplitude 5
#   bash scripts/run_da_waveform.sh --skip-ckpt         # reuse existing checkpoint
#   bash scripts/run_da_waveform.sh --help               # show all options
# ==============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          DA-PFC: DA Waveform Experiment                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

python experiments/da_waveform_exp.py "$@"
