#!/bin/bash
# ============================================================
# Run All Experiments (Exp A → D) in pfc conda environment
# ============================================================
# Usage:
#   pfc                          # activate pfc conda env first
#   bash run_all_experiments.sh  # then run this script
#
# Or in one line:
#   bash -c "source activate pfc && bash run_all_experiments.sh"
# ============================================================

set -e  # Exit on error

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo "  DA-PFC: Running All Experiments (A → D)"
echo "  Project dir: $PROJECT_DIR"
echo "  Python: $(which python)"
echo "  Time: $(date)"
echo "============================================================"

# Verify checkpoint exists
CKPT="checkpoints/ckpt_DA2nM_500s.pkl"
if [ ! -f "$CKPT" ]; then
    echo "❌ Checkpoint not found: $CKPT"
    echo "   Please run: python main.py --da 2.0 --duration 500 --save-ckpt"
    exit 1
fi
echo "✅ Checkpoint found: $CKPT"
echo ""

# ============================================================
# Experiment A: DA Pulse Response
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 [1/4] Experiment A: DA Pulse Response"
echo "   Scientific question: D1/D2 temporal segregation & D1 afterglow window"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m experiments.exp_a_pulse_response --gpu 0
echo ""
echo "✅ Experiment A complete!"
echo ""

# ============================================================
# Experiment B: Dynamic vs Static Model Comparison
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 [2/4] Experiment B: Dynamic vs Static Model Comparison"
echo "   Scientific question: Response delay, overshoot, steady-state convergence"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m experiments.exp_b_dynamic_vs_static --gpu 0
echo ""
echo "✅ Experiment B complete!"
echo ""

# ============================================================
# Experiment C: Frequency Response
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 [3/4] Experiment C: Frequency Response"
echo "   Scientific question: D1/D2 low-pass filtering (Bode plot)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m experiments.exp_c_frequency_response --gpu 0
echo ""
echo "✅ Experiment C complete!"
echo ""

# ============================================================
# Experiment D: Working Memory Gating
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 [4/4] Experiment D: Working Memory Gating"
echo "   Scientific question: DA temporal gating of stimulus response"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python -m experiments.exp_d_working_memory --gpu 0
echo ""
echo "✅ Experiment D complete!"
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "  ✅ All 4 experiments completed successfully!"
echo "  📁 Results saved in: $PROJECT_DIR/outputs/"
echo "  Time: $(date)"
echo "============================================================"
echo ""
echo "Output directories:"
ls -dt outputs/exp_*pulse* outputs/exp_*dynamic* outputs/exp_*freq* outputs/exp_*working* 2>/dev/null | head -8
