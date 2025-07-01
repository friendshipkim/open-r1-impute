#!/bin/bash

# Quick start script for local model comparison (no API costs)
# Compares trained GRPO model with untrained baseline model
# Usage: bash bash_scripts/quick_local_comparison.sh

set -e

# Configuration
TRAINED_MODEL="friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0"
BASELINE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR="local_comparison_results"
NUM_SAMPLES=50

echo "============================================================"
echo "Local Model Comparison: Trained GRPO vs Baseline (No API Costs)"
echo "============================================================"
echo ""
echo "Trained Model: $TRAINED_MODEL"
echo "Baseline Model: $BASELINE_MODEL"
echo "Evaluation Method: Local Heuristics"
echo "Samples: $NUM_SAMPLES"
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "Note: This comparison uses local heuristics and costs nothing!"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run local model comparison
echo "Starting local model comparison..."
python scripts/model_comparison_local.py \
    --model1 "$TRAINED_MODEL" \
    --model2 "$BASELINE_MODEL" \
    --dataset ultrachat \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --show-samples 5

echo ""
echo "============================================================"
echo "Local Model Comparison Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - local_model_comparison_ultrachat.json"
echo ""
echo "To view results, check the JSON file in $OUTPUT_DIR"
echo "This evaluation cost nothing and used local heuristics!"
echo "============================================================" 