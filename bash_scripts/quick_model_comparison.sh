#!/bin/bash

# Quick start script for model comparison using LLM-as-Judge
# Compares trained GRPO model with untrained baseline model
# Usage: bash bash_scripts/quick_model_comparison.sh

set -e

# Configuration
TRAINED_MODEL="friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0"
BASELINE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR="model_comparison_results"
NUM_SAMPLES=50
JUDGE_MODEL="gpt-4"

# Check if API key is set
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set it with: export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

echo "============================================================"
echo "Model Comparison: Trained GRPO vs Baseline"
echo "============================================================"
echo ""
echo "Trained Model: $TRAINED_MODEL"
echo "Baseline Model: $BASELINE_MODEL"
echo "Judge Model: $JUDGE_MODEL"
echo "Samples: $NUM_SAMPLES"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run model comparison
echo "Starting model comparison..."
python scripts/model_comparison.py \
    --model1 "$TRAINED_MODEL" \
    --model2 "$BASELINE_MODEL" \
    --dataset ultrachat \
    --num-samples "$NUM_SAMPLES" \
    --judge-model "$JUDGE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --api-key "$OPENAI_API_KEY" \
    --show-samples 5

echo ""
echo "============================================================"
echo "Model Comparison Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - model_comparison_ultrachat.json"
echo ""
echo "To view results, check the JSON file in $OUTPUT_DIR"
echo "============================================================" 