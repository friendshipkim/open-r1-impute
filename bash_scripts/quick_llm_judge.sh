#!/bin/bash

# Quick start script for LLM-as-Judge evaluation
# Usage: bash bash_scripts/quick_llm_judge.sh

set -e

# Configuration
MODEL_PATH="friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0"
OUTPUT_DIR="llm_judge_results"
NUM_SAMPLES=20
JUDGE_MODEL="gpt-4"

# Check if API key is set
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "Error: OPENAI_API_KEY environment variable is not set."
    echo "Please set it with: export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

echo "============================================================"
echo "LLM-as-Judge Evaluation Quick Start"
echo "============================================================"
echo ""
echo "Model: $MODEL_PATH"
echo "Judge Model: $JUDGE_MODEL"
echo "Samples: $NUM_SAMPLES"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation on different datasets
echo "Running evaluation on UltraChat dataset..."
python scripts/llm_judge_evaluation.py \
    --model-path "$MODEL_PATH" \
    --dataset ultrachat \
    --num-samples "$NUM_SAMPLES" \
    --judge-model "$JUDGE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --api-key "$OPENAI_API_KEY"

echo ""
echo "Running evaluation on custom prompts..."
python scripts/llm_judge_evaluation.py \
    --model-path "$MODEL_PATH" \
    --dataset custom \
    --num-samples "$NUM_SAMPLES" \
    --judge-model "$JUDGE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --api-key "$OPENAI_API_KEY"

echo ""
echo "============================================================"
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - llm_judge_results_ultrachat.json"
echo "  - llm_judge_results_custom.json"
echo ""
echo "To view results, check the JSON files in $OUTPUT_DIR"
echo "============================================================" 