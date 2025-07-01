#!/bin/bash

# Script to evaluate and compare GRPO Oracle vs GRPO Impute models

set -e

# Default values
ORACLE_MODEL_PATH=""
IMPUTE_MODEL_PATH=""
OUTPUT_DIR="evaluation_results"
TASKS="math_500,gpqa:diamond"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --oracle-model)
            ORACLE_MODEL_PATH="$2"
            shift 2
            ;;
        --impute-model)
            IMPUTE_MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --oracle-model PATH --impute-model PATH [--output-dir DIR] [--tasks TASK1,TASK2]"
            echo ""
            echo "Arguments:"
            echo "  --oracle-model PATH    Path to GRPO Oracle model"
            echo "  --impute-model PATH    Path to GRPO Impute model"
            echo "  --output-dir DIR       Output directory (default: evaluation_results)"
            echo "  --tasks TASK1,TASK2    Comma-separated list of tasks (default: math_500,gpqa:diamond)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$ORACLE_MODEL_PATH" ]]; then
    echo "Error: --oracle-model is required"
    exit 1
fi

if [[ -z "$IMPUTE_MODEL_PATH" ]]; then
    echo "Error: --impute-model is required"
    exit 1
fi

# Check if model paths exist
if [[ ! -d "$ORACLE_MODEL_PATH" ]]; then
    echo "Error: Oracle model path does not exist: $ORACLE_MODEL_PATH"
    exit 1
fi

if [[ ! -d "$IMPUTE_MODEL_PATH" ]]; then
    echo "Error: Impute model path does not exist: $IMPUTE_MODEL_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Model Comparison: GRPO Oracle vs GRPO Impute"
echo "============================================================"
echo ""
echo "Oracle Model: $ORACLE_MODEL_PATH"
echo "Impute Model: $IMPUTE_MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Tasks: $TASKS"
echo ""

# Function to evaluate a single model
evaluate_model() {
    local model_path="$1"
    local model_name="$2"
    local output_file="$3"
    
    echo "Evaluating $model_name model..."
    echo "Model path: $model_path"
    echo ""
    
    # Run evaluation using the existing benchmark infrastructure
    python -m lighteval \
        --model_args pretrained=$model_path \
        --tasks $TASKS \
        --output_dir "$OUTPUT_DIR/${model_name}_results" \
        --save_details \
        --max_samples 100 \
        --batch_size 1
    
    echo "$model_name evaluation complete!"
    echo ""
}

# Evaluate both models
evaluate_model "$ORACLE_MODEL_PATH" "oracle" "$OUTPUT_DIR/oracle_results"
evaluate_model "$IMPUTE_MODEL_PATH" "impute" "$OUTPUT_DIR/impute_results"

# Compare results
echo "============================================================"
echo "Comparing Results"
echo "============================================================"

# Simple comparison script
python3 -c "
import json
import os
import sys

def load_results(results_dir):
    \"\"\"Load results from lighteval output.\"\"\"
    results_file = os.path.join(results_dir, 'results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return {}

oracle_results = load_results('$OUTPUT_DIR/oracle_results')
impute_results = load_results('$OUTPUT_DIR/impute_results')

print('\\nComparison Summary:')
print('=' * 60)

for task in ['$TASKS'.split(',')]:
    task = task.strip()
    if task in oracle_results and task in impute_results:
        oracle_acc = oracle_results[task].get('acc_norm', 0)
        impute_acc = impute_results[task].get('acc_norm', 0)
        
        print(f'\\n{task}:')
        print(f'  Oracle Accuracy: {oracle_acc:.3f}')
        print(f'  Impute Accuracy: {impute_acc:.3f}')
        print(f'  Difference: {oracle_acc - impute_acc:.3f}')
        
        if oracle_acc > impute_acc:
            print('  🎉 Oracle performs better!')
        elif impute_acc > oracle_acc:
            print('  🎉 Impute performs better!')
        else:
            print('  🤝 Models perform similarly!')

print('\\n' + '=' * 60)
print('Evaluation complete!')
print(f'Results saved to: $OUTPUT_DIR')
"

echo ""
echo "Evaluation complete! Results saved to: $OUTPUT_DIR" 