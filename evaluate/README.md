# Model Evaluation Tools

This folder contains all the evaluation and comparison tools for GRPO models.

## Files Description

### Core Evaluation Scripts

- **`model_comparison.py`** - LLM-as-Judge evaluation using GPT-4
  - Compares two models using OpenAI's GPT-4 as judge
  - Provides winner determination and detailed explanations
  - No numerical scores, only winner/loser/tie decisions
  - Requires OpenAI API key
  - **NEW**: Integrated VLLM for 5-10x faster inference

- **`model_comparison_local.py`** - Local heuristic-based evaluation
  - Compares two models using local heuristics
  - Provides numerical scores and winner determination
  - No external API required, runs completely locally
  - Faster but potentially less accurate than GPT-4
  - **NEW**: Integrated VLLM for 5-10x faster inference

- **`inference_with_vllm.py`** - Fast inference using VLLM
  - Standalone inference tool for single model evaluation
  - Uses VLLM for high-speed generation
  - Useful for bulk generation or production inference
  - Not required for model comparison

- **`create_report.py`** - HTML report generator
  - Creates shareable HTML reports from evaluation results
  - Converts JSON results to web-friendly format
  - Includes visualizations and detailed comparisons

## Usage Examples

### LLM-as-Judge Evaluation (GPT-4) with VLLM
```bash
cd /root/open-r1-impute
PYTHONPATH=/root/open-r1-impute/src python evaluate/model_comparison.py \
  --grpo-comparison \
  --evaluation-method gpt4 \
  --num-samples 50 \
  --api-key YOUR_OPENAI_API_KEY \
  --use-vllm
```

### Local Evaluation (Free) with VLLM
```bash
cd /root/open-r1-impute
PYTHONPATH=/root/open-r1-impute/src python evaluate/model_comparison_local.py \
  --grpo-comparison \
  --evaluation-method local \
  --num-samples 100 \
  --use-vllm
```

### Fallback to Standard Transformers (if VLLM not available)
```bash
cd /root/open-r1-impute
PYTHONPATH=/root/open-r1-impute/src python evaluate/model_comparison.py \
  --grpo-comparison \
  --evaluation-method gpt4 \
  --num-samples 50 \
  --no-vllm
```

### Generate HTML Report
```bash
cd /root/open-r1-impute
python evaluate/create_report.py \
  --input evaluation_results/grpo_comparison_results/grpo_model_comparison_local.json \
  --output evaluation_results/grpo_comparison_report.html
```

### Fast Inference
```bash
cd /root/open-r1-impute
python evaluate/inference_with_vllm.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt "What is the capital of France?"
```

## VLLM Integration Benefits

### 🚀 **Performance Improvements**
- **5-10x faster inference** compared to standard transformers
- **Batch processing** for efficient GPU utilization
- **Memory optimization** with tensor parallelism
- **Automatic fallback** to transformers if VLLM unavailable

### ⚙️ **Configuration Options**
- `--use-vllm`: Enable VLLM inference (default: True)
- `--no-vllm`: Disable VLLM and use standard transformers
- Automatic detection of VLLM availability

### 🔧 **Technical Details**
- Uses VLLM's optimized CUDA kernels
- Supports tensor parallelism for multi-GPU setups
- Configurable batch sizes and memory utilization
- Maintains compatibility with existing evaluation pipeline

## Output Structure

Results are saved to the `evaluation_results/` folder with the following structure:

```
evaluation_results/
├── grpo_comparison_results/          # GRPO model comparison results
│   ├── grpo_model_comparison_local.json
│   └── grpo_model_comparison_gpt4.json
├── local_comparison_results/         # Other local comparison results
├── grpo_analysis_summary.json        # Analysis summary data
├── grpo_analysis_report.txt          # Detailed text report
├── GRPO_Analysis_Summary.md          # Human-readable summary
└── grpo_comparison_report.html       # Shareable HTML report
```

## Key Features

- **Dual Evaluation Methods**: Choose between GPT-4 (accurate, paid) or local heuristics (free, fast)
- **VLLM Integration**: 5-10x faster inference with automatic fallback
- **Comprehensive Analysis**: Detailed categorization and performance analysis
- **Shareable Reports**: Generate HTML reports for collaboration
- **Validation Prompts**: Uses test_sft split to avoid training data overlap
- **Flexible Configuration**: Customizable model paths, sample sizes, and evaluation parameters 