# LLM-as-Judge Evaluation Guide for GRPO Models

This guide provides a comprehensive approach to implementing LLM-as-Judge evaluation for your trained GRPO model at [friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0](https://huggingface.co/friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0).

## Overview

LLM-as-Judge is a technique that uses a powerful language model (like GPT-4) to evaluate the quality of responses from other models. This approach provides more nuanced and human-like evaluation compared to traditional metrics.

## Prerequisites

### 1. Install Dependencies
```bash
pip install openai numpy tqdm transformers torch datasets
```

### 2. Set Up OpenAI API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Method 1: Quick Evaluation Script
```bash
# Run the quick start script
bash bash_scripts/quick_llm_judge.sh
```

This will:
- Evaluate your model on UltraChat dataset (20 samples)
- Evaluate your model on custom prompts (20 samples)
- Save results to `llm_judge_results/`

### Method 2: Comprehensive Evaluation
```bash
# Run comprehensive evaluation
python scripts/llm_judge_evaluation.py \
    --model-path "friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0" \
    --dataset ultrachat \
    --num-samples 50 \
    --judge-model gpt-4 \
    --output-dir evaluation_results
```

### Method 3: Compare with Baseline
```bash
# Compare with a baseline model
python scripts/llm_judge_evaluation.py \
    --model-path "friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0" \
    --baseline-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset custom \
    --num-samples 30 \
    --judge-model gpt-4
```

## Understanding Your Model

Your model was trained with the following configuration:
- **Base Model**: Qwen2.5-1.5B-Instruct
- **Training Method**: GRPO with imputation
- **Training Data**: UltraChat 200k
- **Reward Model**: QRM (Quality Reward Model)
- **Imputation Settings**: 
  - `start_patch: 0.2` (20% of training steps)
  - `start_pre_patch: 0.0` (no pre-patch)
  - `rho: 0` (correlation threshold)

## Evaluation Datasets

### 1. UltraChat Dataset
- **Source**: HuggingFaceH4/ultrachat_200k
- **Type**: General conversation
- **Format**: User-assistant conversations
- **Use Case**: Evaluate general chat capabilities

### 2. Math Dataset
- **Source**: HuggingFaceH4/MATH-500
- **Type**: Mathematical reasoning
- **Format**: Math problems with solutions
- **Use Case**: Evaluate reasoning capabilities

### 3. Custom Prompts
- **Type**: Curated diverse prompts
- **Categories**: 
  - Technical explanations
  - Creative writing
  - Problem solving
  - Educational content
- **Use Case**: Targeted evaluation of specific skills

## Understanding the Results

### Score Interpretation
- **0.0-0.3**: Poor response
- **0.3-0.6**: Adequate response
- **0.6-0.8**: Good response
- **0.8-1.0**: Excellent response

### Key Metrics
1. **Mean Score**: Overall performance
2. **Standard Deviation**: Consistency
3. **Score Range**: Best and worst performance
4. **Category Analysis**: Performance by task type

### Example Output
```
LLM-as-Judge Evaluation Summary
================================================================================

Model: friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0
Dataset: ultrachat
Samples: 50

Overall Performance:
  Mean Score: 0.723 ± 0.156
  Score Range: 0.400 - 0.900

Performance by Category:
  general_chat: 0.745 ± 0.142 (n=50)
```

## Advanced Usage

### 1. Custom Evaluation Prompts
```python
from open_r1.llm_judge_evaluator import LLMJudgeEvaluator

# Initialize evaluator
evaluator = LLMJudgeEvaluator(judge_model_name="gpt-4")

# Custom prompts
prompts = [
    "Explain quantum computing in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the ethical considerations of AI?"
]

# Generate completions with your model
completions = [...]  # Your model's outputs

# Evaluate
results = evaluator.evaluate_completions(
    prompts=prompts,
    completions=completions,
    model_name="My-GRPO-Model"
)
```

### 2. Model Comparison
```python
# Compare two models
comparison = evaluator.compare_models(
    prompts=prompts,
    completions_model1=model1_completions,
    completions_model2=model2_completions,
    model1_name="GRPO-Model",
    model2_name="Baseline-Model"
)

# Print summary
evaluator.print_summary(comparison)
```

### 3. Batch Evaluation
```bash
# Evaluate on multiple datasets
for dataset in ultrachat math custom; do
    python scripts/llm_judge_evaluation.py \
        --model-path "friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0" \
        --dataset "$dataset" \
        --num-samples 30 \
        --output-dir "batch_evaluation"
done
```

## Cost Considerations

### OpenAI API Costs
- **GPT-4**: ~$0.03 per 1K tokens (input + output)
- **GPT-3.5-turbo**: ~$0.002 per 1K tokens
- **Estimated cost for 50 samples**: $2-5 with GPT-4

### Cost Optimization
1. Use GPT-3.5-turbo for initial testing
2. Use smaller sample sizes for exploration
3. Cache results to avoid re-evaluation
4. Use batch processing for efficiency

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **Model Loading Errors**
   ```bash
   # Check model path
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0')"
   ```

3. **Memory Issues**
   ```bash
   # Use smaller batch sizes
   --num-samples 10
   ```

4. **Rate Limiting**
   ```bash
   # Add delays between requests
   # The script includes built-in rate limiting
   ```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Interpreting Results for Your Model

### Expected Performance
Based on your model's training configuration:
- **Strengths**: 
  - Structured reasoning (think/answer format)
  - General conversation capabilities
  - QRM-optimized responses
- **Areas for Improvement**:
  - Complex reasoning tasks
  - Creative writing
  - Technical explanations

### Comparison Benchmarks
- **Baseline Qwen2.5-1.5B**: ~0.6-0.7 mean score
- **Your GRPO Model**: Expected 0.7-0.8 mean score
- **State-of-the-art**: 0.8-0.9 mean score

## Next Steps

1. **Run Initial Evaluation**
   ```bash
   bash bash_scripts/quick_llm_judge.sh
   ```

2. **Analyze Results**
   - Check category-wise performance
   - Identify strengths and weaknesses
   - Compare with baseline models

3. **Iterate and Improve**
   - Adjust training parameters
   - Try different reward functions
   - Experiment with different datasets

4. **Scale Up**
   - Increase sample sizes
   - Add more evaluation datasets
   - Implement automated evaluation pipelines

## Files Created

- `src/open_r1/llm_judge_evaluator.py`: Core evaluation module
- `scripts/llm_judge_evaluation.py`: Comprehensive evaluation script
- `examples/llm_judge_example.py`: Simple usage example
- `bash_scripts/quick_llm_judge.sh`: Quick start script
- `LLM_JUDGE_GUIDE.md`: This guide

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example scripts
3. Examine the generated JSON results
4. Consult the evaluation guide

This LLM-as-Judge implementation provides a robust framework for evaluating your GRPO model's performance across multiple dimensions and comparing it with baseline models. 