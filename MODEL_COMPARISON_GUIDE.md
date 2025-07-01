# Model Comparison Guide: Trained GRPO vs Baseline

This guide provides a comprehensive approach to comparing your trained GRPO model at [friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0](https://huggingface.co/friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0) with the untrained baseline model using LLM-as-Judge evaluation.

## Overview

This comparison will:
1. Load both the trained GRPO model and the baseline Qwen2.5-1.5B-Instruct model
2. Take 50 validation prompts from the UltraChat dataset
3. Generate completions from both models
4. Use GPT-4 as a judge to determine which output is better
5. Provide detailed analysis and statistics

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

### Method 1: Quick Comparison Script
```bash
# Run the quick start script
bash bash_scripts/quick_model_comparison.sh
```

This will:
- Load both models
- Compare them on 50 UltraChat validation prompts
- Use GPT-4 as judge
- Save results to `model_comparison_results/`

### Method 2: Custom Comparison
```bash
# Run custom comparison
python scripts/model_comparison.py \
    --model1 "friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0" \
    --model2 "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset ultrachat \
    --num-samples 50 \
    --judge-model gpt-4 \
    --output-dir comparison_results \
    --show-samples 5
```

### Method 3: Simple Example
```bash
# Run the example script
python examples/model_comparison_example.py
```

## Understanding the Models

### Trained GRPO Model
- **Model**: `friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0`
- **Base**: Qwen2.5-1.5B-Instruct
- **Training Method**: GRPO with reward imputation
- **Training Data**: UltraChat 200k
- **Reward Model**: QRM (Quality Reward Model)
- **Key Features**:
  - Structured reasoning format (`<think>` and `<answer>` tags)
  - Optimized for quality responses
  - Trained with imputation strategy

### Baseline Model
- **Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Type**: Untrained baseline
- **Key Features**:
  - Standard instruction-tuned model
  - No GRPO training
  - No reward optimization

## How the Comparison Works

### 1. Prompt Selection
- Randomly samples 50 prompts from UltraChat test set
- Ensures diverse evaluation across different conversation types
- Uses reproducible random seed for consistency

### 2. Generation
- Both models generate completions for the same prompts
- Uses consistent generation parameters:
  - Temperature: 0.7
  - Top-p: 0.9
  - Max tokens: 1024

### 3. LLM-as-Judge Evaluation
- GPT-4 compares completions side-by-side
- Provides scores (0-10) for each response
- Determines winner (Model A, Model B, or Tie)
- Gives explanation for the judgment

### 4. Analysis
- Calculates win rates for each model
- Computes average scores and standard deviations
- Provides detailed statistics and insights

## Understanding the Results

### Key Metrics

1. **Win Rate**: Percentage of comparisons won by each model
2. **Average Score**: Mean quality score from GPT-4 judge
3. **Score Difference**: Average difference in scores (GRPO - Baseline)
4. **Consistency**: Standard deviation of scores

### Expected Outcomes

Based on the training configuration, you should expect:

- **Trained GRPO Model**:
  - Higher win rate (60-80%)
  - Better structured responses
  - More consistent quality
  - Better reasoning format

- **Baseline Model**:
  - Lower win rate (20-40%)
  - Less structured responses
  - More variable quality
  - Standard instruction-following

### Example Output
```
Model Comparison Summary
================================================================================

Model 1 (Trained GRPO): friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0
Model 2 (Baseline): Qwen/Qwen2.5-1.5B-Instruct
Judge Model: gpt-4
Total Comparisons: 50

Win Rates:
  Trained GRPO: 72.0% (36 wins)
  Baseline: 24.0% (12 wins)
  Ties: 4.0% (2 ties)

Average Scores:
  Trained GRPO: 0.745 ± 0.142
  Baseline: 0.623 ± 0.189

Score Analysis:
  Average Difference (GRPO - Baseline): 0.122 ± 0.156
  GRPO Better: 38 times
  Baseline Better: 10 times
  Equal: 2 times

🏆 Overall Winner: Trained GRPO Model
```

## Advanced Usage

### 1. Custom Dataset Comparison
```bash
# Compare on different datasets
python scripts/model_comparison.py \
    --model1 "friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0" \
    --model2 "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset custom \
    --num-samples 30
```

### 2. Different Judge Models
```bash
# Use GPT-3.5-turbo for cost efficiency
python scripts/model_comparison.py \
    --judge-model gpt-3.5-turbo \
    --num-samples 50
```

### 3. Batch Comparison
```bash
# Compare multiple models
for baseline in "Qwen/Qwen2.5-1.5B-Instruct" "microsoft/DialoGPT-medium"; do
    python scripts/model_comparison.py \
        --model1 "friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0" \
        --model2 "$baseline" \
        --output-dir "batch_comparison"
done
```

## Cost Considerations

### OpenAI API Costs
- **GPT-4**: ~$0.03 per 1K tokens (input + output)
- **GPT-3.5-turbo**: ~$0.002 per 1K tokens
- **Estimated cost for 50 samples**: $3-8 with GPT-4

### Cost Optimization
1. Use GPT-3.5-turbo for initial testing
2. Start with smaller sample sizes (10-20)
3. Use the same judge model for consistency
4. Cache results to avoid re-evaluation

## Interpreting Results for Your Model

### What to Look For

1. **Win Rate**: Should be >50% for trained model
2. **Score Improvement**: Should show positive score difference
3. **Consistency**: Lower standard deviation indicates more reliable quality
4. **Explanation Quality**: Check if judge explanations make sense

### Expected Performance

Based on your model's training:
- **Win Rate**: 60-80% expected
- **Score Improvement**: 0.1-0.2 points expected
- **Strengths**: Structured reasoning, consistent quality
- **Areas for Improvement**: Creative tasks, complex reasoning

### Red Flags

- Win rate <40% for trained model
- Negative score difference
- High variance in scores
- Inconsistent judge explanations

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Check model paths
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0')"
   ```

2. **Memory Issues**
   ```bash
   # Use smaller batch sizes or fewer samples
   --num-samples 20
   ```

3. **API Rate Limiting**
   ```bash
   # The script includes built-in rate limiting
   # Add delays if needed
   ```

4. **Inconsistent Results**
   ```bash
   # Use fixed random seed
   # Check judge model consistency
   # Verify prompt formatting
   ```

## Files Created

- `scripts/model_comparison.py`: Main comparison script
- `examples/model_comparison_example.py`: Simple usage example
- `bash_scripts/quick_model_comparison.sh`: Quick start script
- `MODEL_COMPARISON_GUIDE.md`: This guide

## Next Steps

1. **Run Initial Comparison**
   ```bash
   bash bash_scripts/quick_model_comparison.sh
   ```

2. **Analyze Results**
   - Check win rates and score differences
   - Review sample comparisons
   - Identify strengths and weaknesses

3. **Iterate and Improve**
   - Adjust training parameters based on results
   - Try different reward functions
   - Experiment with different datasets

4. **Scale Up**
   - Increase sample sizes
   - Add more evaluation datasets
   - Compare with other baseline models

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example scripts
3. Examine the generated JSON results
4. Consult the comparison guide

This model comparison framework provides a robust way to evaluate the effectiveness of your GRPO training and understand how it improves upon the baseline model. 