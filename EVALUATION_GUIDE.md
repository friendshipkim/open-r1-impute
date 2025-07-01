# GRPO Model Evaluation Guide

This guide explains how to evaluate and compare the two different GRPO training approaches in this project:

1. **GRPO Oracle**: Uses only the reward model for all training steps
2. **GRPO Impute**: Uses reward model initially, then switches to a mixture of trained regression model and reward model

## Understanding the Two Approaches

### GRPO Oracle (Pure Reward Model)
- **Configuration**: `recipes/Qwen2.5-1.5B-Instruct/grpo/config_chat_oracle.yaml`
- **Script**: `bash_scripts/grpo_oracle.sh`
- **Behavior**: Always uses the true reward model for computing rewards
- **Key Parameters**:
  ```yaml
  start_patch: -1  # No reward imputation
  start_pre_patch: -1  # No pre-patch training
  rho: -1  # No correlation threshold
  ```

### GRPO Impute (Mixed Reward Model)
- **Configuration**: `recipes/Qwen2.5-1.5B-Instruct/grpo/config_chat.yaml`
- **Script**: `bash_scripts/grpo_impute.sh`
- **Behavior**: 
  - Steps 0-60 (20%): Uses true reward model only
  - Steps 60-120 (20-40%): Uses true reward model + trains pre-patch regression model
  - Steps 120+ (40%+): Uses mixture of regression model and reward model based on correlation
- **Key Parameters**:
  ```yaml
  start_patch: 0.4  # Start imputation at 40% of max_steps
  start_pre_patch: 0.2  # Start pre-patch at 20% of max_steps
  rho: 0  # Correlation threshold for using imputed rewards
  ```

## Where Rewards Are Modified

The rewards fed into the gradient descent objective function are computed in:

### Primary Location: `src/open_r1/grpo_trainer.py`

**Lines 1020-1080** in the `_generate_and_score_completions` method:

```python
# Get true rewards from reward model
outputs = reward_func(**reward_inputs)
true_rewards = outputs.logits[:, 0]  # ← True rewards computed here
rewards_per_func[:, i] = true_rewards  # ← Used for training

# Imputation logic
if self.state.global_step >= self.start_patch and self.start_patch > 0:
    imputed_rewards = self.rhat_model.impute(...)
    if corr_coef > self.rho:
        rewards_per_func[:, i] = imputed_rewards  # ← Override with imputed rewards
```

**Lines 1100-1110** where rewards are aggregated:

```python
# Apply weights and sum rewards
rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

# Compute advantages for gradient descent
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
```

### Reward Functions: `src/open_r1/rewards.py`

Custom reward functions that can be modified:
- `accuracy_reward()`: Checks if completion matches ground truth
- `format_reward()`: Checks for proper formatting
- `qrm_reward()`: Quality Reward Model reward
- `llm_as_judge_reward()`: LLM-as-Judge evaluation (newly added)

## Training the Models

### 1. Train GRPO Oracle Model
```bash
cd /root/open-r1-impute
bash bash_scripts/grpo_oracle.sh
```

### 2. Train GRPO Impute Model
```bash
cd /root/open-r1-impute
bash bash_scripts/grpo_impute.sh
```

## Evaluating the Models

### Method 1: Using Built-in Benchmarks

The project includes built-in evaluation tasks in `src/open_r1/evaluate.py`:
- `math_500`: Mathematical reasoning
- `gpqa:diamond`: Physics questions
- `aime24`, `aime25`: Advanced math competitions

### Method 2: Using the Evaluation Script

```bash
# Make the script executable
chmod +x bash_scripts/evaluate_models.sh

# Run evaluation
bash bash_scripts/evaluate_models.sh \
    --oracle-model /path/to/oracle/model \
    --impute-model /path/to/impute/model \
    --output-dir evaluation_results \
    --tasks math_500,gpqa:diamond
```

### Method 3: LLM-as-Judge Evaluation

For more nuanced evaluation, you can use the LLM-as-Judge approach:

```python
from open_r1.llm_judge_evaluator import LLMJudgeEvaluator

# Initialize evaluator
evaluator = LLMJudgeEvaluator(judge_model_name="gpt-4")

# Compare models
results = evaluator.compare_models(
    prompts=prompts,
    completions_model1=oracle_completions,
    completions_model2=impute_completions,
    model1_name="GRPO Oracle",
    model2_name="GRPO Impute"
)

# Print summary
evaluator.print_summary(results)

# Save results
evaluator.save_results(results, "llm_judge_results.json")
```

## Understanding the Results

### Key Metrics to Compare

1. **Accuracy on Benchmarks**: Standard accuracy on math and physics tasks
2. **Win Rate**: How often one model outperforms the other
3. **Score Distribution**: Mean and standard deviation of scores
4. **Correlation with True Rewards**: How well imputed rewards correlate with true rewards

### Expected Outcomes

**GRPO Oracle**:
- ✅ Always uses true reward model
- ✅ More consistent training signal
- ❌ Slower training (reward model inference)
- ❌ Higher computational cost

**GRPO Impute**:
- ✅ Faster training after initial phase
- ✅ Lower computational cost
- ❌ Potential quality degradation if regression model is poor
- ❌ Depends on correlation threshold

## Configuration Parameters

### Key Parameters in `configs.py`

```python
@dataclass
class GRPOConfig:
    reward_record_window: int = 100  # Steps to record reward outputs
    start_patch: Optional[float] = -1  # When to start imputation
    start_pre_patch: Optional[float] = -1  # When to start pre-patch
    rho: Optional[float] = -1  # Correlation threshold
```

### Training Configuration

```yaml
# Oracle config
start_patch: -1
start_pre_patch: -1
rho: -1

# Impute config  
start_patch: 0.4
start_pre_patch: 0.2
rho: 0
```

## Troubleshooting

### Common Issues

1. **Model Paths**: Ensure model paths exist before evaluation
2. **Dependencies**: Install required packages:
   ```bash
   pip install openai numpy tqdm
   ```
3. **Memory**: Use smaller batch sizes if running out of GPU memory
4. **API Keys**: Set OpenAI API key for LLM-as-Judge evaluation:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

### Debugging Reward Computation

To debug reward computation, add logging in `grpo_trainer.py`:

```python
# Around line 1030
print(f"True rewards: {true_rewards}")
print(f"Imputed rewards: {imputed_rewards}")
print(f"Correlation: {corr_coef}")
print(f"Using imputed: {corr_coef > self.rho}")
```

## Advanced Customization

### Adding New Reward Functions

1. Add function to `src/open_r1/rewards.py`
2. Register in `src/open_r1/grpo.py` in `REWARD_FUNCS_REGISTRY`
3. Update configuration file

### Modifying Imputation Logic

The imputation logic is in `src/open_r1/impute_utils.py`:
- `RewardImputation`: Linear regression for reward prediction
- `CorrImputation`: Lasso regression for correlation prediction

### Custom Evaluation Tasks

Add new tasks to `src/open_r1/evaluate.py` following the existing pattern.

## Conclusion

This evaluation framework allows you to systematically compare the two GRPO approaches and understand their trade-offs. The key is to look at both quantitative metrics (accuracy, win rates) and qualitative aspects (completion quality, training efficiency) to make informed decisions about which approach works better for your specific use case. 