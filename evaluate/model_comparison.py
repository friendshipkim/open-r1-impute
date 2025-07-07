#!/usr/bin/env python3
"""
Model comparison script using LLM-as-Judge evaluation.
Compares trained GRPO model with untrained baseline model.
"""

import os
import json
import argparse
import random
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Import VLLM for fast inference
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: VLLM not available. Falling back to standard transformers inference.")
    VLLM_AVAILABLE = False

# Import our LLM-as-Judge evaluator
import sys
sys.path.append('/root/open-r1-impute/src')
from open_r1.llm_judge_evaluator import LLMJudgeEvaluator, ComparisonResult


class ModelComparator:
    """Comprehensive model comparator using LLM-as-Judge."""
    
    def __init__(self, 
                 model1_path: str, 
                 model2_path: str, 
                 judge_model: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 use_vllm: bool = True,
                 seed: int = 42):
        """
        Initialize the model comparator.
        
        Args:
            model1_path: Path to the first model (trained GRPO model)
            model2_path: Path to the second model (untrained baseline)
            judge_model: Judge model to use (auto-detected if None based on API key)
            api_key: OpenAI or Anthropic API key
            use_vllm: Whether to use VLLM for faster inference
            seed: Random seed for reproducibility
        """
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.api_key = api_key
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.seed = seed
        
        # Auto-detect API type and set appropriate judge model
        if judge_model is None and api_key:
            if api_key.startswith("sk-ant-"):
                self.judge_model = "claude-3-sonnet-20240229"
                print(f"Detected Anthropic API key, using judge model: {self.judge_model}")
            else:
                self.judge_model = "gpt-4"
                print(f"Detected OpenAI API key, using judge model: {self.judge_model}")
        else:
            self.judge_model = judge_model or "gpt-4"
        
        self.model1_name = "Model1"
        self.model2_name = "Model2"
        
        # Set the random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if self.use_vllm:
            print("Using VLLM for fast inference...")
            # Initialize VLLM models
            self.vllm_model1 = LLM(
                model=model1_path,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=8192,
            )
            self.vllm_model2 = LLM(
                model=model2_path,
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=8192,
            )
            # Set sampling parameters
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
            )
        else:
            print("Using standard transformers inference...")
            # Load both models with transformers
            print(f"Loading model 1 (trained): {model1_path}")
            self.tokenizer1 = AutoTokenizer.from_pretrained(model1_path, trust_remote_code=True)
            self.model1 = AutoModelForCausalLM.from_pretrained(
                model1_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"Loading model 2 (baseline): {model2_path}")
            self.tokenizer2 = AutoTokenizer.from_pretrained(model2_path, trust_remote_code=True)
            self.model2 = AutoModelForCausalLM.from_pretrained(
                model2_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Initialize LLM-as-Judge evaluator
        self.llm_judge = LLMJudgeEvaluator(judge_model_name=self.judge_model, api_key=api_key)
        
        # Set up system prompt for chat format
        self.system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
    
    def format_prompt(self, user_message: str, tokenizer) -> str:
        """Format prompt for a specific model."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        if tokenizer is None:
            # For VLLM, use a simple format
            return f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Apply chat template for transformers
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return prompt
    
    def generate_completion(self, prompt: str, model, tokenizer, max_new_tokens: int = 1024) -> str:
        """Generate completion for a given prompt using specified model."""
        if self.use_vllm:
            # Use VLLM for fast inference
            outputs = model.generate([prompt], self.sampling_params)
            completion = outputs[0].outputs[0].text
        else:
            # Use standard transformers inference
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return completion
    
    def generate_completions_batch(self, prompts: List[str], model_num: int = 1, batch_size: int = 8) -> List[str]:
        """Generate completions for a batch of prompts using VLLM."""
        if not self.use_vllm:
            raise ValueError("Batch generation only available with VLLM")
        
        model = self.vllm_model1 if model_num == 1 else self.vllm_model2
        completions = []
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating with Model {model_num}"):
            batch_prompts = prompts[i:i + batch_size]
            outputs = model.generate(batch_prompts, self.sampling_params)
            
            for output in outputs:
                completion = output.outputs[0].text
                completions.append(completion)
        
        return completions
    
    def load_validation_prompts(self, dataset_name: str = "ultrachat", num_samples: int = 50, use_validation_split: bool = True) -> List[str]:
        """Load validation prompts from the chat dataset that weren't used in training."""
        print(f"Loading {num_samples} validation prompts from {dataset_name} dataset...")
        
        if dataset_name == "ultrachat":
            # Load UltraChat dataset
            dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
            
            # Choose the appropriate split to avoid training data
            if use_validation_split and 'validation' in dataset:
                split_name = 'validation'
                print("Using validation split to avoid training data overlap")
            elif 'test' in dataset:
                split_name = 'test'
                print("Using test split to avoid training data overlap")
            else:
                # If no validation/test split, use a portion of train split
                split_name = 'train'
                print("Warning: No validation/test split found. Using portion of train split.")
            
            # Get the appropriate split
            if split_name == 'train':
                # Use the last portion of train data to avoid overlap with training
                dataset_split = dataset[split_name]['messages'][-2000:]  # Take last 2000 examples
            else:
                dataset_split = dataset[split_name]['messages']
            
            # Randomly sample prompts
            random.seed(self.seed)
            indices = random.sample(range(len(dataset_split)), min(num_samples, len(dataset_split)))
            dataset_subset = [dataset_split[i] for i in indices]
            
            prompts = []
            for example in dataset_subset:
                # Extract the first user message as the prompt
                if len(example) >= 1 and example[0]["role"] == "user":
                    user_message = example[0]["content"]
                    prompts.append(user_message)
            
            print(f"Loaded {len(prompts)} validation prompts from {split_name} split")
            return prompts
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def load_grpo_validation_prompts(self, num_samples: int = 100) -> List[str]:
        """Load validation prompts specifically for GRPO model comparison."""
        print(f"Loading {num_samples} validation prompts for GRPO model comparison...")
        
        # Load UltraChat dataset
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
        
        # Use test_sft split specifically for GRPO models (since they were trained on SFT data)
        split_name = 'test_sft'
        print(f"Using {split_name} split to ensure no training data overlap")
        
        # Get prompts from the test_sft split
        prompts = dataset[split_name]['messages']
        
        # Extract user messages (prompts) - take the first user message from each conversation
        validation_prompts = []
        for example in prompts:
            # Find the first user message
            for message in example:
                if message['role'] == 'user':
                    validation_prompts.append(message['content'])
                    break
        
        # Shuffle and take the specified number
        random.seed(self.seed)
        random.shuffle(validation_prompts)
        validation_prompts = validation_prompts[:num_samples]
        
        print(f"Loaded {len(validation_prompts)} validation prompts from {split_name} split")
        return validation_prompts
    
    def compare_models_on_prompts(self, prompts: List[str], model1_name: str = "oracle_model", model2_name: str = "imputed_model") -> List[ComparisonResult]:
        """Compare both models on the given prompts."""
        print("Generating completions and comparing models...")
        
        results = []
        
        if self.use_vllm:
            # Use VLLM batch generation for faster inference
            print("Using VLLM batch generation for faster inference...")
            
            # Format all prompts
            formatted_prompts1 = [self.format_prompt(prompt, None) for prompt in prompts]
            formatted_prompts2 = [self.format_prompt(prompt, None) for prompt in prompts]
            
            # Generate completions in batches
            completions1 = self.generate_completions_batch(formatted_prompts1, model_num=1)
            completions2 = self.generate_completions_batch(formatted_prompts2, model_num=2)
            
            # Compare completions
            for i, (prompt, completion1, completion2) in enumerate(tqdm(zip(prompts, completions1, completions2), 
                                                                       total=len(prompts), desc="Comparing models")):
                try:
                    # Use LLM-as-Judge to compare the completions
                    comparison = self.llm_judge.compare_completions(
                        prompt=prompt,
                        completion1=completion1,
                        completion2=completion2,
                        model1_name=model1_name,
                        model2_name=model2_name
                    )
                    
                    # Create comparison result
                    result = ComparisonResult(
                        prompt=prompt,
                        completion1=completion1,
                        completion2=completion2,
                        winner=comparison.winner,
                        explanation=comparison.explanation,
                        model1_name=model1_name,
                        model2_name=model2_name
                    )
                    
                    results.append(result)
                    
                    # Print progress
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{len(prompts)} prompts")
                    
                except Exception as e:
                    print(f"Error processing prompt {i}: {e}")
                    continue
        else:
            # Use standard sequential generation
            for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Comparing models"):
                try:
                    # Generate completions from both models
                    prompt1 = self.format_prompt(prompt, self.tokenizer1)
                    completion1 = self.generate_completion(prompt1, self.model1, self.tokenizer1)
                    
                    prompt2 = self.format_prompt(prompt, self.tokenizer2)
                    completion2 = self.generate_completion(prompt2, self.model2, self.tokenizer2)
                    
                    # Use LLM-as-Judge to compare the completions
                    comparison = self.llm_judge.compare_completions(
                        prompt=prompt,
                        completion1=completion1,
                        completion2=completion2,
                        model1_name=model1_name,
                        model2_name=model2_name
                    )
                    
                    # Create comparison result
                    result = ComparisonResult(
                        prompt=prompt,
                        completion1=completion1,
                        completion2=completion2,
                        winner=comparison.winner,
                        explanation=comparison.explanation,
                        model1_name=model1_name,
                        model2_name=model2_name
                    )
                    
                    results.append(result)
                    
                    # Print progress
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{len(prompts)} prompts")
                    
                except Exception as e:
                    print(f"Error processing prompt {i}: {e}")
                    continue
        
        return results
    
    def analyze_results(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze the comparison results."""
        if not results:
            return {}
        
        # Count wins
        model1_wins = sum(1 for r in results if r.winner == "model1")
        model2_wins = sum(1 for r in results if r.winner == "model2")
        ties = sum(1 for r in results if r.winner == "tie")
        
        analysis = {
            "total_comparisons": len(results),
            "model1_wins": model1_wins,
            "model2_wins": model2_wins,
            "ties": ties,
            "model1_win_rate": model1_wins / len(results),
            "model2_win_rate": model2_wins / len(results),
            "tie_rate": ties / len(results)
        }
        
        return analysis
    
    def print_summary(self, results: List[ComparisonResult], analysis: Dict[str, Any]):
        """Print comparison summary."""
        print("\n" + "="*80)
        print("Model Comparison Summary")
        print("="*80)
        
        # Get model names from the first result if available
        if results:
            model1_name = results[0].model1_name
            model2_name = results[0].model2_name
        else:
            model1_name = "Model 1"
            model2_name = "Model 2"
        
        print(f"\nModel 1 ({model1_name}): {self.model1_path}")
        print(f"Model 2 ({model2_name}): {self.model2_path}")
        print(f"Judge Model: {self.judge_model}")
        print(f"Total Comparisons: {analysis['total_comparisons']}")
        
        print(f"\nWin Rates:")
        print(f"  {model1_name}: {analysis['model1_win_rate']:.1%} ({analysis['model1_wins']} wins)")
        print(f"  {model2_name}: {analysis['model2_win_rate']:.1%} ({analysis['model2_wins']} wins)")
        print(f"  Ties: {analysis['tie_rate']:.1%} ({analysis['ties']} ties)")
        
        # Determine overall winner
        if analysis['model1_win_rate'] > analysis['model2_win_rate']:
            print(f"\n🏆 Overall Winner: {model1_name}")
        elif analysis['model2_win_rate'] > analysis['model1_win_rate']:
            print(f"\n🏆 Overall Winner: {model2_name}")
        else:
            print(f"\n🤝 Overall Result: Tie")
        
        print("="*80)
    
    def save_results(self, results: List[ComparisonResult], analysis: Dict[str, Any], output_file: str):
        """Save comparison results."""
        # Convert dataclasses to dictionaries
        results_dict = []
        for result in results:
            results_dict.append({
                "prompt": result.prompt,
                "model1_completion": result.completion1,
                "model2_completion": result.completion2,
                "winner": result.winner,
                "explanation": result.explanation
            })
        
        output_data = {
            "model1_path": self.model1_path,
            "model2_path": self.model2_path,
            "judge_model": self.judge_model,
            "api_type": getattr(self.llm_judge, 'api_type', 'unknown'),
            "seed": self.seed,
            "analysis": analysis,
            "results": results_dict
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def print_sample_comparisons(self, results: List[ComparisonResult], num_samples: int = 5):
        """Print sample comparisons for inspection."""
        print(f"\nSample Comparisons (showing first {num_samples}):")
        print("-" * 80)
        
        for i, result in enumerate(results[:num_samples]):
            print(f"\nComparison {i+1}:")
            print(f"Prompt: {result.prompt[:100]}...")
            print(f"Winner: {result.winner}")
            print(f"Explanation: {result.explanation}")
            print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Model comparison using LLM-as-Judge")
    parser.add_argument("--model1", type=str, 
                       default="friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0",
                       help="Path to first model")
    parser.add_argument("--model2", type=str, 
                       default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Path to second model")
    parser.add_argument("--dataset", type=str, default="ultrachat",
                       help="Dataset to use for validation")
    parser.add_argument("--num-samples", type=int, default=50,
                       help="Number of validation samples")
    parser.add_argument("--judge-model", type=str, default=None,
                       help="Judge model to use (auto-detected from API key if not specified)")
    parser.add_argument("--output-dir", type=str, default="comparison_results",
                       help="Output directory for results")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI or Anthropic API key")
    parser.add_argument("--show-samples", type=int, default=5,
                       help="Number of sample comparisons to show")
    parser.add_argument("--grpo-comparison", action="store_true",
                       help="Run GRPO model comparison with validation prompts")
    parser.add_argument("--evaluation-method", choices=["local", "gpt4"], default="local",
                       help="Evaluation method for GRPO comparison")
    parser.add_argument("--use-vllm", action="store_true", default=True,
                       help="Use VLLM for faster inference (default: True)")
    parser.add_argument("--no-vllm", action="store_true",
                       help="Disable VLLM and use standard transformers inference")
    parser.add_argument("--num-runs", type=int, default=1,
                       help="Number of times to run the experiment (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set API key
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Determine VLLM usage
    use_vllm = args.use_vllm and not args.no_vllm
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize comparator
    print("Initializing model comparator...")
    comparator = ModelComparator(
        model1_path=args.model1,
        model2_path=args.model2,
        judge_model=args.judge_model,
        api_key=args.api_key,
        use_vllm=use_vllm,
        seed=args.seed
    )
    
    # Load validation prompts based on mode (ONCE - same prompts for all runs)
    if args.grpo_comparison:
        print("Running GRPO model comparison with validation prompts...")
        prompts = comparator.load_grpo_validation_prompts(num_samples=args.num_samples)
        model1_name = "oracle_model"
        model2_name = "imputed_model"
    else:
        prompts = comparator.load_validation_prompts(
            dataset_name=args.dataset,
            num_samples=args.num_samples
        )
        model1_name = "Trained-GRPO"
        model2_name = "Baseline-Qwen"
    
    print(f"Loaded {len(prompts)} validation prompts (will be used for all {args.num_runs} runs)")
    
    # Run multiple experiments
    for run_num in range(1, args.num_runs + 1):
        print(f"\n{'='*80}")
        print(f"RUN {run_num}/{args.num_runs}")
        print(f"{'='*80}")
        
        # Run comparison
        print("Starting model comparison...")
        results = comparator.compare_models_on_prompts(prompts, model1_name, model2_name)
        
        # Analyze results
        analysis = comparator.analyze_results(results)
        
        # Print summary
        comparator.print_summary(results, analysis)
        
        # Print sample comparisons (only for first run to avoid spam)
        if run_num == 1:
            comparator.print_sample_comparisons(results, args.show_samples)
        
        # Save results with run number
        if args.grpo_comparison:
            # Check if we're comparing oracle vs impute models
            if "oracle" in args.model1 and "imputed" in args.model2 or "oracle" in args.model1 and "ps0.2-preps0.0-rho0" in args.model2:
                # Extract parameters from model paths
                # Oracle model: Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-oracle-lr2e-6-warmup0.05
                # Imputed model: Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0
                
                oracle_params = "lr2e-6-warmup0.05"
                imputed_params = "lr2e-6-warmup0.05-ps0.2-preps0.0-rho0"
                
                # Check if using Claude API and append to filename
                api_suffix = "_claude" if args.api_key and args.api_key.startswith("sk-ant-") else ""
                
                if args.num_runs > 1:
                    output_file = os.path.join(args.output_dir, f"oracle_{oracle_params}_vs_imputed_{imputed_params}_comparison_{args.num_samples}samples_seed{args.seed}{api_suffix}_loop{run_num}.json")
                else:
                    output_file = os.path.join(args.output_dir, f"oracle_{oracle_params}_vs_imputed_{imputed_params}_comparison_{args.num_samples}samples_seed{args.seed}{api_suffix}.json")
            else:
                # Check if using Claude API and append to filename
                api_suffix = "_claude" if args.api_key and args.api_key.startswith("sk-ant-") else ""
                
                if args.num_runs > 1:
                    output_file = os.path.join(args.output_dir, f"grpo_model_comparison_{args.evaluation_method}_{args.num_samples}samples_seed{args.seed}{api_suffix}_loop{run_num}.json")
                else:
                    output_file = os.path.join(args.output_dir, f"grpo_model_comparison_{args.evaluation_method}_{args.num_samples}samples_seed{args.seed}{api_suffix}.json")
        else:
            # Check if using Claude API and append to filename
            api_suffix = "_claude" if args.api_key and args.api_key.startswith("sk-ant-") else ""
            
            if args.num_runs > 1:
                output_file = os.path.join(args.output_dir, f"model_comparison_{args.dataset}_{args.num_samples}samples_seed{args.seed}{api_suffix}_loop{run_num}.json")
            else:
                output_file = os.path.join(args.output_dir, f"model_comparison_{args.dataset}_{args.num_samples}samples_seed{args.seed}{api_suffix}.json")
        
        comparator.save_results(results, analysis, output_file)
        
        print(f"Run {run_num} complete! Results saved to: {output_file}")
        
        # Add a small delay between runs to avoid rate limiting
        if run_num < args.num_runs:
            print("Waiting 2 seconds before next run...")
            time.sleep(2)
    
    print(f"\n{'='*80}")
    print(f"ALL {args.num_runs} RUNS COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main() 