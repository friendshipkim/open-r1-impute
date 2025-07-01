#!/usr/bin/env python3
"""
Comprehensive LLM-as-Judge evaluation script for GRPO models.
Evaluates the trained model against baselines and provides detailed analysis.
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Import our LLM-as-Judge evaluator
from open_r1.llm_judge_evaluator import LLMJudgeEvaluator


class GRPOModelEvaluator:
    """Comprehensive evaluator for GRPO models using LLM-as-Judge."""
    
    def __init__(self, model_path: str, judge_model: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained GRPO model
            judge_model: Judge model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key
        """
        self.model_path = model_path
        self.judge_model = judge_model
        
        # Load model and tokenizer
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Initialize LLM-as-Judge evaluator
        self.llm_judge = LLMJudgeEvaluator(judge_model_name=judge_model, api_key=api_key)
        
        # Set up system prompt for chat format
        self.system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"
    
    def format_prompt(self, user_message: str) -> str:
        """Format prompt for the model."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt
    
    def generate_completion(self, prompt: str, max_new_tokens: int = 1024) -> str:
        """Generate completion for a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        completion = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return completion
    
    def load_evaluation_dataset(self, dataset_name: str = "ultrachat", num_samples: int = 50) -> List[Dict[str, str]]:
        """Load evaluation dataset."""
        print(f"Loading evaluation dataset: {dataset_name}")
        
        if dataset_name == "ultrachat":
            # Load UltraChat dataset
            dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            evaluation_data = []
            for example in dataset:
                # Extract the first user message as the prompt
                messages = example["messages"]
                if len(messages) >= 2 and messages[0]["role"] == "user":
                    user_message = messages[0]["content"]
                    evaluation_data.append({
                        "prompt": user_message,
                        "category": "general_chat"
                    })
            
            return evaluation_data
        
        elif dataset_name == "math":
            # Load math problems
            dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
            dataset = dataset.select(range(min(num_samples, len(dataset))))
            
            evaluation_data = []
            for example in dataset:
                evaluation_data.append({
                    "prompt": f"Solve this math problem: {example['problem']}",
                    "category": "math",
                    "solution": example["solution"]
                })
            
            return evaluation_data
        
        elif dataset_name == "custom":
            # Custom evaluation prompts
            custom_prompts = [
                "Explain quantum computing in simple terms.",
                "Write a short story about a robot learning to paint.",
                "What are the main differences between machine learning and deep learning?",
                "Design a simple algorithm to find the largest number in a list.",
                "Explain the concept of climate change to a 10-year-old.",
                "Write a poem about artificial intelligence.",
                "What are the ethical considerations of autonomous vehicles?",
                "Explain how photosynthesis works.",
                "Write a recipe for chocolate chip cookies.",
                "What is the difference between a virus and a bacterium?"
            ]
            
            evaluation_data = []
            for i, prompt in enumerate(custom_prompts):
                evaluation_data.append({
                    "prompt": prompt,
                    "category": f"custom_{i//3}"  # Group into categories
                })
            
            return evaluation_data
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def evaluate_model(self, dataset_name: str = "ultrachat", num_samples: int = 50) -> Dict[str, Any]:
        """Evaluate the model using LLM-as-Judge."""
        print(f"Evaluating model on {dataset_name} dataset...")
        
        # Load evaluation data
        evaluation_data = self.load_evaluation_dataset(dataset_name, num_samples)
        
        # Generate completions
        prompts = []
        completions = []
        categories = []
        
        print("Generating completions...")
        for item in tqdm(evaluation_data, desc="Generating completions"):
            prompt = self.format_prompt(item["prompt"])
            completion = self.generate_completion(prompt)
            
            prompts.append(item["prompt"])  # Original user message
            completions.append(completion)
            categories.append(item.get("category", "unknown"))
        
        # Evaluate using LLM-as-Judge
        print("Running LLM-as-Judge evaluation...")
        results = self.llm_judge.evaluate_completions(
            prompts=prompts,
            completions=completions,
            model_name=f"GRPO-{Path(self.model_path).name}"
        )
        
        # Analyze results by category
        category_analysis = self.analyze_by_category(results, categories)
        
        return {
            "model_path": self.model_path,
            "dataset": dataset_name,
            "num_samples": num_samples,
            "results": results,
            "category_analysis": category_analysis,
            "summary": {
                "mean_score": np.mean([r.score for r in results]),
                "std_score": np.std([r.score for r in results]),
                "min_score": min([r.score for r in results]),
                "max_score": max([r.score for r in results])
            }
        }
    
    def analyze_by_category(self, results: List, categories: List[str]) -> Dict[str, Any]:
        """Analyze results by category."""
        category_scores = {}
        
        for result, category in zip(results, categories):
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(result.score)
        
        analysis = {}
        for category, scores in category_scores.items():
            analysis[category] = {
                "count": len(scores),
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "min_score": min(scores),
                "max_score": max(scores)
            }
        
        return analysis
    
    def compare_with_baseline(self, baseline_model_path: str, dataset_name: str = "ultrachat", num_samples: int = 30) -> Dict[str, Any]:
        """Compare with a baseline model."""
        print(f"Comparing with baseline model: {baseline_model_path}")
        
        # Load baseline model
        baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_path, trust_remote_code=True)
        baseline_model = AutoModelForCausalLM.from_pretrained(
            baseline_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load evaluation data
        evaluation_data = self.load_evaluation_dataset(dataset_name, num_samples)
        
        # Generate completions for both models
        prompts = []
        grpo_completions = []
        baseline_completions = []
        
        print("Generating completions for both models...")
        for item in tqdm(evaluation_data, desc="Generating completions"):
            # GRPO model
            grpo_prompt = self.format_prompt(item["prompt"])
            grpo_completion = self.generate_completion(grpo_prompt)
            
            # Baseline model
            baseline_prompt = baseline_tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                tokenize=False,
                add_generation_prompt=True
            )
            baseline_inputs = baseline_tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=2048)
            baseline_inputs = {k: v.to(baseline_model.device) for k, v in baseline_inputs.items()}
            
            with torch.no_grad():
                baseline_outputs = baseline_model.generate(
                    **baseline_inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=baseline_tokenizer.eos_token_id
                )
            
            baseline_completion = baseline_tokenizer.decode(
                baseline_outputs[0][baseline_inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            prompts.append(item["prompt"])
            grpo_completions.append(grpo_completion)
            baseline_completions.append(baseline_completion)
        
        # Compare using LLM-as-Judge
        comparison_results = self.llm_judge.compare_models(
            prompts=prompts,
            completions_model1=grpo_completions,
            completions_model2=baseline_completions,
            model1_name=f"GRPO-{Path(self.model_path).name}",
            model2_name=f"Baseline-{Path(baseline_model_path).name}"
        )
        
        return comparison_results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results."""
        # Convert dataclasses to dictionaries for JSON serialization
        def convert_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return obj
        
        # Convert results to JSON-serializable format
        json_results = json.loads(json.dumps(results, default=convert_to_dict))
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("LLM-as-Judge Evaluation Summary")
        print("="*80)
        
        print(f"\nModel: {results['model_path']}")
        print(f"Dataset: {results['dataset']}")
        print(f"Samples: {results['num_samples']}")
        
        summary = results['summary']
        print(f"\nOverall Performance:")
        print(f"  Mean Score: {summary['mean_score']:.3f} ± {summary['std_score']:.3f}")
        print(f"  Score Range: {summary['min_score']:.3f} - {summary['max_score']:.3f}")
        
        if 'category_analysis' in results:
            print(f"\nPerformance by Category:")
            for category, stats in results['category_analysis'].items():
                print(f"  {category}: {stats['mean_score']:.3f} ± {stats['std_score']:.3f} (n={stats['count']})")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation for GRPO models")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the trained GRPO model")
    parser.add_argument("--baseline-model", type=str, default=None,
                       help="Path to baseline model for comparison")
    parser.add_argument("--dataset", type=str, default="ultrachat",
                       choices=["ultrachat", "math", "custom"],
                       help="Evaluation dataset")
    parser.add_argument("--num-samples", type=int, default=50,
                       help="Number of samples to evaluate")
    parser.add_argument("--judge-model", type=str, default="gpt-4",
                       help="Judge model to use")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI API key")
    
    args = parser.parse_args()
    
    # Set API key
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = GRPOModelEvaluator(
        model_path=args.model_path,
        judge_model=args.judge_model,
        api_key=args.api_key
    )
    
    # Run evaluation
    print(f"Starting LLM-as-Judge evaluation...")
    results = evaluator.evaluate_model(
        dataset_name=args.dataset,
        num_samples=args.num_samples
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    output_file = os.path.join(args.output_dir, f"llm_judge_results_{args.dataset}.json")
    evaluator.save_results(results, output_file)
    
    # Compare with baseline if provided
    if args.baseline_model:
        print(f"\nComparing with baseline model...")
        comparison_results = evaluator.compare_with_baseline(
            baseline_model_path=args.baseline_model,
            dataset_name=args.dataset,
            num_samples=min(args.num_samples, 30)  # Use fewer samples for comparison
        )
        
        # Print comparison summary
        evaluator.llm_judge.print_summary(comparison_results)
        
        # Save comparison results
        comparison_file = os.path.join(args.output_dir, f"comparison_results_{args.dataset}.json")
        evaluator.llm_judge.save_results(comparison_results, comparison_file)
    
    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 