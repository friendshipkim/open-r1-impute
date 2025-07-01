#!/usr/bin/env python3
"""
Script to evaluate and compare GRPO Oracle vs GRPO Impute models.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from open_r1.evaluate import TASKS_TABLE
from open_r1.utils import get_tokenizer


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from path."""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer


def generate_completions(model, tokenizer, prompts: List[str], max_new_tokens: int = 1024):
    """Generate completions for given prompts."""
    completions = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        completions.append(completion)
    
    return completions


def evaluate_on_benchmarks(model, tokenizer, tasks: List[str] = None):
    """Evaluate model on specified benchmarks."""
    if tasks is None:
        tasks = ["math_500", "gpqa:diamond"]
    
    results = {}
    
    for task_name in tasks:
        print(f"\nEvaluating on {task_name}...")
        
        # Find the task configuration
        task_config = None
        for task in TASKS_TABLE:
            if task.name == task_name:
                task_config = task
                break
        
        if task_config is None:
            print(f"Task {task_name} not found!")
            continue
        
        # Load dataset
        dataset = load_dataset(task_config.hf_repo, task_config.hf_subset, split=task_config.evaluation_splits[0])
        
        # Take a subset for faster evaluation
        dataset = dataset.select(range(min(100, len(dataset))))
        
        # Generate prompts and get completions
        prompts = []
        for example in dataset:
            doc = task_config.prompt_function(example, task_name)
            prompts.append(doc.query)
        
        completions = generate_completions(model, tokenizer, prompts)
        
        # Calculate metrics
        correct = 0
        total = len(dataset)
        
        for i, example in enumerate(dataset):
            doc = task_config.prompt_function(example, task_name)
            
            # Use the first metric for evaluation
            metric = task_config.metric[0]
            score = metric.compute(predictions=[completions[i]], references=[doc.choices[doc.gold_index]])
            
            if score > 0.5:  # Threshold for correctness
                correct += 1
        
        accuracy = correct / total
        results[task_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "completions": completions[:5]  # Save first 5 for inspection
        }
        
        print(f"  Accuracy: {accuracy:.3f} ({correct}/{total})")
    
    return results


def compare_models(model1_path: str, model2_path: str, output_dir: str = "evaluation_results"):
    """Compare two models on benchmarks."""
    print("="*60)
    print("Model Comparison: GRPO Oracle vs GRPO Impute")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    model1, tokenizer1 = load_model_and_tokenizer(model1_path)
    model2, tokenizer2 = load_model_and_tokenizer(model2_path)
    
    # Evaluate both models
    print("\nEvaluating GRPO Oracle model...")
    results1 = evaluate_on_benchmarks(model1, tokenizer1)
    
    print("\nEvaluating GRPO Impute model...")
    results2 = evaluate_on_benchmarks(model2, tokenizer2)
    
    # Compare results
    comparison = {}
    for task_name in results1.keys():
        if task_name in results2:
            acc1 = results1[task_name]["accuracy"]
            acc2 = results2[task_name]["accuracy"]
            
            comparison[task_name] = {
                "oracle_accuracy": acc1,
                "impute_accuracy": acc2,
                "difference": acc1 - acc2,
                "oracle_better": acc1 > acc2,
                "impute_better": acc2 > acc1
            }
    
    # Save results
    all_results = {
        "oracle_results": results1,
        "impute_results": results2,
        "comparison": comparison
    }
    
    with open(os.path.join(output_dir, "model_comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    
    for task_name, comp in comparison.items():
        print(f"\n{task_name}:")
        print(f"  Oracle Accuracy: {comp['oracle_accuracy']:.3f}")
        print(f"  Impute Accuracy: {comp['impute_accuracy']:.3f}")
        print(f"  Difference: {comp['difference']:.3f}")
        
        if comp['oracle_better']:
            print("  🎉 Oracle performs better!")
        elif comp['impute_better']:
            print("  🎉 Impute performs better!")
        else:
            print("  🤝 Models perform similarly!")
    
    print(f"\nResults saved to: {output_dir}/model_comparison.json")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Compare GRPO Oracle vs GRPO Impute models")
    parser.add_argument("--oracle-model", type=str, required=True, 
                       help="Path to GRPO Oracle model")
    parser.add_argument("--impute-model", type=str, required=True,
                       help="Path to GRPO Impute model")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Check if model paths exist
    if not os.path.exists(args.oracle_model):
        print(f"Error: Oracle model path does not exist: {args.oracle_model}")
        return
    
    if not os.path.exists(args.impute_model):
        print(f"Error: Impute model path does not exist: {args.impute_model}")
        return
    
    # Run comparison
    results = compare_models(args.oracle_model, args.impute_model, args.output_dir)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main() 