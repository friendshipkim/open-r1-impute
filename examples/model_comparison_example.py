#!/usr/bin/env python3
"""
Simple example of model comparison using LLM-as-Judge.
Compares the trained GRPO model with the baseline model.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.model_comparison import ModelComparator

def main():
    # Model paths
    trained_model = "friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0"
    baseline_model = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Set your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Example prompts for comparison
    example_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the main differences between machine learning and deep learning?",
        "Design a simple algorithm to find the largest number in a list.",
        "Explain the concept of climate change to a 10-year-old."
    ]
    
    print("Initializing model comparator...")
    comparator = ModelComparator(
        model1_path=trained_model,
        model2_path=baseline_model,
        judge_model="gpt-4",
        api_key=api_key
    )
    
    print("Running comparison on example prompts...")
    results = comparator.compare_models_on_prompts(example_prompts)
    
    # Analyze results
    analysis = comparator.analyze_results(results)
    
    # Print summary
    comparator.print_summary(results, analysis)
    
    # Print sample comparisons
    comparator.print_sample_comparisons(results, num_samples=3)
    
    # Save results
    output_file = "example_model_comparison_results.json"
    comparator.save_results(results, analysis, output_file)
    
    print(f"\nResults saved to {output_file}")
    print("\nThis demonstrates how the trained GRPO model compares to the baseline model.")
    print("The comparison uses GPT-4 as a judge to evaluate which model produces better responses.")

if __name__ == "__main__":
    main() 