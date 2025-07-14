#!/usr/bin/env python3
"""
LLM-as-Judge evaluation script for cleaned JSON files.
This script evaluates model completions that are already stored in cleaned JSON files,
skipping the model inference and prompt loading steps.
Supports both OpenAI and Anthropic APIs.
"""

import os
import json
import argparse
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.append('/root/open-r1-impute/src')
from open_r1.llm_judge_evaluator import LLMJudgeEvaluator, ComparisonResult


@dataclass
class CleanedComparisonResult:
    """Result from cleaned JSON file."""
    prompt: str
    model1_completion: str
    model2_completion: str


class LLMJudgeEvaluatorFromCleaned:
    """LLM-as-Judge evaluator for cleaned JSON files."""
    
    def __init__(self, 
                 judge_model: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 model1_name: str = "oracle_model",
                 model2_name: str = "imputed_model"):
        """
        Initialize the LLM-as-Judge evaluator.
        
        Args:
            judge_model: Judge model to use (auto-detected if None based on API key)
            api_key: OpenAI or Anthropic API key
            model1_name: Name for the first model
            model2_name: Name for the second model
        """
        self.api_key = api_key
        self.model1_name = model1_name
        self.model2_name = model2_name
        
        # Auto-detect API type and set appropriate judge model
        if judge_model is None and api_key:
            if api_key.startswith("sk-ant-"):
                self.judge_model = "claude-3-5-sonnet-20241022"
                print(f"Detected Anthropic API key, using judge model: {self.judge_model}")
            else:
                self.judge_model = "gpt-4"
                print(f"Detected OpenAI API key, using judge model: {self.judge_model}")
        else:
            self.judge_model = judge_model or "gpt-4"
        
        # Initialize LLM-as-Judge evaluator
        self.llm_judge = LLMJudgeEvaluator(judge_model_name=self.judge_model, api_key=api_key)
    
    def load_cleaned_data(self, json_file: str) -> List[CleanedComparisonResult]:
        """Load cleaned JSON data."""
        print(f"Loading cleaned data from {json_file}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        for result in data.get('results', []):
            cleaned_result = CleanedComparisonResult(
                prompt=result['prompt'],
                model1_completion=result['model1_completion'],
                model2_completion=result['model2_completion']
            )
            results.append(cleaned_result)
        
        print(f"Loaded {len(results)} comparison results")
        return results
    
    def evaluate_completions(self, cleaned_results: List[CleanedComparisonResult]) -> List[ComparisonResult]:
        """Evaluate completions using LLM-as-Judge."""
        print("Evaluating completions using LLM-as-Judge...")
        
        results = []
        
        for i, cleaned_result in enumerate(cleaned_results):
            try:
                # Use LLM-as-Judge to compare the completions
                comparison = self.llm_judge.compare_completions(
                    prompt=cleaned_result.prompt,
                    completion1=cleaned_result.model1_completion,
                    completion2=cleaned_result.model2_completion,
                    model1_name=self.model1_name,
                    model2_name=self.model2_name
                )
                
                # Create comparison result
                result = ComparisonResult(
                    prompt=cleaned_result.prompt,
                    completion1=cleaned_result.model1_completion,
                    completion2=cleaned_result.model2_completion,
                    winner=comparison.winner,
                    explanation=comparison.explanation,
                    model1_name=self.model1_name,
                    model2_name=self.model2_name
                )
                
                results.append(result)
                
                # Print progress
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(cleaned_results)} comparisons")
                
                # Add small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing comparison {i}: {e}")
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
        print("LLM-as-Judge Evaluation Summary")
        print("="*80)
        
        print(f"\nModel 1 ({self.model1_name}): Oracle Model")
        print(f"Model 2 ({self.model2_name}): Imputed Model")
        print(f"Judge Model: {self.judge_model}")
        print(f"API Type: {getattr(self.llm_judge, 'api_type', 'unknown')}")
        print(f"Total Comparisons: {analysis['total_comparisons']}")
        
        print(f"\nWin Rates:")
        print(f"  {self.model1_name}: {analysis['model1_win_rate']:.1%} ({analysis['model1_wins']} wins)")
        print(f"  {self.model2_name}: {analysis['model2_win_rate']:.1%} ({analysis['model2_wins']} wins)")
        print(f"  Ties: {analysis['tie_rate']:.1%} ({analysis['ties']} ties)")
        
        # Determine overall winner
        if analysis['model1_win_rate'] > analysis['model2_win_rate']:
            print(f"\n🏆 Overall Winner: {self.model1_name}")
        elif analysis['model2_win_rate'] > analysis['model1_win_rate']:
            print(f"\n🏆 Overall Winner: {self.model2_name}")
        else:
            print(f"\n🤝 Overall Result: Tie")
        
        print("="*80)
    
    def save_results(self, results: List[ComparisonResult], analysis: Dict[str, Any], output_file: str, original_data: Dict[str, Any]):
        """Save evaluation results in the same format as original JSON file."""
        # Convert dataclasses to dictionaries and add winner/explanation to original results
        results_dict = []
        for i, result in enumerate(results):
            # Get the original result data
            original_result = original_data['results'][i]
            # Add winner and explanation to the original result
            original_result['winner'] = result.winner
            original_result['explanation'] = result.explanation
            results_dict.append(original_result)
        
        # Create output data with the same structure as original file
        output_data = {
            "model1_path": original_data.get('model1_path', ''),
            "model2_path": original_data.get('model2_path', ''),
            "judge_model": self.judge_model,
            "api_type": getattr(self.llm_judge, 'api_type', 'unknown'),
            "seed": original_data.get('seed', 42),
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


def detect_api_type(api_key: str) -> str:
    """Detect API type based on API key format."""
    if api_key.startswith("sk-"):
        if api_key.startswith("sk-ant-"):
            return "anthropic"
        else:
            return "openai"
    else:
        return "unknown"


def get_default_judge_model(api_type: str) -> str:
    """Get default judge model based on API type."""
    # Use gpt-4 as default like in the original model_comparison.py
    return "gpt-4"


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation for cleaned JSON files")
    parser.add_argument("--input-file", type=str, required=True,
                       help="Path to cleaned JSON file")
    parser.add_argument("--judge-model", type=str, default=None,
                       help="Judge model to use (auto-detected from API key if not specified)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="OpenAI or Anthropic API key")
    parser.add_argument("--model1-name", type=str, default="oracle_model",
                       help="Name for the first model")
    parser.add_argument("--model2-name", type=str, default="imputed_model",
                       help="Name for the second model")
    parser.add_argument("--num-runs", type=int, default=1,
                       help="Number of times to run the evaluation (default: 1)")
    parser.add_argument("--show-samples", type=int, default=5,
                       help="Number of sample comparisons to show")
    
    args = parser.parse_args()
    
    # Set API key
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        os.environ["ANTHROPIC_API_KEY"] = args.api_key
    
    # Load original data to preserve structure
    with open(args.input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Determine output directory (same as input file directory)
    input_path = Path(args.input_file)
    output_dir = input_path.parent
    
    # Initialize evaluator
    print("Initializing LLM-as-Judge evaluator...")
    evaluator = LLMJudgeEvaluatorFromCleaned(
        judge_model=args.judge_model,
        api_key=args.api_key,
        model1_name=args.model1_name,
        model2_name=args.model2_name
    )
    
    # Load cleaned data
    cleaned_results = evaluator.load_cleaned_data(args.input_file)
    
    if not cleaned_results:
        print("No data found in the cleaned JSON file!")
        return
    
    # Run multiple evaluations
    for run_num in range(1, args.num_runs + 1):
        print(f"\n{'='*80}")
        print(f"RUN {run_num}/{args.num_runs}")
        print(f"{'='*80}")
        
        # Evaluate completions
        print("Starting LLM-as-Judge evaluation...")
        results = evaluator.evaluate_completions(cleaned_results)
        
        if not results:
            print("No results obtained from evaluation!")
            continue
        
        # Analyze results
        analysis = evaluator.analyze_results(results)
        
        # Print summary
        evaluator.print_summary(results, analysis)
        
        # Print sample comparisons (only for first run to avoid spam)
        if run_num == 1:
            evaluator.print_sample_comparisons(results, args.show_samples)
        
        # Save results with judge model in filename
        output_file = str(output_dir / f"loop{run_num}_{evaluator.judge_model}.json")
        evaluator.save_results(results, analysis, output_file, original_data)
        
        print(f"Run {run_num} complete! Results saved to: {output_file}")
        
        # Add a small delay between runs to avoid rate limiting
        if run_num < args.num_runs:
            print("Waiting 2 seconds before next run...")
            time.sleep(2)
    
    print(f"\n{'='*80}")
    print(f"ALL {args.num_runs} RUNS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main() 