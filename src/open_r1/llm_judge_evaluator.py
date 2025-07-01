"""
LLM-as-Judge evaluation module for comparing GRPO models.
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

try:
    import openai
except ImportError:
    print("Warning: openai package not found. Please install it with: pip install openai")
    openai = None


@dataclass
class EvaluationResult:
    """Result of LLM-as-Judge evaluation."""
    prompt: str
    completion: str
    score: float
    explanation: str
    model_name: str


@dataclass
class ComparisonResult:
    """Result of comparing two completions side by side."""
    prompt: str
    completion1: str
    completion2: str
    winner: str  # "model1", "model2", or "tie"
    explanation: str
    model1_name: str
    model2_name: str


class LLMJudgeEvaluator:
    """
    LLM-as-Judge evaluator for comparing model outputs.
    """
    
    def __init__(self, judge_model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the LLM-as-Judge evaluator.
        
        Args:
            judge_model_name: Name of the judge model to use
            api_key: OpenAI API key (if not set, will use environment variable)
        """
        self.judge_model_name = judge_model_name
        
        if openai is None:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        # Initialize OpenAI client with new API format
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = openai.OpenAI()
    
    def create_judgment_prompt(self, prompt: str, completion: str) -> str:
        """Create a prompt for the judge LLM."""
        return f"""You are an expert evaluator. Please rate the following response to a user query on a scale of 0-10, where 0 is completely incorrect/inappropriate and 10 is perfect.

User Query: {prompt}

Response: {completion}

Please provide your rating (0-10) and a brief explanation:

Rating:"""

    def get_judgment(self, prompt: str, completion: str) -> tuple[float, str]:
        """
        Get judgment score and explanation from the judge LLM.
        
        Returns:
            Tuple of (score, explanation) where score is normalized to 0-1
        """
        try:
            judgment_prompt = self.create_judgment_prompt(prompt, completion)
            
            response = self.client.chat.completions.create(
                model=self.judge_model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Provide only the numerical rating (0-10) followed by a brief explanation."},
                    {"role": "user", "content": judgment_prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract the numerical rating
            rating_match = re.search(r'(\d+(?:\.\d+)?)', response_text)
            if rating_match:
                rating = float(rating_match.group(1))
                # Normalize to 0-1 range
                score = min(max(rating / 10.0, 0.0), 1.0)
            else:
                score = 0.5
            
            # Extract explanation (everything after the rating)
            explanation = response_text
            if rating_match:
                explanation = response_text[rating_match.end():].strip()
                if explanation.startswith('.'):
                    explanation = explanation[1:].strip()
            
            return score, explanation
            
        except Exception as e:
            print(f"Error getting judgment: {e}")
            return 0.5, f"Error: {str(e)}"
    
    def compare_completions(self, 
                           prompt: str, 
                           completion1: str, 
                           completion2: str,
                           model1_name: str = "Model 1",
                           model2_name: str = "Model 2") -> ComparisonResult:
        """
        Compare two completions side by side using LLM-as-Judge.
        
        Args:
            prompt: The original prompt
            completion1: Completion from first model
            completion2: Completion from second model
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            ComparisonResult with winner and explanation
        """
        try:
            # Create comparison prompt
            comparison_prompt = f"""You are an expert evaluator. Please compare two responses to the same user query and determine which one is better.

User Query: {prompt}

Response A ({model1_name}):
{completion1}

Response B ({model2_name}):
{completion2}

Please evaluate both responses and determine which one is better. Consider factors such as:
- Accuracy and relevance to the query
- Completeness of the response
- Clarity and coherence
- Helpfulness and usefulness

Please provide:
1. Which response is better (A, B, or tie)
2. A detailed explanation of why you chose that response

Format your response as:
Winner: [A/B/tie]
Explanation: [detailed explanation of your reasoning]"""

            response = self.client.chat.completions.create(
                model=self.judge_model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Be fair and objective in your comparison. Provide clear reasoning for your choice."},
                    {"role": "user", "content": comparison_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse the response
            winner_match = re.search(r'Winner:\s*(A|B|tie)', response_text, re.IGNORECASE)
            
            # Determine winner
            winner = "tie"
            if winner_match:
                winner_text = winner_match.group(1).lower()
                if winner_text == "a":
                    winner = "model1"
                elif winner_text == "b":
                    winner = "model2"
                else:
                    winner = "tie"
            
            # Extract explanation
            explanation_match = re.search(r'Explanation:\s*(.+)', response_text, re.IGNORECASE | re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
            
            return ComparisonResult(
                prompt=prompt,
                completion1=completion1,
                completion2=completion2,
                winner=winner,
                explanation=explanation,
                model1_name=model1_name,
                model2_name=model2_name
            )
            
        except Exception as e:
            print(f"Error comparing completions: {e}")
            return ComparisonResult(
                prompt=prompt,
                completion1=completion1,
                completion2=completion2,
                winner="tie",
                explanation=f"Error during comparison: {str(e)}",
                model1_name=model1_name,
                model2_name=model2_name
            )
    
    def evaluate_completions(self, 
                           prompts: List[str], 
                           completions: List[str], 
                           model_name: str = "unknown") -> List[EvaluationResult]:
        """
        Evaluate a list of completions using LLM-as-Judge.
        
        Args:
            prompts: List of prompts
            completions: List of completions
            model_name: Name of the model being evaluated
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for prompt, completion in tqdm(zip(prompts, completions), 
                                     total=len(prompts), 
                                     desc=f"Evaluating {model_name}"):
            score, explanation = self.get_judgment(prompt, completion)
            
            result = EvaluationResult(
                prompt=prompt,
                completion=completion,
                score=score,
                explanation=explanation,
                model_name=model_name
            )
            results.append(result)
        
        return results
    
    def compare_models(self, 
                      prompts: List[str],
                      completions_model1: List[str],
                      completions_model2: List[str],
                      model1_name: str = "Model 1",
                      model2_name: str = "Model 2") -> Dict[str, Any]:
        """
        Compare two models using LLM-as-Judge.
        
        Args:
            prompts: List of prompts
            completions_model1: Completions from first model
            completions_model2: Completions from second model
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Dictionary with comparison results
        """
        # Evaluate both models
        results1 = self.evaluate_completions(prompts, completions_model1, model1_name)
        results2 = self.evaluate_completions(prompts, completions_model2, model2_name)
        
        # Calculate statistics
        scores1 = [r.score for r in results1]
        scores2 = [r.score for r in results2]
        
        # Count wins for each model
        wins_model1 = sum(1 for s1, s2 in zip(scores1, scores2) if s1 > s2)
        wins_model2 = sum(1 for s1, s2 in zip(scores1, scores2) if s2 > s1)
        ties = len(scores1) - wins_model1 - wins_model2
        
        comparison_results = {
            "model1": {
                "name": model1_name,
                "mean_score": np.mean(scores1),
                "std_score": np.std(scores1),
                "wins": wins_model1,
                "results": results1
            },
            "model2": {
                "name": model2_name,
                "mean_score": np.mean(scores2),
                "std_score": np.std(scores2),
                "wins": wins_model2,
                "results": results2
            },
            "comparison": {
                "total_examples": len(prompts),
                "wins_model1": wins_model1,
                "wins_model2": wins_model2,
                "ties": ties,
                "win_rate_model1": wins_model1 / len(prompts),
                "win_rate_model2": wins_model2 / len(prompts),
                "score_difference": np.mean(scores1) - np.mean(scores2)
            }
        }
        
        return comparison_results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to a JSON file."""
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
        """Print a summary of the comparison results."""
        print("\n" + "="*60)
        print("LLM-as-Judge Evaluation Summary")
        print("="*60)
        
        model1 = results["model1"]
        model2 = results["model2"]
        comparison = results["comparison"]
        
        print(f"\n{model1['name']}:")
        print(f"  Mean Score: {model1['mean_score']:.3f} ± {model1['std_score']:.3f}")
        print(f"  Wins: {model1['wins']}")
        
        print(f"\n{model2['name']}:")
        print(f"  Mean Score: {model2['mean_score']:.3f} ± {model2['std_score']:.3f}")
        print(f"  Wins: {model2['wins']}")
        
        print(f"\nComparison:")
        print(f"  Total Examples: {comparison['total_examples']}")
        print(f"  Ties: {comparison['ties']}")
        print(f"  Win Rate {model1['name']}: {comparison['win_rate_model1']:.1%}")
        print(f"  Win Rate {model2['name']}: {comparison['win_rate_model2']:.1%}")
        print(f"  Score Difference ({model1['name']} - {model2['name']}): {comparison['score_difference']:.3f}")
        
        if comparison['score_difference'] > 0:
            print(f"\n🎉 {model1['name']} performs better!")
        elif comparison['score_difference'] < 0:
            print(f"\n🎉 {model2['name']} performs better!")
        else:
            print(f"\n🤝 Models perform similarly!")
        
        print("="*60)


def main():
    """Example usage of the LLM-as-Judge evaluator."""
    # Example prompts and completions
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about AI."
    ]
    
    # Example completions from two different models
    completions_model1 = [
        "The capital of France is Paris.",
        "Quantum computing uses quantum bits that can be in multiple states at once, making it much faster than regular computers for certain problems.",
        "Silicon dreams and neural streams,\nAI learns and grows and schemes.\nIn circuits deep, intelligence flows,\nA future bright, the mind bestows."
    ]
    
    completions_model2 = [
        "Paris is the capital city of France.",
        "Quantum computing is like having a super-fast computer that can solve really hard problems by using the weird properties of tiny particles.",
        "Digital minds awake,\nLearning from mistakes,\nGrowing ever stronger,\nAI's future longer."
    ]
    
    # Initialize evaluator
    evaluator = LLMJudgeEvaluator(judge_model_name="gpt-4")
    
    # Compare models
    results = evaluator.compare_models(
        prompts=prompts,
        completions_model1=completions_model1,
        completions_model2=completions_model2,
        model1_name="GRPO Oracle",
        model2_name="GRPO Impute"
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    evaluator.save_results(results, "llm_judge_results.json")


if __name__ == "__main__":
    main() 