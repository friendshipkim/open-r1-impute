#!/usr/bin/env python3
"""
Analysis script for GRPO model comparison results.
Categorizes prompts and analyzes which model performs better on different types.
"""

import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

# Optional imports for visualization (not required for basic analysis)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    # Fallback to basic numpy-like operations
    import math
    VISUALIZATION_AVAILABLE = False
    
    # Simple numpy-like functions
    def mean(values):
        return sum(values) / len(values) if values else 0
    
    def std(values):
        if not values:
            return 0
        avg = mean(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

class GRPOAnalyzer:
    def __init__(self, json_file_path: str):
        """Initialize the analyzer with the JSON results file."""
        self.json_file_path = json_file_path
        self.data = self.load_data()
        self.prompt_categories = {}
        
    def load_data(self) -> Dict:
        """Load the JSON data from file."""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def categorize_prompts(self) -> Dict[str, List[int]]:
        """
        Categorize prompts into different types based on content analysis.
        Returns a dictionary mapping categories to list of indices.
        """
        categories = defaultdict(list)
        
        for idx, result in enumerate(self.data['results']):
            prompt = result['prompt'].lower()
            
            # Technical/Programming prompts
            if any(keyword in prompt for keyword in ['program', 'code', 'algorithm', 'software', 'application', 'matlab', 'c#', 'python', 'database', 'api']):
                categories['Technical/Programming'].append(idx)
            
            # Creative Writing prompts
            elif any(keyword in prompt for keyword in ['write a', 'story', 'poem', 'character', 'narrative', 'creative', 'fiction', 'satirical']):
                categories['Creative Writing'].append(idx)
            
            # Business/Professional prompts
            elif any(keyword in prompt for keyword in ['business', 'company', 'marketing', 'employee', 'professional', 'policy', 'report', 'newsletter', 'consultation']):
                categories['Business/Professional'].append(idx)
            
            # Educational/Academic prompts
            elif any(keyword in prompt for keyword in ['explain', 'analyze', 'research', 'study', 'academic', 'theory', 'principle', 'concept', 'history', 'culture']):
                categories['Educational/Academic'].append(idx)
            
            # Recipe/Cooking prompts
            elif any(keyword in prompt for keyword in ['recipe', 'cooking', 'food', 'ingredients', 'kitchen', 'bake', 'cook', 'dish']):
                categories['Recipe/Cooking'].append(idx)
            
            # Travel/Location prompts
            elif any(keyword in prompt for keyword in ['travel', 'location', 'place', 'visit', 'destination', 'hiking', 'trail', 'city', 'country']):
                categories['Travel/Location'].append(idx)
            
            # Health/Medical prompts
            elif any(keyword in prompt for keyword in ['health', 'medical', 'doctor', 'patient', 'treatment', 'surgery', 'hospital', 'medicine']):
                categories['Health/Medical'].append(idx)
            
            # Current Events/News prompts
            elif any(keyword in prompt for keyword in ['news', 'current', 'recent', 'report', 'investigation', 'audit', 'government', 'policy']):
                categories['Current Events/News'].append(idx)
            
            # How-to/Instructional prompts
            elif any(keyword in prompt for keyword in ['how to', 'steps', 'instructions', 'guide', 'tutorial', 'process', 'method']):
                categories['How-to/Instructional'].append(idx)
            
            # Question/Answer prompts
            elif any(keyword in prompt for keyword in ['what is', 'what are', 'how does', 'why', 'when', 'where', 'can you', 'explain']):
                categories['Question/Answer'].append(idx)
            
            # Analysis/Comparison prompts
            elif any(keyword in prompt for keyword in ['compare', 'analyze', 'examine', 'evaluate', 'assess', 'review', 'investigate']):
                categories['Analysis/Comparison'].append(idx)
            
            else:
                categories['General/Other'].append(idx)
        
        self.prompt_categories = dict(categories)
        return self.prompt_categories
    
    def analyze_model_performance_by_category(self) -> Dict[str, Dict]:
        """
        Analyze which model performs better in each category.
        Returns performance statistics for each category.
        """
        if not self.prompt_categories:
            self.categorize_prompts()
        
        category_performance = {}
        
        for category, indices in self.prompt_categories.items():
            if len(indices) < 3:  # Skip categories with too few samples
                continue
                
            model1_wins = 0
            model2_wins = 0
            ties = 0
            model1_scores = []
            model2_scores = []
            
            for idx in indices:
                result = self.data['results'][idx]
                winner = result['winner']
                model1_score = result['model1_score']
                model2_score = result['model2_score']
                
                model1_scores.append(model1_score)
                model2_scores.append(model2_score)
                
                if winner == 'model1':
                    model1_wins += 1
                elif winner == 'model2':
                    model2_wins += 1
                else:
                    ties += 1
            
            total = len(indices)
            
            # Use appropriate mean and std functions
            if VISUALIZATION_AVAILABLE:
                model1_avg = np.mean(model1_scores)
                model2_avg = np.mean(model2_scores)
                model1_std = np.std(model1_scores)
                model2_std = np.std(model2_scores)
            else:
                model1_avg = mean(model1_scores)
                model2_avg = mean(model2_scores)
                model1_std = std(model1_scores)
                model2_std = std(model2_scores)
            
            category_performance[category] = {
                'total_prompts': total,
                'model1_wins': model1_wins,
                'model2_wins': model2_wins,
                'ties': ties,
                'model1_win_rate': model1_wins / total,
                'model2_win_rate': model2_wins / total,
                'tie_rate': ties / total,
                'model1_avg_score': model1_avg,
                'model2_avg_score': model2_avg,
                'model1_std_score': model1_std,
                'model2_std_score': model2_std,
                'avg_score_difference': model1_avg - model2_avg
            }
        
        return category_performance
    
    def analyze_prompt_length_impact(self) -> Dict[str, Any]:
        """
        Analyze how prompt length affects model performance.
        """
        prompt_lengths = []
        model1_scores = []
        model2_scores = []
        winners = []
        
        for result in self.data['results']:
            prompt_length = len(result['prompt'])
            prompt_lengths.append(prompt_length)
            model1_scores.append(result['model1_score'])
            model2_scores.append(result['model2_score'])
            winners.append(result['winner'])
        
        # Categorize by length
        short_prompts = [i for i, length in enumerate(prompt_lengths) if length < 200]
        medium_prompts = [i for i, length in enumerate(prompt_lengths) if 200 <= length < 500]
        long_prompts = [i for i, length in enumerate(prompt_lengths) if length >= 500]
        
        length_analysis = {}
        for name, indices in [('Short (<200 chars)', short_prompts), 
                             ('Medium (200-500 chars)', medium_prompts), 
                             ('Long (>=500 chars)', long_prompts)]:
            if not indices:
                continue
                
            model1_wins = sum(1 for i in indices if winners[i] == 'model1')
            model2_wins = sum(1 for i in indices if winners[i] == 'model2')
            total = len(indices)
            
            # Use appropriate mean function
            if VISUALIZATION_AVAILABLE:
                avg_model1_score = np.mean([model1_scores[i] for i in indices])
                avg_model2_score = np.mean([model2_scores[i] for i in indices])
            else:
                avg_model1_score = mean([model1_scores[i] for i in indices])
                avg_model2_score = mean([model2_scores[i] for i in indices])
            
            length_analysis[name] = {
                'total': total,
                'model1_wins': model1_wins,
                'model2_wins': model2_wins,
                'model1_win_rate': model1_wins / total,
                'model2_win_rate': model2_wins / total,
                'avg_model1_score': avg_model1_score,
                'avg_model2_score': avg_model2_score
            }
        
        return length_analysis
    
    def analyze_complexity_indicators(self) -> Dict[str, Any]:
        """
        Analyze how prompt complexity affects performance.
        """
        complexity_indicators = {
            'multi_step': [],
            'technical_terms': [],
            'specific_requirements': [],
            'creative_tasks': []
        }
        
        for idx, result in enumerate(self.data['results']):
            prompt = result['prompt'].lower()
            
            # Multi-step tasks (containing numbered steps or multiple requirements)
            if re.search(r'\d+\.|step|first|second|finally|additionally', prompt):
                complexity_indicators['multi_step'].append(idx)
            
            # Technical terms
            if re.search(r'\b(api|algorithm|database|framework|protocol|interface|architecture)\b', prompt):
                complexity_indicators['technical_terms'].append(idx)
            
            # Specific requirements
            if re.search(r'\b(must|should|require|specify|include|format|length|word)\b', prompt):
                complexity_indicators['specific_requirements'].append(idx)
            
            # Creative tasks
            if re.search(r'\b(creative|imaginative|story|poem|character|narrative|satirical)\b', prompt):
                complexity_indicators['creative_tasks'].append(idx)
        
        complexity_analysis = {}
        for complexity_type, indices in complexity_indicators.items():
            if not indices:
                continue
                
            model1_wins = sum(1 for i in indices if self.data['results'][i]['winner'] == 'model1')
            model2_wins = sum(1 for i in indices if self.data['results'][i]['winner'] == 'model2')
            total = len(indices)
            
            complexity_analysis[complexity_type] = {
                'total': total,
                'model1_wins': model1_wins,
                'model2_wins': model2_wins,
                'model1_win_rate': model1_wins / total,
                'model2_win_rate': model2_wins / total
            }
        
        return complexity_analysis
    
    def generate_detailed_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        """
        report = []
        report.append("=" * 80)
        report.append("GRPO MODEL COMPARISON ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        total_comparisons = self.data['analysis']['total_comparisons']
        model1_wins = self.data['analysis']['model1_wins']
        model2_wins = self.data['analysis']['model2_wins']
        
        report.append(f"OVERALL RESULTS:")
        report.append(f"Total comparisons: {total_comparisons}")
        report.append(f"Oracle Model wins: {model1_wins} ({model1_wins/total_comparisons*100:.1f}%)")
        report.append(f"Imputed Model wins: {model2_wins} ({model2_wins/total_comparisons*100:.1f}%)")
        report.append("")
        
        # Category analysis
        category_performance = self.analyze_model_performance_by_category()
        report.append("PERFORMANCE BY PROMPT CATEGORY:")
        report.append("-" * 50)
        
        for category, stats in sorted(category_performance.items(), 
                                    key=lambda x: x[1]['total_prompts'], reverse=True):
            report.append(f"\n{category.upper()} ({stats['total_prompts']} prompts):")
            report.append(f"  Oracle Model: {stats['model1_wins']} wins ({stats['model1_win_rate']*100:.1f}%)")
            report.append(f"  Imputed Model: {stats['model2_wins']} wins ({stats['model2_win_rate']*100:.1f}%)")
            report.append(f"  Average score difference: {stats['avg_score_difference']:.3f}")
            
            if stats['model1_win_rate'] > stats['model2_win_rate']:
                report.append(f"  → Oracle Model performs better")
            elif stats['model2_win_rate'] > stats['model1_win_rate']:
                report.append(f"  → Imputed Model performs better")
            else:
                report.append(f"  → Models perform similarly")
        
        # Length analysis
        length_analysis = self.analyze_prompt_length_impact()
        report.append("\n\nPERFORMANCE BY PROMPT LENGTH:")
        report.append("-" * 40)
        
        for length_type, stats in length_analysis.items():
            report.append(f"\n{length_type} ({stats['total']} prompts):")
            report.append(f"  Oracle Model: {stats['model1_wins']} wins ({stats['model1_win_rate']*100:.1f}%)")
            report.append(f"  Imputed Model: {stats['model2_wins']} wins ({stats['model2_win_rate']*100:.1f}%)")
        
        # Complexity analysis
        complexity_analysis = self.analyze_complexity_indicators()
        report.append("\n\nPERFORMANCE BY COMPLEXITY INDICATORS:")
        report.append("-" * 45)
        
        for complexity_type, stats in complexity_analysis.items():
            report.append(f"\n{complexity_type.replace('_', ' ').title()} ({stats['total']} prompts):")
            report.append(f"  Oracle Model: {stats['model1_wins']} wins ({stats['model1_win_rate']*100:.1f}%)")
            report.append(f"  Imputed Model: {stats['model2_wins']} wins ({stats['model2_win_rate']*100:.1f}%)")
        
        # Key insights
        report.append("\n\nKEY INSIGHTS:")
        report.append("-" * 20)
        
        # Find best performing categories for each model
        if category_performance:
            oracle_best = max(category_performance.items(), 
                             key=lambda x: x[1]['model1_win_rate'])
            imputed_best = max(category_performance.items(), 
                              key=lambda x: x[1]['model2_win_rate'])
            
            report.append(f"• Oracle Model excels at: {oracle_best[0]} ({oracle_best[1]['model1_win_rate']*100:.1f}% win rate)")
            report.append(f"• Imputed Model excels at: {imputed_best[0]} ({imputed_best[1]['model2_win_rate']*100:.1f}% win rate)")
        
        # Overall winner
        if model1_wins > model2_wins:
            report.append(f"• Overall winner: Oracle Model by {model1_wins - model2_wins} wins")
        elif model2_wins > model1_wins:
            report.append(f"• Overall winner: Imputed Model by {model2_wins - model1_wins} wins")
        else:
            report.append("• Overall result: Tie")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_analysis(self, output_file: str = "grpo_analysis_report.txt"):
        """Save the analysis report to a file."""
        report = self.generate_detailed_report()
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Analysis report saved to {output_file}")

def main():
    """Main function to run the analysis."""
    analyzer = GRPOAnalyzer("grpo_comparison_results/grpo_model_comparison_local.json")
    
    # Generate and display the report
    report = analyzer.generate_detailed_report()
    print(report)
    
    # Save the report
    analyzer.save_analysis()
    
    # Also save a JSON summary for further analysis
    summary = {
        'category_performance': analyzer.analyze_model_performance_by_category(),
        'length_analysis': analyzer.analyze_prompt_length_impact(),
        'complexity_analysis': analyzer.analyze_complexity_indicators()
    }
    
    with open("grpo_analysis_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("\nAnalysis summary saved to grpo_analysis_summary.json")

if __name__ == "__main__":
    main() 