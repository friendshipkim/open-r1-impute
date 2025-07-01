#!/usr/bin/env python3
"""
Simple visualization script for GRPO analysis results.
Creates basic charts to visualize the performance differences.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def create_visualizations():
    """Create visualizations from the analysis results."""
    
    # Load the analysis summary
    with open("grpo_analysis_summary.json", 'r') as f:
        data = json.load(f)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GRPO Model Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. Category Performance Chart
    category_data = data['category_performance']
    categories = list(category_data.keys())
    oracle_wins = [category_data[cat]['model1_wins'] for cat in categories]
    imputed_wins = [category_data[cat]['model2_wins'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, oracle_wins, width, label='Oracle Model', color='#2E86AB', alpha=0.8)
    axes[0, 0].bar(x + width/2, imputed_wins, width, label='Imputed Model', color='#A23B72', alpha=0.8)
    
    axes[0, 0].set_xlabel('Prompt Categories')
    axes[0, 0].set_ylabel('Number of Wins')
    axes[0, 0].set_title('Performance by Prompt Category')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([cat.replace('/', '\n') for cat in categories], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Win Rate Comparison
    oracle_rates = [category_data[cat]['model1_win_rate'] * 100 for cat in categories]
    imputed_rates = [category_data[cat]['model2_win_rate'] * 100 for cat in categories]
    
    axes[0, 1].bar(x - width/2, oracle_rates, width, label='Oracle Model', color='#2E86AB', alpha=0.8)
    axes[0, 1].bar(x + width/2, imputed_rates, width, label='Imputed Model', color='#A23B72', alpha=0.8)
    
    axes[0, 1].set_xlabel('Prompt Categories')
    axes[0, 1].set_ylabel('Win Rate (%)')
    axes[0, 1].set_title('Win Rate by Prompt Category')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([cat.replace('/', '\n') for cat in categories], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Prompt Length Analysis
    length_data = data['length_analysis']
    length_types = list(length_data.keys())
    oracle_length_wins = [length_data[lt]['model1_wins'] for lt in length_types]
    imputed_length_wins = [length_data[lt]['model2_wins'] for lt in length_types]
    
    x_length = np.arange(len(length_types))
    
    axes[1, 0].bar(x_length - width/2, oracle_length_wins, width, label='Oracle Model', color='#2E86AB', alpha=0.8)
    axes[1, 0].bar(x_length + width/2, imputed_length_wins, width, label='Imputed Model', color='#A23B72', alpha=0.8)
    
    axes[1, 0].set_xlabel('Prompt Length')
    axes[1, 0].set_ylabel('Number of Wins')
    axes[1, 0].set_title('Performance by Prompt Length')
    axes[1, 0].set_xticks(x_length)
    axes[1, 0].set_xticklabels([lt.replace(' chars', '\nchars') for lt in length_types])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Complexity Analysis
    complexity_data = data['complexity_analysis']
    complexity_types = list(complexity_data.keys())
    oracle_complexity_wins = [complexity_data[ct]['model1_wins'] for ct in complexity_types]
    imputed_complexity_wins = [complexity_data[ct]['model2_wins'] for ct in complexity_types]
    
    x_complexity = np.arange(len(complexity_types))
    
    axes[1, 1].bar(x_complexity - width/2, oracle_complexity_wins, width, label='Oracle Model', color='#2E86AB', alpha=0.8)
    axes[1, 1].bar(x_complexity + width/2, imputed_complexity_wins, width, label='Imputed Model', color='#A23B72', alpha=0.8)
    
    axes[1, 1].set_xlabel('Complexity Indicators')
    axes[1, 1].set_ylabel('Number of Wins')
    axes[1, 1].set_title('Performance by Complexity Indicators')
    axes[1, 1].set_xticks(x_complexity)
    axes[1, 1].set_xticklabels([ct.replace('_', ' ').title() for ct in complexity_types], rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('grpo_analysis_visualization.png', dpi=300, bbox_inches='tight')
    plt.savefig('grpo_analysis_visualization.pdf', bbox_inches='tight')
    
    print("Visualizations saved as:")
    print("- grpo_analysis_visualization.png")
    print("- grpo_analysis_visualization.pdf")
    
    # Create a summary table
    print("\n" + "="*80)
    print("SUMMARY OF KEY FINDINGS")
    print("="*80)
    
    # Find best and worst categories for each model
    oracle_best = max(category_data.items(), key=lambda x: x[1]['model1_win_rate'])
    oracle_worst = min(category_data.items(), key=lambda x: x[1]['model1_win_rate'])
    imputed_best = max(category_data.items(), key=lambda x: x[1]['model2_win_rate'])
    imputed_worst = min(category_data.items(), key=lambda x: x[1]['model2_win_rate'])
    
    print(f"\nOracle Model Performance:")
    print(f"  Best category: {oracle_best[0]} ({oracle_best[1]['model1_win_rate']*100:.1f}% win rate)")
    print(f"  Worst category: {oracle_worst[0]} ({oracle_worst[1]['model1_win_rate']*100:.1f}% win rate)")
    
    print(f"\nImputed Model Performance:")
    print(f"  Best category: {imputed_best[0]} ({imputed_best[1]['model2_win_rate']*100:.1f}% win rate)")
    print(f"  Worst category: {imputed_worst[0]} ({imputed_worst[1]['model2_win_rate']*100:.1f}% win rate)")
    
    # Overall statistics
    total_oracle_wins = sum(cat['model1_wins'] for cat in category_data.values())
    total_imputed_wins = sum(cat['model2_wins'] for cat in category_data.values())
    total_prompts = sum(cat['total_prompts'] for cat in category_data.values())
    
    print(f"\nOverall Statistics:")
    print(f"  Total prompts analyzed: {total_prompts}")
    print(f"  Oracle Model total wins: {total_oracle_wins} ({total_oracle_wins/total_prompts*100:.1f}%)")
    print(f"  Imputed Model total wins: {total_imputed_wins} ({total_imputed_wins/total_prompts*100:.1f}%)")
    
    if total_oracle_wins > total_imputed_wins:
        print(f"  Overall winner: Oracle Model by {total_oracle_wins - total_imputed_wins} wins")
    elif total_imputed_wins > total_oracle_wins:
        print(f"  Overall winner: Imputed Model by {total_imputed_wins - total_oracle_wins} wins")
    else:
        print("  Overall result: Tie")

if __name__ == "__main__":
    try:
        create_visualizations()
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install matplotlib: pip install matplotlib")
    except FileNotFoundError:
        print("Error: grpo_analysis_summary.json not found.")
        print("Please run analyze_grpo_results.py first.") 