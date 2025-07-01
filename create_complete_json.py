import json

# Winners array from the original data
winners = [
    "Model2", "Model2", "Model2", "Model2", "Model1", 
    "Model2", "Model2", "Model2", "Model2", "Model2",
    "Model1", "Model2", "Model1", "Model2", "Model2",
    "Model2", "Model2", "Model1", "Model2", "Model2",
    "Model1", "Model2", "Model2", "Model2", "Model1",
    "Model2", "Model2", "Model2", "Model2", "Model1", 
    "Model2", "Model1", "Model2", "Model2", "Model2",
    "Model2", "Model2", "Model2", "Model2", "Model2",
    "Model2", "Model2", "Model2", "Model1", "Model1",
    "Model2", "Model2", "Model2", "Model2", "Model2",
    "Model2", "Model2", "Model2", "Model1", "Model2",
    "Model2", "Model2", "Model1", "Model2", "Model1",
    "Model2", "Model2", "Model2", "Model1", "Model2",
    "Model2", "Model2", "Model1", "Model2", "Model2",
    "Model2", "Model2", "Model1", "Model2", "Model2",
    "Model2", "Model2", "Model2", "Model2", "Model2",
    "Model2", "Model2", "Model2", "Model2", "Model2",
    "Model1", "Model2", "Model2", "Model2", "Model2",
    "Model2", "Model2", "Model2", "Model2", "Model2",
    "Model2", "Model2", "Model2", "Model1", "Model2"
]

# Reasons array from the original data
reasons = [
    "Model 2 provides better structured report with clearer sections, more comprehensive coverage of different solutions, and better organized recommendations.",
    "Model 1 contains obvious factual errors (Muhammad Ali vs Anthony Joshua in 2024 is fictional). Model 2 has better structure and fewer glaring inaccuracies.",
    "Model 2 provides a complete, well-structured policy brief with clear sections and comprehensive coverage. Model 1's response appears incomplete.",
    "While both have some inaccuracies, Model 2 provides clearer, more organized directions. Model 1's response is confusing with incorrect details.",
    "Model 1 provides more detailed step-by-step process with specific preparation advice. Model 2 is good but less comprehensive.",
    "Model 2 creates a more engaging and vivid narrative with better character development and sensory details.",
    "Model 2 provides more comprehensive explanation of how narrative essays differ from others with clearer examples.",
    "Model 2 gives clearer, more organized response that directly addresses the question. Model 1's response is verbose and circular.",
    "While both have issues, Model 2 attempts a more complete solution and better addresses the requirements.",
    "Model 2 provides more comprehensive guide with better narrative flow and detailed segments.",
    "Model 1 creates more vivid and immersive descriptive narrative with better sensory details and emotional depth.",
    "Model 2 provides more specific and accurate trail recommendations with better descriptions.",
    "Model 1 provides clearer and more accurate paraphrase of the scientific concept. Model 2's explanation is convoluted.",
    "Model 2 provides better formatted, more professional report structure with clearer sections and comprehensive coverage.",
    "Model 2 gives more detailed explanation of lesson relationships and better paraphrasing of the content.",
    "Model 2 provides more comprehensive review with better organization, practical examples, and thorough coverage.",
    "Model 2 offers better structured and more organized exploration of the topic with clearer sections.",
    "Model 1 provides more accurate and focused paraphrase of the poem's central message.",
    "Model 2 gives clearer explanation of the concept with better examples and comprehensive coverage.",
    "Model 2 provides more complete recipe with better step-by-step instructions and practical cooking advice.",
    "Model 1 provides better analysis and summary of the specific content with more accurate capture of main points.",
    "Model 2 gives more thorough and better organized analysis of the research findings.",
    "Model 2 provides more comprehensive list with better descriptions and systematic organization.",
    "Model 2 offers better structured explanation with clearer organization and comprehensive coverage of influence.",
    "Model 1 provides more helpful and practical alternative suggestion that directly addresses the user's need.",
    "Model 2 creates more engaging narrative with better character development and realistic portrayal.",
    "Model 2 provides more detailed and comprehensive analysis with better structured recommendations.",
    "Model 2 offers more comprehensive and better organized analysis with clearer examples of influence.",
    "Model 2 provides more practical and detailed implementation guide with better code examples.",
    "Model 1 does better job of converting casual tone to appropriate formal business communication.",
    "Model 2 creates more heartfelt and engaging letter with better emotional depth and meaningful advice.",
    "Model 1 provides more accurate and focused information about Agave americana specifically.",
    "Model 2 provides better and more accurate paraphrase of the comeback efforts with clearer structure.",
    "Model 2 gives more organized and comprehensive explanation with better structure.",
    "Model 2 provides more accurate explanation of the actual Clue board game rules.",
    "Model 2 creates more engaging and complete narrative with better character development.",
    "Model 2 provides better structured and more comprehensive exploration with clearer organization.",
    "Model 2 gives clearer and more accurate paraphrase with better explanation of life application.",
    "Model 2 provides more comprehensive and better organized feature article with clearer structure.",
    "Model 2 gives more organized and comprehensive explanation with better structure and presentation.",
    "Model 2 provides more detailed and systematic step-by-step guide with better organization.",
    "Model 2 creates more complete and practical recipe with proper ingredients and clear instructions.",
    "Model 2 provides more comprehensive and better structured analysis with clearer organization.",
    "Model 1 provides more comprehensive list of evaluation elements and better explains the process.",
    "Model 1 gives more accurate and comprehensive explanation of religious significance with better understanding.",
    "Model 2 provides more detailed and practical step-by-step guide with better organization.",
    "Model 2 creates more professional and comprehensive newsletter format with better structure.",
    "Model 2 provides better paraphrase that captures the instructor's multicultural approach more clearly.",
    "Model 1 attempts to provide more specific information about the region, though both have inaccuracies.",
    "Model 2 gives more organized and comprehensive explanation with better structure and presentation.",
    "Model 2 provides more complete and better structured translation attempt.",
    "Model 2 creates more engaging narrative with better character development and compelling story arc.",
    "Model 2 provides more focused and accurate analysis with better organization.",
    "Model 1 provides more complete and accurate implementation with better code examples.",
    "Model 2 gives more accurate and comprehensive explanation with better organization.",
    "Model 2 provides more comprehensive and better structured analysis with clearer examples.",
    "Model 2 provides more accurate balsamic vinaigrette recipe. Model 1 incorrectly uses white wine vinegar.",
    "Model 2 creates more detailed and comprehensive documentary concept with better structure.",
    "Model 1 provides more detailed and comprehensive instructions with better step-by-step guidance.",
    "Model 2 provides more accurate and comprehensive analysis with better historical context.",
    "Model 2 gives more comprehensive and practical advice with better organized tips.",
    "Model 2 creates more engaging narrative with better plot development and satisfying resolution.",
    "Model 2 provides more comprehensive and detailed guide with better organization.",
    "Model 2 provides more coherent and better structured summary with clearer interpretation.",
    "Model 2 gives more accurate and complete list of traditional picadillo ingredients.",
    "Model 2 provides more comprehensive and better organized explanation with clearer structure.",
    "Model 2 gives more thoughtful and comprehensive analysis with better reasoning.",
    "Model 2 provides more accurate and comprehensive summary with better organization.",
    "Model 2 gives more comprehensive and better structured analysis with clearer examples.",
    "Model 2 provides more accurate answer about non-registered user access with clearer explanation.",
    "Model 2 provides more comprehensive and better organized advice with clearer guidance.",
    "Model 2 creates more engaging story with better character development and compelling narrative.",
    "Model 1 provides more complete and accurate implementation with better code examples and explanation.",
    "Model 2 gives more accurate and comprehensive explanation with better organization.",
    "Model 2 provides more comprehensive and better structured analysis with clearer examples.",
    "Model 2 gives more detailed and practical step-by-step guide with better organization.",
    "Model 2 creates more professional and comprehensive newsletter with better structure.",
    "Model 2 provides more comprehensive and better organized explanation with clearer structure.",
    "Model 2 creates more effective satirical piece with better humor and compelling critique.",
    "Model 2 provides more realistic and comprehensive character sketch with better emotional depth.",
    "Model 2 gives more accurate and comprehensive answer with better organization.",
    "Model 2 provides more helpful response by offering alternative finishes versus stating none available.",
    "Model 2 provides more comprehensive and detailed summary with better organization.",
    "Model 2 provides more complete and accurate implementation approach with better code structure.",
    "Model 1 provides more detailed and practical logo design concept with better specific recommendations.",
    "Model 2 creates more engaging and effective volunteer recruitment tweet with better language.",
    "Model 2 provides more complete and practical implementation approach with better code examples.",
    "Model 2 gives more comprehensive and detailed explanation with better organization.",
    "Model 2 creates more engaging survival story with better character development and themes.",
    "Model 2 provides more practical and complete implementation with better code structure.",
    "Model 2 gives more comprehensive and better structured analysis of Civil War tensions.",
    "Model 2 provides more accurate and comprehensive information about Gothic buildings in use.",
    "Model 2 gives more comprehensive and better organized explanation of IT program benefits.",
    "Model 2 creates more powerful and emotionally resonant poem about mental health strength.",
    "Model 2 provides more comprehensive and detailed information with better organization.",
    "Model 2 creates more engaging survival story with better plot development and meaningful themes.",
    "Model 2 provides more comprehensive and detailed step-by-step guide with better organization.",
    "Model 2 gives more detailed and practical optimization strategy with better coverage.",
    "Model 1 provides more detailed and comprehensive analysis of climate and geography influence."
]

print(f"Winners length: {len(winners)}")
print(f"Reasons length: {len(reasons)}")

# Ensure both arrays have the same length
min_length = min(len(winners), len(reasons))
winners = winners[:min_length]
reasons = reasons[:min_length]

# Calculate statistics
model1_wins = winners.count("Model1")
model2_wins = winners.count("Model2")
total_comparisons = len(winners)

# Create the JSON structure
json_data = {
    "model1_path": "friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-oracle-lr2e-6-warmup0.05",
    "model2_path": "friendshipkim/Qwen2.5-1.5B-ultrachat-qrm-p16-g8-ts300-lr2e-6-warmup0.05-ps0.2-preps0.0-rho0",
    "judge_model": "claude-3-5-sonnet",
    "analysis": {
        "total_comparisons": total_comparisons,
        "model1_wins": model1_wins,
        "model2_wins": model2_wins,
        "ties": 0,
        "model1_win_rate": round(model1_wins / total_comparisons, 2),
        "model2_win_rate": round(model2_wins / total_comparisons, 2),
        "tie_rate": 0.0
    },
    "results": []
}

# Create placeholder results (since we don't have the actual prompts and completions)
for i in range(total_comparisons):
    result = {
        "prompt": f"Prompt {i+1}",
        "model1_completion": f"Model 1 completion for prompt {i+1}",
        "model2_completion": f"Model 2 completion for prompt {i+1}",
        "winner": winners[i].lower(),
        "explanation": reasons[i]
    }
    json_data["results"].append(result)

# Write to file
output_filename = 'evaluation_results/oracle_lr2e-6-warmup0.05_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples_claude.json'
with open(output_filename, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"Created complete JSON file: {output_filename}")
print(f"Model 1 wins: {model1_wins}")
print(f"Model 2 wins: {model2_wins}")
print(f"Total comparisons: {total_comparisons}") 