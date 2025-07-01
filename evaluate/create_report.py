#!/usr/bin/env python3
"""
Create a shareable HTML report from model comparison results.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

def create_html_report(json_file: str, output_file: str = None):
    """Create an HTML report from JSON results."""
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if output_file is None:
        output_file = json_file.replace('.json', '_report.html')
    
    # Extract key information
    model1_path = data.get('model1_path', 'Unknown')
    model2_path = data.get('model2_path', 'Unknown')
    analysis = data.get('analysis', {})
    results = data.get('results', [])
    
    # Create HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px 15px;
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }}
        .winner {{
            background-color: #27ae60;
            font-size: 1.2em;
            padding: 15px 20px;
        }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .comparison-table th, .comparison-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .comparison-table th {{
            background-color: #34495e;
            color: white;
        }}
        .comparison-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .sample-comparison {{
            background-color: #f8f9fa;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #3498db;
            border-radius: 5px;
        }}
        .prompt {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .winner-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .winner-model1 {{
            background-color: #27ae60;
            color: white;
        }}
        .winner-model2 {{
            background-color: #e74c3c;
            color: white;
        }}
        .tie {{
            background-color: #f39c12;
            color: white;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Model Comparison Report</h1>
        
        <div class="summary-box">
            <h2>📊 Executive Summary</h2>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Evaluation Method:</strong> {data.get('evaluation_method', 'Local Heuristics')}</p>
            <p><strong>Total Comparisons:</strong> {analysis.get('total_comparisons', 0)}</p>
        </div>

        <h2>🏆 Overall Results</h2>
        <div class="summary-box">
            <div class="metric">Model 1 Wins: {analysis.get('model1_wins', 0)} ({analysis.get('model1_win_rate', 0):.1%})</div>
            <div class="metric">Model 2 Wins: {analysis.get('model2_wins', 0)} ({analysis.get('model2_win_rate', 0):.1%})</div>
            <div class="metric">Ties: {analysis.get('ties', 0)} ({analysis.get('tie_rate', 0):.1%})</div>
            <br><br>
            <div class="metric winner">
                {'🏆 Model 1 Wins!' if analysis.get('model1_win_rate', 0) > analysis.get('model2_win_rate', 0) else '🏆 Model 2 Wins!' if analysis.get('model2_win_rate', 0) > analysis.get('model1_win_rate', 0) else '🤝 Tie!'}
            </div>
        </div>

        <h2>📈 Detailed Metrics</h2>
        <table class="comparison-table">
            <tr>
                <th>Metric</th>
                <th>Model 1 (Trained GRPO)</th>
                <th>Model 2 (Baseline)</th>
                <th>Difference</th>
            </tr>
            <tr>
                <td>Average Score</td>
                <td>{analysis.get('model1_avg_score', 0):.3f} ± {analysis.get('model1_std_score', 0):.3f}</td>
                <td>{analysis.get('model2_avg_score', 0):.3f} ± {analysis.get('model2_std_score', 0):.3f}</td>
                <td>{analysis.get('avg_score_difference', 0):.3f}</td>
            </tr>
            <tr>
                <td>Wins</td>
                <td>{analysis.get('model1_wins', 0)}</td>
                <td>{analysis.get('model2_wins', 0)}</td>
                <td>{analysis.get('model1_wins', 0) - analysis.get('model2_wins', 0)}</td>
            </tr>
        </table>

        <h2>🔍 Sample Comparisons</h2>
        <p>Showing first 10 comparisons from the evaluation:</p>
    """
    
    # Add sample comparisons
    for i, result in enumerate(results[:10]):
        winner_class = {
            'model1': 'winner-model1',
            'model2': 'winner-model2', 
            'tie': 'tie'
        }.get(result.get('winner', 'tie'), 'tie')
        
        winner_text = {
            'model1': 'Model 1 Wins',
            'model2': 'Model 2 Wins',
            'tie': 'Tie'
        }.get(result.get('winner', 'tie'), 'Tie')
        
        html_content += f"""
        <div class="sample-comparison">
            <div class="prompt">Prompt {i+1}: {result.get('prompt', '')[:100]}...</div>
            <p><strong>Scores:</strong> Model 1: {result.get('model1_score', 0):.3f}, Model 2: {result.get('model2_score', 0):.3f}</p>
            <p><strong>Winner:</strong> <span class="winner-badge {winner_class}">{winner_text}</span></p>
            <p><strong>Explanation:</strong> {result.get('explanation', 'No explanation provided')}</p>
        </div>
        """
    
    html_content += f"""
        <div class="footer">
            <p>Report generated automatically from model comparison results</p>
            <p>Model 1: {model1_path}</p>
            <p>Model 2: {model2_path}</p>
        </div>
    </div>
</body>
</html>
    """
    
    # Write HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ HTML report created: {output_file}")
    print(f"📊 You can now share this HTML file with your collaborator!")
    print(f"🌐 Open it in any web browser to view the results.")

def main():
    parser = argparse.ArgumentParser(description="Create HTML report from model comparison results")
    parser.add_argument("json_file", help="Path to the JSON results file")
    parser.add_argument("--output", "-o", help="Output HTML file path")
    
    args = parser.parse_args()
    
    create_html_report(args.json_file, args.output)

if __name__ == "__main__":
    main() 