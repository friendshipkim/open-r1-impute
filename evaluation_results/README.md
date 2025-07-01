# Model Comparison Analysis Results

This folder contains all analysis results and tools for comparing GRPO models.

## Folder Structure

```
analysis_results/
├── README.md                           # This file
├── analyze_grpo_results.py             # Main analysis script
├── create_analysis_visualization.py    # Visualization script (requires matplotlib)
├── GRPO_Analysis_Summary.md            # Human-readable analysis summary
├── grpo_analysis_report.txt            # Detailed text report
├── grpo_analysis_summary.json          # Structured data for further analysis
├── grpo_comparison_results/            # GRPO model comparison results
│   └── grpo_model_comparison_local.json
└── local_comparison_results/           # Other local comparison results
```

## Files Description

### Analysis Scripts
- **`analyze_grpo_results.py`**: Main analysis script that categorizes prompts and analyzes model performance by different criteria
- **`create_analysis_visualization.py`**: Creates charts and visualizations from the analysis results (requires matplotlib)

### Results Files
- **`grpo_analysis_summary.json`**: Structured data containing category performance, length analysis, and complexity analysis
- **`grpo_analysis_report.txt`**: Detailed text report with all analysis results
- **`GRPO_Analysis_Summary.md`**: Human-readable summary with key insights and findings

### Data Directories
- **`grpo_comparison_results/`**: Contains the original GRPO model comparison JSON files
- **`local_comparison_results/`**: Contains other local comparison results

## Key Findings

### Overall Results
- **Total Comparisons**: 100 prompts
- **Oracle Model (GRPO-trained)**: 48 wins (48.0%)
- **Imputed Model (baseline)**: 52 wins (52.0%)
- **Overall Winner**: Imputed Model by 4 wins

### Oracle Model Strengths
- Business & Professional Content (75% win rate)
- Technical Programming (61.5% win rate)
- Structured Instructions (71.4% win rate)
- Long-form Content (54.2% win rate)

### Imputed Model Strengths
- Creative Writing (67.6% win rate)
- Travel & Location (83.3% win rate)
- Educational Explanations (60% win rate)
- Short-form Content (59.3% win rate)

## Usage

### Running the Analysis
```bash
cd analysis_results
python analyze_grpo_results.py
```

### Creating Visualizations
```bash
cd analysis_results
python create_analysis_visualization.py
```

### Viewing Results
- Read `GRPO_Analysis_Summary.md` for a comprehensive overview
- Check `grpo_analysis_report.txt` for detailed statistics
- Use `grpo_analysis_summary.json` for programmatic access to the data

## Model Selection Guide

### Choose Oracle Model for:
- Business reports and professional documents
- Technical programming and documentation
- Structured instructional content
- Long-form, detailed responses

### Choose Imputed Model for:
- Creative writing and storytelling
- Travel and location information
- Educational explanations
- Short, direct responses

## Notes

- The analysis uses a local heuristic-based scoring system
- Results are based on 100 validation prompts from the UltraChat dataset
- The GRPO training appears to optimize for structured, professional content rather than creative responses 