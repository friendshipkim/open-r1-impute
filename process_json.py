#!/usr/bin/env python3
"""
Script to clean JSON files by removing specified fields.
"""

import json
import os

def clean_json_file(input_file, output_file):
    """Clean JSON file by removing specified fields."""
    print(f"Processing {input_file}...")
    
    # Read the original JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Remove top-level fields
    fields_to_remove = ["judge_model", "api_type", "analysis"]
    for field in fields_to_remove:
        if field in data:
            del data[field]
            print(f"Removed top-level field: {field}")
    
    # Remove fields from each result
    if "results" in data:
        for i, result in enumerate(data["results"]):
            result_fields_to_remove = ["winner", "explanation"]
            for field in result_fields_to_remove:
                if field in result:
                    del result[field]
            print(f"Cleaned result {i+1}")
    
    # Write the cleaned data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned data saved to {output_file}")

def main():
    # Create the cleaned files
    base_dir = "evaluation_results"
    
    # File 1: oracle_lr2e-6-warmup0.05_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples_loop1
    input_file1 = os.path.join(base_dir, "oracle_lr2e-6-warmup0.05_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples_loop1.json")
    output_file1 = os.path.join(base_dir, "oracle_lr2e-6-warmup0.05_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples_loop1", "oracle_lr2e-6-warmup0.05_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples_loop1_cleaned.json")
    
    # File 2: oracle_lr2e-6-warmup0.05_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples_claude_loop1
    input_file2 = os.path.join(base_dir, "oracle_lr2e-6-warmup0.05_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples_claude_loop1.json")
    output_file2 = os.path.join(base_dir, "oracle_lr2e-6-warmup0.05_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples_claude_loop1", "oracle_lr2e-6-warmup0.05_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples_claude_loop1_cleaned.json")
    
    # File 3: untrained_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples
    input_file3 = os.path.join(base_dir, "untrained_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples.json")
    output_file3 = os.path.join(base_dir, "untrained_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples", "untrained_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples.json")
    
    # Process all files
    if os.path.exists(input_file1):
        clean_json_file(input_file1, output_file1)
    else:
        print(f"Input file not found: {input_file1}")
    
    if os.path.exists(input_file2):
        clean_json_file(input_file2, output_file2)
    else:
        print(f"Input file not found: {input_file2}")
    
    if os.path.exists(input_file3):
        clean_json_file(input_file3, output_file3)
    else:
        print(f"Input file not found: {input_file3}")

if __name__ == "__main__":
    main() 