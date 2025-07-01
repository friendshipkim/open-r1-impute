import json

# Read the original JSON file
with open('evaluation_results/oracle_lr2e-6-warmup0.05_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples.json', 'r') as f:
    data = json.load(f)

# Remove judge_model and analysis from the top level
if 'judge_model' in data:
    del data['judge_model']
if 'analysis' in data:
    del data['analysis']

# Remove winner and explanation from each result
for result in data['results']:
    if 'winner' in result:
        del result['winner']
    if 'explanation' in result:
        del result['explanation']

# Write the modified data to a new file
output_filename = 'evaluation_results/oracle_lr2e-6-warmup0.05_vs_imputed_lr2e-6-warmup0.05-ps0.2-preps0.0-rho0_comparison_100samples_cleaned.json'
with open(output_filename, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Created cleaned version: {output_filename}") 