import json
import os

path = os.path.join('results', 'attack_evaluation', 'attack_results.json')
with open(path, 'r') as f:
    data = json.load(f)

print("Keys:", data.keys())
if 'fgsm' in data:
    print("FGSM Keys:", data['fgsm'].keys())
    first_key = list(data['fgsm'].keys())[0]
    print(f"FGSM['{first_key}'] Keys:", data['fgsm'][first_key].keys())
