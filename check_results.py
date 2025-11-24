"""
Verifica risultati nel file JSON - CORRETTO
Controlla execution_result.passed invece di success (che indica solo generazione ok)
"""

import json

r = json.load(open('results/raw_outputs/full_experiments/full_experiments_gemini25_20251031_135614.json'))

print("="*70)
print("VERIFICA RISULTATI - EXECUTION (test passed)")
print("="*70)

print(f"\nTotal results: {len(r['results'])}")
print(f"Tests PASSED: {sum(1 for x in r['results'] if x.get('execution_result', {}).get('passed', False))}")
print(f"Tests FAILED: {sum(1 for x in r['results'] if not x.get('execution_result', {}).get('passed', False))}")

print(f"\n{'='*70}")
print("By model (all 5 attempts):")
print("="*70)
for model in ['gpt', 'gemini', 'llama', 'qwen']:
    total = sum(1 for x in r['results'] if x['model'] == model)
    passed = sum(1 for x in r['results'] if x['model'] == model and x.get('execution_result', {}).get('passed', False))
    print(f"  {model.upper():8s}: {passed:3d}/{total:3d} = {passed/total*100:5.1f}%")

print(f"\n{'='*70}")
print("Pass@1 (first attempt only):")
print("="*70)
for model in ['gpt', 'gemini', 'llama', 'qwen']:
    first = [x for x in r['results'] if x['model'] == model and x['attempt'] == 1]
    passed = sum(1 for x in first if x.get('execution_result', {}).get('passed', False))
    print(f"  {model.upper():8s}: {passed:2d}/{len(first):2d} = {passed/len(first)*100:5.1f}%")
