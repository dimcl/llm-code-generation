"""
Ispeziona la struttura del file dei risultati
"""

import json

# Carica file
with open('results/raw_outputs/full_experiments/full_experiments_gemini25_20251031_135614.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*70)
print("STRUTTURA FILE JSON")
print("="*70)

# Top level keys
print(f"\nTop-level keys: {list(data.keys())}")

# Results
results = data['results']
print(f"\nTotal results: {len(results)}")

# Esamina primo risultato
sample = results[0]
print(f"\nKeys nel primo risultato:")
for key in sample.keys():
    value = sample[key]
    if isinstance(value, dict):
        print(f"  - {key}: dict con keys {list(value.keys())}")
    elif isinstance(value, str):
        print(f"  - {key}: str (len={len(value)})")
    else:
        print(f"  - {key}: {type(value).__name__} = {value}")

# Verifica execution_result
print(f"\nHa execution_result? {'execution_result' in sample}")

if 'execution_result' in sample:
    exec_result = sample['execution_result']
    print(f"\nexecution_result keys: {list(exec_result.keys())}")
    print(f"  - passed: {exec_result.get('passed')}")
    print(f"  - error: {exec_result.get('error', 'None')[:50] if exec_result.get('error') else 'None'}")
    print(f"  - error_type: {exec_result.get('error_type')}")

# Conta passed vs failed
print(f"\n{'='*70}")
print("STATISTICHE")
print("="*70)

total_with_exec = sum(1 for r in results if 'execution_result' in r)
print(f"Risultati con execution_result: {total_with_exec}/{len(results)}")

if total_with_exec > 0:
    passed_count = sum(1 for r in results if r.get('execution_result', {}).get('passed', False))
    failed_count = total_with_exec - passed_count
    
    print(f"  - Passed: {passed_count}")
    print(f"  - Failed: {failed_count}")
    print(f"  - Pass rate: {passed_count/total_with_exec*100:.1f}%")
    
    # Per modello
    print(f"\nPer modello:")
    for model in ['gemini', 'llama', 'qwen', 'gpt']:
        model_results = [r for r in results if r['model'] == model]
        model_passed = sum(1 for r in model_results if r.get('execution_result', {}).get('passed', False))
        print(f"  {model:8s}: {model_passed}/{len(model_results)} = {model_passed/len(model_results)*100:.1f}%")
else:
    print("NESSUN execution_result trovato!")
    print("\nIl file contiene solo generation results, senza execution!")
