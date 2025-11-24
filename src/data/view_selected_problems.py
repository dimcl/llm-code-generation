"""
Visualizza riepilogo dei 60 problemi selezionati per difficoltà e categoria.
"""
import json
from pathlib import Path
from collections import defaultdict


def main():
    # Carica problemi selezionati
    json_path = Path(__file__).parent / 'selected_problems' / 'selected_problems.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problems = data['problems']
    metadata = data['metadata']
    
    print("\n" + "=" * 80)
    print(" RIEPILOGO 60 PROBLEMI SELEZIONATI PER LA TESI")
    print("=" * 80)
    
    print(f"\n METADATA:")
    print(f"   Totale problemi: {metadata['total_problems']}")
    print(f"   Data selezione: {metadata['selection_date']}")
    
    print(f"\n DISTRIBUZIONE DIFFICOLTÀ:")
    for diff, count in metadata['distribution'].items():
        percentage = (count / metadata['total_problems']) * 100
        print(f"   {diff.upper()}: {count} ({percentage:.1f}%)")
    
    print(f"\n DISTRIBUZIONE ARGOMENTI:")
    for topic, count in sorted(metadata['topics'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / metadata['total_problems']) * 100
        print(f"   {topic}: {count} ({percentage:.1f}%)")
    
    print(f"\n DISTRIBUZIONE SOURCE:")
    for source, count in metadata['sources'].items():
        percentage = (count / metadata['total_problems']) * 100
        print(f"   {source}: {count} ({percentage:.1f}%)")
    
    # Raggruppa per difficoltà
    by_difficulty = defaultdict(list)
    for p in problems:
        by_difficulty[p['difficulty']].append(p)
    
    # Mostra esempi per ogni difficoltà
    print("\n" + "=" * 80)
    print(" ESEMPI PER DIFFICOLTÀ (5 per categoria)")
    print("=" * 80)
    
    for difficulty in ['easy', 'medium', 'hard']:
        probs = by_difficulty[difficulty]
        
        print(f"\n{'='*80}")
        print(f"🔹 {difficulty.upper()} ({len(probs)} problemi)")
        print(f"{'='*80}")
        
        # Mostra primi 5
        for i, p in enumerate(probs[:5], 1):
            print(f"\n{i}. {p['id']} (source: {p['source']}, categoria: {p['category']})")
            print(f"   Task ID: {p['task_id']}")
            
            # Mostra prompt (prime 2 righe)
            prompt_lines = p['prompt'].strip().split('\n')[:3]
            for line in prompt_lines:
                if line.strip():
                    print(f"   {line[:76]}")
            
            if len(prompt_lines) > 3:
                print(f"   ...")
    
    # Distribuzione per categoria e difficoltà
    print(f"\n" + "=" * 80)
    print(" MATRICE CATEGORIA × DIFFICOLTÀ")
    print("=" * 80)
    
    matrix = defaultdict(lambda: defaultdict(int))
    for p in problems:
        matrix[p['category']][p['difficulty']] += 1
    
    # Header
    print(f"\n{'Categoria':<20} {'Easy':<8} {'Medium':<8} {'Hard':<8} {'Totale':<8}")
    print("-" * 60)
    
    # Righe
    for category in sorted(matrix.keys()):
        easy = matrix[category]['easy']
        medium = matrix[category]['medium']
        hard = matrix[category]['hard']
        total = easy + medium + hard
        print(f"{category:<20} {easy:<8} {medium:<8} {hard:<8} {total:<8}")
    
    print("-" * 60)
    
    # Totali
    total_easy = sum(matrix[cat]['easy'] for cat in matrix)
    total_medium = sum(matrix[cat]['medium'] for cat in matrix)
    total_hard = sum(matrix[cat]['hard'] for cat in matrix)
    total_all = total_easy + total_medium + total_hard
    
    print(f"{'TOTALE':<20} {total_easy:<8} {total_medium:<8} {total_hard:<8} {total_all:<8}")
    
    # Stima costi
    print(f"\n" + "=" * 80)
    print(" STIMA COSTI ESPERIMENTI")
    print("=" * 80)
    
    avg_tokens_per_problem = sum(p['estimated_tokens'] for p in problems) / len(problems)
    
    print(f"\nToken stimati per problema: ~{avg_tokens_per_problem:.0f}")
    print(f"Generazioni totali: 60 problemi × 4 modelli × 5 tentativi = 1200")
    print(f"Token totali stimati: ~{avg_tokens_per_problem * 1200:.0f}")
    
    # Costi per modello (dalle metriche test)
    costs = {
        'Gemini 2.0': 0.000287,
        'Llama 3.1': 0.000063,
        'Qwen 3': 0.000171,
        'GPT-4o-mini': 0.000280
    }
    
    print(f"\n Costi per modello (300 generazioni ciascuno):")
    total_cost = 0
    for model, cost_per_gen in costs.items():
        model_cost = cost_per_gen * 300
        total_cost += model_cost
        print(f"   {model:<15}: ${model_cost:.4f}")
    
    print(f"\n COSTO TOTALE STIMATO: ${total_cost:.2f}")
    print(f"   (Coperto dai crediti Azure: $100 disponibili)")
    
    print(f"\n" + "=" * 80)
    print(" SELEZIONE COMPLETATA - PRONTO PER IMPLEMENTARE PIPELINE")
    print("=" * 80)
    print(f"\nFile: {json_path}")
    print(f"Dimensione: {json_path.stat().st_size / 1024:.1f} KB")
    print(f"\n Prossimo step: Implementare prompt_templates.py e code_generation.py")

if __name__ == "__main__":
    main()
