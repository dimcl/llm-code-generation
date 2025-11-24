"""
Pilot test della pipeline completa
Testa 5 problemi facili con tutti i 4 modelli (20 generazioni totali)
"""

import json
import sys
from pathlib import Path

# Aggiungi src al path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from experiments.code_generation import CodeGenerator
from experiments.code_execution import CodeExecutor


def load_problems():
    """Carica i problemi selezionati"""
    problems_file = Path(__file__).parent / "src" / "data" / "selected_problems" / "selected_problems.json"
    
    with open(problems_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['problems']


def run_pilot_test():
    """Esegue pilot test su 5 problemi easy"""
    
    print("\n" + "="*70)
    print(" PILOT TEST - Pipeline Completa")
    print("="*70)
    
    # Carica problemi
    all_problems = load_problems()
    
    # Filtra solo problemi easy
    easy_problems = [p for p in all_problems if p['difficulty'] == 'easy']
    
    # Seleziona primi 5 problemi easy
    test_problems = easy_problems[:5]
    
    print(f"\n Problemi selezionati per test:")
    for i, p in enumerate(test_problems, 1):
        print(f"   {i}. {p['id']} - {p['category']} ({p['source']})")
    
    print(f"\n Modelli: Gemini, Llama, Qwen, GPT-4o-mini")
    print(f" Tentativi: 1 per modello (totale: 5 × 4 = 20 generazioni)")
    print(f" Prompt: basic template")
    
    # Inizializza generatore
    generator = CodeGenerator(
        output_dir="results/raw_outputs/pilot_test",
        temperature=0.2,
        max_tokens=1024
    )
    
    # Genera codice
    print(f"\n{'='*70}")
    print(" FASE 1: GENERAZIONE CODICE")
    print("="*70)
    
    all_results = []
    
    for problem in test_problems:
        results = generator.generate_for_problem(
            problem=problem,
            models=['gemini', 'llama', 'qwen', 'gpt'],
            num_attempts=1,  # Solo 1 tentativo per pilot test
            prompt_type="basic"
        )
        all_results.extend(results)
    
    # Esegui codice generato
    print(f"\n{'='*70}")
    print(" FASE 2: ESECUZIONE E TEST")
    print("="*70)
    
    executor = CodeExecutor(timeout=10)
    
    # Crea dizionario problem_id -> problem
    problems_dict = {p['id']: p for p in test_problems}
    
    # Esegui tutti i codici generati
    executed_results = executor.batch_execute(all_results, problems_dict)
    
    # Analizza risultati
    print(f"\n{'='*70}")
    print(" ANALISI RISULTATI")
    print("="*70)
    
    # Statistiche per modello
    model_stats = {
        'gemini': {'total': 0, 'passed': 0, 'syntax_errors': 0, 'runtime_errors': 0},
        'llama': {'total': 0, 'passed': 0, 'syntax_errors': 0, 'runtime_errors': 0},
        'qwen': {'total': 0, 'passed': 0, 'syntax_errors': 0, 'runtime_errors': 0},
        'gpt': {'total': 0, 'passed': 0, 'syntax_errors': 0, 'runtime_errors': 0}
    }
    
    for result in executed_results:
        model = result['model']
        model_stats[model]['total'] += 1
        
        if result.get('execution_result', {}).get('passed', False):
            model_stats[model]['passed'] += 1
        else:
            error_type = result.get('execution_result', {}).get('error_type', 'unknown')
            if error_type == 'syntax_error':
                model_stats[model]['syntax_errors'] += 1
            else:
                model_stats[model]['runtime_errors'] += 1
    
    print(f"\n{'Modello':<15} {'Total':<10} {'Passed':<10} {'Pass %':<10} {'Syntax':<10} {'Runtime':<10}")
    print("-" * 70)
    
    for model in ['gemini', 'llama', 'qwen', 'gpt']:
        stats = model_stats[model]
        pass_rate = stats['passed'] / max(stats['total'], 1) * 100
        print(f"{model.upper():<15} {stats['total']:<10} {stats['passed']:<10} {pass_rate:<10.1f} {stats['syntax_errors']:<10} {stats['runtime_errors']:<10}")
    
    # Totale
    total = sum(s['total'] for s in model_stats.values())
    total_passed = sum(s['passed'] for s in model_stats.values())
    total_pass_rate = total_passed / max(total, 1) * 100
    
    print("-" * 70)
    print(f"{'TOTALE':<15} {total:<10} {total_passed:<10} {total_pass_rate:<10.1f}")
    
    # Salva risultati
    output_file = Path("results/raw_outputs/pilot_test") / "pilot_test_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_config': {
                'num_problems': len(test_problems),
                'problems': [p['id'] for p in test_problems],
                'models': ['gemini', 'llama', 'qwen', 'gpt'],
                'attempts_per_model': 1,
                'prompt_type': 'basic'
            },
            'statistics': {
                'total_generations': total,
                'passed': total_passed,
                'pass_rate': total_pass_rate,
                'by_model': model_stats
            },
            'results': executed_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n Risultati salvati: {output_file}")
    
    # Mostra esempi di successi/fallimenti
    print(f"\n{'='*70}")
    print(" ESEMPI")
    print("="*70)
    
    successes = [r for r in executed_results if r.get('execution_result', {}).get('passed', False)]
    failures = [r for r in executed_results if not r.get('execution_result', {}).get('passed', False)]
    
    if successes:
        print(f"\n Esempio di successo ({len(successes)} totali):")
        example = successes[0]
        print(f"   Problema: {example['problem_id']}")
        print(f"   Modello: {example['model'].upper()}")
        print(f"   Tempo: {example.get('execution_result', {}).get('execution_time', 0):.3f}s")
        print(f"   Codice:\n")
        code_lines = example['generated_code'].split('\n')[:10]
        for line in code_lines:
            print(f"      {line}")
        if len(example['generated_code'].split('\n')) > 10:
            print("      ...")
    
    if failures:
        print(f"\n Esempio di fallimento ({len(failures)} totali):")
        example = failures[0]
        print(f"   Problema: {example['problem_id']}")
        print(f"   Modello: {example['model'].upper()}")
        print(f"   Errore: {example.get('execution_result', {}).get('error_type', 'unknown')}")
        error_msg = example.get('execution_result', {}).get('error', '')
        if error_msg:
            print(f"   Messaggio: {error_msg[:200]}")
    
    print(f"\n{'='*70}")
    print(" PILOT TEST COMPLETATO")
    print("="*70)
    
    return {
        'total': total,
        'passed': total_passed,
        'pass_rate': total_pass_rate,
        'model_stats': model_stats
    }


if __name__ == "__main__":
    try:
        stats = run_pilot_test()
        
        # Ritorna exit code basato sul success rate
        if stats['pass_rate'] >= 50:
            print(f"\n Pilot test superato! Pass rate: {stats['pass_rate']:.1f}%")
            sys.exit(0)
        else:
            print(f"\n Pilot test parziale. Pass rate: {stats['pass_rate']:.1f}%")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n Errore durante pilot test: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
