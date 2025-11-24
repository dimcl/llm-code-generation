"""
Esperimenti completi su 60 problemi
Genera 1200 soluzioni: 60 problemi × 4 modelli × 5 tentativi
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Aggiungi src al path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from experiments.code_generation import CodeGenerator
from experiments.code_execution import CodeExecutor


def load_all_problems():
    """Carica tutti i 60 problemi selezionati"""
    json_path = Path(__file__).parent / 'src' / 'data' / 'selected_problems' / 'selected_problems.json'
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['problems'], data['metadata']


def run_full_experiments():
    """
    Esegue esperimenti completi su tutti i 60 problemi
    
    Configurazione:
    - 60 problemi (20 easy + 20 medium + 20 hard)
    - 4 modelli (Gemini, Llama, Qwen, GPT-4o-mini)
    - 5 tentativi per problema per Pass@k
    - Totale: 1200 generazioni
    """
    
    print("\n" + "="*70)
    print("ESPERIMENTI COMPLETI - 60 PROBLEMI")
    print("="*70)
    
    # Carica problemi
    all_problems, metadata = load_all_problems()
    
    print(f"\n CONFIGURAZIONE:")
    print(f"   Problemi totali: {len(all_problems)}")
    print(f"   - Easy: {metadata['distribution']['easy']}")
    print(f"   - Medium: {metadata['distribution']['medium']}")
    print(f"   - Hard: {metadata['distribution']['hard']}")
    print(f"   Modelli: Gemini, Llama, Qwen, GPT-4o-mini")
    print(f"   Tentativi per problema: 5 (per Pass@k)")
    print(f"   Generazioni totali: {len(all_problems)} × 4 × 5 = {len(all_problems)*4*5}")
    
    # Stima tempo e costo
    print(f"\n STIMA:")
    print(f"   Tempo (sequenziale): ~1.9 ore")
    print(f"   Costo totale: ~$0.24")
    print(f"   Rate limiting: Gemini 1s, Groq 2s tra richieste")
    
    # Inizializza generatore
    output_dir = Path("results/raw_outputs/full_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = CodeGenerator(
        output_dir=str(output_dir),
        temperature=0.2,
        max_tokens=1024,
        max_retries=3
    )
    
    # Conferma
    print(f"\n   IMPORTANTE:")
    print(f"   - I risultati saranno salvati in: {output_dir}")
    print(f"   - Salvataggio automatico ogni 10 problemi")
    print(f"   - Interrompibile con Ctrl+C (risultati parziali salvati)")
    
    # Auto-conferma per esecuzione automatica
    print(f"\n Avvio esperimenti completi...")
    print(f"   (Per conferma manuale, commentare la riga auto_confirm=True)")
    
    # ========================================================================
    # FASE 1: GENERAZIONE CODICE
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(" FASE 1: GENERAZIONE CODICE")
    print("="*70)
    
    try:
        # Genera codice per tutti i problemi
        batch_stats = generator.generate_batch(
            problems=all_problems,
            models=['gemini', 'llama', 'qwen', 'gpt'],
            num_attempts=5,
            prompt_type='basic',
            save_interval=10
        )
        
    except KeyboardInterrupt:
        print("\n\n   Interruzione rilevata!")
        print("  Risultati parziali salvati")
        sys.exit(1)
    except Exception as e:
        print(f"\n  Errore durante la generazione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # FASE 2: ESECUZIONE E TEST
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(" FASE 2: ESECUZIONE E TEST")
    print("="*70)
    
    # Carica risultati generati - usa il pattern corretto
    # Il file è già stato salvato da generate_batch, lo ricarichiamo
    latest_results = sorted(output_dir.glob("results_*.json"))[-1]
    
    with open(latest_results, 'r', encoding='utf-8') as f:
        data = json.load(f)
        all_results = data['results']
    
    print(f" Caricati {len(all_results)} risultati da: {latest_results.name}")
    
    # Crea dizionario problem_id -> problem
    problems_dict = {p['id']: p for p in all_problems}
    
    # Inizializza executor
    executor = CodeExecutor(timeout=10)
    
    try:
        # Esegui tutti i codici generati
        executed_results = executor.batch_execute(all_results, problems_dict)
        
    except KeyboardInterrupt:
        print("\n\n  Interruzione rilevata durante esecuzione!")
        sys.exit(1)
    except Exception as e:
        print(f"\n Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # FASE 3: ANALISI E SALVATAGGIO
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(" FASE 3: ANALISI RISULTATI")
    print("="*70)
    
    # Calcola statistiche per modello
    model_stats = {
        'gemini': {'total': 0, 'passed': 0, 'failed': 0, 'pass_rate': 0.0},
        'llama': {'total': 0, 'passed': 0, 'failed': 0, 'pass_rate': 0.0},
        'qwen': {'total': 0, 'passed': 0, 'failed': 0, 'pass_rate': 0.0},
        'gpt': {'total': 0, 'passed': 0, 'failed': 0, 'pass_rate': 0.0}
    }
    
    # Statistiche per difficoltà
    difficulty_stats = {
        'easy': {'total': 0, 'passed': 0},
        'medium': {'total': 0, 'passed': 0},
        'hard': {'total': 0, 'passed': 0}
    }
    
    # Conta risultati
    for result in executed_results:
        model = result['model']
        
        if model in model_stats:
            model_stats[model]['total'] += 1
            
            if result.get('execution_result', {}).get('passed', False):
                model_stats[model]['passed'] += 1
            else:
                model_stats[model]['failed'] += 1
    
    # Calcola pass rate
    for model in model_stats:
        if model_stats[model]['total'] > 0:
            model_stats[model]['pass_rate'] = (
                model_stats[model]['passed'] / model_stats[model]['total'] * 100
            )
    
    # Stampa risultati
    print(f"\n{'Modello':<20} {'Total':<10} {'Passed':<10} {'Pass %':<10}")
    print("-" * 70)
    for model, stats in model_stats.items():
        print(f"{model.upper():<20} {stats['total']:<10} {stats['passed']:<10} {stats['pass_rate']:.1f}%")
    print("-" * 70)
    
    total_generations = sum(s['total'] for s in model_stats.values())
    total_passed = sum(s['passed'] for s in model_stats.values())
    overall_pass_rate = (total_passed / total_generations * 100) if total_generations > 0 else 0
    
    print(f"{'TOTALE':<20} {total_generations:<10} {total_passed:<10} {overall_pass_rate:.1f}%")
    
    # Salva risultati finali
    final_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_problems': len(all_problems),
            'total_generations': total_generations,
            'models': ['gemini', 'llama', 'qwen', 'gpt'],
            'attempts_per_problem': 5,
            'prompt_type': 'basic'
        },
        'statistics': {
            'overall_pass_rate': overall_pass_rate,
            'total_passed': total_passed,
            'total_failed': total_generations - total_passed,
            'by_model': model_stats,
            'generation_stats': batch_stats
        },
        'results': executed_results
    }
    
    final_file = output_dir / f"full_experiments_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n Risultati salvati: {final_file}")
    
    # Riepilogo finale
    print(f"\n{'='*70}")
    print(" ESPERIMENTI COMPLETI TERMINATI")
    print("="*70)
    print(f" Generazioni totali: {total_generations}")
    print(f" Test passati: {total_passed} ({overall_pass_rate:.1f}%)")
    print(f" Test falliti: {total_generations - total_passed}")
    print(f"⏱  Tempo totale: {batch_stats.get('total_latency', 0)/60:.1f} minuti")
    print(f" Costo totale: ${batch_stats.get('total_cost', 0):.4f}")
    print(f" File risultati: {final_file.name}")
    print(f"{'='*70}\n")
    
    return {
        'pass_rate': overall_pass_rate,
        'total': total_generations,
        'passed': total_passed,
        'model_stats': model_stats
    }


if __name__ == "__main__":
    try:
        stats = run_full_experiments()
        
        # Successo se pass rate >= 40% (considerando errori normali)
        if stats['pass_rate'] >= 40:
            print(" Esperimenti completati con successo!")
            sys.exit(0)
        else:
            print("  Esperimenti completati ma pass rate basso")
            sys.exit(0)  # Non fallire, i dati sono utili comunque
            
    except Exception as e:
        print(f"\n Errore critico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
