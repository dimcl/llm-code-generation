"""
Analisi Approcci Algoritmici su TUTTI i Risultati
Classifica approcci (iterativo, ricorsivo, funzionale, etc.) per TUTTE le soluzioni corrette.

Questo script analizza tutti i 1200 esperimenti e calcola le percentuali degli approcci
algoritmici utilizzati da ogni modello.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import pandas as pd


def load_results(file_path: str) -> List[dict]:
    """Carica risultati completi da JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']


def has_recursion(code: str) -> bool:
    """
    Rileva ricorsione nel codice
    
    Cerca pattern: def nome(...): ... nome(...)
    """
    func_pattern = r'def\s+(\w+)\s*\([^)]*\):'
    funcs = re.findall(func_pattern, code)
    
    for func_name in funcs:
        # Dividi il codice per isolare il corpo della funzione
        parts = code.split(f'def {func_name}')
        if len(parts) > 1:
            func_body = parts[1]
            # Cerca chiamate ricorsive alla stessa funzione
            if re.search(rf'\b{func_name}\s*\(', func_body):
                return True
    
    return False


def count_code_features(code: str) -> Dict:
    """
    Conta feature caratteristiche del codice
    
    Returns:
        Dict con conteggi di loops, comprehensions, conditionals, etc.
    """
    features = {
        'loops': len(re.findall(r'\b(for|while)\b', code)),
        'comprehensions': len(re.findall(r'\[.+for\s+.+in\s+.+\]|\{.+for\s+.+in\s+.+\}|\(.+for\s+.+in\s+.+\)', code)),
        'conditionals': len(re.findall(r'\bif\b', code)),
        'recursion': has_recursion(code),
        'lambda': len(re.findall(r'\blambda\b', code)),
        'map_filter': len(re.findall(r'\b(map|filter|reduce)\b', code))
    }
    return features


def classify_algorithmic_approach(code: str) -> str:
    """
    Classifica l'approccio algoritmico principale del codice
    
    Categorie:
    - recursive: Usa ricorsione
    - functional: Usa comprehensions/lambda/map/filter senza loops
    - mixed: Usa comprehensions + loops
    - iterative: Usa loops espliciti (for/while)
    - conditional: Principalmente logica condizionale (if/else)
    - direct: Calcolo diretto senza strutture di controllo complesse
    
    Returns:
        Stringa con categoria dell'approccio
    """
    if not code or len(code.strip()) < 10:
        return "empty"
    
    features = count_code_features(code)
    
    # Priorità: ricorsione > funzionale > misto > iterativo > condizionale > diretto
    
    # 1. Ricorsivo
    if features['recursion']:
        return "recursive"
    
    # 2. Funzionale (comprehensions, lambda, map/filter SENZA loops)
    has_functional = (features['comprehensions'] > 0 or 
                     features['lambda'] > 0 or 
                     features['map_filter'] > 0)
    
    if has_functional and features['loops'] == 0:
        return "functional"
    
    # 3. Misto (comprehensions + loops)
    if has_functional and features['loops'] > 0:
        return "mixed"
    
    # 4. Iterativo (loops espliciti)
    if features['loops'] > 0:
        return "iterative"
    
    # 5. Condizionale (if/else dominante)
    if features['conditionals'] > 0:
        return "conditional"
    
    # 6. Diretto (calcolo semplice)
    return "direct"


def analyze_all_approaches(results: List[dict]) -> Dict:
    """
    Analizza approcci algoritmici per tutti i risultati di successo
    
    Returns:
        Dict con conteggi e percentuali per modello
    """
    approaches_count = defaultdict(lambda: defaultdict(int))
    total_success = defaultdict(int)
    
    # Analizza ogni risultato
    for result in results:
        model = result['model']
        exec_result = result.get('execution_result', {})
        
        # Solo codice che ha PASSATO i test
        if exec_result.get('success') and exec_result.get('passed'):
            code = result.get('generated_code', '')
            
            if code:
                approach = classify_algorithmic_approach(code)
                approaches_count[model][approach] += 1
                total_success[model] += 1
    
    # Calcola statistiche
    stats = {}
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        total = total_success[model]
        
        stats[model] = {
            'total_analyzed': total,
            'approaches': {}
        }
        
        # Calcola percentuali per ogni approccio
        for approach in ['recursive', 'functional', 'mixed', 'iterative', 'conditional', 'direct', 'empty']:
            count = approaches_count[model][approach]
            percentage = (count / total * 100) if total > 0 else 0
            
            if count > 0:  # Includi solo approcci effettivamente utilizzati
                stats[model]['approaches'][approach] = {
                    'count': count,
                    'percentage': round(percentage, 2)
                }
        
        # Aggregazioni per tesi (mappatura categorie)
        # Iterativo esplicito = iterative
        # Funzionale/conciso = functional + mixed
        # Ricorsivo = recursive
        
        iterative_count = approaches_count[model]['iterative']
        functional_count = approaches_count[model]['functional'] + approaches_count[model]['mixed']
        recursive_count = approaches_count[model]['recursive']
        
        stats[model]['aggregated'] = {
            'iterative_explicit': {
                'count': iterative_count,
                'percentage': round((iterative_count / total * 100) if total > 0 else 0, 1)
            },
            'functional_concise': {
                'count': functional_count,
                'percentage': round((functional_count / total * 100) if total > 0 else 0, 1)
            },
            'recursive': {
                'count': recursive_count,
                'percentage': round((recursive_count / total * 100) if total > 0 else 0, 1)
            }
        }
    
    return stats


def generate_approaches_table(stats: Dict) -> pd.DataFrame:
    """Genera tabella riassuntiva approcci (formato tesi)"""
    rows = []
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats:
            agg = stats[model]['aggregated']
            rows.append({
                'Modello': model.upper(),
                'Totale Analizzati': stats[model]['total_analyzed'],
                'Iterativo Esplicito (%)': agg['iterative_explicit']['percentage'],
                'Funzionale/Conciso (%)': agg['functional_concise']['percentage'],
                'Ricorsivo (%)': agg['recursive']['percentage']
            })
    
    return pd.DataFrame(rows)


def generate_detailed_approaches_table(stats: Dict) -> pd.DataFrame:
    """Genera tabella dettagliata con tutte le categorie"""
    rows = []
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats:
            s = stats[model]
            
            row = {
                'Modello': model.upper(),
                'Totale': s['total_analyzed']
            }
            
            # Aggiungi ogni approccio
            for approach in ['recursive', 'functional', 'mixed', 'iterative', 'conditional', 'direct']:
                if approach in s['approaches']:
                    row[approach.capitalize()] = f"{s['approaches'][approach]['count']} ({s['approaches'][approach]['percentage']:.1f}%)"
                else:
                    row[approach.capitalize()] = "0 (0.0%)"
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def print_summary(stats: Dict):
    """Stampa sommario leggibile"""
    print("\n" + "="*80)
    print("RIEPILOGO APPROCCI ALGORITMICI (per Tesi)")
    print("="*80)
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats:
            agg = stats[model]['aggregated']
            total = stats[model]['total_analyzed']
            
            print(f"\n {model.upper()} (totale: {total} soluzioni corrette):")
            print(f"   Iterativo esplicito:  {agg['iterative_explicit']['percentage']:5.1f}% ({agg['iterative_explicit']['count']} soluzioni)")
            print(f"   Funzionale/conciso:   {agg['functional_concise']['percentage']:5.1f}% ({agg['functional_concise']['count']} soluzioni)")
            print(f"   Ricorsivo:            {agg['recursive']['percentage']:5.1f}% ({agg['recursive']['count']} soluzioni)")
    
    print("\n" + "="*80)
    print("DETTAGLIO COMPLETO APPROCCI")
    print("="*80)
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats:
            s = stats[model]
            print(f"\n{model.upper()} (totale: {s['total_analyzed']}):")
            
            for approach, data in sorted(s['approaches'].items()):
                print(f"  {approach:15} {data['count']:3} ({data['percentage']:5.1f}%)")


def save_to_csv(df: pd.DataFrame, output_path: str):
    """Salva DataFrame in CSV"""
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f" CSV salvato: {output_path}")


def save_to_latex(df: pd.DataFrame, output_path: str, caption: str, label: str):
    """Salva DataFrame in formato LaTeX"""
    latex = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        position='htbp',
        escape=False
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    
    print(f" LaTeX salvato: {output_path}")


def main():
    """Analizza approcci algoritmici su tutti i risultati"""
    print("="*80)
    print(" ANALISI APPROCCI ALGORITMICI - DATASET COMPLETO")
    print("="*80)
    print("\nAnalizza TUTTI i 1200 esperimenti per classificare gli approcci")
    
    # Percorso file risultati
    results_file = "results/raw_outputs/full_experiments/full_experiments_gemini25_20251031_135614.json"
    
    # Carica dati
    print("\n Caricamento risultati completi...")
    results = load_results(results_file)
    print(f"   Totale esperimenti: {len(results)}")
    
    # Conta successi
    successes = defaultdict(int)
    for r in results:
        if r.get('execution_result', {}).get('passed', False):
            successes[r['model']] += 1
    
    print(f"\n   Soluzioni corrette da analizzare:")
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        print(f"      {model.upper()}: {successes[model]}")
    
    # Crea cartelle output
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Analisi approcci
    print(f"\n{'='*80}")
    print(" CLASSIFICAZIONE APPROCCI")
    print("="*80)
    print("\n Analisi in corso...")
    
    stats = analyze_all_approaches(results)
    
    # Salva JSON
    json_path = output_dir / "algorithmic_approaches.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n Statistiche salvate: {json_path}")
    
    # Stampa sommario
    print_summary(stats)
    
    # Tabella per tesi (aggregata)
    print("\n" + "="*80)
    print(" TABELLA APPROCCI (formato Tesi)")
    print("="*80)
    
    df_thesis = generate_approaches_table(stats)
    print("\n" + df_thesis.to_string(index=False))
    
    csv_path = tables_dir / "algorithmic_approaches_thesis.csv"
    latex_path = tables_dir / "algorithmic_approaches_thesis.tex"
    
    save_to_csv(df_thesis, csv_path)
    save_to_latex(
        df_thesis,
        latex_path,
        caption="Distribuzione approcci algoritmici per modello",
        label="tab:algorithmic_approaches"
    )
    
    # Tabella dettagliata
    print("\n" + "="*80)
    print(" TABELLA DETTAGLIATA (tutte le categorie)")
    print("="*80)
    
    df_detailed = generate_detailed_approaches_table(stats)
    print("\n" + df_detailed.to_string(index=False))
    
    csv_path = tables_dir / "algorithmic_approaches_detailed.csv"
    latex_path = tables_dir / "algorithmic_approaches_detailed.tex"
    
    save_to_csv(df_detailed, csv_path)
    save_to_latex(
        df_detailed,
        latex_path,
        caption="Distribuzione dettagliata approcci algoritmici",
        label="tab:algorithmic_approaches_detailed"
    )
    
    print(f"\n{'='*80}")
    print(" ANALISI COMPLETATA!")
    print(f"{'='*80}")
    print(f"\n  File generati:")
    print(f"   - JSON: results/metrics/algorithmic_approaches.json")
    print(f"   - Tabella tesi (CSV): results/tables/algorithmic_approaches_thesis.csv")
    print(f"   - Tabella tesi (LaTeX): results/tables/algorithmic_approaches_thesis.tex")
    print(f"   - Tabella dettagliata (CSV): results/tables/algorithmic_approaches_detailed.csv")
    print(f"   - Tabella dettagliata (LaTeX): results/tables/algorithmic_approaches_detailed.tex")


if __name__ == "__main__":
    main()
