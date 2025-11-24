"""
Classificazione e analisi errori nelle generazioni di codice.

Categorizza gli errori in:
- Syntax errors: Codice non valido sintatticamente
- Name errors: Variabili/funzioni non definite
- Type errors: Operazioni su tipi incompatibili
- Assertion errors: Test falliti (logica errata)
- Timeout errors: Esecuzione troppo lenta
- Other runtime errors: Altri errori di esecuzione
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import pandas as pd


def load_results(file_path: str) -> Tuple[dict, List[dict]]:
    """Carica risultati da JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['metadata'], data['results']


def classify_error(result: dict) -> str:
    """
    Classifica il tipo di errore basandosi su execution_result
    
    Returns:
        Categoria errore: 'success', 'syntax_error', 'name_error', 
                         'type_error', 'assertion_error', 'timeout', 
                         'other_runtime', 'no_code'
    """
    # Successo
    if result.get('execution_result', {}).get('passed', False):
        return 'success'
    
    # Nessun codice generato
    if not result.get('generated_code'):
        return 'no_code'
    
    # Analizza errore
    exec_result = result.get('execution_result', {})
    error = exec_result.get('error', '')
    error_type = exec_result.get('error_type', '')
    
    # Nessun errore ma test falliti
    if not error and not exec_result.get('passed', False):
        return 'assertion_error'
    
    # Classifica per error_type se disponibile
    if error_type:
        if 'syntax' in error_type.lower():
            return 'syntax_error'
        elif 'name' in error_type.lower():
            return 'name_error'
        elif 'type' in error_type.lower():
            return 'type_error'
        elif 'assertion' in error_type.lower():
            return 'assertion_error'
        elif 'timeout' in error_type.lower():
            return 'timeout'
        elif 'attribute' in error_type.lower():
            return 'attribute_error'
        elif 'index' in error_type.lower():
            return 'index_error'
        elif 'key' in error_type.lower():
            return 'key_error'
        elif 'value' in error_type.lower():
            return 'value_error'
        elif 'zero_division' in error_type.lower() or 'zerodivision' in error_type.lower():
            return 'zero_division_error'
    
    # Classifica per messaggio errore
    error_lower = error.lower()
    
    if 'syntaxerror' in error_lower or 'invalid syntax' in error_lower:
        return 'syntax_error'
    elif 'nameerror' in error_lower or 'is not defined' in error_lower:
        return 'name_error'
    elif 'typeerror' in error_lower:
        return 'type_error'
    elif 'assertionerror' in error_lower or 'assert' in error_lower:
        return 'assertion_error'
    elif 'timeout' in error_lower or 'timed out' in error_lower:
        return 'timeout'
    elif 'attributeerror' in error_lower:
        return 'attribute_error'
    elif 'indexerror' in error_lower:
        return 'index_error'
    elif 'keyerror' in error_lower:
        return 'key_error'
    elif 'valueerror' in error_lower:
        return 'value_error'
    elif 'zerodivisionerror' in error_lower:
        return 'zero_division_error'
    elif 'recursionerror' in error_lower or 'maximum recursion' in error_lower:
        return 'recursion_error'
    elif 'indentationerror' in error_lower:
        return 'indentation_error'
    
    # Default: altro errore runtime
    return 'other_runtime'


def analyze_errors_by_model(results: List[dict]) -> Dict:
    """Analizza distribuzione errori per modello"""
    error_stats = defaultdict(lambda: Counter())
    
    for result in results:
        model = result['model']
        error_type = classify_error(result)
        error_stats[model][error_type] += 1
    
    return dict(error_stats)


def analyze_errors_by_difficulty(results: List[dict], problems: Dict[str, dict]) -> Dict:
    """Analizza distribuzione errori per difficoltà"""
    error_stats = defaultdict(lambda: defaultdict(lambda: Counter()))
    
    for result in results:
        model = result['model']
        problem_id = result['problem_id']
        difficulty = problems.get(problem_id, {}).get('difficulty', 'unknown')
        error_type = classify_error(result)
        
        error_stats[model][difficulty][error_type] += 1
    
    return dict(error_stats)


def analyze_errors_by_category(results: List[dict], problems: Dict[str, dict]) -> Dict:
    """Analizza distribuzione errori per categoria problema"""
    error_stats = defaultdict(lambda: defaultdict(lambda: Counter()))
    
    for result in results:
        model = result['model']
        problem_id = result['problem_id']
        category = problems.get(problem_id, {}).get('category', 'unknown')
        error_type = classify_error(result)
        
        error_stats[model][category][error_type] += 1
    
    return dict(error_stats)


def generate_error_summary_table(error_stats: Dict) -> pd.DataFrame:
    """Genera tabella riassuntiva errori per modello"""
    rows = []
    
    error_types = ['success', 'assertion_error', 'syntax_error', 'name_error', 
                   'type_error', 'attribute_error', 'index_error', 'timeout', 
                   'other_runtime']
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in error_stats:
            row = {'Modello': model.upper()}
            total = sum(error_stats[model].values())
            
            for error_type in error_types:
                count = error_stats[model].get(error_type, 0)
                percentage = (count / total * 100) if total > 0 else 0.0
                row[error_type.replace('_', ' ').title()] = f"{count} ({percentage:.1f}%)"
            
            row['Total'] = total
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def generate_failure_analysis_table(error_stats: Dict) -> pd.DataFrame:
    """Genera tabella solo errori (esclude successi)"""
    rows = []
    
    failure_types = ['assertion_error', 'syntax_error', 'name_error', 'type_error',
                     'attribute_error', 'index_error', 'value_error', 'timeout', 
                     'recursion_error', 'other_runtime']
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in error_stats:
            row = {'Modello': model.upper()}
            
            # Totale fallimenti (esclusi successi)
            total_failures = sum(error_stats[model].get(et, 0) for et in failure_types)
            
            if total_failures > 0:
                for error_type in failure_types:
                    count = error_stats[model].get(error_type, 0)
                    percentage = (count / total_failures * 100) if total_failures > 0 else 0.0
                    
                    if count > 0:  # Mostra solo errori presenti
                        row[error_type.replace('_', ' ').title()] = f"{count} ({percentage:.1f}%)"
                
                row['Total Failures'] = total_failures
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.fillna('-')  # Sostituisci NaN con '-'
    return df


def find_common_error_patterns(results: List[dict], model: str, error_type: str, limit: int = 5) -> List[str]:
    """Trova pattern comuni per tipo di errore e modello"""
    error_messages = []
    
    for result in results:
        if result['model'] == model and classify_error(result) == error_type:
            error = result.get('execution_result', {}).get('error', '')
            if error:
                # Estrai messaggio principale (prima riga significativa)
                lines = error.split('\n')
                for line in lines:
                    if line.strip() and not line.strip().startswith('File'):
                        error_messages.append(line.strip())
                        break
    
    # Conta pattern più comuni
    counter = Counter(error_messages)
    return counter.most_common(limit)


def save_to_csv(df: pd.DataFrame, output_path: str):
    """Salva DataFrame in CSV"""
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f" Tabella CSV salvata: {output_path}")


def save_to_latex(df: pd.DataFrame, output_path: str, caption: str, label: str):
    """Salva DataFrame in formato LaTeX"""
    latex = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        position='htbp',
        escape=False  # Permette % e caratteri speciali
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    
    print(f" Tabella LaTeX salvata: {output_path}")


def main():
    """Analizza tutti gli errori"""
    print("="*80)
    print("🔍 ANALISI E CLASSIFICAZIONE ERRORI")
    print("="*80)
    
    # Percorsi file
    results_file = "results/raw_outputs/full_experiments/full_experiments_gemini25_20251031_135614.json"
    problems_file = "src/data/selected_problems/selected_problems.json"
    
    # Carica dati
    print("\n Caricamento dati...")
    metadata, results = load_results(results_file)
    
    with open(problems_file, 'r', encoding='utf-8') as f:
        problems_data = json.load(f)
    
    problems = {}
    for p in problems_data['problems']:
        problems[p['id']] = p
    
    print(f"   Total results: {len(results)}")
    print(f"   Total problems: {len(problems)}")
    
    # Crea cartelle output
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Analisi per modello
    print(f"\n{'='*80}")
    print(" DISTRIBUZIONE ERRORI PER MODELLO")
    print("="*80)
    
    error_by_model = analyze_errors_by_model(results)
    
    # Salva JSON
    json_path = output_dir / "error_distribution.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(error_by_model, f, indent=2, ensure_ascii=False)
    print(f"\n Statistiche salvate: {json_path}")
    
    # Tabella riassuntiva
    print("\n Distribuzione Errori per Modello:")
    df_errors = generate_error_summary_table(error_by_model)
    print(df_errors.to_string(index=False))
    
    # Salva
    csv_path = tables_dir / "error_distribution_summary.csv"
    latex_path = tables_dir / "error_distribution_summary.tex"
    
    save_to_csv(df_errors, csv_path)
    save_to_latex(
        df_errors,
        latex_path,
        caption="Error distribution across all models",
        label="tab:error_distribution"
    )
    
    # Tabella solo fallimenti
    print("\n Analisi Fallimenti (solo errori):")
    df_failures = generate_failure_analysis_table(error_by_model)
    print(df_failures.to_string(index=False))
    
    csv_path = tables_dir / "failure_analysis.csv"
    latex_path = tables_dir / "failure_analysis.tex"
    
    save_to_csv(df_failures, csv_path)
    save_to_latex(
        df_failures,
        latex_path,
        caption="Failure type distribution (excluding successes)",
        label="tab:failure_analysis"
    )
    
    # Statistiche dettagliate per modello
    print(f"\n{'='*80}")
    print(" STATISTICHE DETTAGLIATE PER MODELLO")
    print("="*80)
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        print(f"\n {model.upper()}:")
        
        total = sum(error_by_model[model].values())
        successes = error_by_model[model].get('success', 0)
        failures = total - successes
        
        print(f"   Total: {total}")
        print(f"   Successi: {successes} ({successes/total*100:.1f}%)")
        print(f"   Fallimenti: {failures} ({failures/total*100:.1f}%)")
        
        if failures > 0:
            print(f"\n   Distribuzione fallimenti:")
            failure_types = [(k, v) for k, v in error_by_model[model].items() 
                           if k != 'success' and v > 0]
            failure_types.sort(key=lambda x: x[1], reverse=True)
            
            for error_type, count in failure_types:
                percentage = count / failures * 100
                print(f"      {error_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    print(f"\n{'='*80}")
    print(" ANALISI COMPLETATA!")
    print(f"{'='*80}")
    print(f"\n File generati:")
    print(f"   - Statistiche JSON: results/metrics/error_distribution.json")
    print(f"   - Tabelle CSV: results/tables/error_*.csv")
    print(f"   - Tabelle LaTeX: results/tables/error_*.tex")


if __name__ == "__main__":
    main()
