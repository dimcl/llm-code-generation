"""
Calcolo metriche di correttezza: Pass@1, Pass@3, Pass@5

Formula Pass@k: 
  Percentuale di problemi per cui almeno 1 soluzione su k è corretta.
  
  Per ogni problema:
  - Se almeno 1/k soluzioni passa tutti i test → successo
  - Altrimenti → fallimento
  
  Pass@k = (problemi con ≥1 soluzione corretta) / (totale problemi) * 100
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd


def load_results(file_path: str) -> Tuple[dict, List[dict]]:
    """Carica risultati da JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['metadata'], data['results']


def load_problems(file_path: str) -> Dict[str, dict]:
    """Carica info problemi (difficoltà, categoria)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problems = {}
    for problem in data['problems']:
        problems[problem['id']] = {
            'difficulty': problem['difficulty'],
            'category': problem['category'],
            'source': problem['source']
        }
    return problems


def calculate_pass_at_k(results: List[dict], problems: Dict[str, dict], k: int = 5) -> Dict:
    """
    Calcola Pass@k per ogni modello e breakdown per difficoltà/categoria
    
    Args:
        results: Lista risultati esperimenti
        problems: Dizionario info problemi
        k: Numero di tentativi da considerare (1, 3, o 5)
    
    Returns:
        Dict con metriche Pass@k per modello, difficoltà, categoria
    """
    # Raggruppa risultati per (modello, problema_id)
    by_model_problem = defaultdict(list)
    
    for result in results:
        model = result['model']
        problem_id = result['problem_id']
        attempt = result['attempt']
        
        # Considera solo i primi k tentativi
        if attempt <= k:
            passed = result.get('execution_result', {}).get('passed', False)
            by_model_problem[(model, problem_id)].append(passed)
    
    # Calcola Pass@k
    stats = {
        'overall': {},
        'by_difficulty': {},
        'by_category': {}
    }
    
    models = ['gemini', 'llama', 'qwen', 'gpt']
    
    for model in models:
        # Pass@k globale
        total_problems = 0
        solved_problems = 0
        
        # Per difficoltà
        by_diff = defaultdict(lambda: {'total': 0, 'solved': 0})
        
        # Per categoria
        by_cat = defaultdict(lambda: {'total': 0, 'solved': 0})
        
        for problem_id in problems:
            attempts = by_model_problem[(model, problem_id)]
            
            if len(attempts) > 0:
                total_problems += 1
                
                # Se almeno 1 tentativo su k è corretto → problema risolto
                if any(attempts):
                    solved_problems += 1
                    
                    # Breakdown difficoltà
                    diff = problems[problem_id]['difficulty']
                    by_diff[diff]['solved'] += 1
                    
                    # Breakdown categoria
                    cat = problems[problem_id]['category']
                    by_cat[cat]['solved'] += 1
                
                # Conta totali
                diff = problems[problem_id]['difficulty']
                by_diff[diff]['total'] += 1
                
                cat = problems[problem_id]['category']
                by_cat[cat]['total'] += 1
        
        # Calcola percentuali
        pass_rate = (solved_problems / total_problems * 100) if total_problems > 0 else 0.0
        
        stats['overall'][model] = {
            'total_problems': total_problems,
            'solved_problems': solved_problems,
            'pass_rate': round(pass_rate, 1)
        }
        
        # Difficoltà
        stats['by_difficulty'][model] = {}
        for diff in ['easy', 'medium', 'hard']:
            if diff in by_diff:
                d = by_diff[diff]
                rate = (d['solved'] / d['total'] * 100) if d['total'] > 0 else 0.0
                stats['by_difficulty'][model][diff] = {
                    'total': d['total'],
                    'solved': d['solved'],
                    'pass_rate': round(rate, 1)
                }
        
        # Categoria
        stats['by_category'][model] = {}
        for cat in ['strings', 'lists', 'math', 'algorithms']:
            if cat in by_cat:
                c = by_cat[cat]
                rate = (c['solved'] / c['total'] * 100) if c['total'] > 0 else 0.0
                stats['by_category'][model][cat] = {
                    'total': c['total'],
                    'solved': c['solved'],
                    'pass_rate': round(rate, 1)
                }
    
    return stats


def generate_summary_table(stats: Dict, k: int) -> pd.DataFrame:
    """Genera tabella riassuntiva Pass@k per modello"""
    rows = []
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats['overall']:
            s = stats['overall'][model]
            rows.append({
                'Modello': model.upper(),
                f'Pass@{k} (%)': s['pass_rate'],
                'Problemi Risolti': f"{s['solved_problems']}/{s['total_problems']}"
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(f'Pass@{k} (%)', ascending=False)
    return df


def generate_difficulty_table(stats: Dict, k: int) -> pd.DataFrame:
    """Genera tabella Pass@k per difficoltà"""
    rows = []
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats['by_difficulty']:
            row = {'Modello': model.upper()}
            
            for diff in ['easy', 'medium', 'hard']:
                if diff in stats['by_difficulty'][model]:
                    d = stats['by_difficulty'][model][diff]
                    row[f'{diff.capitalize()}'] = d['pass_rate']
                else:
                    row[f'{diff.capitalize()}'] = 0.0
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def generate_category_table(stats: Dict, k: int) -> pd.DataFrame:
    """Genera tabella Pass@k per categoria"""
    rows = []
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats['by_category']:
            row = {'Modello': model.upper()}
            
            for cat in ['strings', 'lists', 'math', 'algorithms']:
                if cat in stats['by_category'][model]:
                    c = stats['by_category'][model][cat]
                    row[f'{cat.capitalize()}'] = c['pass_rate']
                else:
                    row[f'{cat.capitalize()}'] = 0.0
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def save_to_csv(df: pd.DataFrame, output_path: str):
    """Salva DataFrame in CSV"""
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f" Tabella CSV salvata: {output_path}")


def save_to_latex(df: pd.DataFrame, output_path: str, caption: str, label: str):
    """Salva DataFrame in formato LaTeX"""
    latex = df.to_latex(
        index=False,
        float_format="%.1f",
        caption=caption,
        label=label,
        position='htbp'
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    
    print(f" Tabella LaTeX salvata: {output_path}")


def main():
    """Calcola tutte le metriche Pass@k"""
    print("="*80)
    print(" CALCOLO METRICHE DI CORRETTEZZA (Pass@k)")
    print("="*80)
    
    # Percorsi file
    results_file = "results/raw_outputs/full_experiments/full_experiments_gemini25_20251031_135614.json"
    problems_file = "src/data/selected_problems/selected_problems.json"
    
    # Carica dati
    print("\n Caricamento dati...")
    metadata, results = load_results(results_file)
    problems = load_problems(problems_file)
    
    print(f"   Total results: {len(results)}")
    print(f"   Total problems: {len(problems)}")
    print(f"   Models: {metadata['models']}")
    
    # Crea cartelle output
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Calcola Pass@k per k=1, 3, 5
    for k in [1, 3, 5]:
        print(f"\n{'='*80}")
        print(f" CALCOLO Pass@{k}")
        print(f"{'='*80}")
        
        stats = calculate_pass_at_k(results, problems, k=k)
        
        # Salva JSON
        json_path = output_dir / f"pass_at_{k}_stats.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n Statistiche salvate: {json_path}")
        
        # Tabella riassuntiva
        print(f"\n Pass@{k} per Modello:")
        df_summary = generate_summary_table(stats, k)
        print(df_summary.to_string(index=False))
        
        # Salva CSV e LaTeX
        csv_path = tables_dir / f"pass_at_{k}_summary.csv"
        latex_path = tables_dir / f"pass_at_{k}_summary.tex"
        
        save_to_csv(df_summary, csv_path)
        save_to_latex(
            df_summary, 
            latex_path,
            caption=f"Pass@{k} performance comparison across models",
            label=f"tab:pass_at_{k}_summary"
        )
        
        # Tabella per difficoltà
        print(f"\n Pass@{k} per Difficoltà:")
        df_diff = generate_difficulty_table(stats, k)
        print(df_diff.to_string(index=False))
        
        csv_path = tables_dir / f"pass_at_{k}_by_difficulty.csv"
        latex_path = tables_dir / f"pass_at_{k}_by_difficulty.tex"
        
        save_to_csv(df_diff, csv_path)
        save_to_latex(
            df_diff,
            latex_path,
            caption=f"Pass@{k} breakdown by problem difficulty",
            label=f"tab:pass_at_{k}_difficulty"
        )
        
        # Tabella per categoria
        print(f"\n Pass@{k} per Categoria:")
        df_cat = generate_category_table(stats, k)
        print(df_cat.to_string(index=False))
        
        csv_path = tables_dir / f"pass_at_{k}_by_category.csv"
        latex_path = tables_dir / f"pass_at_{k}_by_category.tex"
        
        save_to_csv(df_cat, csv_path)
        save_to_latex(
            df_cat,
            latex_path,
            caption=f"Pass@{k} breakdown by problem category",
            label=f"tab:pass_at_{k}_category"
        )
    
    print(f"\n{'='*80}")
    print(" CALCOLO COMPLETATO!")
    print(f"{'='*80}")
    print(f"\n File generati:")
    print(f"   - Metriche JSON: results/metrics/pass_at_*_stats.json")
    print(f"   - Tabelle CSV: results/tables/pass_at_*.csv")
    print(f"   - Tabelle LaTeX: results/tables/pass_at_*.tex")


if __name__ == "__main__":
    main()
