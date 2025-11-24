"""
Analisi qualità del codice generato:
- Complessità ciclomatica (McCabe)
- Lines of Code (LOC)
- Maintainability Index
- Halstead metrics (difficulty, effort, volume)

Analizza solo il codice che ha passato i test (generazioni di successo).
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd

# Radon imports
from radon.complexity import cc_visit, average_complexity
from radon.metrics import mi_visit, h_visit
from radon.raw import analyze


def load_results(file_path: str) -> Tuple[dict, List[dict]]:
    """Carica risultati da JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['metadata'], data['results']


def count_lines_of_code(code: str) -> Dict:
    """Conta linee di codice usando radon"""
    try:
        analysis = analyze(code)
        return {
            'loc': analysis.loc,  # Lines of code (totali)
            'lloc': analysis.lloc,  # Logical lines of code
            'sloc': analysis.sloc,  # Source lines of code
            'comments': analysis.comments,
            'multi': analysis.multi,  # Multi-line strings
            'blank': analysis.blank
        }
    except Exception as e:
        return {
            'loc': 0,
            'lloc': 0,
            'sloc': 0,
            'comments': 0,
            'multi': 0,
            'blank': 0,
            'error': str(e)
        }


def calculate_cyclomatic_complexity(code: str) -> Dict:
    """Calcola complessità ciclomatica usando radon"""
    try:
        complexity_blocks = cc_visit(code)
        
        if not complexity_blocks:
            return {
                'average': 0,
                'max': 0,
                'total': 0,
                'functions': 0
            }
        
        complexities = [block.complexity for block in complexity_blocks]
        
        return {
            'average': statistics.mean(complexities) if complexities else 0,
            'median': statistics.median(complexities) if complexities else 0,
            'max': max(complexities) if complexities else 0,
            'min': min(complexities) if complexities else 0,
            'total': sum(complexities),
            'functions': len(complexity_blocks)
        }
    except Exception as e:
        return {
            'average': 0,
            'median': 0,
            'max': 0,
            'min': 0,
            'total': 0,
            'functions': 0,
            'error': str(e)
        }


def calculate_maintainability_index(code: str) -> float:
    """Calcola Maintainability Index usando radon"""
    try:
        mi = mi_visit(code, multi=True)
        if mi:
            # Radon ritorna una lista di MI per ogni blocco
            # Prendiamo la media
            mi_values = [block.mi for block in mi]
            return statistics.mean(mi_values) if mi_values else 0.0
        return 0.0
    except Exception as e:
        return 0.0


def calculate_halstead_metrics(code: str) -> Dict:
    """Calcola Halstead metrics usando radon"""
    try:
        halstead = h_visit(code)
        
        if not halstead:
            return {
                'volume': 0,
                'difficulty': 0,
                'effort': 0,
                'time': 0,
                'bugs': 0
            }
        
        # Radon ritorna lista di report Halstead
        # Aggreghiamo i valori
        total = halstead.total
        
        return {
            'volume': total.volume if total else 0,
            'difficulty': total.difficulty if total else 0,
            'effort': total.effort if total else 0,
            'time': total.time if total else 0,
            'bugs': total.bugs if total else 0
        }
    except Exception as e:
        return {
            'volume': 0,
            'difficulty': 0,
            'effort': 0,
            'time': 0,
            'bugs': 0,
            'error': str(e)
        }


def analyze_code_quality(results: List[dict]) -> Dict:
    """
    Analizza qualità del codice per ogni modello.
    Considera SOLO le generazioni di successo (test passati).
    """
    stats = defaultdict(lambda: {
        'total_analyzed': 0,
        'loc_list': [],
        'sloc_list': [],
        'complexity_avg_list': [],
        'complexity_max_list': [],
        'abs_max_complexity_list': [],  # Track absolute max for finding overall max
        'mi_list': [],
        'halstead_volume_list': [],
        'halstead_difficulty_list': [],
        'halstead_effort_list': []
    })
    
    for result in results:
        # Analizza solo codice che ha passato i test
        if not result.get('execution_result', {}).get('passed', False):
            continue
        
        code = result.get('generated_code', '')
        if not code:
            continue
        
        model = result['model']
        
        # Lines of code
        loc_metrics = count_lines_of_code(code)
        if 'error' not in loc_metrics:
            stats[model]['loc_list'].append(loc_metrics['loc'])
            stats[model]['sloc_list'].append(loc_metrics['sloc'])
        
        # Cyclomatic complexity
        cc_metrics = calculate_cyclomatic_complexity(code)
        if 'error' not in cc_metrics and cc_metrics['average'] > 0:
            stats[model]['complexity_avg_list'].append(cc_metrics['average'])
            stats[model]['complexity_max_list'].append(cc_metrics['max'])
            stats[model]['abs_max_complexity_list'].append(cc_metrics['max'])  # Track all max values
        
        # Maintainability Index
        mi = calculate_maintainability_index(code)
        if mi > 0:
            stats[model]['mi_list'].append(mi)
        
        # Halstead metrics
        halstead = calculate_halstead_metrics(code)
        if 'error' not in halstead and halstead['volume'] > 0:
            stats[model]['halstead_volume_list'].append(halstead['volume'])
            stats[model]['halstead_difficulty_list'].append(halstead['difficulty'])
            stats[model]['halstead_effort_list'].append(halstead['effort'])
        
        stats[model]['total_analyzed'] += 1
    
    # Calcola statistiche aggregate
    for model in stats:
        s = stats[model]
        
        # LOC
        s['avg_loc'] = statistics.mean(s['loc_list']) if s['loc_list'] else 0
        s['median_loc'] = statistics.median(s['loc_list']) if s['loc_list'] else 0
        s['std_loc'] = statistics.stdev(s['loc_list']) if len(s['loc_list']) > 1 else 0
        
        s['avg_sloc'] = statistics.mean(s['sloc_list']) if s['sloc_list'] else 0
        
        # Cyclomatic Complexity
        s['avg_complexity'] = statistics.mean(s['complexity_avg_list']) if s['complexity_avg_list'] else 0
        s['median_complexity'] = statistics.median(s['complexity_avg_list']) if s['complexity_avg_list'] else 0
        s['std_complexity'] = statistics.stdev(s['complexity_avg_list']) if len(s['complexity_avg_list']) > 1 else 0
        
        s['avg_max_complexity'] = statistics.mean(s['complexity_max_list']) if s['complexity_max_list'] else 0
        s['absolute_max_complexity'] = max(s['abs_max_complexity_list']) if s['abs_max_complexity_list'] else 0
        
        # Maintainability Index
        s['avg_mi'] = statistics.mean(s['mi_list']) if s['mi_list'] else 0
        s['median_mi'] = statistics.median(s['mi_list']) if s['mi_list'] else 0
        s['std_mi'] = statistics.stdev(s['mi_list']) if len(s['mi_list']) > 1 else 0
        
        # Halstead
        s['avg_halstead_volume'] = statistics.mean(s['halstead_volume_list']) if s['halstead_volume_list'] else 0
        s['avg_halstead_difficulty'] = statistics.mean(s['halstead_difficulty_list']) if s['halstead_difficulty_list'] else 0
        s['avg_halstead_effort'] = statistics.mean(s['halstead_effort_list']) if s['halstead_effort_list'] else 0
        
        # Rimuovi liste (troppo grandi)
        for key in list(s.keys()):
            if key.endswith('_list'):
                del s[key]
    
    return dict(stats)


def generate_quality_summary_table(stats: Dict) -> pd.DataFrame:
    """Genera tabella riassuntiva qualità codice"""
    rows = []
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats:
            s = stats[model]
            rows.append({
                'Modello': model.upper(),
                'Soluzioni Analizzate': s['total_analyzed'],
                'LOC Media': f"{s['avg_loc']:.1f}",
                'SLOC Media': f"{s['avg_sloc']:.1f}",
                'Complessità Media': f"{s['avg_complexity']:.2f}",
                'Complessità Max Media': f"{s['avg_max_complexity']:.1f}",
                'Maintainability Index': f"{s['avg_mi']:.1f}"
            })
    
    df = pd.DataFrame(rows)
    return df


def generate_halstead_table(stats: Dict) -> pd.DataFrame:
    """Genera tabella Halstead metrics"""
    rows = []
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats:
            s = stats[model]
            rows.append({
                'Modello': model.upper(),
                'Volume': f"{s['avg_halstead_volume']:.1f}",
                'Difficulty': f"{s['avg_halstead_difficulty']:.2f}",
                'Effort': f"{s['avg_halstead_effort']:.0f}"
            })
    
    df = pd.DataFrame(rows)
    return df


def interpret_metrics(stats: Dict):
    """Interpreta le metriche e fornisce insights"""
    print("\n INTERPRETAZIONE METRICHE:")
    print("\n1. Complessità Ciclomatica (McCabe):")
    print("   - 1-10: Semplice, basso rischio")
    print("   - 11-20: Moderato, moderato rischio")
    print("   - 21-50: Complesso, alto rischio")
    print("   - >50: Non testabile, molto alto rischio")
    
    print("\n2. Maintainability Index:")
    print("   - 85-100: Alta manutenibilità")
    print("   - 65-84: Moderata manutenibilità")
    print("   - <65: Difficile da mantenere")
    
    print("\n3. Lines of Code (LOC):")
    print("   - Codice più conciso spesso indica migliore qualità")
    print("   - Ma troppo conciso può essere oscuro")
    
    print("\n CONFRONTO MODELLI:")
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats:
            s = stats[model]
            print(f"\n🤖 {model.upper()}:")
            print(f"   Complessità: {s['avg_complexity']:.2f} ", end="")
            
            if s['avg_complexity'] < 5:
                print("( Semplice)")
            elif s['avg_complexity'] < 10:
                print("( Buono)")
            elif s['avg_complexity'] < 15:
                print("( Moderato)")
            else:
                print("( Complesso)")
            
            print(f"   Manutenibilità: {s['avg_mi']:.1f} ", end="")
            
            if s['avg_mi'] >= 85:
                print("( Alta)")
            elif s['avg_mi'] >= 65:
                print("( Moderata)")
            else:
                print("( Bassa)")
            
            print(f"   LOC media: {s['avg_loc']:.1f}")


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
        escape=False
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    
    print(f" Tabella LaTeX salvata: {output_path}")


def main():
    """Analizza qualità del codice"""
    print("="*80)
    print(" ANALISI QUALITÀ CODICE GENERATO")
    print("="*80)
    print("\n Nota: Analizza SOLO codice che ha passato i test")
    
    # Percorsi file
    results_file = "results/raw_outputs/full_experiments/full_experiments_gemini25_20251031_135614.json"
    
    # Carica dati
    print("\n Caricamento dati...")
    metadata, results = load_results(results_file)
    
    print(f"   Total results: {len(results)}")
    
    # Conta successi per modello
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
    
    # Analisi qualità
    print(f"\n{'='*80}")
    print(" CALCOLO METRICHE QUALITÀ")
    print("="*80)
    print("\n Analisi in corso (può richiedere 1-2 minuti)...")
    
    quality_stats = analyze_code_quality(results)
    
    # Salva JSON
    json_path = output_dir / "quality_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(quality_stats, f, indent=2, ensure_ascii=False)
    print(f"\n Statistiche salvate: {json_path}")
    
    # Tabella riassuntiva qualità
    print("\n Metriche Qualità per Modello:")
    df_quality = generate_quality_summary_table(quality_stats)
    print(df_quality.to_string(index=False))
    
    csv_path = tables_dir / "code_quality_summary.csv"
    latex_path = tables_dir / "code_quality_summary.tex"
    
    save_to_csv(df_quality, csv_path)
    save_to_latex(
        df_quality,
        latex_path,
        caption="Code quality metrics for successful solutions",
        label="tab:code_quality"
    )
    
    # Tabella Halstead
    print("\n Halstead Metrics:")
    df_halstead = generate_halstead_table(quality_stats)
    print(df_halstead.to_string(index=False))
    
    csv_path = tables_dir / "halstead_metrics.csv"
    latex_path = tables_dir / "halstead_metrics.tex"
    
    save_to_csv(df_halstead, csv_path)
    save_to_latex(
        df_halstead,
        latex_path,
        caption="Halstead complexity metrics",
        label="tab:halstead_metrics"
    )
    
    # Interpretazione
    interpret_metrics(quality_stats)
    
    print(f"\n{'='*80}")
    print(" ANALISI COMPLETATA!")
    print(f"{'='*80}")
    print(f"\n File generati:")
    print(f"   - Statistiche JSON: results/metrics/quality_metrics.json")
    print(f"   - Tabelle CSV: results/tables/code_quality_*.csv")
    print(f"   - Tabelle LaTeX: results/tables/code_quality_*.tex")


if __name__ == "__main__":
    main()
