"""
Analisi costi e performance:
- Token usage totale e medio
- Latency (tempo generazione)
- Costo API totale
- Costo per problema risolto (ROI)
- Rapporto efficienza costo/performance
"""

import json
import statistics
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd


def load_results(file_path: str) -> Tuple[dict, List[dict]]:
    """Carica risultati da JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['metadata'], data['results']


def analyze_costs_by_model(results: List[dict]) -> Dict:
    """Analizza token usage, latency, costi per modello"""
    stats = defaultdict(lambda: {
        'total_generations': 0,
        'successful_generations': 0,
        'total_tokens': 0,
        'tokens_list': [],
        'total_latency': 0.0,
        'latency_list': [],
        'total_cost': 0.0
    })
    
    for result in results:
        model = result['model']
        
        stats[model]['total_generations'] += 1
        
        # Conta successi
        if result.get('execution_result', {}).get('passed', False):
            stats[model]['successful_generations'] += 1
        
        # Token usage
        tokens = result.get('tokens', 0)
        if tokens > 0:
            stats[model]['total_tokens'] += tokens
            stats[model]['tokens_list'].append(tokens)
        
        # Latency
        latency = result.get('latency', 0.0)
        if latency > 0:
            stats[model]['total_latency'] += latency
            stats[model]['latency_list'].append(latency)
        
        # Cost
        cost = result.get('cost', 0.0)
        if cost > 0:
            stats[model]['total_cost'] += cost
    
    # Calcola medie e statistiche
    for model in stats:
        s = stats[model]
        
        # Media tokens
        s['avg_tokens'] = s['total_tokens'] / s['total_generations'] if s['total_generations'] > 0 else 0
        s['median_tokens'] = statistics.median(s['tokens_list']) if s['tokens_list'] else 0
        s['std_tokens'] = statistics.stdev(s['tokens_list']) if len(s['tokens_list']) > 1 else 0
        
        # Media latency
        s['avg_latency'] = s['total_latency'] / s['total_generations'] if s['total_generations'] > 0 else 0
        s['median_latency'] = statistics.median(s['latency_list']) if s['latency_list'] else 0
        s['std_latency'] = statistics.stdev(s['latency_list']) if len(s['latency_list']) > 1 else 0
        
        # Costo per generazione
        s['cost_per_generation'] = s['total_cost'] / s['total_generations'] if s['total_generations'] > 0 else 0
        
        # Costo per successo (ROI)
        s['cost_per_success'] = s['total_cost'] / s['successful_generations'] if s['successful_generations'] > 0 else 0
        
        # Pass rate
        s['pass_rate'] = (s['successful_generations'] / s['total_generations'] * 100) if s['total_generations'] > 0 else 0
        
        # Rimuovi liste (troppo grandi per JSON)
        del s['tokens_list']
        del s['latency_list']
    
    return dict(stats)


def generate_cost_summary_table(stats: Dict) -> pd.DataFrame:
    """Genera tabella riassuntiva costi per modello"""
    rows = []
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats:
            s = stats[model]
            rows.append({
                'Modello': model.upper(),
                'Generazioni': s['total_generations'],
                'Successi': s['successful_generations'],
                'Pass Rate (%)': f"{s['pass_rate']:.1f}",
                'Token Totali': f"{s['total_tokens']:,}",
                'Token Medi': f"{s['avg_tokens']:.0f}",
                'Costo Totale ($)': f"{s['total_cost']:.4f}",
                'Costo/Gen ($)': f"{s['cost_per_generation']:.6f}",
                'Costo/Successo ($)': f"{s['cost_per_success']:.6f}"
            })
    
    df = pd.DataFrame(rows)
    return df


def generate_latency_table(stats: Dict) -> pd.DataFrame:
    """Genera tabella latency per modello"""
    rows = []
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats:
            s = stats[model]
            rows.append({
                'Modello': model.upper(),
                'Latency Media (s)': f"{s['avg_latency']:.2f}",
                'Latency Mediana (s)': f"{s['median_latency']:.2f}",
                'Std Dev (s)': f"{s['std_latency']:.2f}",
                'Latency Totale (s)': f"{s['total_latency']:.1f}",
                'Latency Totale (min)': f"{s['total_latency']/60:.1f}"
            })
    
    df = pd.DataFrame(rows)
    return df


def generate_efficiency_table(stats: Dict) -> pd.DataFrame:
    """
    Genera tabella efficienza: rapporto performance/costo
    
    Efficienza = Pass Rate / Costo per successo
    (maggiore è meglio: alta performance a basso costo)
    """
    rows = []
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model in stats:
            s = stats[model]
            
            # Efficienza assoluta (pass rate / costo per successo)
            # Normalizzata per confronto (maggiore = migliore)
            if s['cost_per_success'] > 0:
                efficiency_score = s['pass_rate'] / (s['cost_per_success'] * 1000)  # Scala per leggibilità
            else:
                efficiency_score = 0
            
            rows.append({
                'Modello': model.upper(),
                'Pass Rate (%)': f"{s['pass_rate']:.1f}",
                'Costo/Successo ($)': f"{s['cost_per_success']:.6f}",
                'Score Efficienza': f"{efficiency_score:.1f}",
                'Problemi Risolti': s['successful_generations'],
                'Costo Totale ($)': f"{s['total_cost']:.4f}"
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Score Efficienza', ascending=False)
    return df


def calculate_total_project_cost(stats: Dict) -> Dict:
    """Calcola costo totale progetto"""
    total_cost = sum(s['total_cost'] for s in stats.values())
    total_generations = sum(s['total_generations'] for s in stats.values())
    total_successes = sum(s['successful_generations'] for s in stats.values())
    total_tokens = sum(s['total_tokens'] for s in stats.values())
    total_time = sum(s['total_latency'] for s in stats.values())
    
    return {
        'total_cost': total_cost,
        'total_generations': total_generations,
        'total_successes': total_successes,
        'total_tokens': total_tokens,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'total_time_hours': total_time / 3600,
        'avg_cost_per_generation': total_cost / total_generations if total_generations > 0 else 0,
        'avg_cost_per_success': total_cost / total_successes if total_successes > 0 else 0
    }


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
    """Analizza costi e performance"""
    print("="*80)
    print(" ANALISI COSTI E PERFORMANCE")
    print("="*80)
    
    # Percorsi file
    results_file = "results/raw_outputs/full_experiments/full_experiments_gemini25_20251031_135614.json"
    
    # Carica dati
    print("\n Caricamento dati...")
    metadata, results = load_results(results_file)
    
    print(f"   Total results: {len(results)}")
    print(f"   Models: {metadata['models']}")
    
    # Crea cartelle output
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Analisi costi
    print(f"\n{'='*80}")
    print(" ANALISI COSTI PER MODELLO")
    print("="*80)
    
    cost_stats = analyze_costs_by_model(results)
    
    # Salva JSON
    json_path = output_dir / "cost_analysis.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cost_stats, f, indent=2, ensure_ascii=False)
    print(f"\n Statistiche salvate: {json_path}")
    
    # Tabella riassuntiva costi
    print("\n Riepilogo Costi e Token Usage:")
    df_cost = generate_cost_summary_table(cost_stats)
    print(df_cost.to_string(index=False))
    
    csv_path = tables_dir / "cost_summary.csv"
    latex_path = tables_dir / "cost_summary.tex"
    
    save_to_csv(df_cost, csv_path)
    save_to_latex(
        df_cost,
        latex_path,
        caption="Cost and token usage analysis across models",
        label="tab:cost_summary"
    )
    
    # Tabella latency
    print("\n Analisi Latency (Tempo Generazione):")
    df_latency = generate_latency_table(cost_stats)
    print(df_latency.to_string(index=False))
    
    csv_path = tables_dir / "latency_analysis.csv"
    latex_path = tables_dir / "latency_analysis.tex"
    
    save_to_csv(df_latency, csv_path)
    save_to_latex(
        df_latency,
        latex_path,
        caption="Latency analysis across models",
        label="tab:latency_analysis"
    )
    
    # Tabella efficienza
    print("\n Score Efficienza (Performance/Costo):")
    df_efficiency = generate_efficiency_table(cost_stats)
    print(df_efficiency.to_string(index=False))
    
    csv_path = tables_dir / "efficiency_analysis.csv"
    latex_path = tables_dir / "efficiency_analysis.tex"
    
    save_to_csv(df_efficiency, csv_path)
    save_to_latex(
        df_efficiency,
        latex_path,
        caption="Cost-effectiveness analysis: performance vs cost ratio",
        label="tab:efficiency_analysis"
    )
    
    # Costo totale progetto
    print(f"\n{'='*80}")
    print(" COSTO TOTALE PROGETTO")
    print("="*80)
    
    project_cost = calculate_total_project_cost(cost_stats)
    
    print(f"\n   Generazioni totali: {project_cost['total_generations']:,}")
    print(f"   Successi totali: {project_cost['total_successes']:,}")
    print(f"   Token totali: {project_cost['total_tokens']:,}")
    print(f"   Tempo totale: {project_cost['total_time_minutes']:.1f} min ({project_cost['total_time_hours']:.2f} ore)")
    print(f"\n   COSTO TOTALE: ${project_cost['total_cost']:.4f}")
    print(f"   Costo medio per generazione: ${project_cost['avg_cost_per_generation']:.6f}")
    print(f"   Costo medio per successo: ${project_cost['avg_cost_per_success']:.6f}")
    
    # Salva riepilogo progetto
    project_json = output_dir / "project_cost_summary.json"
    with open(project_json, 'w', encoding='utf-8') as f:
        json.dump(project_cost, f, indent=2, ensure_ascii=False)
    print(f"\n Riepilogo progetto salvato: {project_json}")
    
    print(f"\n{'='*80}")
    print(" ANALISI COMPLETATA!")
    print(f"{'='*80}")
    print(f"\n File generati:")
    print(f"   - Statistiche JSON: results/metrics/cost_analysis.json")
    print(f"   - Riepilogo progetto: results/metrics/project_cost_summary.json")
    print(f"   - Tabelle CSV: results/tables/cost_*.csv")
    print(f"   - Tabelle LaTeX: results/tables/cost_*.tex")


if __name__ == "__main__":
    main()
