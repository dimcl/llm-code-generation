"""
Generate Final Summary Table for Thesis
Genera tabella riassuntiva finale con tutti i risultati per la tesi
"""

import json
import pandas as pd
from pathlib import Path


def generate_final_summary_table():
    """Genera tabella riassuntiva completa per la tesi"""
    
    print("="*80)
    print("GENERAZIONE TABELLA RIASSUNTIVA FINALE")
    print("="*80)
    
    # Carica metriche
    metrics_dir = Path("results/metrics")
    
    # 1. Pass@k metrics
    pass_at_1 = json.load(open(metrics_dir / "pass_at_1_stats.json"))
    pass_at_3 = json.load(open(metrics_dir / "pass_at_3_stats.json"))
    pass_at_5 = json.load(open(metrics_dir / "pass_at_5_stats.json"))
    
    # 2. Cost analysis
    cost = json.load(open(metrics_dir / "cost_analysis.json"))
    
    # 3. Quality metrics
    quality = json.load(open(metrics_dir / "quality_metrics.json"))
    
    # 4. Error distribution
    errors = json.load(open(metrics_dir / "error_distribution.json"))
    
    # Crea tabella
    models = ['gpt', 'gemini', 'llama', 'qwen']
    model_names = {
        'gpt': 'GPT-4o-mini',
        'gemini': 'Gemini 2.5 Flash',
        'llama': 'Llama 3.3 70B',
        'qwen': 'Qwen 2.5 Coder 32B'
    }
    
    rows = []
    
    for model in models:
        row = {
            'Model': model_names[model],
            
            # Correctness
            'Pass@1 (%)': f"{pass_at_1['overall'][model]['pass_rate']:.1f}",
            'Pass@3 (%)': f"{pass_at_3['overall'][model]['pass_rate']:.1f}",
            'Pass@5 (%)': f"{pass_at_5['overall'][model]['pass_rate']:.1f}",
            'Problems Solved': f"{pass_at_1['overall'][model]['solved_problems']}/60",
            
            # Quality (solo per soluzioni corrette)
            'Avg Cyclomatic': f"{quality[model]['avg_complexity']:.1f}" if model in quality else 'N/A',
            'Avg Maintainability': f"{quality[model]['avg_mi']:.1f}" if model in quality else 'N/A',
            
            # Cost & Efficiency
            'Avg Tokens': f"{cost[model]['avg_tokens']:.0f}",
            'Avg Latency (s)': f"{cost[model]['avg_latency']:.2f}",
            'Total Cost ($)': f"{cost[model]['total_cost']:.4f}",
            'Cost per Success ($)': f"{cost[model]['cost_per_success']:.4f}" if cost[model].get('cost_per_success') else 'N/A',
            
            # Errors
            'Total Errors': sum(v for k, v in errors[model].items() if k != 'success'),
            'Syntax Errors': errors[model].get('syntax_error', 0),
            'Runtime Errors': errors[model].get('assertion_error', 0) + 
                            errors[model].get('runtime_error', 0) +
                            errors[model].get('value_error', 0),
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Salva in vari formati
    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV
    csv_file = output_dir / "final_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nCSV salvato: {csv_file}")
    
    # LaTeX
    latex_file = output_dir / "final_summary.tex"
    with open(latex_file, 'w', encoding='utf-8') as f:
        latex = df.to_latex(index=False, caption="Comprehensive Results Summary", label="tab:final_summary")
        f.write(latex)
    print(f" LaTeX salvato: {latex_file}")
    
    # Markdown
    md_file = output_dir / "final_summary.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Final Results Summary\n\n")
        f.write(df.to_markdown(index=False))
    print(f" Markdown salvato: {md_file}")
    
    # Print to console
    print(f"\n{'='*80}")
    print("TABELLA RIASSUNTIVA FINALE")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))
    
    # Statistiche aggiuntive
    print(f"\n{'='*80}")
    print("STATISTICHE AGGIUNTIVE")
    print(f"{'='*80}\n")
    
    # Overall pass rates
    print("Overall Pass Rates:")
    for model in models:
        print(f"  {model_names[model]:25s}: {pass_at_1['overall'][model]['pass_rate']:.1f}%")
    
    print(f"\nTotal Experiments: 1200 (60 problems × 4 models × 5 attempts)")
    print(f"Total Cost: ${sum(cost[m]['total_cost'] for m in models):.4f}")
    
    # Best performing model
    best_model = max(models, key=lambda m: pass_at_1['overall'][m]['pass_rate'])
    print(f"\nBest Overall Performance: {model_names[best_model]} ({pass_at_1['overall'][best_model]['pass_rate']:.1f}%)")
    
    # Most cost-efficient
    cost_efficiency = {m: cost[m].get('cost_per_success', float('inf')) for m in models}
    best_cost = min(cost_efficiency, key=cost_efficiency.get)
    if cost_efficiency[best_cost] != float('inf'):
        print(f"Most Cost-Efficient: {model_names[best_cost]} (${cost_efficiency[best_cost]:.4f} per success)")
    
    print(f"\n{'='*80}\n")


def generate_case_studies_summary():
    """Genera tabella riassuntiva dei case studies"""
    
    print("GENERAZIONE TABELLA CASE STUDIES")
    print("="*80)
    
    # Carica case studies
    cs_file = Path("results/analysis/selected_case_studies.json")
    cs_data = json.load(open(cs_file))
    
    rows = []
    for cs in cs_data['case_studies']:
        row = {
            'Problem ID': cs['problem_id'],
            'Difficulty': cs['difficulty'].capitalize(),
            'Category': cs['category'].capitalize(),
            'GPT (%)': f"{cs['pass_rates']['gpt']:.0f}",
            'Gemini (%)': f"{cs['pass_rates']['gemini']:.0f}",
            'Llama (%)': f"{cs['pass_rates']['llama']:.0f}",
            'Qwen (%)': f"{cs['pass_rates']['qwen']:.0f}",
            'Reason': cs['reason']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Salva
    output_dir = Path("results/tables")
    
    # CSV
    csv_file = output_dir / "case_studies_selection.csv"
    df.to_csv(csv_file, index=False)
    print(f" CSV salvato: {csv_file}")
    
    # LaTeX
    latex_file = output_dir / "case_studies_selection.tex"
    with open(latex_file, 'w', encoding='utf-8') as f:
        latex = df.to_latex(index=False, caption="Selected Case Studies for Qualitative Analysis", 
                          label="tab:case_studies")
        f.write(latex)
    print(f" LaTeX salvato: {latex_file}")
    
    # Markdown
    md_file = output_dir / "case_studies_selection.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Selected Case Studies\n\n")
        f.write(df.to_markdown(index=False))
    print(f" Markdown salvato: {md_file}")
    
    print(f"\n{df.to_string(index=False)}\n")
    print("="*80 + "\n")


def main():
    generate_final_summary_table()
    print()
    generate_case_studies_summary()
    
    print(" TUTTE LE TABELLE GENERATE CON SUCCESSO")
    print("\n File disponibili in: results/tables/")
    print("   - final_summary.csv/.tex/.md")
    print("   - case_studies_selection.csv/.tex/.md")


if __name__ == "__main__":
    main()
