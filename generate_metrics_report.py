"""
Script master per generare report completo di tutte le metriche.

Esegue in sequenza:
1. Correctness metrics (Pass@k)
2. Error classification
3. Cost analysis
4. Quality metrics

Genera un report riassuntivo completo.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


def run_script(script_path: str, description: str) -> bool:
    """Esegue uno script Python e ritorna True se successo"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n{description} - COMPLETATO")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{description} - ERRORE")
        print(f"   {e}")
        return False


def generate_summary_report():
    """Genera report riassuntivo combinando tutti i risultati"""
    print(f"\n{'='*80}")
    print("GENERAZIONE REPORT RIASSUNTIVO")
    print(f"{'='*80}\n")
    
    metrics_dir = Path("results/metrics")
    
    # Carica tutte le metriche
    try:
        # Pass@k stats
        with open(metrics_dir / "pass_at_1_stats.json", 'r') as f:
            pass_at_1 = json.load(f)
        with open(metrics_dir / "pass_at_3_stats.json", 'r') as f:
            pass_at_3 = json.load(f)
        with open(metrics_dir / "pass_at_5_stats.json", 'r') as f:
            pass_at_5 = json.load(f)
        
        # Error distribution
        with open(metrics_dir / "error_distribution.json", 'r') as f:
            errors = json.load(f)
        
        # Cost analysis
        with open(metrics_dir / "cost_analysis.json", 'r') as f:
            costs = json.load(f)
        
        # Quality metrics
        with open(metrics_dir / "quality_metrics.json", 'r') as f:
            quality = json.load(f)
        
        # Project summary
        with open(metrics_dir / "project_cost_summary.json", 'r') as f:
            project = json.load(f)
        
        # Genera report
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_generations': project['total_generations'],
                'total_successes': project['total_successes'],
                'overall_pass_rate': round(project['total_successes'] / project['total_generations'] * 100, 1),
                'total_cost': project['total_cost'],
                'total_time_minutes': project['total_time_minutes']
            },
            'models_ranking': {},
            'pass_at_k': {
                'pass_at_1': pass_at_1['overall'],
                'pass_at_3': pass_at_3['overall'],
                'pass_at_5': pass_at_5['overall']
            },
            'errors': errors,
            'costs': costs,
            'quality': quality
        }
        
        # Ranking modelli
        models = ['gpt', 'gemini', 'llama', 'qwen']
        for i, model in enumerate(sorted(models, 
                                         key=lambda m: pass_at_1['overall'][m]['pass_rate'], 
                                         reverse=True), 1):
            report['models_ranking'][model] = {
                'rank': i,
                'pass_at_1': pass_at_1['overall'][model]['pass_rate'],
                'total_cost': costs[model]['total_cost'],
                'avg_complexity': quality[model]['avg_complexity'],
                'avg_loc': quality[model]['avg_loc']
            }
        
        # Salva report
        report_path = metrics_dir / "complete_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Report completo salvato: {report_path}")
        
        # Stampa sommario
        print(f"\n{'='*80}")
        print("SOMMARIO RISULTATI")
        print(f"{'='*80}\n")
        
        print(f"   RISULTATI GLOBALI:")
        print(f"   Generazioni totali: {report['summary']['total_generations']:,}")
        print(f"   Successi totali: {report['summary']['total_successes']:,}")
        print(f"   Pass rate complessivo: {report['summary']['overall_pass_rate']}%")
        print(f"   Costo totale: ${report['summary']['total_cost']:.4f}")
        print(f"   Tempo totale: {report['summary']['total_time_minutes']:.1f} min")
        
        print(f"\nRANKING MODELLI (Pass@1):")
        for model, stats in sorted(report['models_ranking'].items(), 
                                   key=lambda x: x[1]['rank']):
            medal = {1: 'primo', 2: 'secondo', 3: 'terzo'}.get(stats['rank'], '  ')
            print(f"   {medal} {stats['rank']}. {model.upper()}: {stats['pass_at_1']}%")
        
        print(f"\n COSTI PER MODELLO:")
        for model in ['gpt', 'gemini', 'llama', 'qwen']:
            print(f"   {model.upper()}: ${costs[model]['total_cost']:.4f}")
        
        print(f"\n QUALITÀ CODICE (Complessità Media):")
        for model in ['gpt', 'gemini', 'llama', 'qwen']:
            cc = quality[model]['avg_complexity']
            status = "successo" if cc < 5 else "attenzione" if cc < 10 else "errore"
            print(f"   {model.upper()}: {cc:.2f} {status}")
        
        return True
        
    except Exception as e:
        print(f" Errore nella generazione del report: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Esegue tutti gli script di analisi"""
    print("="*80)
    print(" GENERAZIONE REPORT COMPLETO - ANALISI ESPERIMENTI")
    print("="*80)
    print(f"\nData: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    scripts = [
        ("src/evaluation/correctness_metrics.py", "Calcolo Pass@1, Pass@3, Pass@5"),
        ("src/evaluation/error_classifier.py", "Classificazione errori"),
        ("src/evaluation/cost_analysis.py", "Analisi costi e performance"),
        ("src/evaluation/quality_metrics.py", "Analisi qualità codice")
    ]
    
    success_count = 0
    
    for script_path, description in scripts:
        if run_script(script_path, description):
            success_count += 1
        else:
            print(f"\n Continuando nonostante l'errore...")
    
    print(f"\n{'='*80}")
    print(f" SCRIPTS ESEGUITI: {success_count}/{len(scripts)}")
    print(f"{'='*80}")
    
    # Genera report riassuntivo
    if success_count == len(scripts):
        generate_summary_report()
    
    print(f"\n{'='*80}")
    print(" PROCESSO COMPLETATO!")
    print(f"{'='*80}\n")
    
    print(" File generati:")
    print("   results/metrics/")
    print("      - pass_at_*_stats.json (metriche correttezza)")
    print("      - error_distribution.json (classificazione errori)")
    print("      - cost_analysis.json (analisi costi)")
    print("      - quality_metrics.json (qualità codice)")
    print("      - complete_report.json (report completo)")
    print("\n   results/tables/")
    print("      - pass_at_*.csv e .tex (tabelle Pass@k)")
    print("      - error_*.csv e .tex (tabelle errori)")
    print("      - cost_*.csv e .tex (tabelle costi)")
    print("      - code_quality_*.csv e .tex (tabelle qualità)")


if __name__ == "__main__":
    main()
