"""
Extract Case Study Solutions
Estrae le soluzioni generate dai 4 modelli per i 6 case studies selezionati
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_case_studies() -> Dict:
    """Carica case studies selezionati"""
    path = Path("results/analysis/selected_case_studies.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_results() -> Dict:
    """Carica risultati completi"""
    path = Path("results/raw_outputs/full_experiments/full_experiments_gemini25_20251031_135614.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_problems() -> Dict:
    """Carica problemi originali"""
    path = Path("src/data/selected_problems/selected_problems.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_solutions_for_problem(
    problem_id: str,
    results: List[Dict],
    problem_info: Dict
) -> Dict:
    """
    Estrae tutte le soluzioni per un problema specifico
    
    Returns:
        Dict con soluzioni per modello e statistiche
    """
    solutions = {
        'problem_id': problem_id,
        'difficulty': problem_info['difficulty'],
        'category': problem_info['category'],
        'prompt': problem_info['prompt'],
        'test_count': len(problem_info.get('test_list', [])) if problem_info['source'] == 'mbpp' else 1,
        'canonical_solution': problem_info.get('canonical_solution', ''),
        'models': {}
    }
    
    # Estrai soluzioni per ogni modello
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        model_results = [
            r for r in results
            if r['problem_id'] == problem_id and r['model'] == model
        ]
        
        # Ordina per attempt
        model_results.sort(key=lambda x: x['attempt'])
        
        attempts = []
        for result in model_results:
            exec_result = result.get('execution_result', {})
            
            attempt_data = {
                'attempt': result['attempt'],
                'passed': exec_result.get('passed', False),
                'code': result.get('generated_code', ''),
                'error': exec_result.get('error', None),
                'error_type': exec_result.get('error_type', None),
                'execution_time': exec_result.get('execution_time', 0.0),
                'latency': result.get('latency', 0.0),
                'tokens': result.get('tokens', 0),
                'cost': result.get('cost', 0.0)
            }
            attempts.append(attempt_data)
        
        # Statistiche modello
        passed_count = sum(1 for a in attempts if a['passed'])
        
        solutions['models'][model] = {
            'attempts': attempts,
            'passed_count': passed_count,
            'pass_rate': passed_count / len(attempts) * 100 if attempts else 0,
            'first_success': next((a['attempt'] for a in attempts if a['passed']), None),
            'avg_tokens': sum(a['tokens'] for a in attempts) / len(attempts) if attempts else 0,
            'avg_latency': sum(a['latency'] for a in attempts) / len(attempts) if attempts else 0
        }
    
    return solutions


def create_comparative_analysis(solutions: Dict) -> Dict:
    """
    Crea analisi comparativa delle soluzioni
    
    Confronta:
    - Codice generato (approcci diversi)
    - Pattern di errori
    - Complessità (tokens, linee)
    - Performance (tempo esecuzione)
    """
    analysis = {
        'problem_id': solutions['problem_id'],
        'difficulty': solutions['difficulty'],
        'category': solutions['category'],
        'comparison': {}
    }
    
    # Confronta per ogni modello
    for model, data in solutions['models'].items():
        # Prendi il primo tentativo riuscito (o il primo se nessuno riesce)
        best_attempt = None
        for attempt in data['attempts']:
            if attempt['passed']:
                best_attempt = attempt
                break
        
        if not best_attempt:
            best_attempt = data['attempts'][0] if data['attempts'] else None
        
        if best_attempt:
            code = best_attempt['code']
            
            analysis['comparison'][model] = {
                'success': best_attempt['passed'],
                'attempt_number': best_attempt['attempt'],
                'code_lines': len([line for line in code.split('\n') if line.strip()]),
                'code_length': len(code),
                'has_comments': '#' in code,
                'has_docstring': '"""' in code or "'''" in code,
                'error_type': best_attempt['error_type'],
                'execution_time': best_attempt['execution_time'],
                'tokens_used': best_attempt['tokens']
            }
    
    return analysis


def generate_case_study_report(case_study_id: str, solutions: Dict, analysis: Dict) -> str:
    """Genera report testuale per un case study"""
    
    report = f"""
{'='*80}
CASE STUDY: {case_study_id}
{'='*80}

Difficulty: {solutions['difficulty']}
Category: {solutions['category']}

PROBLEM DESCRIPTION:
{solutions['prompt'][:500]}...

{'='*80}
RESULTS SUMMARY
{'='*80}

"""
    
    # Tabella risultati
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        model_data = solutions['models'][model]
        comp_data = analysis['comparison'].get(model, {})
        
        report += f"\n{model.upper():8s}:\n"
        report += f"  Pass Rate: {model_data['pass_rate']:.1f}% ({model_data['passed_count']}/5)\n"
        report += f"  First Success: Attempt #{model_data['first_success']}" if model_data['first_success'] else "  First Success: None\n"
        report += f"\n  Avg Tokens: {model_data['avg_tokens']:.0f}\n"
        report += f"  Avg Latency: {model_data['avg_latency']:.2f}s\n"
        
        if comp_data.get('success'):
            report += f"  Success on attempt #{comp_data['attempt_number']}\n"
            report += f"  Code: {comp_data['code_lines']} lines, {comp_data['code_length']} chars\n"
        else:
            report += f"  Failed: {comp_data.get('error_type', 'unknown')}\n"
    
    return report


def main():
    print("="*80)
    print(" ESTRAZIONE SOLUZIONI CASE STUDIES")
    print("="*80)
    
    # Carica dati
    print("\n Caricamento dati...")
    case_studies = load_case_studies()
    results_data = load_results()
    problems_data = load_problems()
    
    results = results_data['results']
    problems = {p['id']: p for p in problems_data['problems']}
    
    print(f"    {len(case_studies['case_studies'])} case studies")
    print(f"    {len(results)} risultati totali")
    print(f"    {len(problems)} problemi")
    
    # Estrai soluzioni per ogni case study
    print("\n Estrazione soluzioni...")
    
    all_solutions = []
    all_analyses = []
    all_reports = []
    
    for idx, case_study in enumerate(case_studies['case_studies'], 1):
        problem_id = case_study['problem_id']
        
        print(f"\n   [{idx}/6] {problem_id} ({case_study['difficulty']}, {case_study['category']})")
        
        # Estrai soluzioni
        solutions = extract_solutions_for_problem(
            problem_id,
            results,
            problems[problem_id]
        )
        
        # Analisi comparativa
        analysis = create_comparative_analysis(solutions)
        
        # Report testuale
        report = generate_case_study_report(case_study['reason'], solutions, analysis)
        
        all_solutions.append(solutions)
        all_analyses.append(analysis)
        all_reports.append(report)
        
        # Print breve summary
        for model in ['gpt', 'gemini', 'llama', 'qwen']:
            passed = solutions['models'][model]['passed_count']
            print(f"      {model:8s}: {passed}/5 passed", end="")
            if passed > 0:
                first = solutions['models'][model]['first_success']
                print(f" (first at #{first})")
            else:
                print()
    
    # Salva risultati
    print("\n Salvataggio risultati...")
    
    output_dir = Path("results/analysis/case_studies")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON completo con tutte le soluzioni
    solutions_file = output_dir / "detailed_solutions.json"
    with open(solutions_file, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': case_studies['generated_at'],
            'case_studies': all_solutions
        }, f, indent=2, ensure_ascii=False)
    
    print(f"    Soluzioni dettagliate: {solutions_file}")
    
    # 2. JSON con analisi comparative
    analysis_file = output_dir / "comparative_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': case_studies['generated_at'],
            'analyses': all_analyses
        }, f, indent=2, ensure_ascii=False)
    
    print(f"    Analisi comparative: {analysis_file}")
    
    # 3. Report testuali
    report_file = output_dir / "case_studies_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_reports))
    
    print(f"    Report testuale: {report_file}")
    
    # 4. CSV summary per tabella tesi
    summary_data = []
    for solution in all_solutions:
        for model in ['gpt', 'gemini', 'llama', 'qwen']:
            model_data = solution['models'][model]
            summary_data.append({
                'problem_id': solution['problem_id'],
                'difficulty': solution['difficulty'],
                'category': solution['category'],
                'model': model,
                'pass_rate': model_data['pass_rate'],
                'first_success': model_data['first_success'],
                'avg_tokens': model_data['avg_tokens'],
                'avg_latency': model_data['avg_latency']
            })
    
    df = pd.DataFrame(summary_data)
    csv_file = output_dir / "case_studies_summary.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"    CSV summary: {csv_file}")
    
    print("\n" + "="*80)
    print(" ESTRAZIONE COMPLETATA")
    print("="*80)
    print(f"\n Directory output: {output_dir}")
    print(f" Prossimo step: Analisi qualitativa dettagliata del codice")


if __name__ == "__main__":
    main()
