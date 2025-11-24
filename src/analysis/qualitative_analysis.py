"""
Qualitative Code Analysis for Case Studies
Analisi qualitativa approfondita del codice generato per i 6 case studies

Analizza:
1. Approcci algoritmici (iterativo vs ricorsivo, list comprehension, etc.)
2. Pattern di errori (syntax, logic, edge cases)
3. Stile e leggibilità (nomi variabili, commenti, struttura)
4. Complessità e efficienza
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import ast


def load_detailed_solutions() -> Dict:
    """Carica soluzioni estratte"""
    path = Path("results/analysis/case_studies/detailed_solutions.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_code_structure(code: str) -> Dict:
    """
    Analizza la struttura del codice
    
    Returns:
        Dict con metriche strutturali
    """
    analysis = {
        'valid_syntax': False,
        'lines_of_code': 0,
        'blank_lines': 0,
        'comment_lines': 0,
        'has_docstring': False,
        'functions_count': 0,
        'loops_count': 0,
        'conditionals_count': 0,
        'comprehensions_count': 0,
        'recursion': False,
        'imports': [],
        'complexity_indicators': {}
    }
    
    if not code or not code.strip():
        return analysis
    
    lines = code.split('\n')
    analysis['lines_of_code'] = len([l for l in lines if l.strip()])
    analysis['blank_lines'] = len([l for l in lines if not l.strip()])
    analysis['comment_lines'] = len([l for l in lines if l.strip().startswith('#')])
    
    # Docstring check
    analysis['has_docstring'] = '"""' in code or "'''" in code
    
    # Pattern matching (fallback se AST fallisce)
    analysis['loops_count'] = len(re.findall(r'\b(for|while)\b', code))
    analysis['conditionals_count'] = len(re.findall(r'\bif\b', code))
    analysis['comprehensions_count'] = len(re.findall(r'\[.*for.*in.*\]', code))
    
    # Recursion detection
    func_names = re.findall(r'def\s+(\w+)\s*\(', code)
    for func_name in func_names:
        if func_name in code[code.find(f'def {func_name}'):]:
            # Cerca chiamate ricorsive
            pattern = rf'\b{func_name}\s*\('
            if len(re.findall(pattern, code)) > 1:
                analysis['recursion'] = True
                break
    
    # Try parsing with AST for more detailed analysis
    try:
        tree = ast.parse(code)
        analysis['valid_syntax'] = True
        
        # Conta funzioni
        analysis['functions_count'] = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        
        # Import statements
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                analysis['imports'].extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                analysis['imports'].append(node.module)
        
    except SyntaxError as e:
        analysis['valid_syntax'] = False
        analysis['syntax_error'] = str(e)
    
    return analysis


def classify_approach(code: str, structure: Dict) -> str:
    """
    Classifica l'approccio algoritmico
    
    Returns:
        Descrizione dell'approccio (es. 'iterative', 'recursive', 'functional')
    """
    if not code or not code.strip():
        return "empty"
    
    if structure.get('recursion'):
        return "recursive"
    
    if structure.get('comprehensions_count', 0) > 0:
        if structure.get('loops_count', 0) == 0:
            return "functional (comprehensions)"
        else:
            return "mixed (comprehensions + loops)"
    
    if structure.get('loops_count', 0) > 0:
        return "iterative"
    
    if structure.get('conditionals_count', 0) > 0:
        return "conditional logic"
    
    return "direct computation"


def analyze_variable_naming(code: str) -> Dict:
    """Analizza qualità dei nomi delle variabili"""
    
    # Estrai nomi variabili (semplificato)
    var_pattern = r'\b([a-z_][a-z0-9_]*)\s*='
    variables = re.findall(var_pattern, code)
    
    analysis = {
        'total_variables': len(variables),
        'single_char_vars': 0,
        'descriptive_vars': 0,
        'avg_var_length': 0,
        'common_vars': []
    }
    
    if variables:
        analysis['single_char_vars'] = len([v for v in variables if len(v) == 1])
        analysis['descriptive_vars'] = len([v for v in variables if len(v) > 3])
        analysis['avg_var_length'] = sum(len(v) for v in variables) / len(variables)
        analysis['common_vars'] = [v for v, count in Counter(variables).most_common(5)]
    
    return analysis


def compare_solutions(solutions: Dict) -> Dict:
    """
    Confronta le 4 soluzioni (una per modello)
    
    Returns:
        Analisi comparativa dettagliata
    """
    comparison = {
        'problem_id': solutions['problem_id'],
        'difficulty': solutions['difficulty'],
        'category': solutions['category'],
        'models_analysis': {},
        'comparative_insights': {
            'approaches': {},
            'success_patterns': [],
            'failure_patterns': [],
            'quality_ranking': []
        }
    }
    
    # Analizza ogni modello
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        model_data = solutions['models'][model]
        
        # Prendi il primo tentativo riuscito, o il primo se nessuno riesce
        best_attempt = None
        for attempt in model_data['attempts']:
            if attempt['passed']:
                best_attempt = attempt
                break
        
        if not best_attempt and model_data['attempts']:
            best_attempt = model_data['attempts'][0]
        
        if best_attempt:
            code = best_attempt['code']
            
            # Analisi strutturale
            structure = analyze_code_structure(code)
            approach = classify_approach(code, structure)
            naming = analyze_variable_naming(code)
            
            comparison['models_analysis'][model] = {
                'success': best_attempt['passed'],
                'attempt_number': best_attempt['attempt'],
                'code': code,
                'structure': structure,
                'approach': approach,
                'naming_quality': naming,
                'error_info': {
                    'type': best_attempt['error_type'],
                    'message': best_attempt['error'][:200] if best_attempt['error'] else None
                }
            }
            
            # Aggiungi agli insights
            if approach not in comparison['comparative_insights']['approaches']:
                comparison['comparative_insights']['approaches'][approach] = []
            comparison['comparative_insights']['approaches'][approach].append(model)
    
    # Identifica pattern di successo/fallimento
    successful_models = [m for m in ['gpt', 'gemini', 'llama', 'qwen'] 
                        if comparison['models_analysis'].get(m, {}).get('success')]
    failed_models = [m for m in ['gpt', 'gemini', 'llama', 'qwen']
                    if not comparison['models_analysis'].get(m, {}).get('success')]
    
    if successful_models:
        # Pattern di successo
        success_approaches = [comparison['models_analysis'][m]['approach'] for m in successful_models]
        most_common_success = Counter(success_approaches).most_common(1)[0][0]
        comparison['comparative_insights']['success_patterns'].append(
            f"Successful approach: {most_common_success} (used by {', '.join(successful_models)})"
        )
    
    if failed_models:
        # Pattern di fallimento
        error_types = [comparison['models_analysis'][m]['error_info']['type'] 
                      for m in failed_models if comparison['models_analysis'][m]['error_info']['type']]
        if error_types:
            most_common_error = Counter(error_types).most_common(1)[0][0]
            comparison['comparative_insights']['failure_patterns'].append(
                f"Common error: {most_common_error} (in {', '.join(failed_models)})"
            )
    
    # Ranking qualità (basato su successo, complessità, naming)
    quality_scores = {}
    for model, analysis in comparison['models_analysis'].items():
        score = 0
        
        # Successo = peso maggiore
        if analysis['success']:
            score += 100
        
        # Codice valido sintatticamente
        if analysis['structure'].get('valid_syntax'):
            score += 20
        
        # Naming quality
        naming = analysis['naming_quality']
        if naming['total_variables'] > 0:
            desc_ratio = naming['descriptive_vars'] / naming['total_variables']
            score += desc_ratio * 10
        
        # Docstring
        if analysis['structure'].get('has_docstring'):
            score += 5
        
        # Commenti
        if analysis['structure'].get('comment_lines', 0) > 0:
            score += 3
        
        quality_scores[model] = score
    
    ranked = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
    comparison['comparative_insights']['quality_ranking'] = [
        {'model': m, 'score': s} for m, s in ranked
    ]
    
    return comparison


def generate_qualitative_report(comparison: Dict) -> str:
    """Genera report qualitativo testuale"""
    
    report = f"""
{'='*80}
QUALITATIVE ANALYSIS: {comparison['problem_id']}
{'='*80}

Difficulty: {comparison['difficulty']}
Category: {comparison['category']}

{'='*80}
APPROACHES COMPARISON
{'='*80}

"""
    
    for approach, models in comparison['comparative_insights']['approaches'].items():
        report += f"{approach.upper()}: {', '.join(models)}\n"
    
    report += f"\n{'='*80}\n"
    report += "DETAILED MODEL ANALYSIS\n"
    report += f"{'='*80}\n\n"
    
    for model in ['gpt', 'gemini', 'llama', 'qwen']:
        if model not in comparison['models_analysis']:
            continue
        
        analysis = comparison['models_analysis'][model]
        
        report += f"{model.upper()}:\n"
        report += f"  Status: {'✅ PASSED' if analysis['success'] else '❌ FAILED'}\n"
        report += f"  Attempt: #{analysis['attempt_number']}\n"
        report += f"  Approach: {analysis['approach']}\n"
        
        struct = analysis['structure']
        report += f"  Structure:\n"
        report += f"    - Lines of code: {struct['lines_of_code']}\n"
        report += f"    - Functions: {struct['functions_count']}\n"
        report += f"    - Loops: {struct['loops_count']}\n"
        report += f"    - Conditionals: {struct['conditionals_count']}\n"
        report += f"    - Comprehensions: {struct['comprehensions_count']}\n"
        report += f"    - Has docstring: {struct['has_docstring']}\n"
        report += f"    - Comment lines: {struct['comment_lines']}\n"
        
        naming = analysis['naming_quality']
        if naming['total_variables'] > 0:
            report += f"  Variable Naming:\n"
            report += f"    - Total variables: {naming['total_variables']}\n"
            report += f"    - Descriptive: {naming['descriptive_vars']} ({naming['descriptive_vars']/naming['total_variables']*100:.1f}%)\n"
            report += f"    - Avg length: {naming['avg_var_length']:.1f} chars\n"
        
        if not analysis['success']:
            error = analysis['error_info']
            report += f"  Error:\n"
            report += f"    - Type: {error['type']}\n"
            if error['message']:
                report += f"    - Message: {error['message'][:100]}...\n"
        
        report += "\n"
    
    report += f"{'='*80}\n"
    report += "INSIGHTS\n"
    report += f"{'='*80}\n\n"
    
    for pattern in comparison['comparative_insights']['success_patterns']:
        report += f" {pattern}\n"
    
    for pattern in comparison['comparative_insights']['failure_patterns']:
        report += f" {pattern}\n"
    
    report += f"\nQuality Ranking:\n"
    for idx, item in enumerate(comparison['comparative_insights']['quality_ranking'], 1):
        report += f"  {idx}. {item['model'].upper()}: {item['score']:.1f} points\n"
    
    return report


def main():
    print("="*80)
    print(" ANALISI QUALITATIVA CASE STUDIES")
    print("="*80)
    
    # Carica soluzioni
    print("\n Caricamento soluzioni...")
    data = load_detailed_solutions()
    case_studies = data['case_studies']
    
    print(f"    {len(case_studies)} case studies caricati")
    
    # Analizza ogni case study
    print("\n Analisi qualitativa in corso...")
    
    all_comparisons = []
    all_reports = []
    
    for idx, solutions in enumerate(case_studies, 1):
        problem_id = solutions['problem_id']
        
        print(f"\n   [{idx}/6] {problem_id} ({solutions['difficulty']}, {solutions['category']})")
        
        # Confronta soluzioni
        comparison = compare_solutions(solutions)
        
        # Genera report
        report = generate_qualitative_report(comparison)
        
        all_comparisons.append(comparison)
        all_reports.append(report)
        
        # Print breve insight
        approaches = comparison['comparative_insights']['approaches']
        print(f"      Approaches: {', '.join(approaches.keys())}")
        
        quality_top = comparison['comparative_insights']['quality_ranking'][0]
        print(f"      Best quality: {quality_top['model'].upper()} ({quality_top['score']:.0f} pts)")
    
    # Salva risultati
    print("\n Salvataggio analisi...")
    
    output_dir = Path("results/analysis/case_studies")
    
    # 1. JSON con analisi qualitativa completa
    analysis_file = output_dir / "qualitative_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': data['generated_at'],
            'qualitative_analyses': all_comparisons
        }, f, indent=2, ensure_ascii=False)
    
    print(f"    Analisi qualitativa: {analysis_file}")
    
    # 2. Report testuali dettagliati
    report_file = output_dir / "qualitative_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_reports))
    
    print(f"    Report qualitativo: {report_file}")
    
    # 3. Summary insights (per tesi)
    insights_file = output_dir / "key_insights.md"
    with open(insights_file, 'w', encoding='utf-8') as f:
        f.write("# Key Insights from Case Studies\n\n")
        
        for idx, comp in enumerate(all_comparisons, 1):
            f.write(f"## Case Study {idx}: {comp['problem_id']}\n\n")
            f.write(f"**Difficulty:** {comp['difficulty']} | **Category:** {comp['category']}\n\n")
            
            f.write("### Approaches\n\n")
            for approach, models in comp['comparative_insights']['approaches'].items():
                f.write(f"- **{approach}**: {', '.join(models).upper()}\n")
            
            f.write("\n### Success Patterns\n\n")
            for pattern in comp['comparative_insights']['success_patterns']:
                f.write(f"- {pattern}\n")
            
            f.write("\n### Failure Patterns\n\n")
            for pattern in comp['comparative_insights']['failure_patterns']:
                f.write(f"- {pattern}\n")
            
            f.write("\n### Quality Ranking\n\n")
            for idx_rank, item in enumerate(comp['comparative_insights']['quality_ranking'], 1):
                f.write(f"{idx_rank}. **{item['model'].upper()}**: {item['score']:.1f} points\n")
            
            f.write("\n---\n\n")
    
    print(f"    Key insights (Markdown): {insights_file}")
    
    print("\n" + "="*80)
    print(" ANALISI QUALITATIVA COMPLETATA")
    print("="*80)
    print(f"\n Directory output: {output_dir}")
    print(f" File generati:")
    print(f"   - qualitative_analysis.json (analisi completa)")
    print(f"   - qualitative_report.txt (report dettagliato)")
    print(f"   - key_insights.md (insights per tesi)")


if __name__ == "__main__":
    main()
