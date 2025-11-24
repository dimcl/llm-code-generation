"""
Case Studies Selection - Automatic Selection of Representative Problems

Seleziona automaticamente 6-8 problemi rappresentativi per analisi qualitativa:
- Problemi con performance diverse tra modelli
- Copertura di tutte le difficoltà (easy/medium/hard)
- Copertura di tutte le categorie (strings/lists/math/algorithms)
- Problemi che mostrano pattern interessanti

Output: Lista problemi selezionati con motivazioni
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_results() -> Dict:
    """Carica risultati esperimenti."""
    results_path = Path("results/raw_outputs/full_experiments/full_experiments_gemini25_20251031_135614.json")
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_problems() -> Dict:
    """Carica problemi selezionati."""
    problems_path = Path("src/data/selected_problems/selected_problems.json")
    with open(problems_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_problem_statistics(results: Dict, problems: Dict) -> pd.DataFrame:
    """
    Calcola statistiche per ogni problema:
    - Success rate per modello
    - Variance tra modelli
    - Complessità problema
    """
    
    problem_stats = []
    
    for problem in problems['problems']:
        problem_id = problem['id']
        
        # Conta successi per modello (test PASSED, non solo generazione ok)
        model_success = {'gpt': 0, 'gemini': 0, 'llama': 0, 'qwen': 0}
        total_attempts = {'gpt': 0, 'gemini': 0, 'llama': 0, 'qwen': 0}
        
        for result in results['results']:
            if result['problem_id'] == problem_id:
                model = result['model']
                total_attempts[model] += 1
                # FIX: Usa execution_result.passed invece di success (che indica solo API ok)
                if result.get('execution_result', {}).get('passed', False):
                    model_success[model] += 1
        
        # Calcola pass rate per modello (su 5 tentativi)
        pass_rates = {
            model: (model_success[model] / total_attempts[model] * 100) if total_attempts[model] > 0 else 0
            for model in model_success.keys()
        }
        
        # Calcola variance (indica quanto i modelli performano diversamente)
        rates = list(pass_rates.values())
        mean_rate = sum(rates) / len(rates)
        variance = sum((r - mean_rate) ** 2 for r in rates) / len(rates)
        
        # Conta quanti modelli hanno successo almeno 1 volta
        models_with_success = sum(1 for count in model_success.values() if count > 0)
        
        problem_stats.append({
            'problem_id': problem_id,
            'difficulty': problem['difficulty'],
            'category': problem['category'],
            'source': problem['source'],
            'gpt_pass_rate': pass_rates['gpt'],
            'gemini_pass_rate': pass_rates['gemini'],
            'llama_pass_rate': pass_rates['llama'],
            'qwen_pass_rate': pass_rates['qwen'],
            'mean_pass_rate': mean_rate,
            'variance': variance,
            'models_with_success': models_with_success,
            'title': problem.get('title', problem_id)
        })
    
    return pd.DataFrame(problem_stats)


def select_case_studies(df: pd.DataFrame) -> List[Dict]:
    """
    Seleziona 6-8 problemi rappresentativi seguendo criteri:
    
    1. EASY problems (2):
       - 1 con successo universale (tutti modelli ok)
       - 1 con solo Qwen fallimento (mostrare limite Qwen)
    
    2. MEDIUM problems (2-3):
       - 1 con alta variance (modelli performano molto diversamente)
       - 1 con performance moderate (tutti intorno al 50-70%)
       - Opzionale: 1 con fallimento totale
    
    3. HARD problems (2-3):
       - 1 con solo GPT successo (mostrare superiorità GPT)
       - 1 con vari successi (confronto approcci)
       - 1 con fallimento totale (problema troppo difficile)
    
    Criteri secondari:
    - Copertura categorie (almeno 1 per categoria)
    - Problemi HumanEval (più noti e citabili)
    """
    
    selected = []
    
    # ========================================================================
    # 1. EASY PROBLEMS
    # ========================================================================
    
    easy_df = df[df['difficulty'] == 'easy'].copy()
    
    # Easy #1: Successo universale (tutti ≥60%)
    universal_success = easy_df[
        (easy_df['gpt_pass_rate'] >= 60) &
        (easy_df['gemini_pass_rate'] >= 60) &
        (easy_df['llama_pass_rate'] >= 60) &
        (easy_df['qwen_pass_rate'] >= 60)
    ].sort_values('mean_pass_rate', ascending=False)
    
    if len(universal_success) > 0:
        best = universal_success.iloc[0]
        selected.append({
            'problem_id': best['problem_id'],
            'difficulty': 'easy',
            'category': best['category'],
            'title': best['title'],
            'reason': f"Successo universale (mean={best['mean_pass_rate']:.1f}%) - Mostra che problema è effettivamente facile",
            'pass_rates': {
                'gpt': best['gpt_pass_rate'],
                'gemini': best['gemini_pass_rate'],
                'llama': best['llama_pass_rate'],
                'qwen': best['qwen_pass_rate']
            }
        })
    
    # Easy #2: Solo Qwen fallisce (altri ≥40%, Qwen ≤40%)
    qwen_fails = easy_df[
        (easy_df['gpt_pass_rate'] >= 40) &
        (easy_df['gemini_pass_rate'] >= 40) &
        (easy_df['llama_pass_rate'] >= 40) &
        (easy_df['qwen_pass_rate'] <= 40)
    ].sort_values('variance', ascending=False)
    
    if len(qwen_fails) > 0:
        worst_qwen = qwen_fails.iloc[0]
        selected.append({
            'problem_id': worst_qwen['problem_id'],
            'difficulty': 'easy',
            'category': worst_qwen['category'],
            'title': worst_qwen['title'],
            'reason': f"Evidenzia limite Qwen su problema easy (variance={worst_qwen['variance']:.1f})",
            'pass_rates': {
                'gpt': worst_qwen['gpt_pass_rate'],
                'gemini': worst_qwen['gemini_pass_rate'],
                'llama': worst_qwen['llama_pass_rate'],
                'qwen': worst_qwen['qwen_pass_rate']
            }
        })
    
    # ========================================================================
    # 2. MEDIUM PROBLEMS
    # ========================================================================
    
    medium_df = df[df['difficulty'] == 'medium'].copy()
    
    # Medium #1: Alta variance (modelli molto diversi)
    high_variance = medium_df.sort_values('variance', ascending=False).head(5)
    
    if len(high_variance) > 0:
        diverse = high_variance.iloc[0]
        selected.append({
            'problem_id': diverse['problem_id'],
            'difficulty': 'medium',
            'category': diverse['category'],
            'title': diverse['title'],
            'reason': f"Massima divergenza modelli (variance={diverse['variance']:.1f}) - Confronto approcci",
            'pass_rates': {
                'gpt': diverse['gpt_pass_rate'],
                'gemini': diverse['gemini_pass_rate'],
                'llama': diverse['llama_pass_rate'],
                'qwen': diverse['qwen_pass_rate']
            }
        })
    
    # Medium #2: Performance moderate (tutti 40-70%)
    moderate = medium_df[
        (medium_df['mean_pass_rate'] >= 40) &
        (medium_df['mean_pass_rate'] <= 70) &
        (medium_df['variance'] < 500)  # Non troppa variance
    ].sort_values('mean_pass_rate')
    
    if len(moderate) > 0:
        balanced = moderate.iloc[len(moderate)//2]  # Prendi quello centrale
        selected.append({
            'problem_id': balanced['problem_id'],
            'difficulty': 'medium',
            'category': balanced['category'],
            'title': balanced['title'],
            'reason': f"Performance moderate bilanciate (mean={balanced['mean_pass_rate']:.1f}%, var={balanced['variance']:.1f})",
            'pass_rates': {
                'gpt': balanced['gpt_pass_rate'],
                'gemini': balanced['gemini_pass_rate'],
                'llama': balanced['llama_pass_rate'],
                'qwen': balanced['qwen_pass_rate']
            }
        })
    
    # ========================================================================
    # 3. HARD PROBLEMS
    # ========================================================================
    
    hard_df = df[df['difficulty'] == 'hard'].copy()
    
    # Hard #1: Solo GPT successo (GPT ≥40%, altri ≤30%)
    gpt_dominance = hard_df[
        (hard_df['gpt_pass_rate'] >= 40) &
        (hard_df['gemini_pass_rate'] <= 30) &
        (hard_df['llama_pass_rate'] <= 30)
    ].sort_values('gpt_pass_rate', ascending=False)
    
    if len(gpt_dominance) > 0:
        gpt_best = gpt_dominance.iloc[0]
        selected.append({
            'problem_id': gpt_best['problem_id'],
            'difficulty': 'hard',
            'category': gpt_best['category'],
            'title': gpt_best['title'],
            'reason': f"GPT domina (GPT={gpt_best['gpt_pass_rate']:.1f}% vs altri ≤40%) - Mostra superiorità",
            'pass_rates': {
                'gpt': gpt_best['gpt_pass_rate'],
                'gemini': gpt_best['gemini_pass_rate'],
                'llama': gpt_best['llama_pass_rate'],
                'qwen': gpt_best['qwen_pass_rate']
            }
        })
    
    # Hard #2: Vari successi (almeno 2 modelli con ≥20%)
    multiple_success = hard_df[
        hard_df['models_with_success'] >= 2
    ].sort_values('variance', ascending=False)
    
    if len(multiple_success) > 0:
        diverse_hard = multiple_success.iloc[0]
        selected.append({
            'problem_id': diverse_hard['problem_id'],
            'difficulty': 'hard',
            'category': diverse_hard['category'],
            'title': diverse_hard['title'],
            'reason': f"Vari modelli riescono (variance={diverse_hard['variance']:.1f}) - Confronto approcci hard",
            'pass_rates': {
                'gpt': diverse_hard['gpt_pass_rate'],
                'gemini': diverse_hard['gemini_pass_rate'],
                'llama': diverse_hard['llama_pass_rate'],
                'qwen': diverse_hard['qwen_pass_rate']
            }
        })
    
    # Hard #3: Fallimento totale (tutti ≤20%)
    total_failure = hard_df[
        (hard_df['gpt_pass_rate'] <= 20) &
        (hard_df['gemini_pass_rate'] <= 20) &
        (hard_df['llama_pass_rate'] <= 20) &
        (hard_df['qwen_pass_rate'] <= 20)
    ].sort_values('mean_pass_rate')
    
    if len(total_failure) > 0:
        hardest = total_failure.iloc[0]
        selected.append({
            'problem_id': hardest['problem_id'],
            'difficulty': 'hard',
            'category': hardest['category'],
            'title': hardest['title'],
            'reason': f"Fallimento universale (mean={hardest['mean_pass_rate']:.1f}%) - Limite attuale LLM",
            'pass_rates': {
                'gpt': hardest['gpt_pass_rate'],
                'gemini': hardest['gemini_pass_rate'],
                'llama': hardest['llama_pass_rate'],
                'qwen': hardest['qwen_pass_rate']
            }
        })
    
    return selected


def verify_category_coverage(selected: List[Dict]) -> Dict:
    """Verifica copertura categorie."""
    categories = {'strings': 0, 'lists': 0, 'math': 0, 'algorithms': 0}
    
    for case in selected:
        cat = case['category']
        if cat in categories:
            categories[cat] += 1
    
    return categories


def save_case_studies(selected: List[Dict], output_path: Path):
    """Salva case studies selezionati."""
    output = {
        'generated_at': '2025-10-31T17:00:00',
        'total_selected': len(selected),
        'selection_criteria': [
            'Copertura difficoltà: 2 easy + 2-3 medium + 2-3 hard',
            'Performance diverse tra modelli (alta variance)',
            'Successi universali e fallimenti totali',
            'Problemi che evidenziano forze/debolezze specifiche'
        ],
        'case_studies': selected
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def print_summary(selected: List[Dict], categories: Dict):
    """Stampa riassunto selezione."""
    print("=" * 80)
    print(" CASE STUDIES SELEZIONATI")
    print("=" * 80)
    print()
    
    print(f"Totale selezionati: {len(selected)}")
    print()
    
    # Raggruppa per difficoltà
    by_difficulty = {'easy': [], 'medium': [], 'hard': []}
    for case in selected:
        by_difficulty[case['difficulty']].append(case)
    
    for difficulty in ['easy', 'medium', 'hard']:
        cases = by_difficulty[difficulty]
        if cases:
            print(f"{'=' * 80}")
            print(f" {difficulty.upper()} ({len(cases)} problemi)")
            print(f"{'=' * 80}")
            
            for i, case in enumerate(cases, 1):
                print(f"\n{i}. {case['problem_id']} - {case['title']}")
                print(f"   Categoria: {case['category']}")
                print(f"   Motivo: {case['reason']}")
                print(f"   Pass rates:")
                for model in ['gpt', 'gemini', 'llama', 'qwen']:
                    rate = case['pass_rates'][model]
                    bar = '█' * int(rate / 5)
                    print(f"     {model:6s}: {rate:5.1f}% {bar}")
            print()
    
    print("=" * 80)
    print(" COPERTURA CATEGORIE")
    print("=" * 80)
    for cat, count in categories.items():
        print(f"  {cat:12s}: {count} problemi")
    print()
    
    missing = [cat for cat, count in categories.items() if count == 0]
    if missing:
        print(f"  Categorie mancanti: {', '.join(missing)}")
        print("   Considera aggiungere 1-2 problemi per coprire tutte le categorie")
    else:
        print(" Tutte le categorie coperte!")
    print()


def main():
    """Main execution."""
    print("=" * 80)
    print(" SELEZIONE AUTOMATICA CASE STUDIES")
    print("=" * 80)
    print()
    
    # Load data
    print(" Caricamento dati...")
    results = load_results()
    problems = load_problems()
    print(f"    {len(results['results'])} risultati")
    print(f"    {len(problems['problems'])} problemi")
    print()
    
    # Calculate statistics
    print(" Calcolo statistiche per problema...")
    df = calculate_problem_statistics(results, problems)
    print(f"    Statistiche calcolate per {len(df)} problemi")
    print()
    
    # Select case studies
    print(" Selezione case studies...")
    selected = select_case_studies(df)
    print(f"    {len(selected)} problemi selezionati")
    print()
    
    # Verify coverage
    categories = verify_category_coverage(selected)
    
    # Save results
    output_path = Path("results/analysis/selected_case_studies.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_case_studies(selected, output_path)
    print(f"    Salvato: {output_path}")
    print()
    
    # Print summary
    print_summary(selected, categories)
    
    print("=" * 80)
    print(" SELEZIONE COMPLETATA!")
    print("=" * 80)
    print()
    print(f" File generato: {output_path}")
    print(f" Prossimo step: Analisi qualitativa dettagliata di {len(selected)} problemi")
    print()


if __name__ == "__main__":
    main()
