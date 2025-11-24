"""
Test statistici per confronto performance modelli LLM.

Implementa:
1. ANOVA one-way: confronto medie tra modelli
2. Kruskal-Wallis: test non-parametrico (se dati non normali)
3. Post-hoc tests: Tukey HSD, Mann-Whitney U
4. Normalità: Shapiro-Wilk test
5. Omogeneità varianze: Levene test

Obiettivo: Verificare se le differenze tra modelli sono statisticamente significative.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Statistical tests
from scipy import stats
from scipy.stats import shapiro, levene, f_oneway, kruskal
from scipy.stats import mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def load_results(file_path: str) -> Tuple[dict, List[dict]]:
    """Carica risultati da JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['metadata'], data['results']


def load_problems(file_path: str) -> Dict[str, dict]:
    """Carica info problemi"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problems = {}
    for problem in data['problems']:
        problems[problem['id']] = {
            'difficulty': problem['difficulty'],
            'category': problem['category']
        }
    return problems


def extract_success_rates_by_problem(results: List[dict]) -> Dict[str, Dict[str, float]]:
    """
    Estrae success rate per ogni (modello, problema).
    
    Returns:
        Dict[model][problem_id] = success_rate (0.0-1.0)
    """
    # Raggruppa per (modello, problema)
    by_model_problem = defaultdict(list)
    
    for result in results:
        model = result['model']
        problem_id = result['problem_id']
        passed = result.get('execution_result', {}).get('passed', False)
        by_model_problem[(model, problem_id)].append(1 if passed else 0)
    
    # Calcola success rate per ogni combinazione
    success_rates = defaultdict(dict)
    
    for (model, problem_id), outcomes in by_model_problem.items():
        success_rates[model][problem_id] = np.mean(outcomes)
    
    return dict(success_rates)


def prepare_data_for_anova(success_rates: Dict[str, Dict[str, float]], 
                           models: List[str] = None) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    Prepara dati per ANOVA: matrice modelli × problemi.
    
    Returns:
        - DataFrame con success rates
        - Lista di array (uno per modello) per ANOVA
    """
    if models is None:
        models = ['gpt', 'gemini', 'llama', 'qwen']
    
    # Crea DataFrame
    data = {}
    for model in models:
        data[model] = []
        for problem_id in sorted(success_rates[models[0]].keys()):
            data[model].append(success_rates[model][problem_id])
    
    df = pd.DataFrame(data, index=sorted(success_rates[models[0]].keys()))
    
    # Prepara liste per ANOVA
    groups = [df[model].values for model in models]
    
    return df, groups


def test_normality(data: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Test normalità Shapiro-Wilk.
    
    H0: dati seguono distribuzione normale
    """
    statistic, p_value = shapiro(data)
    
    return {
        'test': 'Shapiro-Wilk',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_normal': bool(p_value > alpha),
        'alpha': alpha
    }


def test_variance_homogeneity(groups: List[np.ndarray], alpha: float = 0.05) -> Dict:
    """
    Test omogeneità varianze (Levene).
    
    H0: varianze sono omogenee tra i gruppi
    """
    statistic, p_value = levene(*groups)
    
    return {
        'test': 'Levene',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_homogeneous': bool(p_value > alpha),
        'alpha': alpha
    }


def anova_one_way(groups: List[np.ndarray], 
                  group_names: List[str] = None,
                  alpha: float = 0.05) -> Dict:
    """
    ANOVA one-way: confronto medie tra gruppi.
    
    H0: le medie dei gruppi sono uguali
    H1: almeno una media è diversa
    """
    if group_names is None:
        group_names = ['gpt', 'gemini', 'llama', 'qwen']
    
    # Test ANOVA
    f_statistic, p_value = f_oneway(*groups)
    
    # Calcola statistiche descrittive
    means = [np.mean(g) for g in groups]
    stds = [np.std(g, ddof=1) for g in groups]
    
    return {
        'test': 'ANOVA one-way',
        'f_statistic': float(f_statistic),
        'p_value': float(p_value),
        'is_significant': bool(p_value < alpha),
        'alpha': alpha,
        'groups': {
            name: {
                'mean': float(mean),
                'std': float(std),
                'n': len(groups[i])
            }
            for i, (name, mean, std) in enumerate(zip(group_names, means, stds))
        },
        'interpretation': (
            f"Le differenze tra i modelli sono {'statisticamente significative' if p_value < alpha else 'NON significative'} "
            f"(p={p_value:.4f}, α={alpha})"
        )
    }


def kruskal_wallis_test(groups: List[np.ndarray],
                        group_names: List[str] = None,
                        alpha: float = 0.05) -> Dict:
    """
    Test Kruskal-Wallis: alternativa non-parametrica ad ANOVA.
    
    H0: le distribuzioni dei gruppi sono identiche
    H1: almeno una distribuzione è diversa
    """
    if group_names is None:
        group_names = ['gpt', 'gemini', 'llama', 'qwen']
    
    # Test Kruskal-Wallis
    h_statistic, p_value = kruskal(*groups)
    
    # Calcola mediane
    medians = [np.median(g) for g in groups]
    
    return {
        'test': 'Kruskal-Wallis',
        'h_statistic': float(h_statistic),
        'p_value': float(p_value),
        'is_significant': bool(p_value < alpha),
        'alpha': alpha,
        'groups': {
            name: {
                'median': float(median),
                'n': len(groups[i])
            }
            for i, (name, median) in enumerate(zip(group_names, medians))
        },
        'interpretation': (
            f"Le differenze tra i modelli sono {'statisticamente significative' if p_value < alpha else 'NON significative'} "
            f"(p={p_value:.4f}, α={alpha})"
        )
    }


def tukey_hsd_test(df: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """
    Test post-hoc Tukey HSD: confronti a coppie dopo ANOVA.
    
    Identifica quali coppie di modelli differiscono significativamente.
    """
    # Prepara dati in formato long
    data_long = []
    for model in df.columns:
        for value in df[model]:
            data_long.append({'model': model, 'success_rate': value})
    
    df_long = pd.DataFrame(data_long)
    
    # Test Tukey HSD
    tukey = pairwise_tukeyhsd(
        endog=df_long['success_rate'],
        groups=df_long['model'],
        alpha=alpha
    )
    
    # Parse risultati
    comparisons = []
    for i in range(len(tukey.summary().data) - 1):  # Skip header
        row = tukey.summary().data[i + 1]
        group1, group2, meandiff, p_adj, lower, upper, reject = row
        
        comparisons.append({
            'group1': str(group1),
            'group2': str(group2),
            'mean_diff': float(meandiff),
            'p_value': float(p_adj),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'significant': bool(reject)
        })
    
    return {
        'test': 'Tukey HSD',
        'alpha': alpha,
        'comparisons': comparisons,
        'summary': str(tukey)
    }


def mann_whitney_pairwise(groups: List[np.ndarray],
                          group_names: List[str] = None,
                          alpha: float = 0.05) -> Dict:
    """
    Test Mann-Whitney U: confronti a coppie (non-parametrico).
    
    Post-hoc per Kruskal-Wallis.
    """
    if group_names is None:
        group_names = ['gpt', 'gemini', 'llama', 'qwen']
    
    comparisons = []
    
    # Confronto a coppie
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            u_stat, p_value = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
            
            # Bonferroni correction
            n_comparisons = len(groups) * (len(groups) - 1) / 2
            p_corrected = min(p_value * n_comparisons, 1.0)
            
            comparisons.append({
                'group1': group_names[i],
                'group2': group_names[j],
                'u_statistic': float(u_stat),
                'p_value': float(p_value),
                'p_corrected': float(p_corrected),
                'significant': bool(p_corrected < alpha)
            })
    
    return {
        'test': 'Mann-Whitney U (pairwise)',
        'alpha': alpha,
        'correction': 'Bonferroni',
        'comparisons': comparisons
    }


def effect_size_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calcola Cohen's d (effect size).
    
    Interpetazione:
    - |d| < 0.2: piccolo
    - 0.2 ≤ |d| < 0.5: medio
    - 0.5 ≤ |d| < 0.8: grande
    - |d| ≥ 0.8: molto grande
    """
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
    
    return mean_diff / pooled_std if pooled_std > 0 else 0.0


def comprehensive_statistical_analysis(df: pd.DataFrame,
                                      groups: List[np.ndarray],
                                      group_names: List[str] = None,
                                      alpha: float = 0.05) -> Dict:
    """
    Analisi statistica completa:
    1. Test normalità
    2. Test omogeneità varianze
    3. ANOVA (se assunzioni soddisfatte)
    4. Kruskal-Wallis (alternativa non-parametrica)
    5. Post-hoc tests
    """
    if group_names is None:
        group_names = ['gpt', 'gemini', 'llama', 'qwen']
    
    results = {
        'alpha': alpha,
        'n_groups': len(groups),
        'group_names': group_names,
        'n_observations_per_group': [len(g) for g in groups]
    }
    
    # 1. Test normalità per ogni gruppo
    print("\n 1. TEST NORMALITÀ (Shapiro-Wilk)")
    print("="*70)
    
    normality_tests = {}
    all_normal = True
    
    for i, (name, group) in enumerate(zip(group_names, groups)):
        norm_test = test_normality(group, alpha)
        normality_tests[name] = norm_test
        
        status = " Normale" if norm_test['is_normal'] else "❌ Non normale"
        print(f"  {name.upper()}: p={norm_test['p_value']:.4f} {status}")
        
        if not norm_test['is_normal']:
            all_normal = False
    
    results['normality_tests'] = normality_tests
    results['all_normal'] = all_normal
    
    # 2. Test omogeneità varianze
    print("\n 2. TEST OMOGENEITÀ VARIANZE (Levene)")
    print("="*70)
    
    homogeneity_test = test_variance_homogeneity(groups, alpha)
    results['homogeneity_test'] = homogeneity_test
    
    status = " Omogenee" if homogeneity_test['is_homogeneous'] else "❌ Non omogenee"
    print(f"  p={homogeneity_test['p_value']:.4f} {status}")
    
    # 3. ANOVA (se assunzioni OK) o Kruskal-Wallis
    use_parametric = all_normal and homogeneity_test['is_homogeneous']
    
    if use_parametric:
        print("\n 3. ANOVA ONE-WAY (Assunzioni soddisfatte)")
        print("="*70)
        
        anova_results = anova_one_way(groups, group_names, alpha)
        results['primary_test'] = anova_results
        
        print(f"  F-statistic: {anova_results['f_statistic']:.4f}")
        print(f"  p-value: {anova_results['p_value']:.4f}")
        print(f"  {anova_results['interpretation']}")
        
        # Post-hoc Tukey HSD
        if anova_results['is_significant']:
            print("\n 4. POST-HOC: TUKEY HSD")
            print("="*70)
            
            tukey_results = tukey_hsd_test(df, alpha)
            results['post_hoc'] = tukey_results
            
            for comp in tukey_results['comparisons']:
                sig = " SIG" if comp['significant'] else "   "
                print(f"  {comp['group1'].upper()} vs {comp['group2'].upper()}: "
                      f"diff={comp['mean_diff']:.3f}, p={comp['p_value']:.4f} {sig}")
    
    else:
        print("\n 3. KRUSKAL-WALLIS (Test non-parametrico)")
        print("="*70)
        print("   Assunzioni ANOVA non soddisfatte, uso test non-parametrico")
        
        kruskal_results = kruskal_wallis_test(groups, group_names, alpha)
        results['primary_test'] = kruskal_results
        
        print(f"  H-statistic: {kruskal_results['h_statistic']:.4f}")
        print(f"  p-value: {kruskal_results['p_value']:.4f}")
        print(f"  {kruskal_results['interpretation']}")
        
        # Post-hoc Mann-Whitney U
        if kruskal_results['is_significant']:
            print("\n 4. POST-HOC: MANN-WHITNEY U (pairwise)")
            print("="*70)
            
            mw_results = mann_whitney_pairwise(groups, group_names, alpha)
            results['post_hoc'] = mw_results
            
            for comp in mw_results['comparisons']:
                sig = " SIG" if comp['significant'] else "   "
                print(f"  {comp['group1'].upper()} vs {comp['group2'].upper()}: "
                      f"p={comp['p_value']:.4f} (corr={comp['p_corrected']:.4f}) {sig}")
    
    # 5. Effect sizes (Cohen's d)
    print("\n 5. EFFECT SIZES (Cohen's d)")
    print("="*70)
    
    effect_sizes = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            d = effect_size_cohens_d(groups[i], groups[j])
            
            magnitude = (
                "trascurabile" if abs(d) < 0.2 else
                "piccolo" if abs(d) < 0.5 else
                "medio" if abs(d) < 0.8 else
                "grande"
            )
            
            effect_sizes.append({
                'group1': group_names[i],
                'group2': group_names[j],
                'cohens_d': float(d),
                'magnitude': magnitude
            })
            
            print(f"  {group_names[i].upper()} vs {group_names[j].upper()}: "
                  f"d={d:.3f} ({magnitude})")
    
    results['effect_sizes'] = effect_sizes
    
    return results


def save_results_to_json(results: Dict, output_path: str):
    """Salva risultati in JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n Risultati salvati: {output_path}")


def generate_summary_table(results: Dict) -> pd.DataFrame:
    """Genera tabella riassuntiva confronti a coppie"""
    if 'post_hoc' not in results:
        return None
    
    comparisons = results['post_hoc']['comparisons']
    
    rows = []
    for comp in comparisons:
        # Trova effect size
        effect_size = next(
            (es for es in results['effect_sizes'] 
             if (es['group1'] == comp['group1'] and es['group2'] == comp['group2']) or
                (es['group1'] == comp['group2'] and es['group2'] == comp['group1'])),
            None
        )
        
        row = {
            'Confronto': f"{comp['group1'].upper()} vs {comp['group2'].upper()}",
            'p-value': comp.get('p_value', comp.get('p_corrected')),
            'Significativo': 'Si' if comp['significant'] else 'No',
            "Cohen's d": effect_size['cohens_d'] if effect_size else 0,
            'Magnitude': effect_size['magnitude'] if effect_size else 'N/A'
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def main():
    """Esegue analisi statistica completa"""
    print("="*80)
    print(" ANALISI STATISTICA COMPARATIVA MODELLI LLM")
    print("="*80)
    
    # Carica dati
    print("\n  Caricamento dati...")
    results_file = "results/raw_outputs/full_experiments/full_experiments_gemini25_20251031_135614.json"
    
    metadata, results = load_results(results_file)
    print(f"   Total results: {len(results)}")
    print(f"   Models: {metadata['models']}")
    
    # Estrai success rates per problema
    print("\n  Preparazione dati...")
    success_rates = extract_success_rates_by_problem(results)
    
    models = ['gpt', 'gemini', 'llama', 'qwen']
    df, groups = prepare_data_for_anova(success_rates, models)
    
    print(f"   Problemi analizzati: {len(df)}")
    print(f"   Modelli confrontati: {len(groups)}")
    
    # Statistiche descrittive
    print("\n  STATISTICHE DESCRITTIVE")
    print("="*70)
    for model in models:
        values = df[model].values
        print(f"  {model.upper()}:")
        print(f"    Media: {np.mean(values):.3f}")
        print(f"    Mediana: {np.median(values):.3f}")
        print(f"    Std Dev: {np.std(values, ddof=1):.3f}")
        print(f"    Min: {np.min(values):.3f}")
        print(f"    Max: {np.max(values):.3f}")
    
    # Analisi statistica completa
    print("\n" + "="*80)
    print("  ANALISI STATISTICA INFERENZIALE")
    print("="*80)
    
    alpha = 0.05
    stat_results = comprehensive_statistical_analysis(df, groups, models, alpha)
    
    # Salva risultati
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / "statistical_analysis.json"
    save_results_to_json(stat_results, json_path)
    
    # Tabella riassuntiva
    summary_table = generate_summary_table(stat_results)
    if summary_table is not None:
        print("\n  TABELLA RIASSUNTIVA CONFRONTI")
        print("="*70)
        print(summary_table.to_string(index=False))
        
        # Salva tabella
        csv_path = Path("results/tables") / "statistical_comparisons.csv"
        summary_table.to_csv(csv_path, index=False)
        print(f"\n  Tabella salvata: {csv_path}")
    
    # Conclusioni
    print("\n" + "="*80)
    print("  CONCLUSIONI")
    print("="*80)
    
    test_type = stat_results['primary_test']['test']
    is_sig = stat_results['primary_test']['is_significant']
    p_val = stat_results['primary_test']['p_value']
    
    print(f"\n  Test utilizzato: {test_type}")
    print(f"   p-value: {p_val:.4f}")
    print(f"   Significatività: {'Si (p < 0.05)' if is_sig else 'No (p ≥ 0.05)'}")
    
    if is_sig and 'post_hoc' in stat_results:
        sig_pairs = [c for c in stat_results['post_hoc']['comparisons'] if c['significant']]
        print(f"\n  Coppie significativamente diverse: {len(sig_pairs)}/{len(stat_results['post_hoc']['comparisons'])}")
        
        for comp in sig_pairs:
            print(f"   Si {comp['group1'].upper()} ≠ {comp['group2'].upper()}")
    
    print("\n" + "="*80)
    print("  ANALISI STATISTICA COMPLETATA!")
    print("="*80)
    print(f"\n  File generati:")
    print(f"   - results/metrics/statistical_analysis.json")
    print(f"   - results/tables/statistical_comparisons.csv")


if __name__ == "__main__":
    main()
