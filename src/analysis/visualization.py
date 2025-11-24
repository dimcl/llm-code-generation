"""
Visualization Module - Comprehensive Plots for Thesis

Genera tutti i grafici necessari per l'analisi dei risultati:
1. Pass@k comparison (bar charts + box plots)
2. Quality metrics comparison (box plots)
3. Performance heatmaps (by category and difficulty)
4. Cost vs Accuracy scatter plots
5. Code complexity distributions (violin plots)
6. Error distribution pie charts
7. Latency analysis (box plots)
8. Token usage comparison (bar charts)

Output: PNG + PDF in results/figures/
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configurazione stile grafici
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Colori consistenti per i 4 modelli
MODEL_COLORS = {
    'gpt': '#1f77b4',      # Blu
    'gemini': '#ff7f0e',   # Arancione
    'llama': '#2ca02c',    # Verde
    'qwen': '#d62728'      # Rosso
}

MODEL_NAMES = {
    'gpt': 'GPT-4o-mini',
    'gemini': 'Gemini 2.5 Flash-Lite',
    'llama': 'Llama 3.1 8B',
    'qwen': 'Qwen 2.5 Coder 32B'
}


def load_results() -> Dict:
    """Carica risultati esperimenti completi."""
    results_path = Path("results/raw_outputs/full_experiments/full_experiments_gemini25_20251031_135614.json")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_metrics() -> Dict:
    """Carica tutte le metriche pre-calcolate."""
    metrics_dir = Path("results/metrics")
    
    metrics = {}
    for json_file in metrics_dir.glob("*.json"):
        metric_name = json_file.stem
        with open(json_file, 'r', encoding='utf-8') as f:
            metrics[metric_name] = json.load(f)
    
    return metrics


def create_output_dir() -> Path:
    """Crea directory output per figure."""
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(fig: plt.Figure, name: str, output_dir: Path):
    """Salva figura in PNG e PDF."""
    # PNG per preview
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches='tight')
    # PDF per LaTeX
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches='tight')
    print(f" Salvato: {name}.png + {name}.pdf")
    plt.close(fig)


# ============================================================================
# 1. PASS@K COMPARISON
# ============================================================================

def plot_pass_at_k_comparison(metrics: Dict, output_dir: Path):
    """Bar chart confronto Pass@1, Pass@3, Pass@5 per modello."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = ['gpt', 'gemini', 'llama', 'qwen']
    k_values = [1, 3, 5]
    
    # Prepara dati
    data = {model: [] for model in models}
    for k in k_values:
        pass_at_k = metrics[f'pass_at_{k}_stats']
        for model in models:
            data[model].append(pass_at_k['overall'][model]['pass_rate'])
    
    # Plot grouped bar chart
    x = np.arange(len(k_values))
    width = 0.2
    
    for i, model in enumerate(models):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, data[model], width, 
                     label=MODEL_NAMES[model],
                     color=MODEL_COLORS[model],
                     alpha=0.8)
        
        # Aggiungi valori sopra le barre
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('k (number of attempts)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Pass@k Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Pass@{k}' for k in k_values])
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    save_figure(fig, 'pass_at_k_comparison', output_dir)


def plot_pass_at_1_by_difficulty(metrics: Dict, output_dir: Path):
    """Heatmap Pass@1 per difficulty level."""
    
    pass_at_1 = metrics['pass_at_1_stats']
    models = ['gpt', 'gemini', 'llama', 'qwen']
    difficulties = ['easy', 'medium', 'hard']
    
    # Crea matrice dati
    data = []
    for model in models:
        row = []
        for diff in difficulties:
            rate = pass_at_1['by_difficulty'][model][diff]['pass_rate']
            row.append(rate)
        data.append(row)
    
    df = pd.DataFrame(data, 
                     index=[MODEL_NAMES[m] for m in models],
                     columns=['Easy', 'Medium', 'Hard'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', 
               cbar_kws={'label': 'Pass Rate (%)'}, 
               linewidths=0.5, ax=ax, vmin=0, vmax=100)
    
    ax.set_title('Pass@1 Performance by Difficulty Level', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Difficulty', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    save_figure(fig, 'pass_at_1_by_difficulty_heatmap', output_dir)


def plot_pass_at_1_by_category(metrics: Dict, output_dir: Path):
    """Heatmap Pass@1 per categoria di problema."""
    
    pass_at_1 = metrics['pass_at_1_stats']
    models = ['gpt', 'gemini', 'llama', 'qwen']
    categories = ['strings', 'lists', 'math', 'algorithms']
    
    # Crea matrice dati
    data = []
    for model in models:
        row = []
        for cat in categories:
            rate = pass_at_1['by_category'][model][cat]['pass_rate']
            row.append(rate)
        data.append(row)
    
    df = pd.DataFrame(data,
                     index=[MODEL_NAMES[m] for m in models],
                     columns=['Strings', 'Lists', 'Math', 'Algorithms'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn',
               cbar_kws={'label': 'Pass Rate (%)'}, 
               linewidths=0.5, ax=ax, vmin=0, vmax=100)
    
    ax.set_title('Pass@1 Performance by Problem Category',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    save_figure(fig, 'pass_at_1_by_category_heatmap', output_dir)


# ============================================================================
# 2. QUALITY METRICS
# ============================================================================

def plot_quality_metrics_comparison(metrics: Dict, output_dir: Path):
    """Box plots per complessità ciclomatica e LOC."""
    
    quality = metrics['quality_metrics']
    models = ['gpt', 'gemini', 'llama', 'qwen']
    
    # Usa statistiche pre-calcolate per creare approssimazione di distribuzione
    # (Per un box plot ideale servirebbero i dati raw, ma usiamo media/mediana/std)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Complessità (bar chart con error bars)
    complexities_avg = [quality[m]['avg_complexity'] for m in models]
    complexities_std = [quality[m]['std_complexity'] for m in models]
    
    x = np.arange(len(models))
    ax1.bar(x, complexities_avg, yerr=complexities_std, 
           color=[MODEL_COLORS[m] for m in models],
           alpha=0.7, capsize=5, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Cyclomatic Complexity', fontsize=11, fontweight='bold')
    ax1.set_title('Average Code Complexity', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Aggiungi valori
    for i, (avg, std) in enumerate(zip(complexities_avg, complexities_std)):
        ax1.text(i, avg + std + 0.2, f'{avg:.2f}±{std:.2f}',
                ha='center', fontsize=9)
    
    # Plot 2: LOC (bar chart con error bars)
    locs_avg = [quality[m]['avg_loc'] for m in models]
    locs_std = [quality[m]['std_loc'] for m in models]
    
    ax2.bar(x, locs_avg, yerr=locs_std,
           color=[MODEL_COLORS[m] for m in models],
           alpha=0.7, capsize=5, edgecolor='black', linewidth=1)
    
    ax2.set_ylabel('Lines of Code', fontsize=11, fontweight='bold')
    ax2.set_title('Average Code Length', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=15)
    ax2.grid(axis='y', alpha=0.3)
    
    # Aggiungi valori
    for i, (avg, std) in enumerate(zip(locs_avg, locs_std)):
        ax2.text(i, avg + std + 0.5, f'{avg:.1f}±{std:.1f}',
                ha='center', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, 'quality_metrics_comparison', output_dir)


def plot_halstead_metrics(metrics: Dict, output_dir: Path):
    """Bar chart per Halstead metrics."""
    
    quality = metrics['quality_metrics']
    models = ['gpt', 'gemini', 'llama', 'qwen']
    
    # Metriche Halstead da visualizzare
    halstead_metrics = ['volume', 'difficulty', 'effort']
    metric_keys = ['avg_halstead_volume', 'avg_halstead_difficulty', 'avg_halstead_effort']
    
    # Prepara dati medi per modello
    data = {metric: [] for metric in halstead_metrics}
    
    for model in models:
        data['volume'].append(quality[model]['avg_halstead_volume'])
        data['difficulty'].append(quality[model]['avg_halstead_difficulty'])
        data['effort'].append(quality[model]['avg_halstead_effort'])
    
    # Normalizza per visualizzazione comparabile
    for metric in halstead_metrics:
        max_val = max(data[metric]) if max(data[metric]) > 0 else 1
        data[metric] = [v / max_val * 100 for v in data[metric]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(halstead_metrics))
    width = 0.2
    
    for i, model in enumerate(models):
        offset = (i - 1.5) * width
        values = [data[metric][i] for metric in halstead_metrics]
        ax.bar(x + offset, values, width,
              label=MODEL_NAMES[model],
              color=MODEL_COLORS[model],
              alpha=0.8)
    
    ax.set_xlabel('Halstead Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Value (0-100)', fontsize=12, fontweight='bold')
    ax.set_title('Halstead Software Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Volume', 'Difficulty', 'Effort'])
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3)
    
    save_figure(fig, 'halstead_metrics_comparison', output_dir)


# ============================================================================
# 3. COST AND EFFICIENCY
# ============================================================================

def plot_cost_vs_accuracy(metrics: Dict, output_dir: Path):
    """Scatter plot: Costo totale vs Pass@1."""
    
    cost = metrics['cost_analysis']
    pass_at_1 = metrics['pass_at_1_stats']
    models = ['gpt', 'gemini', 'llama', 'qwen']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model in models:
        x = cost[model]['total_cost']
        y = pass_at_1['overall'][model]['pass_rate']
        
        ax.scatter(x, y, s=300, color=MODEL_COLORS[model],
                  alpha=0.7, edgecolors='black', linewidth=1.5,
                  label=MODEL_NAMES[model], zorder=3)
        
        # Aggiungi etichetta
        ax.annotate(MODEL_NAMES[model], (x, y),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor=MODEL_COLORS[model], 
                            alpha=0.3))
    
    ax.set_xlabel('Total Cost ($)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pass@1 Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cost-Efficiency Analysis: Accuracy vs Price',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_ylim(0, 100)
    
    # Calcola media overall
    avg_rate = sum(pass_at_1['overall'][m]['pass_rate'] for m in models) / len(models)
    ax.axhline(y=avg_rate, 
              color='gray', linestyle='--', alpha=0.5,
              label=f'Average ({avg_rate:.1f}%)')
    
    ax.legend(loc='lower right', frameon=True, shadow=True)
    
    save_figure(fig, 'cost_vs_accuracy_scatter', output_dir)


def plot_latency_comparison(metrics: Dict, output_dir: Path):
    """Box plot: Distribuzione latenze per modello."""
    
    cost = metrics['cost_analysis']
    models = ['gpt', 'gemini', 'llama', 'qwen']
    
    # Carica risultati raw per latenze individuali
    results = load_results()
    
    latency_data = []
    for model in models:
        latencies = []
        for result in results['results']:
            if result['model'] == model and result['success']:
                latencies.append(result['latency'])
        latency_data.append(latencies)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(latency_data, labels=[MODEL_NAMES[m] for m in models],
                   patch_artist=True, notch=True, showmeans=True)
    
    for patch, model in zip(bp['boxes'], models):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Response Time Distribution by Model', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)
    
    # Aggiungi statistiche
    for i, model in enumerate(models):
        median = cost[model]['avg_latency']
        ax.text(i+1, ax.get_ylim()[1]*0.95, f'Med: {median:.2f}s',
               ha='center', fontsize=9, bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.7))
    
    save_figure(fig, 'latency_comparison_boxplot', output_dir)


def plot_token_usage(metrics: Dict, output_dir: Path):
    """Bar chart: Token usage totale per modello."""
    
    cost = metrics['cost_analysis']
    models = ['gpt', 'gemini', 'llama', 'qwen']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    total_tokens = [cost[m]['total_tokens'] for m in models]
    avg_tokens = [cost[m]['avg_tokens'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, total_tokens, width, label='Total Tokens',
                  color=[MODEL_COLORS[m] for m in models], alpha=0.8,
                  edgecolor='black', linewidth=1)
    
    # Secondary axis for average
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, avg_tokens, width, label='Avg per Generation',
                   color='lightgray', alpha=0.6, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Total Tokens', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Tokens/Generation', fontsize=12, fontweight='bold')
    ax.set_title('Token Usage Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    # Legenda combinata
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    save_figure(fig, 'token_usage_comparison', output_dir)


# ============================================================================
# 4. ERROR ANALYSIS
# ============================================================================

def plot_error_distribution(metrics: Dict, output_dir: Path):
    """Stacked bar chart: Distribuzione errori per modello."""
    
    errors = metrics['error_distribution']
    models = ['gpt', 'gemini', 'llama', 'qwen']
    
    # Raccogli tutti i tipi di errori (escluding 'success')
    error_types = set()
    for model in models:
        for error_type in errors[model].keys():
            if error_type != 'success':
                error_types.add(error_type)
    
    error_types = sorted(list(error_types))
    
    # Prepara dati
    data = {et: [] for et in error_types}
    for model in models:
        for et in error_types:
            count = errors[model].get(et, 0)
            data[et].append(count)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.6
    bottom = np.zeros(len(models))
    
    # Colori diversi per ogni tipo di errore
    colors = plt.cm.Set3(np.linspace(0, 1, len(error_types)))
    
    for i, (et, color) in enumerate(zip(error_types, colors)):
        label = et.replace('_', ' ').title()
        p = ax.bar(x, data[et], width, bottom=bottom,
                  label=label, color=color, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
        bottom += data[et]
    
    ax.set_ylabel('Number of Errors', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution by Type and Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES[m] for m in models], rotation=15)
    ax.legend(loc='upper right', frameon=True, shadow=True, ncol=2)
    ax.grid(axis='y', alpha=0.3)
    
    save_figure(fig, 'error_distribution_stacked', output_dir)


def plot_failure_analysis(metrics: Dict, output_dir: Path):
    """Pie charts: Cause dei fallimenti per ogni modello (2x2 grid)."""
    
    errors = metrics['error_distribution']
    models = ['gpt', 'gemini', 'llama', 'qwen']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, model in enumerate(models):
        ax = axes[idx]
        
        # Filtra errori (escludi 'success')
        labels = []
        sizes = []
        for error_type, count in errors[model].items():
            if error_type != 'success' and count > 0:
                label = error_type.replace('_', ' ').title()
                labels.append(label)
                sizes.append(count)
        
        if sizes:
            colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                              colors=colors, startangle=90,
                                              textprops={'fontsize': 9})
            
            # Bold percentages
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        ax.set_title(f'{MODEL_NAMES[model]}\n({sum(sizes)} failures)',
                    fontsize=11, fontweight='bold', pad=10)
    
    plt.suptitle('Failure Analysis: Error Types Distribution', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_figure(fig, 'failure_analysis_pie_charts', output_dir)


# ============================================================================
# 5. COMPREHENSIVE SUMMARY
# ============================================================================

def plot_overall_summary(metrics: Dict, output_dir: Path):
    """Dashboard riassuntivo con metriche chiave."""
    
    report = metrics['complete_report']
    models = ['gpt', 'gemini', 'llama', 'qwen']
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Pass@1 Ranking (top-left, large)
    ax1 = fig.add_subplot(gs[0, :2])
    pass_rates = [report['models_ranking'][m]['pass_at_1'] for m in models]
    bars = ax1.barh([MODEL_NAMES[m] for m in models], pass_rates,
                   color=[MODEL_COLORS[m] for m in models], alpha=0.8,
                   edgecolor='black', linewidth=1)
    
    for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
        ax1.text(rate + 1, i, f'{rate:.1f}%', va='center', fontweight='bold')
    
    ax1.set_xlabel('Pass@1 Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Model Ranking by Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Total Cost (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    costs = [report['models_ranking'][m]['total_cost'] for m in models]
    ax2.bar(range(len(models)), costs, color=[MODEL_COLORS[m] for m in models],
           alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Cost ($)', fontsize=10, fontweight='bold')
    ax2.set_title('Total Cost', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([MODEL_NAMES[m].split()[0] for m in models], fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Complexity (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])
    complexities = [report['models_ranking'][m]['avg_complexity'] for m in models]
    ax3.bar(range(len(models)), complexities, color=[MODEL_COLORS[m] for m in models],
           alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_ylabel('Avg Complexity', fontsize=10, fontweight='bold')
    ax3.set_title('Code Complexity', fontsize=11, fontweight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels([MODEL_NAMES[m].split()[0] for m in models], fontsize=9, rotation=15)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. LOC (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])
    locs = [report['models_ranking'][m]['avg_loc'] for m in models]
    ax4.bar(range(len(models)), locs, color=[MODEL_COLORS[m] for m in models],
           alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_ylabel('Avg LOC', fontsize=10, fontweight='bold')
    ax4.set_title('Code Length', fontsize=11, fontweight='bold')
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels([MODEL_NAMES[m].split()[0] for m in models], fontsize=9, rotation=15)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Success Rate by Difficulty (middle-right + bottom)
    pass_at_1 = metrics['pass_at_1_stats']
    difficulties = ['easy', 'medium', 'hard']
    
    ax5 = fig.add_subplot(gs[1, 2])
    x = np.arange(len(difficulties))
    width = 0.2
    
    for i, model in enumerate(models):
        rates = [pass_at_1['by_difficulty'][model][d]['pass_rate'] for d in difficulties]
        offset = (i - 1.5) * width
        ax5.bar(x + offset, rates, width, label=MODEL_NAMES[model].split()[0],
               color=MODEL_COLORS[model], alpha=0.8)
    
    ax5.set_ylabel('Pass Rate (%)', fontsize=10, fontweight='bold')
    ax5.set_title('By Difficulty', fontsize=11, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(['Easy', 'Med', 'Hard'], fontsize=9)
    ax5.legend(fontsize=8, loc='upper right')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Statistical Significance (bottom row)
    ax6 = fig.add_subplot(gs[2, :])
    
    # Matrice significatività (da statistical_comparisons.csv)
    stat_comp = pd.read_csv('results/tables/statistical_comparisons.csv')
    
    # Crea testo summary
    sig_text = "Statistical Significance (Mann-Whitney U with Bonferroni correction, α=0.05):\n\n"
    sig_pairs = stat_comp[stat_comp['Significativo'] == '✅']
    
    if len(sig_pairs) > 0:
        sig_text += "Significant differences found:\n"
        for _, row in sig_pairs.iterrows():
            cohens_d = row["Cohen's d"]
            sig_text += f"  • {row['Confronto']}: p={row['p-value']:.2e}, Cohen's d={cohens_d:.3f} ({row['Magnitude']})\n"
    
    non_sig_pairs = stat_comp[stat_comp['Significativo'] == '❌']
    if len(non_sig_pairs) > 0:
        sig_text += f"\nNo significant differences ({len(non_sig_pairs)} pairs):\n"
        for _, row in non_sig_pairs.iterrows():
            sig_text += f"  • {row['Confronto']}: p={row['p-value']:.3f}\n"
    
    ax6.text(0.05, 0.95, sig_text, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax6.axis('off')
    
    plt.suptitle('Comprehensive Analysis Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
    
    save_figure(fig, 'overall_summary_dashboard', output_dir)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Genera tutti i grafici."""
    print("=" * 80)
    print(" GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("=" * 80)
    print()
    
    # Setup
    output_dir = create_output_dir()
    print(f" Output directory: {output_dir}")
    print()
    
    # Load data
    print(" Loading data...")
    metrics = load_metrics()
    print(f"   Loaded {len(metrics)} metric files")
    print()
    
    # Generate plots
    print(" Generating plots...")
    print()
    
    print("1️  Pass@k Comparison...")
    plot_pass_at_k_comparison(metrics, output_dir)
    
    print("2️  Pass@1 by Difficulty...")
    plot_pass_at_1_by_difficulty(metrics, output_dir)
    
    print("3️  Pass@1 by Category...")
    plot_pass_at_1_by_category(metrics, output_dir)
    
    print("4️  Quality Metrics Comparison...")
    plot_quality_metrics_comparison(metrics, output_dir)
    
    print("5️  Halstead Metrics...")
    plot_halstead_metrics(metrics, output_dir)
    
    print("6️  Cost vs Accuracy...")
    plot_cost_vs_accuracy(metrics, output_dir)
    
    print("7️  Latency Comparison...")
    plot_latency_comparison(metrics, output_dir)
    
    print("8️  Token Usage...")
    plot_token_usage(metrics, output_dir)
    
    print("9️  Error Distribution...")
    plot_error_distribution(metrics, output_dir)
    
    print("10 Failure Analysis...")
    plot_failure_analysis(metrics, output_dir)
    
    print("11 Overall Summary Dashboard...")
    plot_overall_summary(metrics, output_dir)
    
    print()
    print("=" * 80)
    print(" VISUALIZATION COMPLETE!")
    print("=" * 80)
    print()
    print(f" All figures saved in: {output_dir}")
    print("   - 11 visualizations generated")
    print("   - Format: PNG (300 DPI) + PDF (LaTeX-ready)")
    print()


if __name__ == "__main__":
    main()
