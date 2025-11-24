"""
Quick Analysis - Show Most Interesting Problems for Case Studies

Mostra problemi ordinati per divergenza tra modelli
"""

import json
from pathlib import Path
import pandas as pd


def main():
    # Load
    results_path = Path("results/raw_outputs/full_experiments/full_experiments_gemini25_20251031_135614.json")
    problems_path = Path("src/data/selected_problems/selected_problems.json")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    with open(problems_path, 'r') as f:
        problems = json.load(f)
    
    # Calculate
    problem_stats = []
    
    for problem in problems['problems']:
        pid = problem['id']
        
        # Count ONLY first attempt (attempt=1) for Pass@1
        # FIX: Use execution_result.passed instead of success
        first_attempt_success = {'gpt': 0, 'gemini': 0, 'llama': 0, 'qwen': 0}
        
        for result in results['results']:
            if result['problem_id'] == pid and result['attempt'] == 1:
                if result.get('execution_result', {}).get('passed', False):
                    first_attempt_success[result['model']] = 1
        
        # Pass@1 rates (0 or 100%)
        rates = {m: first_attempt_success[m]*100 for m in first_attempt_success.keys()}
        mean_rate = sum(rates.values()) / 4
        variance = sum((r - mean_rate)**2 for r in rates.values()) / 4
        
        problem_stats.append({
            'id': pid,
            'diff': problem['difficulty'],
            'cat': problem['category'],
            'gpt': rates['gpt'],
            'gemini': rates['gemini'],
            'llama': rates['llama'],
            'qwen': rates['qwen'],
            'mean': mean_rate,
            'var': variance
        })
    
    df = pd.DataFrame(problem_stats)
    
    print("\n=== PROBLEMS WITH VARIANCE > 0 (Different Performance) ===\n")
    varied = df[df['var'] > 0].sort_values('var', ascending=False)
    print(f"Total problems with variance: {len(varied)}/{len(df)}\n")
    
    for _, row in varied.head(30).iterrows():
        print(f"{row['id']:25s} [{row['diff']:6s}] [{row['cat']:10s}] "
              f"GPT:{row['gpt']:5.0f}% Gem:{row['gemini']:5.0f}% "
              f"Lla:{row['llama']:5.0f}% Qwen:{row['qwen']:5.0f}% "
              f"(var={row['var']:6.1f})")
    
    print("\n=== SUMMARY STATISTICS ===\n")
    print(f"Problems with 100% success rate (all models): {len(df[df['var'] == 0])}")
    print(f"Problems with variance > 0: {len(df[df['var'] > 0])}")
    print(f"Problems with variance > 500: {len(df[df['var'] > 500])}")
    print(f"Problems with variance > 1000: {len(df[df['var'] > 1000])}")
    
    print("\n=== BY DIFFICULTY ===\n")
    for diff in ['easy', 'medium', 'hard']:
        subset = df[df['diff'] == diff].sort_values('var', ascending=False).head(5)
        print(f"\n{diff.upper()} - Top 5:")
        for _, row in subset.iterrows():
            print(f"  {row['id']:25s} [{row['cat']:10s}] "
                  f"GPT:{row['gpt']:5.0f}% Gem:{row['gemini']:5.0f}% "
                  f"Lla:{row['llama']:5.0f}% Qwen:{row['qwen']:5.0f}% "
                  f"(var={row['var']:6.1f})")


if __name__ == "__main__":
    main()
