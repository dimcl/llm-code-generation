"""
Esplora i dataset HumanEval e MBPP per selezionare problemi rappresentativi.
"""
import json
import gzip
from pathlib import Path
from collections import Counter
import re


def load_humaneval():
    """Carica dataset HumanEval"""
    humaneval_path = Path(__file__).parent / "humaneval" / "HumanEval.jsonl.gz"
    
    problems = []
    with gzip.open(humaneval_path, 'rt', encoding='utf-8') as f:
        for line in f:
            problems.append(json.loads(line))
    
    return problems


def load_mbpp():
    """Carica dataset MBPP"""
    mbpp_path = Path(__file__).parent / "mbpp" / "mbpp.jsonl"
    
    problems = []
    with open(mbpp_path, 'r', encoding='utf-8') as f:
        for line in f:
            problems.append(json.loads(line))
    
    return problems


def estimate_difficulty(problem, source='humaneval'):
    """
    Stima difficoltà basandosi su:
    - Lunghezza prompt
    - Parole chiave
    - Complessità descrizione
    """
    if source == 'humaneval':
        text = problem.get('prompt', '')
    else:  # mbpp
        text = problem.get('text', '')
    
    text_lower = text.lower()
    
    # Parole chiave per difficoltà
    easy_keywords = ['sum', 'add', 'reverse', 'count', 'find', 'simple', 'basic', 
                     'check', 'is_', 'convert', 'get', 'return']
    hard_keywords = ['tree', 'graph', 'dynamic', 'optimize', 'algorithm', 'recursive',
                     'parse', 'complex', 'minimum', 'maximum', 'path', 'search']
    
    # Conta keywords
    easy_count = sum(1 for kw in easy_keywords if kw in text_lower)
    hard_count = sum(1 for kw in hard_keywords if kw in text_lower)
    
    # Lunghezza testo
    text_len = len(text)
    
    # Score
    difficulty_score = 0
    
    # Lunghezza
    if text_len < 200:
        difficulty_score -= 2
    elif text_len > 400:
        difficulty_score += 2
    
    # Keywords
    difficulty_score += (hard_count * 2) - easy_count
    
    # Presenza di esempi complessi
    if 'example' in text_lower and text_len > 300:
        difficulty_score += 1
    
    # Classificazione
    if difficulty_score <= -1:
        return 'easy'
    elif difficulty_score <= 2:
        return 'medium'
    else:
        return 'hard'


def categorize_topic(problem, source='humaneval'):
    """Categorizza per argomento"""
    if source == 'humaneval':
        text = problem.get('prompt', '') + problem.get('entry_point', '')
    else:
        text = problem.get('text', '')
    
    text_lower = text.lower()
    
    # Categorie e keywords
    categories = {
        'strings': ['string', 'char', 'word', 'letter', 'palindrome', 'reverse', 
                   'substring', 'concatenate', 'split'],
        'lists': ['list', 'array', 'element', 'sort', 'filter', 'append', 
                 'remove', 'index'],
        'math': ['number', 'prime', 'factorial', 'sum', 'product', 'digit',
                'calculate', 'math', 'average', 'median'],
        'algorithms': ['search', 'sort', 'find', 'minimum', 'maximum', 'binary',
                      'algorithm', 'optimize'],
        'data_structures': ['dict', 'set', 'tuple', 'stack', 'queue', 'hash',
                           'tree', 'node'],
        'recursion': ['recursive', 'recursion', 'fibonacci', 'tree', 'nested']
    }
    
    # Conta match per categoria
    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score
    
    # Ritorna categoria con score più alto
    if scores:
        return max(scores, key=scores.get)
    else:
        return 'other'


def analyze_humaneval():
    """Analizza dataset HumanEval"""
    print("=" * 70)
    print("HUMANEVAL DATASET ANALYSIS")
    print("=" * 70)
    
    problems = load_humaneval()
    print(f"\n Total problems: {len(problems)}")
    
    # Statistiche lunghezza
    lengths = [len(p['prompt']) for p in problems]
    print(f" Prompt length: avg={sum(lengths)/len(lengths):.0f}, "
          f"min={min(lengths)}, max={max(lengths)}")
    
    # Categorizza
    difficulties = Counter()
    topics = Counter()
    
    for p in problems:
        diff = estimate_difficulty(p, 'humaneval')
        topic = categorize_topic(p, 'humaneval')
        difficulties[diff] += 1
        topics[topic] += 1
    
    print(f"\n Difficulty distribution:")
    for diff, count in difficulties.most_common():
        print(f"   {diff}: {count}")
    
    print(f"\n Topic distribution:")
    for topic, count in topics.most_common():
        print(f"   {topic}: {count}")
    
    # Mostra esempi per difficoltà
    print("\n" + "=" * 70)
    print("EXAMPLES BY DIFFICULTY")
    print("=" * 70)
    
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n🔹 {difficulty.upper()} examples:")
        examples = [p for p in problems if estimate_difficulty(p, 'humaneval') == difficulty][:5]
        
        for i, p in enumerate(examples, 1):
            task_id = p['task_id']
            prompt_preview = p['prompt'].replace('\n', ' ')[:100]
            topic = categorize_topic(p, 'humaneval')
            print(f"\n   {i}. {task_id} [{topic}]")
            print(f"      {prompt_preview}...")
    
    return problems


def analyze_mbpp():
    """Analizza dataset MBPP"""
    print("\n\n" + "=" * 70)
    print("MBPP DATASET ANALYSIS")
    print("=" * 70)
    
    problems = load_mbpp()
    print(f"\n Total problems: {len(problems)}")
    
    # Statistiche
    texts = [p.get('text', '') for p in problems]
    lengths = [len(t) for t in texts]
    print(f" Text length: avg={sum(lengths)/len(lengths):.0f}, "
          f"min={min(lengths)}, max={max(lengths)}")
    
    # Categorizza
    difficulties = Counter()
    topics = Counter()
    
    for p in problems:
        diff = estimate_difficulty(p, 'mbpp')
        topic = categorize_topic(p, 'mbpp')
        difficulties[diff] += 1
        topics[topic] += 1
    
    print(f"\n Difficulty distribution:")
    for diff, count in difficulties.most_common():
        print(f"   {diff}: {count}")
    
    print(f"\n Topic distribution:")
    for topic, count in topics.most_common():
        print(f"   {topic}: {count}")
    
    # Mostra esempi per difficoltà
    print("\n" + "=" * 70)
    print("EXAMPLES BY DIFFICULTY")
    print("=" * 70)
    
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n🔹 {difficulty.upper()} examples:")
        examples = [p for p in problems if estimate_difficulty(p, 'mbpp') == difficulty][:5]
        
        for i, p in enumerate(examples, 1):
            task_id = p.get('task_id', f"mbpp_{i}")
            text_preview = p.get('text', '').replace('\n', ' ')[:100]
            topic = categorize_topic(p, 'mbpp')
            print(f"\n   {i}. {task_id} [{topic}]")
            print(f"      {text_preview}...")
    
    return problems


def main():
    """Main analysis"""
    print("\n DATASET EXPLORATION\n")
    
    # Analizza entrambi
    humaneval_problems = analyze_humaneval()
    mbpp_problems = analyze_mbpp()
    
    print("\n\n" + "=" * 70)
    print(" ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nTotal problems available: {len(humaneval_problems) + len(mbpp_problems)}")
    print(f"  - HumanEval: {len(humaneval_problems)}")
    print(f"  - MBPP: {len(mbpp_problems)}")
    print("\nReady for problem selection! ")


if __name__ == "__main__":
    main()
