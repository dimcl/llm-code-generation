"""
Analizza la distribuzione di problemi per categoria e difficoltà nei dataset.
"""
import json
import gzip
from pathlib import Path
from collections import defaultdict


def load_humaneval():
    """Carica HumanEval"""
    humaneval_path = Path(__file__).parent / "humaneval" / "HumanEval.jsonl.gz"
    problems = []
    with gzip.open(humaneval_path, 'rt', encoding='utf-8') as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def load_mbpp():
    """Carica MBPP"""
    mbpp_path = Path(__file__).parent / "mbpp" / "mbpp.jsonl"
    problems = []
    with open(mbpp_path, 'r', encoding='utf-8') as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def estimate_difficulty(problem, source='humaneval'):
    """Stima difficoltà"""
    if source == 'humaneval':
        text = problem.get('prompt', '')
    else:
        text = problem.get('text', '')
    
    text_lower = text.lower()
    
    easy_keywords = ['sum', 'add', 'reverse', 'count', 'find', 'simple', 'basic', 
                     'check', 'is_', 'convert', 'get', 'return']
    hard_keywords = ['tree', 'graph', 'dynamic', 'optimize', 'algorithm', 'recursive',
                     'parse', 'complex', 'minimum', 'maximum', 'path', 'search']
    
    easy_count = sum(1 for kw in easy_keywords if kw in text_lower)
    hard_count = sum(1 for kw in hard_keywords if kw in text_lower)
    text_len = len(text)
    
    difficulty_score = 0
    if text_len < 200:
        difficulty_score -= 2
    elif text_len > 400:
        difficulty_score += 2
    
    difficulty_score += (hard_count * 2) - easy_count
    
    if 'example' in text_lower and text_len > 300:
        difficulty_score += 1
    
    if difficulty_score <= -1:
        return 'easy'
    elif difficulty_score <= 2:
        return 'medium'
    else:
        return 'hard'


def categorize_topic(problem, source='humaneval'):
    """Categorizza argomento - VERSIONE MIGLIORATA"""
    if source == 'humaneval':
        text = problem.get('prompt', '') + problem.get('entry_point', '')
    else:
        text = problem.get('text', '')
    
    text_lower = text.lower()
    
    # Priorità: controlla in ordine per evitare overlapping
    # 1. Algoritmi specifici (priorità alta)
    if any(kw in text_lower for kw in ['binary search', 'sort', 'quicksort', 'mergesort', 
                                         'heap', 'graph', 'tree traversal', 'bfs', 'dfs',
                                         'dynamic programming', 'dp', 'greedy']):
        return 'algorithms'
    
    # 2. Strutture dati (priorità alta)
    if any(kw in text_lower for kw in ['dictionary', 'dict', 'hash map', 'set', 
                                        'stack', 'queue', 'linked list', 'tree', 'graph']):
        return 'data_structures'
    
    # 3. Matematica (priorità media)
    if any(kw in text_lower for kw in ['prime', 'factorial', 'fibonacci', 'gcd', 'lcm',
                                        'modulo', 'divisor', 'multiple', 'digit',
                                        'square', 'cube', 'power', 'root']):
        return 'math'
    
    # 4. Stringhe (priorità media-bassa)
    if any(kw in text_lower for kw in ['string', 'char', 'word', 'letter', 'palindrome',
                                        'substring', 'prefix', 'suffix', 'vowel', 'consonant',
                                        'uppercase', 'lowercase', 'concatenate']):
        return 'strings'
    
    # 5. Liste/Array (priorità bassa - molto comune)
    if any(kw in text_lower for kw in ['list', 'array', 'element', 'filter',
                                        'map', 'reduce', 'index']):
        return 'lists'
    
    # 6. Ricorsione
    if any(kw in text_lower for kw in ['recursive', 'recursion']):
        return 'recursion'
    
    return 'other'


def main():
    """Analizza distribuzione categorie × difficoltà"""
    
    print("="*80)
    print(" ANALISI DISTRIBUZIONE CATEGORIE × DIFFICOLTÀ")
    print("="*80)
    
    humaneval = load_humaneval()
    mbpp = load_mbpp()
    
    # Analizza HumanEval
    print(f"\n HUMANEVAL ({len(humaneval)} problemi)")
    print("-"*80)
    
    he_matrix = defaultdict(lambda: defaultdict(int))
    for p in humaneval:
        diff = estimate_difficulty(p, 'humaneval')
        topic = categorize_topic(p, 'humaneval')
        he_matrix[topic][diff] += 1
    
    # Stampa matrice HumanEval
    print(f"\n{'Categoria':<20} {'Easy':<8} {'Medium':<8} {'Hard':<8} {'Totale':<8}")
    print("-"*60)
    
    for topic in sorted(he_matrix.keys()):
        easy = he_matrix[topic]['easy']
        medium = he_matrix[topic]['medium']
        hard = he_matrix[topic]['hard']
        total = easy + medium + hard
        print(f"{topic:<20} {easy:<8} {medium:<8} {hard:<8} {total:<8}")
    
    # Totali HumanEval
    total_easy = sum(he_matrix[t]['easy'] for t in he_matrix)
    total_medium = sum(he_matrix[t]['medium'] for t in he_matrix)
    total_hard = sum(he_matrix[t]['hard'] for t in he_matrix)
    print("-"*60)
    print(f"{'TOTALE':<20} {total_easy:<8} {total_medium:<8} {total_hard:<8} {total_easy+total_medium+total_hard:<8}")
    
    # Analizza MBPP
    print(f"\n\n MBPP ({len(mbpp)} problemi)")
    print("-"*80)
    
    mbpp_matrix = defaultdict(lambda: defaultdict(int))
    for p in mbpp:
        diff = estimate_difficulty(p, 'mbpp')
        topic = categorize_topic(p, 'mbpp')
        mbpp_matrix[topic][diff] += 1
    
    # Stampa matrice MBPP
    print(f"\n{'Categoria':<20} {'Easy':<8} {'Medium':<8} {'Hard':<8} {'Totale':<8}")
    print("-"*60)
    
    for topic in sorted(mbpp_matrix.keys()):
        easy = mbpp_matrix[topic]['easy']
        medium = mbpp_matrix[topic]['medium']
        hard = mbpp_matrix[topic]['hard']
        total = easy + medium + hard
        print(f"{topic:<20} {easy:<8} {medium:<8} {hard:<8} {total:<8}")
    
    # Totali MBPP
    total_easy_m = sum(mbpp_matrix[t]['easy'] for t in mbpp_matrix)
    total_medium_m = sum(mbpp_matrix[t]['medium'] for t in mbpp_matrix)
    total_hard_m = sum(mbpp_matrix[t]['hard'] for t in mbpp_matrix)
    print("-"*60)
    print(f"{'TOTALE':<20} {total_easy_m:<8} {total_medium_m:<8} {total_hard_m:<8} {total_easy_m+total_medium_m+total_hard_m:<8}")
    
    # Raccomandazione distribuzione
    print(f"\n\n{'='*80}")
    print(" DISTRIBUZIONE BILANCIATA CONSIGLIATA (60 problemi totali)")
    print("="*80)
    
    print(f"\n Target: 20 easy / 20 medium / 20 hard")
    print(f"\n Distribuzione per categoria (bilanciata):")
    print("-"*60)
    
    # Proporzione ideale: 4 categorie principali
    categories = ['strings', 'lists', 'math', 'algorithms']
    
    # Per ogni livello di difficoltà, distribuisci equamente
    print(f"\n{'Categoria':<20} {'Easy':<8} {'Medium':<8} {'Hard':<8} {'Totale':<8}")
    print("-"*60)
    
    distributions = {
        'strings': {'easy': 5, 'medium': 5, 'hard': 5, 'total': 15},
        'lists': {'easy': 5, 'medium': 5, 'hard': 5, 'total': 15},
        'math': {'easy': 5, 'medium': 5, 'hard': 5, 'total': 15},
        'algorithms': {'easy': 5, 'medium': 5, 'hard': 5, 'total': 15},
    }
    
    for cat, dist in distributions.items():
        print(f"{cat:<20} {dist['easy']:<8} {dist['medium']:<8} {dist['hard']:<8} {dist['total']:<8}")
    
    print("-"*60)
    print(f"{'TOTALE':<20} {20:<8} {20:<8} {20:<8} {60:<8}")
    
    print(f"\n Questo garantisce:")
    print(f"   - Bilanciamento perfetto difficoltà (20/20/20)")
    print(f"   - Bilanciamento perfetto categorie (15/15/15/15)")
    print(f"   - Ogni categoria testata a tutti i livelli (5 per livello)")
    print(f"   - Coverage completo delle capacità LLM")
    
    # Verifica disponibilità
    print(f"\n\n{'='*80}")
    print(" VERIFICA DISPONIBILITÀ NEI DATASET")
    print("="*80)
    
    for cat in categories:
        print(f"\n {cat.upper()}:")
        he_avail = f"HE: {he_matrix[cat]['easy']}E / {he_matrix[cat]['medium']}M / {he_matrix[cat]['hard']}H"
        mbpp_avail = f"MBPP: {mbpp_matrix[cat]['easy']}E / {mbpp_matrix[cat]['medium']}M / {mbpp_matrix[cat]['hard']}H"
        print(f"   {he_avail}")
        print(f"   {mbpp_avail}")
        
        # Check se abbiamo abbastanza
        total_easy = he_matrix[cat]['easy'] + mbpp_matrix[cat]['easy']
        total_medium = he_matrix[cat]['medium'] + mbpp_matrix[cat]['medium']
        total_hard = he_matrix[cat]['hard'] + mbpp_matrix[cat]['hard']
        
        status_e = "✅" if total_easy >= 5 else f"⚠️ (solo {total_easy})"
        status_m = "✅" if total_medium >= 5 else f"⚠️ (solo {total_medium})"
        status_h = "✅" if total_hard >= 5 else f"⚠️ (solo {total_hard})"
        
        print(f"   Disponibili: {total_easy}E {status_e} / {total_medium}M {status_m} / {total_hard}H {status_h}")


if __name__ == "__main__":
    main()
