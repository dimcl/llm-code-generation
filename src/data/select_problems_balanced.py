"""
Seleziona 60 problemi PERFETTAMENTE BILANCIATI:
- 20 facili, 20 medi, 20 difficili
- 4 categorie: strings, lists, math, algorithms
- 5 problemi per categoria per livello (5×4×3=60)
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
    """Categorizza argomento - VERSIONE MIGLIORATA per bilanciamento"""
    if source == 'humaneval':
        text = problem.get('prompt', '') + problem.get('entry_point', '')
    else:
        text = problem.get('text', '')
    
    text_lower = text.lower()
    
    # Priorità: controlla in ordine per evitare overlapping
    # 1. Algoritmi specifici (priorità MASSIMA)
    if any(kw in text_lower for kw in ['binary search', 'quicksort', 'mergesort', 
                                         'heap', 'graph', 'tree traversal', 'bfs', 'dfs',
                                         'dynamic programming', 'dp', 'greedy', 'optimal']):
        return 'algorithms'
    
    # 2. Matematica (priorità alta - prima di list per evitare overlap con sum/product)
    if any(kw in text_lower for kw in ['prime', 'factorial', 'fibonacci', 'gcd', 'lcm',
                                        'modulo', 'divisor', 'multiple', 'digit',
                                        'square', 'cube', 'power', 'root', 'even', 'odd']):
        return 'math'
    
    # 3. Stringhe (priorità media-alta)
    if any(kw in text_lower for kw in ['string', 'char', 'word', 'letter', 'palindrome',
                                        'substring', 'prefix', 'suffix', 'vowel', 'consonant',
                                        'uppercase', 'lowercase', 'concatenate']) and \
       'list' not in text_lower and 'array' not in text_lower:
        return 'strings'
    
    # 4. Liste/Array (priorità media)
    if any(kw in text_lower for kw in ['list', 'array', 'element', 'filter',
                                        'map', 'reduce', 'index', 'append', 'remove']):
        return 'lists'
    
    # 5. Strutture dati
    if any(kw in text_lower for kw in ['dictionary', 'dict', 'hash', 'set', 
                                        'stack', 'queue', 'tree', 'node']):
        return 'data_structures'
    
    # 6. Ricorsione
    if any(kw in text_lower for kw in ['recursive', 'recursion']):
        return 'recursion'
    
    return 'other'


def create_problem_entry(problem, source, idx, difficulty, category):
    """Crea entry per selected_problems.json"""
    if source == 'humaneval':
        entry = {
            'id': f'humaneval_{idx:03d}',
            'source': 'humaneval',
            'task_id': problem['task_id'],
            'difficulty': difficulty,
            'category': category,
            'prompt': problem['prompt'],
            'entry_point': problem['entry_point'],
            'canonical_solution': problem['canonical_solution'],
            'test': problem['test'],
            'estimated_tokens': len(problem['prompt']) // 4
        }
    else:  # mbpp
        entry = {
            'id': f'mbpp_{idx:03d}',
            'source': 'mbpp',
            'task_id': problem.get('task_id', idx),
            'difficulty': difficulty,
            'category': category,
            'prompt': problem.get('text', ''),
            'code': problem.get('code', ''),
            'test_list': problem.get('test_list', []),
            'test_setup_code': problem.get('test_setup_code', ''),
            'challenge_test_list': problem.get('challenge_test_list', []),
            'estimated_tokens': len(problem.get('text', '')) // 4
        }
    
    return entry


def main():
    """Selezione BILANCIATA dei 60 problemi"""
    print("\n" + "=" * 70)
    print(" SELEZIONE 60 PROBLEMI PERFETTAMENTE BILANCIATI")
    print("=" * 70)
    
    # Carica dataset
    humaneval = load_humaneval()
    mbpp = load_mbpp()
    
    print(f"\n Dataset caricati:")
    print(f"   HumanEval: {len(humaneval)} problemi")
    print(f"   MBPP: {len(mbpp)} problemi")
    
    print(f"\n Target: 20 easy / 20 medium / 20 hard")
    print(f" Per categoria: 5 easy + 5 medium + 5 hard = 15 per categoria")
    print(f" Categorie: strings, lists, math, algorithms")
    
    # Raccogli tutti i problemi classificati
    all_problems = []
    for p in humaneval + mbpp:
        source = 'humaneval' if p in humaneval else 'mbpp'
        diff = estimate_difficulty(p, source)
        cat = categorize_topic(p, source)
        all_problems.append((p, source, diff, cat))
    
    # Organizza per difficoltà e categoria
    organized = {
        'easy': {'strings': [], 'lists': [], 'math': [], 'algorithms': []},
        'medium': {'strings': [], 'lists': [], 'math': [], 'algorithms': []},
        'hard': {'strings': [], 'lists': [], 'math': [], 'algorithms': []}
    }
    
    for p, source, diff, cat in all_problems:
        if cat in organized[diff]:
            organized[diff][cat].append((p, source))
    
    # Seleziona 5 per categoria per difficoltà
    selected_problems = []
    idx = 0
    
    categories = ['strings', 'lists', 'math', 'algorithms']
    difficulties = ['easy', 'medium', 'hard']
    
    print(f"\n{'='*70}")
    print(" SELEZIONE PER CATEGORIA E DIFFICOLTÀ")
    print("="*70)
    
    for diff in difficulties:
        print(f"\n{diff.upper()}:")
        diff_count = 0
        
        for cat in categories:
            available = organized[diff][cat]
            
            # Se non ci sono abbastanza, prendi da difficoltà adiacente
            if len(available) < 5:
                if diff == 'hard':
                    # Prendi medium complessi (i più lunghi)
                    medium_candidates = organized['medium'][cat]
                    medium_candidates.sort(key=lambda x: len(x[0].get('prompt', x[0].get('text', ''))), reverse=True)
                    needed = 5 - len(available)
                    available.extend(medium_candidates[:needed])
                    print(f"   {cat}: {len(organized[diff][cat])} {diff} + {needed} medium complessi")
                elif diff == 'medium' and len(available) < 5:
                    # Prendi da easy complessi
                    easy_candidates = organized['easy'][cat]
                    easy_candidates.sort(key=lambda x: len(x[0].get('prompt', x[0].get('text', ''))), reverse=True)
                    needed = 5 - len(available)
                    available.extend(easy_candidates[:needed])
                    print(f"   {cat}: {len(organized[diff][cat])} {diff} + {needed} easy complessi")
                elif diff == 'easy' and len(available) < 5:
                    print(f"    {cat}: solo {len(available)} disponibili (servono 5)")
            else:
                print(f"   {cat}: 5 problemi")
            
            # Seleziona ESATTAMENTE 5 (o quanti disponibili)
            to_select = min(5, len(available))
            for p, source in available[:to_select]:
                entry = create_problem_entry(p, source, idx, diff, cat)
                selected_problems.append(entry)
                idx += 1
                diff_count += 1
        
        print(f"    Totale {diff}: {diff_count}")
    
    # STATISTICHE FINALI
    print(f"\n" + "=" * 70)
    print(" STATISTICHE SELEZIONE FINALE")
    print("=" * 70)
    
    difficulties_count = defaultdict(int)
    topics_count = defaultdict(int)
    sources_count = defaultdict(int)
    
    for p in selected_problems:
        difficulties_count[p['difficulty']] += 1
        topics_count[p['category']] += 1
        sources_count[p['source']] += 1
    
    print(f"\n Per difficoltà:")
    for diff in ['easy', 'medium', 'hard']:
        count = difficulties_count[diff]
        print(f"   {diff.upper()}: {count}")
    
    print(f"\n Per argomento:")
    for topic in sorted(topics_count.keys()):
        count = topics_count[topic]
        print(f"   {topic}: {count}")
    
    print(f"\n Per source:")
    for source, count in sources_count.items():
        print(f"   {source}: {count}")
    
    # Matrice categoria × difficoltà
    print(f"\n" + "=" * 70)
    print(" MATRICE CATEGORIA × DIFFICOLTÀ")
    print("=" * 70)
    
    matrix = defaultdict(lambda: defaultdict(int))
    for p in selected_problems:
        matrix[p['category']][p['difficulty']] += 1
    
    print(f"\n{'Categoria':<20} {'Easy':<8} {'Medium':<8} {'Hard':<8} {'Totale':<8}")
    print("-" * 60)
    
    for category in sorted(matrix.keys()):
        easy = matrix[category]['easy']
        medium = matrix[category]['medium']
        hard = matrix[category]['hard']
        total = easy + medium + hard
        print(f"{category:<20} {easy:<8} {medium:<8} {hard:<8} {total:<8}")
    
    print("-" * 60)
    total_easy = sum(matrix[cat]['easy'] for cat in matrix)
    total_medium = sum(matrix[cat]['medium'] for cat in matrix)
    total_hard = sum(matrix[cat]['hard'] for cat in matrix)
    total_all = total_easy + total_medium + total_hard
    print(f"{'TOTALE':<20} {total_easy:<8} {total_medium:<8} {total_hard:<8} {total_all:<8}")
    
    # CREA FILE JSON
    output = {
        'metadata': {
            'total_problems': len(selected_problems),
            'selection_date': '2025-10-31',
            'distribution': {
                'easy': difficulties_count['easy'],
                'medium': difficulties_count['medium'],
                'hard': difficulties_count['hard']
            },
            'topics': dict(topics_count),
            'sources': dict(sources_count)
        },
        'problems': selected_problems
    }
    
    output_path = Path(__file__).parent / 'selected_problems' / 'selected_problems.json'
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n File salvato: {output_path}")
    
    # Verifica bilanciamento
    print(f"\n" + "=" * 70)
    print(" VERIFICA BILANCIAMENTO")
    print("=" * 70)
    
    all_perfect = True
    
    # Check difficoltà
    if total_easy == 20 and total_medium == 20 and total_hard == 20:
        print(f" Difficoltà: PERFETTO (20/20/20)")
    else:
        print(f" Difficoltà: SBILANCIATO ({total_easy}/{total_medium}/{total_hard})")
        all_perfect = False
    
    # Check categorie
    target_per_cat = 15
    for cat in categories:
        total_cat = matrix[cat]['easy'] + matrix[cat]['medium'] + matrix[cat]['hard']
        if total_cat == target_per_cat:
            print(f" {cat}: PERFETTO (15 problemi: {matrix[cat]['easy']}E/{matrix[cat]['medium']}M/{matrix[cat]['hard']}H)")
        else:
            print(f" {cat}: {total_cat} problemi (target: 15)")
            all_perfect = False
    
    if all_perfect:
        print(f"\n BILANCIAMENTO PERFETTO RAGGIUNTO!")
    else:
        print(f"\n Alcuni sbilanciamenti presenti (dovuti a disponibilità limitata)")
    
    print(f"\n Pronto per implementare la pipeline!")

if __name__ == "__main__":
    main()
