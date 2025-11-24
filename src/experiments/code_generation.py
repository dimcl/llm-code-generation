"""
Code generation pipeline for LLM experiments
Gestisce la generazione di codice dai 4 modelli LLM con retry, timeout e logging
"""

import json
import time
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import traceback

from .llm_clients.gemini_client import GeminiClient
from .llm_clients.llama_client import GroqLlamaClient
from .llm_clients.qwen_client import QwenClient
from .llm_clients.gpt_client import GPTClient
from .prompt_templates import PromptTemplate


def extract_python_code(text: str) -> str:
    """
    Estrae codice Python dal testo generato dal LLM
    
    Gestisce:
    - Tag <think> di Qwen (rimuove il reasoning)
    - Blocchi markdown con ```python ... ```
    - Blocchi markdown con ``` ... ``` (generici)
    - Testo grezzo senza markdown
    - Rimuove spiegazioni prima/dopo il codice
    
    Args:
        text: Testo generato dal LLM
        
    Returns:
        Codice Python estratto
    """
    if not text or not text.strip():
        return ""
    
    # Rimuovi tag <think> di Qwen (modello reasoning)
    # Pattern: tutto ciò che viene prima di def o dopo </think>
    if '<think>' in text.lower():
        # Cerca la fine del tag think
        think_end = re.search(r'</think>', text, re.IGNORECASE)
        if think_end:
            # Prendi solo il codice dopo </think>
            text = text[think_end.end():]
    
    # Cerca blocchi markdown con ```python ... ```
    python_blocks = re.findall(r'```python\s*(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if python_blocks:
        # Prendi il primo blocco Python trovato
        return python_blocks[0].strip()
    
    # Cerca blocchi markdown generici ``` ... ```
    generic_blocks = re.findall(r'```\s*(.*?)```', text, re.DOTALL)
    if generic_blocks:
        # Prendi il primo blocco trovato
        return generic_blocks[0].strip()
    
    # Se non ci sono blocchi markdown, prova a trovare il codice
    # cercando le righe che iniziano con 'def ' (funzioni Python)
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        # Inizia quando trova 'def '
        if line.strip().startswith('def '):
            in_code = True
        
        # Se siamo nel codice, aggiungi la riga
        if in_code:
            code_lines.append(line)
            
            # Interrompi se troviamo righe vuote consecutive dopo il codice
            if not line.strip() and code_lines and len(code_lines) > 3:
                # Controlla se le ultime 2 righe sono vuote
                if all(not l.strip() for l in code_lines[-2:]):
                    break
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # Se nessuna estrazione funziona, restituisci il testo originale ripulito
    return text.strip()


class CodeGenerator:
    """
    Pipeline per generare codice dai 4 LLM
    """
    
    def __init__(
        self,
        output_dir: str = "results/raw_outputs",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Inizializza il generatore di codice
        
        Args:
            output_dir: Directory per salvare i risultati
            temperature: Temperatura per la generazione (0.0-1.0)
            max_tokens: Numero massimo di token generati
            max_retries: Numero massimo di tentativi in caso di errore
            retry_delay: Secondi di attesa tra i tentativi
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Inizializza i 4 client LLM
        print("🔧 Inizializzazione client LLM...")
        self.clients = {
            'gemini': GeminiClient(),
            'llama': GroqLlamaClient(),
            'qwen': QwenClient(),
            'gpt': GPTClient()
        }
        print(" 4 client LLM inizializzati")
    
    def generate_code(
        self,
        problem: Dict,
        model_name: str,
        prompt_type: str = "basic",
        attempt: int = 1
    ) -> Dict:
        """
        Genera codice per un problema usando un LLM specifico
        
        Args:
            problem: Dizionario con informazioni sul problema
            model_name: Nome del modello ('gemini', 'llama', 'qwen', 'gpt')
            prompt_type: Tipo di prompt ('basic', 'few_shot', 'chain_of_thought')
            attempt: Numero del tentativo (1-5 per Pass@k)
        
        Returns:
            Dizionario con risultati della generazione
        """
        if model_name not in self.clients:
            raise ValueError(f"Modello sconosciuto: {model_name}")
        
        client = self.clients[model_name]
        
        # Prepara il prompt
        if problem['source'] == 'humaneval':
            prompt_text = problem['prompt']
            function_signature = f"def {problem['entry_point']}("
        else:  # mbpp
            prompt_text = problem['prompt']
            function_signature = None
        
        # Crea prompt con template
        if prompt_type == "basic":
            prompt = PromptTemplate.basic(prompt_text, function_signature)
        elif prompt_type == "chain_of_thought":
            prompt = PromptTemplate.chain_of_thought(prompt_text, function_signature)
        else:
            prompt = PromptTemplate.basic(prompt_text, function_signature)
        
        # Genera codice con retry logic
        result = {
            'problem_id': problem['id'],
            'model': model_name,
            'attempt': attempt,
            'prompt_type': prompt_type,
            'success': False,
            'error': None,
            'generated_code': None,
            'tokens': 0,
            'latency': 0.0,
            'cost': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        for retry in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Genera codice
                response = client.generate_code(
                    prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                latency = time.time() - start_time
                
                # Estrai codice Python da possibili blocchi markdown o testo extra
                extracted_code = extract_python_code(response.code)
                
                # Salva risultati (response è GenerationResult)
                # Gestisce sia la struttura di base_client (tokens_used, time_seconds)
                # che quella custom dei client (total_tokens, latency_seconds)
                result['success'] = True
                result['generated_code'] = extracted_code  # Usa codice estratto
                result['raw_response'] = response.code     # Salva anche risposta originale
                result['tokens'] = getattr(response, 'tokens_used', getattr(response, 'total_tokens', 0))
                result['latency'] = getattr(response, 'time_seconds', getattr(response, 'latency_seconds', latency))
                result['cost'] = response.cost_usd
                
                print(f"   {model_name.upper()}: {latency:.2f}s, {result['tokens']} tokens")
                break
                
            except Exception as e:
                result['error'] = str(e)
                print(f"   {model_name.upper()} errore (tentativo {retry+1}/{self.max_retries}): {str(e)}")
                
                if retry < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    print(f"   {model_name.upper()} fallito dopo {self.max_retries} tentativi")
        
        return result
    
    def generate_for_problem(
        self,
        problem: Dict,
        models: List[str] = None,
        num_attempts: int = 5,
        prompt_type: str = "basic"
    ) -> List[Dict]:
        """
        Genera k tentativi di codice per un problema con tutti i modelli
        
        Args:
            problem: Dizionario con informazioni sul problema
            models: Lista modelli da usare (None = tutti)
            num_attempts: Numero di tentativi per Pass@k (default: 5)
            prompt_type: Tipo di prompt da usare
        
        Returns:
            Lista di dizionari con risultati
        """
        if models is None:
            models = ['gemini', 'llama', 'qwen', 'gpt']
        
        print(f"\n{'='*70}")
        print(f" Problema: {problem['id']} ({problem['difficulty']}, {problem['category']})")
        print(f"   Source: {problem['source']}")
        print(f"{'='*70}")
        
        all_results = []
        
        for model in models:
            print(f"\n Modello: {model.upper()}")
            
            for attempt in range(1, num_attempts + 1):
                print(f"  Tentativo {attempt}/{num_attempts}...", end=" ")
                
                result = self.generate_code(
                    problem=problem,
                    model_name=model,
                    prompt_type=prompt_type,
                    attempt=attempt
                )
                
                all_results.append(result)
                
                # Rate limiting (evita di superare i limiti API)
                if model == 'gemini':
                    time.sleep(1.0)  # 60 req/min → ~1s tra richieste
                elif model in ['llama', 'qwen']:
                    time.sleep(2.0)  # 30 req/min → ~2s tra richieste
                # GPT-4o-mini Azure non ha limiti stretti
        
        return all_results
    
    def generate_batch(
        self,
        problems: List[Dict],
        models: List[str] = None,
        num_attempts: int = 5,
        prompt_type: str = "basic",
        save_interval: int = 10
    ) -> Dict:
        """
        Genera codice per un batch di problemi
        
        Args:
            problems: Lista di problemi
            models: Lista modelli da usare (None = tutti)
            num_attempts: Numero di tentativi per Pass@k
            prompt_type: Tipo di prompt
            save_interval: Salva risultati ogni N problemi
        
        Returns:
            Dizionario con statistiche e risultati
        """
        print(f"\n{'='*70}")
        print(f" INIZIO GENERAZIONE BATCH")
        print(f"{'='*70}")
        print(f" Problemi: {len(problems)}")
        print(f" Modelli: {models if models else ['gemini', 'llama', 'qwen', 'gpt']}")
        print(f" Tentativi per problema: {num_attempts}")
        print(f" Tipo prompt: {prompt_type}")
        print(f" Salvataggio ogni: {save_interval} problemi")
        
        total_generations = len(problems) * (len(models) if models else 4) * num_attempts
        print(f" Generazioni totali: {total_generations}")
        print(f"{'='*70}\n")
        
        all_results = []
        batch_stats = {
            'total_problems': len(problems),
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_tokens': 0,
            'total_latency': 0.0,
            'total_cost': 0.0,
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
        
        start_time = time.time()
        
        for idx, problem in enumerate(problems, 1):
            # Genera per questo problema
            results = self.generate_for_problem(
                problem=problem,
                models=models,
                num_attempts=num_attempts,
                prompt_type=prompt_type
            )
            
            all_results.extend(results)
            
            # Aggiorna statistiche
            for result in results:
                batch_stats['total_generations'] += 1
                if result['success']:
                    batch_stats['successful_generations'] += 1
                    batch_stats['total_tokens'] += result['tokens']
                    batch_stats['total_latency'] += result['latency']
                    batch_stats['total_cost'] += result['cost']
                else:
                    batch_stats['failed_generations'] += 1
            
            # Salva risultati parziali ogni N problemi
            if idx % save_interval == 0 or idx == len(problems):
                self._save_results(all_results, batch_stats, partial=True)
                print(f"\n Salvati risultati parziali ({idx}/{len(problems)} problemi)")
        
        batch_stats['end_time'] = datetime.now().isoformat()
        elapsed_time = time.time() - start_time
        
        # Salva risultati finali
        self._save_results(all_results, batch_stats, partial=False)
        
        # Stampa riepilogo
        print(f"\n{'='*70}")
        print(f" GENERAZIONE BATCH COMPLETATA")
        print(f"{'='*70}")
        print(f"  Tempo totale: {elapsed_time/60:.1f} minuti")
        print(f" Successi: {batch_stats['successful_generations']}/{batch_stats['total_generations']}")
        print(f" Fallimenti: {batch_stats['failed_generations']}/{batch_stats['total_generations']}")
        print(f" Token totali: {batch_stats['total_tokens']:,}")
        print(f" Costo totale: ${batch_stats['total_cost']:.4f}")
        print(f" Latenza media: {batch_stats['total_latency']/max(batch_stats['successful_generations'],1):.2f}s")
        print(f"{'='*70}\n")
        
        return batch_stats
    
    def _save_results(self, results: List[Dict], stats: Dict, partial: bool = False):
        """Salva risultati su disco"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_partial" if partial else ""
        
        # Salva risultati completi
        results_file = self.output_dir / f"results_{timestamp}{suffix}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'statistics': stats,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"  💾 Risultati salvati: {results_file}")


if __name__ == "__main__":
    # Test su un singolo problema
    print(" Test CodeGenerator")
    
    # Carica un problema di esempio
    from pathlib import Path
    import json
    
    problems_file = Path(__file__).parent.parent / "data" / "selected_problems" / "selected_problems.json"
    
    if problems_file.exists():
        with open(problems_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            test_problem = data['problems'][0]  # Primo problema (easy)
        
        generator = CodeGenerator(
            output_dir="results/raw_outputs/test",
            temperature=0.2,
            max_tokens=1024
        )
        
        # Test su un solo modello con 2 tentativi
        results = generator.generate_for_problem(
            problem=test_problem,
            models=['gemini'],  # Solo Gemini per test veloce
            num_attempts=2,
            prompt_type="basic"
        )
        
        print(f"\n Test completato: {len(results)} generazioni")
        for r in results:
            print(f"  - Attempt {r['attempt']}: {'successo' if r['success'] else 'fallito'}")
    else:
        print(f" File problemi non trovato: {problems_file}")
