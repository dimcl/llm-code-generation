"""
Code execution sandbox for testing generated code
Esegue codice generato in modo sicuro con timeout e isolamento
"""

import subprocess
import tempfile
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import traceback


class CodeExecutor:
    """
    Sandbox sicuro per eseguire codice Python generato
    """
    
    def __init__(
        self,
        timeout: int = 10,
        memory_limit_mb: int = 512
    ):
        """
        Inizializza l'executor
        
        Args:
            timeout: Timeout in secondi per l'esecuzione
            memory_limit_mb: Limite di memoria in MB (non implementato su Windows)
        """
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
    
    def execute_humaneval_tests(
        self,
        generated_code: str,
        problem: Dict
    ) -> Dict:
        """
        Esegue i test di HumanEval per il codice generato
        
        Args:
            generated_code: Codice Python generato
            problem: Dizionario con informazioni sul problema HumanEval
        
        Returns:
            Dizionario con risultati dell'esecuzione
        """
        result = {
            'success': False,
            'passed': False,
            'error': None,
            'error_type': None,
            'execution_time': 0.0,
            'test_results': []
        }
        
        try:
            # Crea file temporaneo con codice + test
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as tmp_file:
                # Scrivi codice generato
                tmp_file.write(generated_code)
                tmp_file.write('\n\n')
                
                # Scrivi test da HumanEval
                tmp_file.write(problem['test'])
                tmp_file.write('\n\n')
                
                # Aggiungi chiamata test
                tmp_file.write('check(' + problem['entry_point'] + ')\n')
                
                tmp_path = tmp_file.name
            
            # Esegui in subprocess con timeout
            start_time = time.time()
            
            try:
                process = subprocess.run(
                    [sys.executable, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=os.path.dirname(tmp_path)
                )
                
                execution_time = time.time() - start_time
                result['execution_time'] = execution_time
                
                # Controlla risultato
                if process.returncode == 0:
                    result['success'] = True
                    result['passed'] = True
                else:
                    result['success'] = True  # Esecuzione completata
                    result['passed'] = False
                    result['error'] = process.stderr
                    result['error_type'] = self._classify_error(process.stderr)
                
            except subprocess.TimeoutExpired:
                result['error'] = f"Timeout dopo {self.timeout} secondi"
                result['error_type'] = 'timeout'
                execution_time = self.timeout
                result['execution_time'] = execution_time
            
            except Exception as e:
                result['error'] = str(e)
                result['error_type'] = 'runtime_error'
            
            # Pulisci file temporaneo
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        except Exception as e:
            result['error'] = str(e)
            result['error_type'] = 'execution_error'
        
        return result
    
    def execute_mbpp_tests(
        self,
        generated_code: str,
        problem: Dict
    ) -> Dict:
        """
        Esegue i test di MBPP per il codice generato
        
        Args:
            generated_code: Codice Python generato
            problem: Dizionario con informazioni sul problema MBPP
        
        Returns:
            Dizionario con risultati dell'esecuzione
        """
        result = {
            'success': False,
            'passed': False,
            'error': None,
            'error_type': None,
            'execution_time': 0.0,
            'test_results': [],
            'tests_passed': 0,
            'tests_total': 0
        }
        
        try:
            # Crea file temporaneo con codice + test
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as tmp_file:
                # Setup code (se presente)
                if problem.get('test_setup_code'):
                    tmp_file.write(problem['test_setup_code'])
                    tmp_file.write('\n\n')
                
                # Codice generato
                tmp_file.write(generated_code)
                tmp_file.write('\n\n')
                
                # Test assertions
                test_list = problem.get('test_list', [])
                result['tests_total'] = len(test_list)
                
                for test in test_list:
                    tmp_file.write(f"assert {test}\n")
                
                tmp_path = tmp_file.name
            
            # Esegui in subprocess con timeout
            start_time = time.time()
            
            try:
                process = subprocess.run(
                    [sys.executable, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=os.path.dirname(tmp_path)
                )
                
                execution_time = time.time() - start_time
                result['execution_time'] = execution_time
                
                # Controlla risultato
                if process.returncode == 0:
                    result['success'] = True
                    result['passed'] = True
                    result['tests_passed'] = result['tests_total']
                else:
                    result['success'] = True  # Esecuzione completata
                    result['passed'] = False
                    result['error'] = process.stderr
                    result['error_type'] = self._classify_error(process.stderr)
                    
                    # Conta quanti test sono passati (approssimazione)
                    if 'AssertionError' in process.stderr:
                        # Almeno un test fallito
                        result['tests_passed'] = 0
                
            except subprocess.TimeoutExpired:
                result['error'] = f"Timeout dopo {self.timeout} secondi"
                result['error_type'] = 'timeout'
                execution_time = self.timeout
                result['execution_time'] = execution_time
            
            except Exception as e:
                result['error'] = str(e)
                result['error_type'] = 'runtime_error'
            
            # Pulisci file temporaneo
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        except Exception as e:
            result['error'] = str(e)
            result['error_type'] = 'execution_error'
        
        return result
    
    def execute_code(
        self,
        generated_code: str,
        problem: Dict
    ) -> Dict:
        """
        Esegue il codice generato con i test appropriati
        
        Args:
            generated_code: Codice Python generato
            problem: Dizionario con informazioni sul problema
        
        Returns:
            Dizionario con risultati dell'esecuzione
        """
        # Controlla sintassi prima di eseguire
        syntax_check = self._check_syntax(generated_code)
        
        if not syntax_check['valid']:
            return {
                'success': False,
                'passed': False,
                'error': syntax_check['error'],
                'error_type': 'syntax_error',
                'execution_time': 0.0
            }
        
        # Esegui test basati sul source
        if problem['source'] == 'humaneval':
            return self.execute_humaneval_tests(generated_code, problem)
        elif problem['source'] == 'mbpp':
            return self.execute_mbpp_tests(generated_code, problem)
        else:
            return {
                'success': False,
                'passed': False,
                'error': f"Source sconosciuto: {problem['source']}",
                'error_type': 'unknown_source',
                'execution_time': 0.0
            }
    
    def _check_syntax(self, code: str) -> Dict:
        """
        Controlla la sintassi del codice senza eseguirlo
        
        Args:
            code: Codice Python da controllare
        
        Returns:
            Dizionario con validità e eventuale errore
        """
        try:
            compile(code, '<string>', 'exec')
            return {'valid': True, 'error': None}
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"SyntaxError: {e.msg} at line {e.lineno}"
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f"Compilation error: {str(e)}"
            }
    
    def _classify_error(self, error_message: str) -> str:
        """
        Classifica il tipo di errore dall'output stderr
        
        Args:
            error_message: Messaggio di errore
        
        Returns:
            Tipo di errore classificato
        """
        error_lower = error_message.lower()
        
        if 'syntaxerror' in error_lower:
            return 'syntax_error'
        elif 'indentationerror' in error_lower:
            return 'indentation_error'
        elif 'nameerror' in error_lower:
            return 'name_error'
        elif 'typeerror' in error_lower:
            return 'type_error'
        elif 'valueerror' in error_lower:
            return 'value_error'
        elif 'assertionerror' in error_lower:
            return 'assertion_error'
        elif 'indexerror' in error_lower or 'keyerror' in error_lower:
            return 'index_error'
        elif 'zerodivisionerror' in error_lower:
            return 'zero_division_error'
        elif 'recursionerror' in error_lower:
            return 'recursion_error'
        elif 'memoryerror' in error_lower:
            return 'memory_error'
        elif 'importerror' in error_lower or 'modulenotfounderror' in error_lower:
            return 'import_error'
        else:
            return 'runtime_error'
    
    def batch_execute(
        self,
        results: List[Dict],
        problems: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Esegue un batch di risultati di generazione
        
        Args:
            results: Lista di risultati da code_generation
            problems: Dizionario problem_id -> problem_data
        
        Returns:
            Lista di risultati con execution_results aggiunti
        """
        print(f"\n{'='*70}")
        print(f" INIZIO ESECUZIONE BATCH")
        print(f"{'='*70}")
        print(f" Generazioni da testare: {len(results)}")
        
        executed_results = []
        stats = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'syntax_errors': 0,
            'runtime_errors': 0,
            'timeouts': 0
        }
        
        for idx, result in enumerate(results, 1):
            # Salta se la generazione è fallita
            if not result['success'] or not result['generated_code']:
                result['execution_result'] = {
                    'success': False,
                    'passed': False,
                    'error': 'No code generated',
                    'error_type': 'no_code'
                }
                executed_results.append(result)
                stats['failed'] += 1
                continue
            
            problem = problems.get(result['problem_id'])
            if not problem:
                result['execution_result'] = {
                    'success': False,
                    'passed': False,
                    'error': 'Problem not found',
                    'error_type': 'problem_not_found'
                }
                executed_results.append(result)
                stats['failed'] += 1
                continue
            
            # Esegui codice
            if idx % 10 == 0:
                print(f"  Esecuzione {idx}/{len(results)}...")
            
            execution_result = self.execute_code(
                result['generated_code'],
                problem
            )
            
            result['execution_result'] = execution_result
            executed_results.append(result)
            
            # Aggiorna statistiche
            stats['total'] += 1
            if execution_result['passed']:
                stats['passed'] += 1
            else:
                stats['failed'] += 1
                error_type = execution_result.get('error_type', 'unknown')
                if error_type == 'syntax_error':
                    stats['syntax_errors'] += 1
                elif error_type == 'timeout':
                    stats['timeouts'] += 1
                else:
                    stats['runtime_errors'] += 1
        
        # Stampa riepilogo
        print(f"\n{'='*70}")
        print(f" ESECUZIONE BATCH COMPLETATA")
        print(f"{'='*70}")
        print(f" Test passati: {stats['passed']}/{stats['total']} ({stats['passed']/max(stats['total'],1)*100:.1f}%)")
        print(f" Test falliti: {stats['failed']}/{stats['total']}")
        print(f"  - Syntax errors: {stats['syntax_errors']}")
        print(f"  - Runtime errors: {stats['runtime_errors']}")
        print(f"  - Timeouts: {stats['timeouts']}")
        print(f"{'='*70}\n")
        
        return executed_results


if __name__ == "__main__":
    # Test executor
    print(" Test CodeExecutor")
    
    # Test semplice
    executor = CodeExecutor(timeout=5)
    
    # Codice corretto
    correct_code = """
def add_numbers(a, b):
    return a + b
"""
    
    # Codice con errore
    wrong_code = """
def add_numbers(a, b):
    return a + c  # NameError: c non definito
"""
    
    # Test sintassi
    print("\n1. Test sintassi codice corretto:")
    syntax_result = executor._check_syntax(correct_code)
    print(f"    Valido: {syntax_result['valid']}")
    
    print("\n2. Test sintassi codice con errore:")
    syntax_result = executor._check_syntax(wrong_code)
    print(f"    Valido: {syntax_result['valid']}")
    print(f"   Errore: {syntax_result.get('error', 'N/A')}")
    
    # Test esecuzione (richiede un problema reale)
    from pathlib import Path
    import json
    
    problems_file = Path(__file__).parent.parent / "data" / "selected_problems" / "selected_problems.json"
    
    if problems_file.exists():
        with open(problems_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            test_problem = data['problems'][0]
        
        print(f"\n3. Test esecuzione su problema reale:")
        print(f"   Problema: {test_problem['id']}")
        
        # Usa la soluzione canonica come test
        if test_problem['source'] == 'humaneval':
            test_code = test_problem['canonical_solution']
            exec_result = executor.execute_humaneval_tests(test_code, test_problem)
            print(f"    Passed: {exec_result['passed']}")
            print(f"   ⏱  Tempo: {exec_result['execution_time']:.3f}s")
            if exec_result['error']:
                print(f"    Errore: {exec_result['error'][:100]}")
    else:
        print(f"\n  File problemi non trovato: {problems_file}")
