"""
Script per scaricare e preparare i dataset HumanEval e MBPP
Da eseguire dopo aver installato Python e le dipendenze
"""

import os
import json
import urllib.request
from pathlib import Path


def download_humaneval():
    """Scarica il dataset HumanEval"""
    print("Scaricando HumanEval...")
    
    humaneval_dir = Path("code/data/humaneval")
    humaneval_dir.mkdir(parents=True, exist_ok=True)
    
    # URL del dataset HumanEval
    url = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
    output_file = humaneval_dir / "HumanEval.jsonl.gz"
    
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"HumanEval scaricato in: {output_file}")
        
        # Decomprimi il file
        import gzip
        import shutil
        
        with gzip.open(output_file, 'rb') as f_in:
            with open(humaneval_dir / "HumanEval.jsonl", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"HumanEval decompresso")
        return True
        
    except Exception as e:
        print(f"Errore durante download HumanEval: {e}")
        print("Soluzione alternativa: git clone https://github.com/openai/human-eval code/data/humaneval")
        return False


def download_mbpp():
    """Scarica il dataset MBPP usando HuggingFace datasets"""
    print("\nScaricando MBPP...")
    
    try:
        from datasets import load_dataset
        
        mbpp_dir = Path("code/data/mbpp")
        mbpp_dir.mkdir(parents=True, exist_ok=True)
        
        # Scarica MBPP
        dataset = load_dataset("mbpp", "sanitized")
        
        # Salva in formato JSON
        output_file = mbpp_dir / "mbpp_sanitized.json"
        
        all_problems = []
        for split in dataset.keys():
            for item in dataset[split]:
                all_problems.append(item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_problems, f, indent=2)
        
        print(f"MBPP scaricato: {len(all_problems)} problemi")
        print(f"Salvato in: {output_file}")
        return True
        
    except ImportError:
        print("Libreria 'datasets' non installata")
        print("Installa con: pip install datasets")
        return False
    except Exception as e:
        print(f"Errore durante download MBPP: {e}")
        return False


def analyze_datasets():
    """Analizza i dataset scaricati"""
    print("\nAnalisi dataset...")
    
    # Analizza HumanEval
    humaneval_file = Path("code/data/humaneval/HumanEval.jsonl")
    if humaneval_file.exists():
        with open(humaneval_file, 'r') as f:
            humaneval_problems = [json.loads(line) for line in f]
        
        print(f"\n🔹 HumanEval:")
        print(f"  - Totale problemi: {len(humaneval_problems)}")
        print(f"  - Esempio task_id: {humaneval_problems[0]['task_id']}")
    
    # Analizza MBPP
    mbpp_file = Path("code/data/mbpp/mbpp_sanitized.json")
    if mbpp_file.exists():
        with open(mbpp_file, 'r') as f:
            mbpp_problems = json.load(f)
        
        print(f"\n🔹 MBPP:")
        print(f"  - Totale problemi: {len(mbpp_problems)}")
        print(f"  - Esempio task_id: {mbpp_problems[0].get('task_id', 'N/A')}")
    
    print("\nDataset pronti per la selezione dei problemi!")


def main():
    """Funzione principale"""
    print("=" * 60)
    print("DOWNLOAD DATASET - TESI LLM CODE GENERATION")
    print("=" * 60)
    
    # Verifica che siamo nella directory corretta
    if not Path("code").exists():
        print("Directory 'code' non trovata!")
        print("Assicurati di eseguire questo script dalla root del progetto")
        return
    
    # Scarica i dataset
    humaneval_ok = download_humaneval()
    mbpp_ok = download_mbpp()
    
    # Analizza se entrambi sono stati scaricati
    if humaneval_ok or mbpp_ok:
        analyze_datasets()
    
    print("\n" + "=" * 60)
    print("PROSSIMI STEP:")
    print("1. Dataset scaricati")
    print("2. Seleziona 40-50 problemi rappresentativi")
    print("3. Crea code/data/selected_problems/problems.json")
    print("4. Configura API keys in .env")
    print("5. Testa i client LLM")
    print("=" * 60)


if __name__ == "__main__":
    main()
