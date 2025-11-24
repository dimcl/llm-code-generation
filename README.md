# LLM Code Generation - Studio Comparativo

**Progetto di Tesi**: Valutazione comparativa di Large Language Models per la generazione automatica di codice

## 📋 Panoramica del Progetto

Questo progetto confronta le prestazioni di 4 diversi LLM su task di generazione di codice:
- **GPT-4o-mini** (OpenAI via Azure)
- **Gemini 2.5 Flash-Lite** (Google)
- **Llama 3.1 8B Instant** (Meta via Groq)
- **Qwen 2.5 Coder 32B** (Alibaba via Groq)

Lo studio valuta 60 problemi di programmazione accuratamente selezionati dai dataset HumanEval e MBPP, bilanciati per:
- **Livelli di difficoltà**: Facile (20), Medio (20), Difficile (20)
- **Categorie**: Stringhe (15), Liste (15), Matematica (15), Algoritmi (15)

Esperimenti totali: **1.200 generazioni di codice** (60 problemi × 4 modelli × 5 tentativi per le metriche Pass@k)

## 🚀 Avvio Rapido

### Prerequisiti
- Python 3.10+
- API keys per: Google (Gemini), Groq (Llama/Qwen), Azure OpenAI (GPT)

### Installazione

1. Clona il repository:
```bash
git clone https://github.com/dimcl/llm-code-generation.git
cd llm-code-generation
```

2. Crea e attiva l'ambiente virtuale:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

4. Configura le API keys:
```bash
# Copia il file di esempio
cp .env.example .env

# Modifica .env e aggiungi le tue API keys
# GOOGLE_API_KEY=tua-chiave-qui
# GROQ_API_KEY=tua-chiave-qui
# AZURE_OPENAI_ENDPOINT=tuo-endpoint
# AZURE_OPENAI_KEY=tua-chiave-qui
```

### Download Dataset

```bash
python download_datasets.py
```

Questo scarica:
- HumanEval (164 problemi)
- MBPP (974 problemi)
- Subset di problemi selezionati (60 problemi)

### Esecuzione Esperimenti

**Test Pilota** (5 problemi, ~5 minuti):
```bash
python run_pilot_test.py
```

**Esperimenti Completi** (60 problemi, ~2 ore):
```bash
python run_full_experiments.py
```

### Generazione Analisi

```bash
# Genera tutte le metriche
python generate_metrics_report.py

# Genera tabelle (CSV + LaTeX)
python generate_final_tables.py

# Genera visualizzazioni
python generate_balance_figures.py
```

## Struttura del Progetto

```
llm-code-generation/
├── src/
│   ├── experiments/          # Generazione ed esecuzione codice
│   │   ├── code_generation.py
│   │   ├── code_execution.py
│   │   ├── prompt_templates.py
│   │   └── llm_clients/      # Client API per ogni modello
│   ├── evaluation/           # Calcolo metriche
│   │   ├── correctness_metrics.py
│   │   ├── quality_metrics.py
│   │   ├── cost_analysis.py
│   │   └── error_classifier.py
│   ├── analysis/             # Analisi statistica e visualizzazione
│   │   ├── statistical_tests.py
│   │   ├── visualization.py
│   │   └── case_studies/
│   └── data/                 # Dataset
│       ├── humaneval/
│       ├── mbpp/
│       └── selected_problems/
├── results/
│   ├── raw_outputs/          # Output grezzi dei modelli (JSON)
│   ├── metrics/              # Metriche calcolate (JSON)
│   ├── tables/               # Tabelle (CSV + LaTeX)
│   ├── figures/              # Visualizzazioni (PNG + PDF)
│   └── analysis/             # Case studies e analisi qualitativa
├── config.yaml               # Configurazione esperimenti
├── requirements.txt          # Dipendenze Python
└── README.md                 # Questo file
```

## 📊 Analisi Disponibili

### Metriche Quantitative
- **Pass@k**: Tasso di successo con k tentativi (k=1,3,5)
- **Qualità del Codice**: Complessità ciclomatica, metriche di Halstead, indice di manutenibilità
- **Efficienza**: Utilizzo token, latenza, costo per problema
- **Analisi Errori**: Classificazione di 12 tipi di errori

### Test Statistici
- Test chi-quadrato per distribuzioni categoriali
- Test H di Kruskal-Wallis per confronti non parametrici
- Test post-hoc di Dunn con correzione di Bonferroni
- Calcolo effect size (V di Cramér, epsilon-quadrato)

### Visualizzazioni
- Grafici di confronto Pass@k
- Heatmap tasso di successo (difficoltà × categoria)
- Scatter plot costo vs accuratezza
- Barre impilate distribuzione errori
- Box plot metriche di qualità

Tutti i risultati sono disponibili in:
- `results/tables/` - 32 tabelle in formato CSV e LaTeX
- `results/figures/` - 11 figure in formato PNG e PDF
- `results/metrics/` - File JSON con metriche dettagliate

## 🔬 Metodologia

1. **Selezione Problemi**: Campionamento stratificato per garantire rappresentazione bilanciata
2. **Generazione Codice**: 5 tentativi indipendenti per problema per modello
3. **Esecuzione Sandbox**: Esecuzione isolata sicura con timeout di 10s
4. **Analisi Qualità**: Analisi statica usando Radon, Pylint
5. **Validazione Statistica**: Test non parametrici (p < 0.05)


- Dataset HumanEval: [OpenAI](https://github.com/openai/human-eval)
- Dataset MBPP: [Google Research](https://github.com/google-research/google-research/tree/master/mbpp)
- Provider modelli: OpenAI (Azure), Google, Meta, Alibaba
- API: Azure OpenAI, Google AI Studio, Groq
