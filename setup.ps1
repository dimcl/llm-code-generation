# Setup automatico per LLM Code Generation
# Eseguire DOPO aver installato Python

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SETUP AUTOMATICO LLM" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Verifica che Python sia installato
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python trovato" -ForegroundColor Green
    Write-Host $pythonVersion -ForegroundColor Gray
    Write-Host ""
} catch {
    Write-Host "[ERRORE] Python non trovato!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Installa Python 3.11 o 3.12 da:" -ForegroundColor Yellow
    Write-Host "https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Poi esegui di nuovo questo script." -ForegroundColor Yellow
    Read-Host "Premi ENTER per uscire"
    exit 1
}

# Crea ambiente virtuale
Write-Host "[1/6] Creazione ambiente virtuale..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "[SKIP] venv esiste gia" -ForegroundColor Gray
} else {
    python -m venv venv
    Write-Host "[OK] venv creato" -ForegroundColor Green
}
Write-Host ""

# Attiva ambiente virtuale
Write-Host "[2/6] Attivazione ambiente virtuale..." -ForegroundColor Yellow
try {
    & ".\venv\Scripts\Activate.ps1"
    Write-Host "[OK] venv attivato" -ForegroundColor Green
} catch {
    Write-Host "[ERRORE] Impossibile attivare venv" -ForegroundColor Red
    Write-Host ""
    Write-Host "Prova a eseguire:" -ForegroundColor Yellow
    Write-Host "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Poi riprova questo script." -ForegroundColor Yellow
    Read-Host "Premi ENTER per uscire"
    exit 1
}
Write-Host ""

# Aggiorna pip
Write-Host "[3/6] Aggiornamento pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "[OK] pip aggiornato" -ForegroundColor Green
Write-Host ""

# Installa dipendenze
Write-Host "[4/6] Installazione dipendenze..." -ForegroundColor Yellow
Write-Host "Questo puo richiedere 5-10 minuti..." -ForegroundColor Gray
pip install -r requirements.txt
Write-Host "[OK] Dipendenze installate" -ForegroundColor Green
Write-Host ""

# Crea file .env
Write-Host "[5/6] Configurazione file .env..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "[SKIP] .env esiste gia" -ForegroundColor Gray
} else {
    Copy-Item ".env.example" ".env"
    Write-Host "[OK] .env creato" -ForegroundColor Green
    Write-Host ""
    Write-Host "IMPORTANTE: Modifica .env e inserisci le tue API keys!" -ForegroundColor Magenta
    Write-Host "Apro il file..." -ForegroundColor Gray
    Start-Sleep -Seconds 2
    notepad .env
}
Write-Host ""

# Scarica dataset
Write-Host "[6/6] Download dataset..." -ForegroundColor Yellow
Write-Host ""
$download = Read-Host "Vuoi scaricare i dataset ora? (s/n)"
if ($download -eq "s" -or $download -eq "S") {
    Write-Host ""
    Write-Host "Scaricamento dataset..." -ForegroundColor Gray
    python download_datasets.py
} else {
    Write-Host "[SKIP] Dataset - esegui manualmente: python download_datasets.py" -ForegroundColor Gray
}
Write-Host ""

# Test setup
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETATO!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Prossimi passi:" -ForegroundColor Yellow
Write-Host "1. Configura API keys in .env"
Write-Host "2. Scarica dataset: python download_datasets.py"
Write-Host "3. Testa client: python code/experiments/llm_clients/gpt_client.py"
Write-Host ""
Write-Host "Per attivare venv in futuro:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host ""
Read-Host "Premi ENTER per uscire"
