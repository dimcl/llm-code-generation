@echo off
REM Setup automatico per LLM Code Generation
REM Eseguire DOPO aver installato Python

echo ========================================
echo SETUP AUTOMATICO LLM
echo ========================================
echo.

REM Verifica che Python sia installato
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato!
    echo.
    echo Installa Python 3.11 o 3.12 da:
    echo https://www.python.org/downloads/
    echo.
    echo Poi esegui di nuovo questo script.
    pause
    exit /b 1
)

echo [OK] Python trovato
python --version
echo.

REM Crea ambiente virtuale
echo [1/6] Creazione ambiente virtuale...
if exist venv (
    echo [SKIP] venv esiste gia
) else (
    python -m venv venv
    echo [OK] venv creato
)
echo.

REM Attiva ambiente virtuale
echo [2/6] Attivazione ambiente virtuale...
call venv\Scripts\activate.bat
echo [OK] venv attivato
echo.

REM Aggiorna pip
echo [3/6] Aggiornamento pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip aggiornato
echo.

REM Installa dipendenze
echo [4/6] Installazione dipendenze...
echo Questo puo richiedere 5-10 minuti...
pip install -r requirements.txt --quiet
echo [OK] Dipendenze installate
echo.

REM Crea file .env
echo [5/6] Configurazione file .env...
if exist .env (
    echo [SKIP] .env esiste gia
) else (
    copy .env.example .env >nul
    echo [OK] .env creato
    echo.
    echo IMPORTANTE: Modifica .env e inserisci le tue API keys!
    echo Apro il file...
    timeout /t 2 >nul
    notepad .env
)
echo.

REM Scarica dataset
echo [6/6] Download dataset...
echo.
set /p download="Vuoi scaricare i dataset ora? (s/n): "
if /i "%download%"=="s" (
    echo.
    echo Scaricamento dataset...
    python download_datasets.py
) else (
    echo [SKIP] Dataset - esegui manualmente: python download_datasets.py
)
echo.

REM Test setup
echo ========================================
echo SETUP COMPLETATO!
echo ========================================
echo.
echo Prossimi passi:
echo 1. Configura API keys in .env
echo 2. Scarica dataset: python download_datasets.py
echo 3. Testa client: python code/experiments/llm_clients/gpt_client.py
echo.
echo Per attivare venv in futuro:
echo   venv\Scripts\activate
echo.
pause
