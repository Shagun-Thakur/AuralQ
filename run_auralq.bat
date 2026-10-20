@echo off
setlocal enabledelayedexpansion
title AuralQ Studio - Local Audio Intelligence Cockpit
color 0B

cd /d "%~dp0"

echo =======================================================
echo          AuralQ Studio - Local Audio Intelligence
echo   Deterministic Signal Processing with Grounded AI
echo =======================================================
echo.

REM 1. Activate Virtual Environment if present
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment: .venv
    call .venv\Scripts\activate.bat
    goto :env_ready
)
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment: venv
    call venv\Scripts\activate.bat
    goto :env_ready
)
echo [INFO] Using system Python environment...
:env_ready

REM 2. Verify Python is installed and accessible
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    where py >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Python was not found on your system PATH!
        echo Please install Python 3.10+ from https://www.python.org/
        echo and ensure "Add Python to PATH" is checked during installation.
        echo.
        pause
        exit /b 1
    )
    set "PY_CMD=py"
) else (
    set "PY_CMD=python"
)

REM 3. Pre-flight check for Ollama daemon
echo [INFO] Checking local Ollama service status...
curl -s http://127.0.0.1:11434/api/tags >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Ollama is not detected at http://127.0.0.1:11434
    echo [WARNING] Local models will be unavailable until Ollama is started.
    echo [WARNING] You can still use Cloud Demo Mode in the sidebar.
    echo.
) else (
    echo [OK] Ollama daemon is running and reachable.
    echo.
)

REM 4. Launch Streamlit Cockpit
echo [INFO] Launching AuralQ Studio in your default browser...
echo [INFO] (Keep this window open while using AuralQ. Press Ctrl+C to stop)
echo.

%PY_CMD% -m streamlit run app/streamlit_app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to start Streamlit app (exit code %ERRORLEVEL%).
    echo If dependencies are missing, try running: pip install -r requirements.txt
    echo.
    pause
)
