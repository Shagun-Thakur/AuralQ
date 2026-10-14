@echo off
TITLE AuralQ Studio — Local Audio Intelligence Cockpit
COLOR 0B

echo =======================================================
echo          AuralQ Studio — Local Audio Intelligence
echo   Deterministic Signal Processing with Grounded AI
echo =======================================================
echo.

:: 1. Check for Virtual Environment
IF EXIST ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) ELSE IF EXIST "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment (venv)...
    call venv\Scripts\activate.bat
) ELSE (
    echo [INFO] Using global Python environment...
)

:: 2. Pre-flight check for Ollama daemon
echo [INFO] Checking local Ollama service status...
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Ollama is not detected at http://127.0.0.1:11434
    echo [WARNING] Local models will be unavailable until Ollama is started.
    echo [WARNING] You can still use Cloud Demo Mode in the sidebar.
    echo.
) ELSE (
    echo [OK] Ollama daemon is running and reachable.
    echo.
)

:: 3. Launch Streamlit Cockpit
echo [INFO] Launching AuralQ Studio in your default browser...
echo.
streamlit run app/streamlit_app.py

pause

