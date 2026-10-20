#!/usr/bin/env bash
# AuralQ Studio — Local Audio Intelligence Cockpit Launcher

echo "======================================================="
echo "         AuralQ Studio — Local Audio Intelligence"
echo "  Deterministic Signal Processing with Grounded AI"
echo "======================================================="
echo ""

# Navigate to script directory
cd "$(dirname "$0")" || exit 1

# 1. Activate virtual environment if available
if [ -f ".venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    echo "[INFO] Activating virtual environment (venv)..."
    source venv/bin/activate
fi

# 2. Pre-flight check for Ollama daemon
echo "[INFO] Checking local Ollama service status..."
if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "[OK] Ollama daemon is running and reachable."
else
    echo "[WARNING] Ollama is not detected at http://127.0.0.1:11434"
    echo "[WARNING] Local models will be unavailable until Ollama is started."
    echo "[WARNING] You can still use Cloud Demo Mode in the sidebar."
fi
echo ""

# 3. Launch Streamlit Cockpit
echo "[INFO] Launching AuralQ Studio in your default browser..."
if command -v python3 &> /dev/null; then
    python3 -m streamlit run app/streamlit_app.py
else
    streamlit run app/streamlit_app.py
fi
