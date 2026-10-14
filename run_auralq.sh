#!/usr/bin/env bash
# AuralQ Studio — Local Audio Intelligence Cockpit Launcher

echo "======================================================="
echo "         AuralQ Studio — Local Audio Intelligence"
echo "  Deterministic Signal Processing with Grounded AI"
echo "======================================================="
echo ""

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
    echo "[WARNING] You can still use Cloud Demo Mode in the sidebar."
fi
echo ""

# 3. Launch Streamlit Cockpit
echo "[INFO] Launching AuralQ Studio in your default browser..."
streamlit run app/streamlit_app.py

