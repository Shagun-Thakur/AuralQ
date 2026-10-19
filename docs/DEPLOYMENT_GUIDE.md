# AuralQ Deployment & Distribution Guide

This document describes how to run AuralQ locally on consumer hardware (100% air-gapped) or deploy it to public cloud hosts (Streamlit Community Cloud or HuggingFace Spaces) in ephemeral demo mode.

---

## 1. Local On-Premise Execution (100% Private & Air-Gapped)

Local mode is recommended for proprietary audio files, sensitive voice recordings, and research datasets. Audio files never leave your machine, and inference runs on your local GPU/CPU via Ollama.

### 1.1 Prerequisites
- **Python**: Version 3.10 or 3.11.
- **Ollama**: Installed from [ollama.ai](https://ollama.ai).
- **Target Model**: `llama3.2:3b-instruct-q4_K_M` (or any installed model).
  ```bash
  ollama pull llama3.2:3b-instruct-q4_K_M
  ```
- **Hardware**: Consumer NVIDIA GPU (RTX 3050 4 GB VRAM or higher) or modern multi-core CPU with 8+ GB RAM.

### 1.2 One-Click Execution

#### Windows
Double-click `run_auralq.bat` or execute in Command Prompt:
```cmd
run_auralq.bat
```
*What it does*:
1. Automatically detects and activates your virtual environment (`.venv` or `venv`).
2. Performs a pre-flight health check on Ollama (`127.0.0.1:11434`).
3. Launches the AuralQ Studio UI at `http://localhost:8501`.

#### Linux & macOS
Run the startup shell script in your terminal:
```bash
chmod +x run_auralq.sh
./run_auralq.sh
```

---

## 2. Public Cloud Deployment (Zero-Retention Ephemeral Demo)

When deploying to public cloud servers, AuralQ automatically shifts to **Cloud Demo Mode**. No audio or chat history is written to disk; all data resides strictly in volatile container RAM and is cleared when the user leaves.

### 2.1 Deploying to Streamlit Community Cloud

1. **Push your repository to GitHub**:
   Ensure `requirements.txt`, `.streamlit/config.toml`, and the codebase are pushed.
2. **Connect to Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Click **New app**.
   - Select your repository, branch (`main`), and set the main file path to:
     ```text
     app/streamlit_app.py
     ```
3. **Configure API Secrets**:
   Under **Advanced Settings** -> **Secrets**, enter your default cloud API key:
   ```toml
   GEMINI_API_KEY = "your-google-gemini-api-key"
   # Or for Groq:
   # GROQ_API_KEY = "your-groq-api-key"
   ```
4. **Deploy**:
   Click **Deploy**. Streamlit Cloud will automatically build the environment, read `.streamlit/config.toml` for the dark theme, and launch AuralQ Studio. Users can either use the default hosted API key or provide their own ephemeral key in the sidebar.

---

### 2.2 Deploying to HuggingFace Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Choose **Streamlit** as the Space SDK.
3. Add repository files (or configure Git remote):
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push space main
   ```
4. Under **Settings** -> **Variables and secrets**, add:
   - Name: `GEMINI_API_KEY`
   - Value: `<Your Google Gemini API Key>`
5. Your Space will build and serve AuralQ Studio with full cloud fallback support.

---

## 3. Configuration & Tuning Reference

All global system parameters can be adjusted in [`configs/default.yaml`](file:///c:/MyProjects/AuralQ/AuralQ/configs/default.yaml):

```yaml
audio:
  target_sr: 22050
  max_duration_seconds: 30.0
  normalize_peak: 0.95
  remove_dc: true

llm:
  model: "llama3.2:3b-instruct-q4_K_M"
  temperature: 0.0
  max_retries: 2
  options:
    num_ctx: 2048

validator:
  tolerance_percent: 5.0
  strict_mode: true
```

