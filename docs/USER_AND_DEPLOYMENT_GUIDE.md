# AuralQ: User Operations & Deployment Guide

> **Practical Guide for Operating, Benchmarking, and Deploying AuralQ Studio**  
> *Covers offline air-gapped workstations, the interactive Ender-Acoustic Cockpit, dataset profiling, and public cloud deployment.*

---

## 1. Hardware & Software Requirements

AuralQ is intentionally engineered to run on affordable consumer workstations and standard laptops.

### 1.1 Minimum Hardware Specifications
- **GPU (Recommended for Local LLM)**: NVIDIA GeForce RTX 3050 (4 GB VRAM) or any modern NVIDIA GPU with CUDA support.
  - *Apple Silicon*: M1/M2/M3 with 8 GB+ unified memory.
  - *CPU Fallback*: Supported on modern x86_64 / ARM64 CPUs (AMD Ryzen 5 / Intel Core i5 with 16 GB system RAM).
- **Disk Space**: ~3.5 GB (Ollama + `llama3.2:3b-instruct-q4_K_M` model weight: ~2.0 GB, Python virtual environment: ~1.5 GB).
- **Operating Systems**: Windows 10/11, Ubuntu 20.04/22.04/24.04 LTS, macOS Monterey or newer.

### 1.2 Software Prerequisites
- **Python**: Version 3.10 or 3.11 (Python 3.11 is strongly recommended).
- **Ollama**: Local LLM server (download from [ollama.ai](https://ollama.ai)).

---

## 2. Installation & Quickstart

### Step 1: Clone Repository
```bash
git clone https://github.com/Shagun-Thakur/AuralQ.git
cd AuralQ
```

### Step 2: Set Up Python Virtual Environment
```bash
# Create environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Install and Pull Local Model
Start the Ollama daemon and pull the lightweight instruction-tuned model:
```bash
# In a terminal or background service
ollama serve

# Pull the primary model (~2.0 GB download)
ollama pull llama3.2:3b-instruct-q4_K_M
```

---

## 3. Launching AuralQ

AuralQ includes one-click executable scripts that automatically handle environment activation, model verification, and browser startup:

### Windows (One-Click Batch File)
Double-click `run_auralq.bat` in Windows File Explorer, or execute via PowerShell:
```powershell
.\run_auralq.bat
```

### Linux & macOS (Shell Script)
```bash
chmod +x run_auralq.sh
./run_auralq.sh
```

### Direct Streamlit Command
```bash
streamlit run app/streamlit_app.py
```
The studio interface will open automatically in your browser at `http://localhost:8501`.

---

## 4. Interactive Studio Cockpit Walkthrough

The interface follows the **Ender-Acoustic Console** design system, structured as an asymmetric two-column workspace.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ AuralQ Studio   Deterministic Audio Intelligence with Grounding  [ LOCAL • AIR ]│
├──────────────────────────────────────┬──────────────────────────────────────────┤
│ SOURCE AUDIO & INGESTION (37%)       │ CONVERSATION & ANALYTICS (63%)           │
│                                      │                                          │
│ [Drag & Drop Audio Hopper]           │ [Acoustic Query Terminal: Type question] │
│ [ 1k Sine ]  [ Snare ]  [ Noise ]    │ [ IGNITE ACOUSTIC RUN ]                  │
│                                      │                                          │
│ ● ACTIVE: clean_sine_1khz.wav        │ ✦ CITATION VERIFIED: All numbers grounded│
│ [ HTML5 Audio Player Controls ]      │                                          │
│                                      │ ┌──────────────────────────────────────┐ │
│ ┌──────────────┬───────────────────┐ │ │ Grounded Answer Panel:               │ │
│ │ Rate: 44.1k  │ Duration: 3.00s   │ │ │ "The audio exhibits pure tonal       │ │
│ ├──────────────┼───────────────────┤ │ │ structure with spectral flatness     │ │
│ │ RMS: -3.0dB  │ Status: CLEAN     │ │ │ of 0.0031..."                        │ │
│ └──────────────┴───────────────────┘ │ └──────────────────────────────────────┘ │
│                                      │                                          │
│ [Batch Dataset Directory Audit]      │ [▶ Inspect Tool Trace & Evidence JSON]   │
│                                      │                                          │
│                                      │ [ Spectrogram | Waveform | Telemetry ]   │
│                                      │   • Logarithmic STFT Spectrogram         │
│                                      │   • Sub-band Energy Partition Bins       │
│                                      │   • 20ms Transient Waveform Zoom         │
└──────────────────────────────────────┴──────────────────────────────────────────┘
```

### 4.1 Ingestion & Audio Source (Left Column)
- **Hopper Upload Slot**: Drag and drop any `.wav`, `.mp3`, `.flac`, or `.ogg` file.
  - Automatically runs Layer 1 hygiene checks: DC-offset blocking, silence gating (fails if $<-60\text{ dBFS}$), and flat-top saturation clipping detection ($\ge 3$ continuous samples at $|\tilde{y}| \ge 0.99$).
- **Hotbar Presets**: Instant one-click test fixtures:
  - `1k Sine`: Analytically pure 1,000 Hz sinusoidal tone (tests tonal flatness $\approx 0.00$).
  - `Snare`: Saturated percussive strike with transient clipping (tests clipping warnings and high ZCR).
  - `Noise`: Gaussian white noise (tests high spectral flatness $\approx 0.85$ and wide bandwidth).
- **Active Jukebox & Audio Player**: Inline HTML5 playback controller displaying the active file name.
- **Instantaneous Acoustic Metrics**:
  - Sample Rate (Hz), Duration (s), RMS Volume (dBFS), and Signal Quality (`CLEAN` or `CLIPPING WARNING`).

### 4.2 Query Execution & Grounding (Right Column)
- **Acoustic Query Terminal**: Enter natural language questions such as:
  - *"Is this recording noisy or does it have tonal harmonic structure?"*
  - *"Does this sound bright or dull? What is the center frequency?"*
  - *"Is the amplitude consistent or does it fluctuate?"*
  - *"Extract MFCCs to describe the timbral texture."*
- **Ignite Acoustic Run**: Dispatches Layer 2 intent parsing, triggers deterministic Layer 3 DSP compute, packages evidence, and validates citations.
- **Verification Banner**: Shows a teal `✦ CITATION VERIFIED` badge if all numerical claims match computed DSP evidence within 5% tolerance, or a purple `🛡️ REPAIRED` badge if hallucination repair was applied.
- **Tool Trace Expander**: Collapsible JSON inspector showing dispatched tool names, arguments, execution latencies, and raw numeric evidence packages.

### 4.3 Deep Visualizations & Diagnostics
- **STFT Spectrogram Tab**: Interactive logarithmic time-frequency color map, including quantitative sub-band power partitions (Bass $<500\text{ Hz}$, Mid-range $500–4000\text{ Hz}$, Treble $>4000\text{ Hz}$).
- **Waveform Tab**: Full temporal waveform with interactive **20 ms Zoom** to inspect individual wave cycles, zero-crossings, and peak clipping saturation.
- **Telemetry Tab**: Live diagnostic readouts showing GPU memory allocation, Ollama daemon ping latency, DSP calculation time, and active turn token count.

### 4.4 Past Sessions Management (Sidebar)
- **Multi-Turn Continuity**: Prior queries and answers are stored locally in SQLite (`data/sessions/auralq_sessions.db`).
- **Load / Delete**: Switch between saved sessions or click **Del** to cascade delete a session and all its turn logs.
- **Export Report**: Export the entire session's grounded findings as a clean Markdown report with a single click.

---

## 5. Streaming Dataset Profiler Guide

When analyzing large corpora of audio files (e.g. training datasets for Whisper, AudioLDM, or classification models), AuralQ's dataset profiler inspects directories out-of-core:

### How to Profile an Audio Dataset
1. In the left column, scroll to **Batch Dataset Audit**.
2. Enter the local path to your audio directory (e.g. `data/benchmark/dataset_audit_fixture` or `C:/MyAudioCorpus`).
3. In the query terminal, type:
   > *"Profile this dataset folder and recommend what preprocessing pipeline I should apply."*
4. Click **Ignite Acoustic Run**.

### Understanding Preprocessing Recommendations
The profiler evaluates distributions across files using streaming online Welford accumulation and emits deterministic recommendations:

| Recommendation | Trigger Condition | Automated Remediation |
| :--- | :--- | :--- |
| **`RESAMPLE`** | File sample rates diverge across files | Resample corpus to uniform standard rate (e.g. 22,050 Hz or 44,100 Hz via `soxr`) |
| **`MONO_DOWNMIX`** | Multi-channel/stereo files detected | Downmix channels using equal-power formula $0.707 \times (L + R)$ |
| **`CHUNKING_OR_PADDING`** | Duration variance $(\max - \min)/\text{median} > 0.5$ | Segment long clips or pad with zeros to median length |
| **`LOUDNESS_NORMALIZATION`**| Loudness std-dev $>6.0\text{ dB}$ across files | Apply EBU R128 loudness normalization to $-20.0\text{ dBFS}$ |
| **`HIGH_PASS_FILTER`** | Mean spectral centroid $<800\text{ Hz}$ | Apply 2nd-order Butterworth high-pass filter ($60\text{ Hz}$ cutoff) to strip rumble |
| **`PRUNE_SILENCE`** | Files with $\text{RMS} < -60\text{ dBFS}$ | Exact file paths flagged for deletion to prevent training collapse |
| **`PRUNE_OR_DECLIP`** | Files with saturated clipping $>0.1\%$ | Flagged for cubic-spline declipping or exclusion |

---

## 6. Dual-Mode Deployment Configurations

AuralQ provides seamless dual-engine operation:

### Mode A: 100% Air-Gapped Local Workstation
- **Privacy**: 100% offline. Zero network packets leave the machine.
- **LLM Engine**: Local Ollama daemon (`llama3.2:3b-instruct-q4_K_M`).
- **Persistence**: SQLite with Write-Ahead Logging (`auralq_sessions.db`).
- **Status Indicator**: Teal `[ LOCAL • AIR-GAPPED ]` pill in top navigation.

### Mode B: Public Cloud Demo Mode (Streamlit Community Cloud)
- **Purpose**: Public demonstration without requiring users to have a local GPU or Ollama installed.
- **LLM Engine**: Google Gemini API (`gemini-2.5-flash`) or Groq Cloud API (`llama-3.3-70b-versatile`).
- **Persistence**: Strict **ephemeral session memory** (no database files created on host).
- **Status Indicator**: Purple `[ CLOUD • EPHEMERAL ]` pill in top navigation.

#### Configuring Cloud Secrets (`.streamlit/secrets.toml`)
When deploying on Streamlit Community Cloud:
1. In the app settings on Streamlit Community Cloud, navigate to **Secrets**.
2. Add your hosted API key:
   ```toml
   # Google Gemini API (Recommended for free tier)
   GEMINI_API_KEY = "AIzaSy..."

   # Or Groq Cloud API
   GROQ_API_KEY = "gsk_..."
   ```
3. Users visiting the app automatically connect to the hosted key, with the option to provide their own temporary session override key in the sidebar.

---

## 7. Production Deployment (Headless Server)

For deploying AuralQ on a dedicated Ubuntu/Debian server with GPU acceleration:

### Systemd Service Configuration
Create `/etc/systemd/system/auralq.service`:
```ini
[Unit]
Description=AuralQ Studio Streamlit Application
After=network.target

[Service]
Type=simple
User=auralq-user
WorkingDirectory=/opt/auralq
ExecStart=/opt/auralq/.venv/bin/streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
Restart=always
RestartSec=5
Environment=PATH=/opt/auralq/.venv/bin:/usr/local/bin:/usr/bin

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now auralq
```

---

## 8. Troubleshooting & FAQ

### Q: "Ollama is Offline" warning in sidebar?
- Ensure the Ollama daemon is running: open a new terminal and run `ollama serve`.
- Verify Ollama port: default is `http://127.0.0.1:11434`.

### Q: Out of Memory (OOM) on 4 GB GPU?
- Verify context length: AuralQ locks context size to `num_ctx: 2048` in `src/llm/client.py`.
- Ensure no other heavy VRAM applications (games, browser hardware acceleration, local Stable Diffusion) are running concurrently.

### Q: How do I run the full test suite?
Run the pytest suite to verify all 74 unit, DSP, agent, and grounding tests:
```bash
python -m pytest tests/
```
All tests use analytical closed-form signals and complete in $\approx 35\text{ seconds}$ without external dependencies.

