# AuralQ (Ask Your Audio — Query Intelligently)

> **Natural-Language Audio Analysis via LLM-Orchestrated Deterministic DSP Tools**  
> *A tool-augmented audio intelligence architecture designed to run locally on consumer hardware without cloud API dependencies.*

---

## 1. System Philosophy & Core Principle

Audio signal processing provides measurable, interpretable evidence about the acoustic properties of a recording, but accessing this evidence typically requires deep domain knowledge to select analytical operations, execute them correctly, and interpret raw matrices or numerical spectra.

```
       User: "Is this audio recording noise-like or does it have tonal structure?"
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             AuralQ Orchestration Layer                │
       │   (Local LLM: llama3.2:3b-instruct-q4_K_M)             │
       │   Intent Parsing  ──►  Tool Dispatch: spectral_flatness │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │             Layer 3: Deterministic DSP                 │
       │         spectral_flatness(y, sr) = 0.0031              │
       │         Perceptual Anchor: "tonal / harmonic"          │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
       ┌────────────────────────────────────────────────────────┐
       │          Evidence Grounding & Verification             │
       │   Response: "The audio exhibits strong tonal structure │
       │   with a measured spectral flatness of 0.0031..."      │
       │   Citation Validator: [0.0031 verified in evidence]    │
       └────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Non-Negotiable Rule**: The LLM is strictly an orchestration, interpretation, and translation layer — **NEVER** an acoustic compute engine. Every numeric value in an AuralQ response originates from a deterministic DSP calculation (`librosa`/`scipy`/`numpy`), verified by a deterministic citation validator before reaching the user.

---

## 2. Hardware Constraints & Deployment Envelope

AuralQ is architected from first principles to execute entirely on budget consumer hardware:

- **Target Hardware**: Laptop NVIDIA GeForce RTX 3050 (4 GB VRAM), AMD Ryzen 5 5600H, 16 GB RAM.
- **Primary Model**: `llama3.2:3b-instruct-q4_K_M` served locally via [Ollama](https://ollama.ai).
- **VRAM Shield**: Context size strictly locked to `num_ctx: 2048` with `temperature: 0.0` to eliminate VRAM paging and guarantee 100% GPU-resident inference.
- **Latency Budget**: Single-shot tool calling replaces open-ended multi-turn ReAct loops for standard queries, targeting an end-to-end response time of **< 6.0 seconds**.
- **Privacy & Autonomy**: 100% offline execution. Zero audio or telemetry data leaves the host machine.

---

## 3. Five-Layer Architecture

Code is strictly partitioned across five isolated architectural layers:

```
src/
├── audio/          # LAYER 1: Untrusted file ingestion, format hygiene, silence/clipping validation
├── tools/          # LAYER 3: Pure deterministic DSP functions (librosa/scipy/numpy) & strict registry
├── llm/            # LAYER 2: Ollama HTTP client, Pydantic tool schemas, tool-selection prompts
├── agent/          # LAYER 4: State management, single-shot dispatch, evidence packaging, numeric validator
├── evaluation/     # EXPERIMENTS: Baselines B1-B4, metric computation, benchmark runners
└── utils/          # Configuration loaders, logging, shared helpers
app/                # LAYER 5: Streamlit cockpit UI and visualization dashboard
```

### Module Boundary Guarantees
1. `src/tools/` **must never import** `src/llm` or `src/agent/`. All DSP tools are pure mathematical functions testable in complete isolation.
2. `src/agent/validator.py` is purely deterministic (regex token matching and numeric tolerance bounds) — no secondary LLM judge calls allowed.
3. Every DSP tool accepts `(waveform: np.ndarray, sr: int, args: ToolArgsModel)` and returns a structured `ToolResultModel`.

---

## 4. Deterministic DSP Tool Catalogue

AuralQ provides 8 core deterministic DSP tools with strict Pydantic v2 schemas:

| Tool | Category | Analytical Focus | Example Query | Perceptual Anchors |
| :--- | :--- | :--- | :--- | :--- |
| `rms_energy` | Energy | Frame-by-frame RMS amplitude & dynamic range CV | *"Is this recording consistently loud?"* | `consistent`, `fluctuating` |
| `zero_crossing_rate` | Temporal | Rate of signal sign changes across frames | *"Is this audio smooth or percussive?"* | `smooth / tonal`, `noisy / transient` |
| `spectral_centroid` | Spectral | Weighted mean frequency (spectral center) | *"Does this sound bright or dull?"* | `dull` (<1200 Hz), `bright` (>3500 Hz) |
| `spectral_bandwidth` | Spectral | Frequency spread ($p$-norm) around centroid | *"Is the frequency content narrow or wide?"* | `narrow`, `wide` |
| `spectral_rolloff` | Spectral | Energy cutoff frequency (e.g., 85% energy) | *"Where is most energy concentrated?"* | `low-frequency`, `high-frequency` |
| `spectral_flatness` | Spectral | Ratio of geometric to arithmetic spectral mean | *"Is this noise-like or tonal?"* | `tonal` (≤0.05), `noise-like` (≥0.25) |
| `mfcc` | Timbre | 13 Mel-Frequency Cepstral Coefficients | *"What is the timbral color or texture?"* | `static`, `dynamic timbre` |
| `spectrogram` | Visual | STFT power sub-band ratios & headless plot | *"Show me frequency content over time"* | `bass`, `treble`, `mid-range balanced` |

---

## 5. Audio Hygiene & Defensive Validation (Layer 1)

All audio input is treated as untrusted data:

1. **Format Validation**: Strict allowlist (`.wav`, `.mp3`, `.flac`, `.ogg`).
2. **Duration Guardrails**: Enforces bounds $[0.5\text{s}, 300.0\text{s}]$. Files outside bounds trigger an explicit `ABSTAIN`.
3. **DC-Offset Blocking**: Applies mean-subtraction prior to any energy or silence evaluation.
4. **Silence Gating**: Measures $\text{RMS dBFS} = 20 \log_{10}(\text{RMS} + \epsilon)$. If $\text{dBFS} < -60.0$, the pipeline safely aborts with `status="ABSTAIN"` and code `"AUDIO_SILENT"`.
5. **Consecutive Clipping Detection**: Identifies flat-topped saturation runs ($\ge 3$ consecutive samples at $|y[n]| \ge 0.99$). When clipping exceeds $0.1\%$ of the signal, a structured `CLIPPING_WARNING` is injected into the evidence payload to alert the LLM that high-frequency harmonics may be contaminated by non-linear distortion.

---

## 6. Anti-Hallucination & Evidence Grounding

To guarantee scientific rigor for non-expert users:

- **Deterministic Perceptual Anchors**: Qualitative terms (`"bright"`, `"tonal"`, `"consistent"`) are computed deterministically by the tools—preventing the LLM from inventing subjective descriptors.
- **Dual Citation Aliasing**: Tools supply canonical string tokens (e.g. `["4210.3", "4210", "4210 Hz", "4.2 kHz"]`) alongside raw numeric floats. This prevents regex citation checkers from incorrectly rejecting valid engineering unit representations.
- **Explicit Abstention**: If audio fails quality checks or the tool outputs do not contain sufficient evidence to answer the user question, the system returns a formal `ABSTAIN` status.

---

## 7. Installation & Quickstart

### Prerequisites
- Python 3.10 or 3.11 (Python 3.11 recommended)
- [Ollama](https://ollama.ai) installed with `llama3.2:3b-instruct-q4_K_M` pulled

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/shagunthakur/AuralQ.git
cd AuralQ

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Analytical Verification Suite

All core DSP and audio ingestion tests use **analytically verifiable closed-form synthetic signals** (pure sine waves, Gaussian white noise, silence, clipped waveforms). No external media downloads are required:

```bash
python -m pytest -v
```

Expected output:
```text
tests/test_audio.py::test_dc_block_removes_bias PASSED
tests/test_audio.py::test_silence_detection PASSED
tests/test_audio.py::test_consecutive_clipping_detection PASSED
tests/test_audio.py::test_validate_duration PASSED
tests/test_audio.py::test_validate_format PASSED
tests/test_audio.py::test_load_audio_file_silence_abstention PASSED
tests/test_audio.py::test_load_audio_file_clipped_warning_injection PASSED
tests/test_tools.py::test_rms_energy_sine_analytical PASSED
tests/test_tools.py::test_zero_crossing_rate PASSED
tests/test_tools.py::test_spectral_centroid_analytical PASSED
tests/test_tools.py::test_spectral_flatness_analytical PASSED
tests/test_tools.py::test_spectral_bandwidth_and_rolloff PASSED
tests/test_tools.py::test_mfcc_extraction PASSED
tests/test_tools.py::test_spectrogram_headless PASSED
tests/test_tools.py::test_insufficient_samples_guard PASSED
tests/test_tools.py::test_tool_registry_and_dispatcher PASSED

======================= 16 passed in ~3s =======================
```

---

## 8. Evolutionary Roadmap

| Evolution | Name | Status | Key Deliverables |
| :--- | :--- | :---: | :--- |
| **E1** | **Audio Foundation** | ✅ **COMPLETE** | Ingestion pipeline, audio hygiene, 8 core DSP tools, synthetic test suite |
| **E2** | **NL Interface** | 🔄 *Next* | Ollama wrapper, Pydantic tool calling prompt, single-shot intent parser |
| **E3** | **Tool Calling & Execution** | 📅 *Planned* | Dispatch engine with structured Pydantic retry loop (max 2 retries) |
| **E4** | **Agent Loop & State** | 📅 *Planned* | Agent state manager, evidence aggregator, multi-tool coordination |
| **E5** | **Evidence Grounding** | 📅 *Planned* | Rule-based numeric citation validator, strict abstention protocols |
| **E6** | **Evaluation & Benchmarks**| 📅 *Planned* | Baselines B1–B4, ablation studies, accuracy & latency metrics |
| **E7** | **Interactive Application** | 📅 *Planned* | Streamlit cockpit interface with tool trace visualizer & audio playback |

---

## 9. Evaluation Framework & Baselines

AuralQ will be quantitatively evaluated across 4 experimental baselines on benchmark cases (`data/benchmark/cases.json`):

- **B1 (Manual Analysis)**: Human DSP expert reference calculation (upper bound).
- **B2 (LLM-Only)**: LLM receives audio metadata but no DSP tools (measures hallucination baseline).
- **B3 (Fixed Pipeline + LLM)**: All 8 core tools run unconditionally; LLM interprets results (measures utility of dynamic tool selection).
- **B4 (AuralQ Proposed)**: Dynamic single-shot tool selection, deterministic computation, and rule-based numeric citation grounding.

---

## 10. License & Academic Attribution

This project is developed as an undergraduate major project in Computer Science & Engineering.  
Licensed under the [MIT License](LICENSE).

