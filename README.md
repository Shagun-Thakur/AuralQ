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
       │             AuralQ Orchestration Layer                 │
       │   (Local LLM: llama3.2:3b-instruct-q4_K_M)             │
       │   Intent Parsing  ──►  Tool Dispatch: spectral_flatness│
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
| `dataset_profiler` | Dataset | Stream-profiler & deterministic preprocessing advisor | *"Profile my dataset and tell me what preprocessing to use"* | `action_required`, `sound` |

---

## 5. Dual-Mode Audio Hygiene & Dataset Profiling (Layer 1 & Layer 3)

AuralQ provides unified ingestion supporting both **Single Audio Files** and **Full Audio Datasets**:

### 5.1 Single-File Audio Hygiene
1. **Format Validation**: Strict allowlist (`.wav`, `.mp3`, `.flac`, `.ogg`).
2. **Duration Guardrails**: Enforces bounds $[0.5\text{s}, 300.0\text{s}]$. Files outside bounds trigger an explicit `ABSTAIN`.
3. **DC-Offset Blocking**: Applies mean-subtraction prior to any energy or silence evaluation.
4. **Silence Gating**: Measures $\text{RMS dBFS} = 20 \log_{10}(\text{RMS} + \epsilon)$. If $\text{dBFS} < -60.0$, the pipeline safely aborts with `status="ABSTAIN"` and code `"AUDIO_SILENT"`.
5. **Consecutive Clipping Detection**: Identifies flat-topped saturation runs ($\ge 3$ consecutive samples at $|y[n]| \ge 0.99$). When clipping exceeds $0.1\%$ of the signal, a structured `CLIPPING_WARNING` is injected into the evidence payload.

### 5.2 Streaming Dataset Profiler & Preprocessing Advisor (`dataset_profiler`)
When given a dataset directory (up to 2,000 files / 5 GB on disk), AuralQ runs an out-of-core streaming acoustic audit:
- **Sub-100 MB RAM Budget**: Files are processed sequentially using online Welford accumulation ($O(1)$ space for mean and sample variance). No full-corpus waveform loading occurs.
- **Two-Tier Audit**:
  - *Tier 1 (Fast Header Scan)*: Gathers sample rate distributions, channel layouts, and duration percentiles at ~1000 files/sec.
  - *Tier 2 (Acoustic Hygiene Pass)*: Measures RMS dynamic range spread, clipping ratios, silence count, spectral centroid, and spectral flatness.
- **Deterministic Preprocessing Recommendations**: Evaluates mathematical rule thresholds to recommend grounded actions:
  - `RESAMPLE`: Triggered when sample rates diverge across files.
  - `MONO_DOWNMIX`: Triggered when stereo/multi-channel files are present.
  - `CHUNKING_OR_PADDING`: Triggered when duration disparity $(\max - \min)/\text{median} > 0.5$.
  - `LOUDNESS_NORMALIZATION`: Triggered when loudness standard deviation $> 6.0\text{ dB}$.
  - `HIGH_PASS_FILTER`: Triggered when low-frequency rumble dominates (< 800 Hz centroid).
  - `PRUNE_SILENCE` & `PRUNE_OR_DECLIP`: Flags unreadable, silent, or severely clipped recordings with exact paths.

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
tests/test_agent.py::test_session_state_tool_caching_invariant_keys PASSED
tests/test_agent.py::test_session_state_cache_invalidation_on_audio_change PASSED
tests/test_agent.py::test_session_state_history_context PASSED
tests/test_agent.py::test_synthesizer_abstention_fast_path PASSED
tests/test_agent.py::test_synthesizer_deterministic_fallback PASSED
tests/test_agent.py::test_agent_query_with_mocked_llm_and_cache_hit PASSED
tests/test_agent.py::test_agent_abstention_flow PASSED
tests/test_agent.py::test_agent_dataset_profiler_flow PASSED
tests/test_agent.py::test_agent_ingestion_warning_propagation PASSED
tests/test_agent.py::test_live_ollama_agent_multiturn PASSED
tests/test_audio.py::test_dc_block_removes_bias PASSED
tests/test_audio.py::test_silence_detection PASSED
tests/test_audio.py::test_consecutive_clipping_detection PASSED
tests/test_audio.py::test_validate_duration PASSED
tests/test_audio.py::test_validate_format PASSED
tests/test_audio.py::test_load_audio_file_silence_abstention PASSED
tests/test_audio.py::test_load_audio_file_clipped_warning_injection PASSED
tests/test_dataset_profiler.py::test_dual_mode_loader_distinction PASSED
tests/test_dataset_profiler.py::test_streaming_dataset_profiler_metrics PASSED
tests/test_dataset_profiler.py::test_deterministic_preprocessing_advisor PASSED
tests/test_dataset_profiler.py::test_registry_dataset_profiler_dispatch PASSED
tests/test_dataset_profiler.py::test_empty_dataset_handling PASSED
tests/test_executor.py::test_executor_single_tool PASSED
tests/test_executor.py::test_executor_multi_tool PASSED
tests/test_executor.py::test_executor_dataset_profiler_routing PASSED
tests/test_executor.py::test_executor_abstention_passthrough PASSED
tests/test_executor.py::test_executor_warning_propagation PASSED
tests/test_executor.py::test_live_e2e_pipeline PASSED
tests/test_llm.py::test_clean_json_text PASSED
tests/test_llm.py::test_intent_schema_valid PASSED
tests/test_llm.py::test_intent_schema_invalid_tool_name PASSED
tests/test_llm.py::test_intent_schema_max_tool_budget PASSED
tests/test_llm.py::test_prompt_construction_contains_all_tools PASSED
tests/test_llm.py::test_client_self_healing_retry PASSED
tests/test_llm.py::test_client_connection_error_graceful_handling PASSED
tests/test_llm.py::test_live_ollama_single_tool_dispatch PASSED
tests/test_llm.py::test_live_ollama_abstention PASSED
tests/test_llm.py::test_live_ollama_dataset_intent PASSED
tests/test_tools.py::test_rms_energy_sine_analytical PASSED
tests/test_tools.py::test_zero_crossing_rate PASSED
tests/test_tools.py::test_spectral_centroid_analytical PASSED
tests/test_tools.py::test_spectral_flatness_analytical PASSED
tests/test_tools.py::test_spectral_bandwidth_and_rolloff PASSED
tests/test_tools.py::test_mfcc_extraction PASSED
tests/test_tools.py::test_spectrogram_headless PASSED
tests/test_tools.py::test_insufficient_samples_guard PASSED
tests/test_validator.py::test_parse_numeric_candidate PASSED
tests/test_validator.py::test_extract_numeric_tokens PASSED
tests/test_validator.py::test_validator_exact_alias_and_tolerance_match PASSED
tests/test_validator.py::test_validator_detects_hallucinated_numbers PASSED
tests/test_validator.py::test_validator_query_echo_grounding PASSED
tests/test_validator.py::test_validator_abstention_and_qualitative_passes PASSED
tests/test_validator.py::test_validator_deterministic_repair PASSED
tests/test_validator.py::test_agent_with_hallucinated_response_triggers_repair PASSED
tests/test_validator.py::test_agent_with_grounded_response_passes_unmodified PASSED
tests/test_db.py::test_session_db_initialization PASSED
tests/test_db.py::test_save_and_retrieve_turn PASSED
tests/test_db.py::test_session_deletion_cascade PASSED
tests/test_db.py::test_ephemeral_cloud_mode PASSED
tests/test_evaluation.py::test_compute_baseline_summary_statistics PASSED
tests/test_evaluation.py::test_benchmark_comparison_report_markdown_generation PASSED
tests/test_evaluation.py::test_run_baseline_b1_oracle PASSED
tests/test_evaluation.py::test_run_baseline_b1_abstention PASSED
tests/test_evaluation.py::test_run_baseline_b2_llm_only_catches_hallucination PASSED
tests/test_evaluation.py::test_run_baseline_b3_fixed_pipeline_runs_all_tools PASSED
tests/test_evaluation.py::test_run_baseline_b4_auralq_dynamic PASSED

======================= 67 passed in ~39s =======================
```

### Launching the Interactive Studio Cockpit

AuralQ provides a zero-setup desktop launcher and standard web startup:

```bash
# Windows (One-Click Launcher)
run_auralq.bat

# Linux / macOS
chmod +x run_auralq.sh
./run_auralq.sh

# Or direct Streamlit command:
streamlit run app/streamlit_app.py
```

---

## 8. Evolutionary Roadmap

| Evolution | Name | Status | Key Deliverables |
| :--- | :--- | :---: | :--- |
| **E1** | **Audio Foundation** | ✅ **COMPLETE** | Ingestion pipeline, audio hygiene, 8 core DSP tools, synthetic test suite |
| **E2** | **NL Interface** | ✅ **COMPLETE** | Ollama client, Pydantic single-shot intent parser, 20-case benchmark |
| **E3** | **Tool Calling & Execution** | ✅ **COMPLETE** | Tool executor, argument correction retry loop, evidence packaging, 100% benchmark |
| **E4** | **Agent Loop & State** | ✅ **COMPLETE** | Agent orchestrator (`AuralQAgent`), session state, tool caching, response synthesizer |
| **E5** | **Evidence Grounding** | ✅ **COMPLETE** | Rule-based numeric citation validator, adaptive tolerance matching, hallucination repair |
| **E6** | **Evaluation & Benchmarks**| ✅ **COMPLETE** | Baselines B1–B4 benchmark, ablation metrics, 99.0% citation grounding, 80.8% tool bloat reduction |
| **E7** | **Interactive Application** | ✅ **COMPLETE** | Streamlit cockpit, Dark Studio UI, dual-engine mode, SQLite local persistence, tool traces |

---

## 9. Evaluation Framework & Baselines (B1–B4 Empirical Results)

AuralQ is quantitatively evaluated across 4 experimental baselines on 20 canonical benchmark cases (`data/benchmark/cases.json`) on consumer hardware (AMD Ryzen 5 5600H, RTX 3050 4 GB, Ollama `llama3.2:3b-instruct-q4_K_M`):

- **B1 (Oracle / DSP Ground Truth)**: Direct analytical execution of ground-truth DSP tools with zero LLM variance (theoretical upper bound).
- **B2 (LLM-Only)**: Audio metadata only, zero DSP tools (evaluates raw LLM hallucination rate on acoustic physics).
- **B3 (Fixed Pipeline + LLM)**: Unconditional brute-force execution of all 8 core tools per query (measures computational bloat and token overhead).
- **B4 (AuralQ Proposed Full Pipeline)**: Dynamic single-shot intent parsing ($\le 3$ tools), Layer 3 cached DSP computation, Layer 4 grounded synthesis, and Layer 5 deterministic citation validation (`CitationValidator`) with fallback repair.

### 9.1 Benchmark Summary Comparison (B1–B4)

| Evaluation Metric | B1: Oracle / DSP Ground Truth | B2: LLM-Only (No DSP) | B3: Fixed Pipeline (All Tools) | B4: AuralQ Proposed |
| :--- | :---: | :---: | :---: | :---: |
| **Citation Grounding Rate** | 87.8% | 90.0% | 94.4% | **99.0%** |
| **Numeric Hallucination Rate** | 20.0% | 10.0% | 10.0% | **5.0%** |
| **Abstention Accuracy** | 100.0% | 90.0% | 90.0% | **100.0%** |
| **Avg Tools Executed / Query** | 1.15 | 0.00 | 7.30 | **1.40** (📉 80.8% reduction vs. B3) |
| **Avg DSP Compute Time** | 10.63 ms | 0.00 ms | 25.75 ms | **27.02 ms** |
| **Avg End-to-End Latency** | 13.61 ms | 3005.06 ms | 3469.77 ms | **6702.00 ms** |
| **Deterministic Fallback Rate** | 0.0% | 0.0% | 0.0% | **5.0%** (1/20 cases caught & repaired) |

---

### 9.2 Granular 20-Case Benchmark Execution Matrix

Complete results across all 20 canonical benchmark cases evaluated on consumer hardware:

| Case ID | Query / Analytical Focus | Expected Tools (B1) | B4 Tools Dispatched | Grounding | Latency | Verification Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **BC_001** | Is this audio recording noise-like or does it have tonal structure? | `spectral_flatness` | `spectral_flatness` | 100% | 6302 ms | ✅ Verified Grounded |
| **BC_002** | Does this recording sound bright or dull? | `spectral_centroid` | `spectral_centroid` | 100% | 5214 ms | ✅ Verified Grounded |
| **BC_003** | Is this audio consistently loud or fluctuating in amplitude? | `rms_energy` | `rms_energy` | 100% | 5409 ms | ✅ Verified Grounded |
| **BC_004** | Is the frequency content of this audio spread out wide or narrow across the spectrum? | `spectral_bandwidth` | `spectral_bandwidth` | 100% | 5301 ms | ✅ Verified Grounded |
| **BC_005** | Where is the high frequency energy concentrated and what is the rolloff cutoff? | `spectral_rolloff` | `spectral_rolloff, spectral_bandwidth` | 100% | 6819 ms | ✅ Verified Grounded |
| **BC_006** | Is this audio smooth and continuous, or does it have rapid transient sign changes and percussiveness? | `zero_crossing_rate` | `zero_crossing_rate, spectral_centroid` | 100% | 8554 ms | ✅ Verified Grounded |
| **BC_007** | What is the timbral texture and spectral envelope of this instrument sound? | `mfcc` | `mfcc` | 100% | 6775 ms | ✅ Verified Grounded |
| **BC_008** | Can you generate a spectrogram visual plot showing time-frequency energy distribution? | `spectrogram` | `spectrogram` | 100% | 7002 ms | ✅ Verified Grounded |
| **BC_009** | Does this sound bright and is it consistently loud throughout? | `spectral_centroid, rms_energy` | `spectral_centroid, rms_energy` | 100% | 7224 ms | ✅ Verified Grounded |
| **BC_010** | Tell me if this audio has strong tonal harmonics and whether it contains percussive transients. | `spectral_flatness, zero_crossing_rate` | `spectral_flatness, zero_crossing_rate` | 100% | 8445 ms | ✅ Verified Grounded |
| **BC_011** | What is the spectral center frequency and what is the bandwidth spread around it? | `spectral_centroid, spectral_bandwidth` | `spectral_centroid, spectral_bandwidth` | 100% | 7150 ms | ✅ Verified Grounded |
| **BC_012** | Please audit my audio dataset directory and recommend what preprocessing pipeline I should apply. | `dataset_profiler` | `dataset_profiler` | 80% | 7299 ms | 🛡️ Repaired (Fallback) |
| **BC_013** | Check my dataset folder for any silent or clipped audio files that should be cleaned. | `dataset_profiler` | `dataset_profiler` | 100% | 6608 ms | ✅ Verified Grounded |
| **BC_014** | What is the cutoff frequency below which 85% of total audio energy is concentrated? | `spectral_rolloff` | `spectral_rolloff` | 100% | 5999 ms | ✅ Verified Grounded |
| **BC_015** | Measure the volume level, dynamic range, and overall amplitude consistency of this clip. | `rms_energy` | `rms_energy, spectral_centroid, zero_crossing_rate` | 100% | 9213 ms | ✅ Verified Grounded |
| **BC_016** | Does this recording sound like white noise hiss or pure musical tones? | `spectral_flatness` | `spectral_flatness, zero_crossing_rate` | 100% | 8053 ms | ✅ Verified Grounded |
| **BC_017** | Extract the mel cepstral coefficients to analyze how timbre evolves over time. | `mfcc` | `mfcc` | 100% | 7973 ms | ✅ Verified Grounded |
| **BC_018** | Tell me if this sound is dull, quiet in amplitude, and smooth without transients. | `spectral_centroid, rms_energy, zero_crossing_rate` | `spectral_centroid, rms_energy, zero_crossing_rate` | 100% | 9772 ms | ✅ Verified Grounded |
| **BC_019** | What is the capital city of France? | `*(None)*` | `*(None)*` | 100% | 2289 ms | 🛑 Abstained (Zero Tools) |
| **BC_020** | Can you diagnose what respiratory illness this coughing audio indicates? | `*(None)*` | `*(None)*` | 100% | 2640 ms | 🛑 Abstained (Zero Tools) |

---

### 9.3 System Verification Suite (Milestones E1–E6)

The entire analytical and integration test suite across all six project milestones passes with 100% compliance:

| Milestone | Subsystem / Test Module | Test File | Tests Passed | Focus & Scope |
| :--- | :--- | :--- | :---: | :--- |
| **E1** | Ingestion & Audio Hygiene | `tests/test_audio.py` | **7 / 7** | DC-blocking, silence gating, consecutive clipping detection, format guardrails |
| **E1** | Core DSP Tool Analytical Suite | `tests/test_tools.py` | **9 / 9** | Closed-form synthetic sine/noise checks for RMS, ZCR, Centroid, Flatness, MFCC, etc. |
| **E1-Ext**| Streaming Dataset Profiler | `tests/test_dataset_profiler.py`| **5 / 5** | Streaming memory bounds, Welford variance, dual-mode loader, advisory rules |
| **E2** | NL Interface & Intent Parser | `tests/test_llm.py` | **10 / 10** | Pydantic schema validation, tool budget, self-healing retry loop, abstention |
| **E3** | Tool Calling & Execution Engine | `tests/test_executor.py` | **6 / 6** | Dynamic tool invocation, argument correction retries, evidence packaging |
| **E4** | Agent Loop & State Management | `tests/test_agent.py` | **10 / 10** | Invariant tool caching, multi-turn history, session isolation, synthesizer fast-paths |
| **E5** | Evidence Grounding & Verification | `tests/test_validator.py` | **9 / 9** | Regex citation extraction, adaptive tolerances, unit scaling, fallback repair |
| **E6** | Evaluation Framework & Baselines | `tests/test_evaluation.py` | **7 / 7** | Metric aggregation, markdown report formatting, B1–B4 execution logic |
| **E7** | Session Persistence & SQLite Engine | `tests/test_db.py` | **4 / 4** | SQLite turn persistence, session cascading deletion, ephemeral cloud mode |
| **Total** | **Full System Regression Suite** | **9 Test Modules** | **67 / 67** | **100% Passed in ~39.6s** |

---

### 9.4 Streaming Dataset Preprocessing Decision Matrix

The deterministic preprocessing advisor (`dataset_profiler`) evaluates statistical distributions over multi-audio datasets and generates grounded recommendations using exact mathematical rules:

| Action | Priority | Trigger Condition | Recommended Parameters & Remediation |
| :--- | :---: | :--- | :--- |
| **`RESAMPLE`** | **CRITICAL** | File sample rates diverge ($\ge 2$ distinct rates) | Target most common or standard sample rate (e.g. 22050 or 44100 Hz via `soxr`) |
| **`PRUNE_SILENCE`** | **CRITICAL** | Audio files with $\text{RMS dBFS} < -60\text{ dBFS}$ | Exact file paths flagged for deletion to prevent vanishing gradients |
| **`MONO_DOWNMIX`** | **RECOMMENDED**| Stereo or multi-channel files present | Downmix using equal-power formula $0.707 \times (L + R)$ |
| **`CHUNKING_OR_PADDING`** | **RECOMMENDED**| $(\text{Max Duration} - \text{Min Duration}) / \text{Median} > 0.5$ | Segment long clips or apply constant zero-padding to median duration |
| **`LOUDNESS_NORMALIZATION`** | **RECOMMENDED**| Loudness standard deviation $> 6.0\text{ dB}$ across files | Standardize to target $-20.0\text{ dBFS}$ via EBU R128 loudness normalization |
| **`PRUNE_OR_DECLIP`** | **RECOMMENDED**| Saturated runs $\ge 3$ consecutive samples exceed $0.1\%$ | Cubic-spline declipping or exclusion of corrupted recordings |
| **`HIGH_PASS_FILTER`** | **OPTIONAL** | Dataset mean spectral centroid $< 800\text{ Hz}$ | 2nd-order Butterworth high-pass filter ($60\text{ Hz}$ cutoff) to remove rumble |

---

## 10. License & Academic Attribution

This project is developed as an undergraduate major project in Computer Science & Engineering.  
Licensed under the [MIT License](LICENSE).

