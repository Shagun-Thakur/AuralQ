# Milestone E7 Specification: Interactive Streamlit Studio Cockpit

## 1. Architectural Scope & Product Design

Milestone E7 delivers the **Interactive Audio Intelligence Cockpit (`app/streamlit_app.py`)** for AuralQ. It translates the 5-layer deterministic audio processing and citation-verified agent architecture into an industrial-grade, dark-themed visual workspace designed for researchers, audio engineers, and public evaluators.

### Core Architectural Pillars:

1. **Dual-Engine Execution**:
   - **Local Studio Mode (Default / 100% Air-Gapped)**: Connects to local Ollama on consumer hardware (RTX 3050). Auto-discovers installed models via `GET /api/tags` and offers dynamic model switching.
   - **Cloud Demo Mode**: Connects to high-speed cloud LLMs (Google Gemini 2.5 Flash / Groq) for public web hosting without GPU requirements.
2. **Strict Privacy Boundary**:
   - **Local Workspace**: Conversational turns, evidence packages, and audio metadata are stored locally in a zero-dependency SQLite database (`data/auralq_sessions.db`) on the user's hard drive.
   - **Cloud Demo Mode**: Complete **Zero-Retention Ephemeral Mode**. No cloud database is used. Files and chats exist strictly in temporary browser RAM and vanish upon session termination.
3. **Dual-Mode Audio Ingestion**:
   - **Single Audio File**: Full audio hygiene pipeline, browser audio playback (`st.audio`), acoustic specification table, and dark-themed STFT power spectrogram (dBFS).
   - **Multi-Audio Dataset**: Directory path ingestion and streaming hygiene profiling (Welford variance, < 100 MB RAM budget) rendering prioritized preprocessing recommendation cards (`RESAMPLE`, `MONO_DOWNMIX`, `PRUNE_SILENCE`, `LOUDNESS_NORMALIZATION`, `PRUNE_OR_DECLIP`, `HIGH_PASS_FILTER`).
4. **Verifiable Grounding & Tool Trace Transparency**:
   - Every assistant response carries visual verification badges:
     - 🟢 `Verified DSP Grounding: 100% (N verified numbers)`
     - 🛡️ `Grounding Repaired: N unverified claim(s) corrected`
     - 🛑 `Abstained: Non-Acoustic Scope`
   - Expandable **Inspect Tool Trace Drawer** disclosing dispatched tools, parameter dictionaries, latency breakdowns, and raw mathematical evidence JSON.
5. **One-Click Desktop Launchers**:
   - `run_auralq.bat` (Windows) and `run_auralq.sh` (Mac/Linux) for zero-command launching.

---

## 2. Component Implementation Summary

| Component | File Path | Role & Technology |
| :--- | :--- | :--- |
| **Streamlit Cockpit UI** | [`app/streamlit_app.py`](file:///c:/MyProjects/AuralQ/AuralQ/app/streamlit_app.py) | Dark Studio CSS, session state coordination, audio player, chat stream, tool traces |
| **SQLite Session Store**| [`src/agent/db.py`](file:///c:/MyProjects/AuralQ/AuralQ/src/agent/db.py) | Zero-dependency persistence for local private workspace; bypasses in Cloud Demo |
| **Cloud Fallback Client**| [`src/llm/cloud_client.py`](file:///c:/MyProjects/AuralQ/AuralQ/src/llm/cloud_client.py) | Gemini 2.5 Flash / Groq REST adapter for GPU-free public cloud demo hosting |
| **Ollama Discovery Helper**| [`src/utils/ollama_helper.py`](file:///c:/MyProjects/AuralQ/AuralQ/src/utils/ollama_helper.py) | Auto-discovers local models via `/api/tags` and handles model pull bootstraps |
| **Demo Sample Showcase**| [`src/utils/generate_samples.py`](file:///c:/MyProjects/AuralQ/AuralQ/src/utils/generate_samples.py) | Generates synthetic audio presets (`clean_sine`, `noisy_speech`, `clipped_snare`, `sample_dataset`) |
| **Windows Desktop Launcher**| [`run_auralq.bat`](file:///c:/MyProjects/AuralQ/AuralQ/run_auralq.bat) | Pre-flight Ollama checks, virtualenv activation, browser launch |
| **Unix Desktop Launcher** | [`run_auralq.sh`](file:///c:/MyProjects/AuralQ/AuralQ/run_auralq.sh) | Unix shell script for single-click execution |

---

## 3. System Verification Suite (67 / 67 Passed)

The entire analytical, execution, persistence, and evaluation test suite passed with 100% compliance:

```text
======================= 67 passed, 4 warnings in 39.63s =======================
```

- `tests/test_audio.py`: Ingestion hygiene, silence, DC bias, clipping (7/7 passed)
- `tests/test_tools.py`: Core 8 DSP tools analytical accuracy (9/9 passed)
- `tests/test_dataset_profiler.py`: Dataset profiler & preprocessing advisor (5/5 passed)
- `tests/test_db.py`: SQLite session persistence, turn deserialization, ephemeral mode (4/4 passed)
- `tests/test_llm.py`: Schema validation, self-healing retry loop (10/10 passed)
- `tests/test_executor.py`: Tool calling, execution engine, warning propagation (6/6 passed)
- `tests/test_agent.py`: Agent orchestrator, session caching, state transitions (10/10 passed)
- `tests/test_validator.py`: Regex extraction, tolerance matching, fallback repair (9/9 passed)
- `tests/test_evaluation.py`: Baselines B1–B4 execution and metric aggregators (7/7 passed)

---

## 4. Milestone Conclusion

With Milestone E7 completed, AuralQ stands as a **complete, end-to-end, production-ready audio intelligence system**:
1. Grounded in deterministic signal processing (librosa/scipy/numpy).
2. Guided by a local or cloud LLM orchestration layer.
3. Protected by a deterministic Layer 5 citation validator that mathematically prevents hallucination.
4. Delivered through a sleek, dark-themed interactive studio cockpit that guarantees 100% privacy on local workstations and universal zero-retention accessibility on the web.

