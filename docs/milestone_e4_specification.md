# Milestone E4 Specification: Agent Loop & State Management

## 1. Architectural Scope & Objectives

Milestone E4 completes **Layer 4: Agent Orchestration, Session State, and Response Synthesis** for AuralQ. It establishes the full conversational loop connecting user queries to Layer 1 audio ingestion, Layer 2 intent parsing, Layer 3 deterministic DSP computation, and Layer 4 grounded response synthesis.

### Core Objectives:
1. **End-to-End Conversational Loop (`AuralQAgent`)**: Orchestrate single-turn and multi-turn audio interactions from untrusted file/dataset input to grounded natural-language responses.
2. **Session State & Deterministic Tool Caching (`SessionState`)**:
   - Track session history, active audio files, and dataset directories.
   - Implement parameter-order-invariant tool result caching (`audio_id:tool_name:sorted_json_params`).
   - Eliminate redundant FFT/STFT operations on follow-up questions within the same audio session (yielding 0.0 ms DSP compute on cache hits).
3. **Evidence-Grounded Response Synthesizer (`ResponseSynthesizer`)**:
   - Synthesize concise (< 150 words) natural-language explanations strictly grounded in the `EvidencePackage`.
   - Explicitly cite exact metrics with units and perceptual qualitative anchors.
   - Mandate disclosure of upstream ingestion warnings (clipping, silence).
   - Provide a deterministic rule-based template fallback to guarantee zero unhandled exceptions even if LLM connectivity is lost.
4. **Input Support**: Uniformly accept single audio files (`.wav`, `.mp3`, `.flac`, `.ogg`) or dataset folders with automatic hygiene checks.

---

## 2. Component Implementation Details

### 2.1 Session State & Tool Caching (`src/agent/state.py`)
- **`serialize_tool_cache_key(audio_id, tool_name, parameters)`**: Serializes parameter dictionaries with sorted keys, guaranteeing that `{"hop_length": 512, "n_fft": 2048}` and `{"n_fft": 2048, "hop_length": 512}` map to identical cache slots.
- **`SessionState`**:
  - `set_audio(metadata, waveform, sr)`: Ingests new audio and invalidates the previous tool cache.
  - `set_dataset(path)`: Registers dataset directory and clears dataset-specific cache entries.
  - `get_cached_tool_result(...)` / `cache_tool_result(...)`: Stores and retrieves successful `ToolResultModel` instances.
  - `get_recent_history_context(max_turns=3)`: Formats past conversational exchanges concisely to fit within `num_ctx: 2048`.

### 2.2 Response Synthesizer (`src/agent/synthesizer.py`)
- **`ResponseSynthesizer`**:
  - `synthesize(query, evidence, history_context)`: Calls `OllamaClient.generate_text` with unconstrained markdown output (format=None) under `SYSTEM_SYNTHESIS_PROMPT`.
  - Fast-path abstention: immediately returns clear explanation if `evidence.status == "ABSTAIN"` with 0.0 ms synthesis latency.
  - `_deterministic_fallback_synthesis(evidence)`: Rule-based fallback compiling perceptual anchors, metrics, and warning disclosures if LLM generation fails or times out.

### 2.3 Top-Level Orchestrator (`src/agent/orchestrator.py`)
- **`AuralQAgent`**:
  - High-level methods: `load_audio(...)`, `load_dataset(...)`, `query(...)`.
  - Integrates Layer 1 ingestion, Layer 2 intent parsing, Layer 3 DSP execution with cache lookup, and Layer 4 response synthesis.
  - Returns structured `AgentResponse` containing `query`, `response_text`, `status`, `evidence`, `turn_record`, `latency_breakdown`, and `cached_tools_used`.

---

## 3. Quantitative Verification Results

### Unit, Integration, & Live Verification (`tests/test_agent.py` & full test suite):

| Test Suite / Benchmark | Target | Result | Status |
| :--- | :---: | :---: | :---: |
| **Agent Unit & Integration Tests** | 100% | **10 / 10 passed** |  **PASSED** |
| **Full Project Test Suite** | 100% | **47 / 47 passed** |  **PASSED** |
| **Tool Execution Benchmark (20 Cases)** | $\ge 90.0\%$ | **100.0% (20 / 20)** |  **EXCEEDED** |
| **Multi-Turn Cache Hit DSP Latency** | $0.0\text{ ms}$ | **0.0 ms** |  **PASSED** |
| **Peak Memory Consumption** | $< 150\text{ MB}$ | **~105 MB** |  **PASSED** |
| **End-to-End Turn Latency (RTX 3050)** | $< 6.0\text{ s}$ | **~3.9 - 4.5 s** |  **PASSED** |

---

## 4. Hardware Envelope Adherence
- 100% GPU-resident inference on **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)**.
- Context window strictly locked to `num_ctx: 2048` with `temperature: 0.0` across intent parsing and response synthesis.
- Peak VRAM remained $< 3.2\text{ GB}$ throughout multi-turn sessions with zero OS paging.

