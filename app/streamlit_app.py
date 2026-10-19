"""AuralQ Interactive Studio Cockpit (Milestone E7).

A professional, dual-engine audio intelligence cockpit combining local privacy-first
signal processing with a public cloud demo capability. Built with custom Dark Studio
glassmorphism styling, interactive spectrograms, streaming dataset profiling,
and verifiable deterministic citation traces.
"""

import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Ensure project root is in sys.path for cloud deployment
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st

from src.agent.db import SessionDatabase
from src.agent.orchestrator import AuralQAgent
from src.agent.schemas import AgentResponse, GroundingReport, TurnRecord
from src.agent.state import SessionState
from src.audio.loader import load_audio_file
from src.audio.validation import AudioMetadata
from src.llm.client import OllamaClient
from src.llm.cloud_client import CloudClient
from src.utils.generate_samples import ensure_showcase_samples
from src.utils.ollama_helper import check_ollama_status, get_installed_ollama_models, pull_ollama_model


# ---------------------------------------------------------------------------
# Page Configuration & Dark Studio CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AuralQ — Audio Intelligence Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_STUDIO_CSS = """
<style>
/* Main app backgrounds */
.stApp {
    background-color: #0B0F17;
    color: #E2E8F0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Glassmorphism sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Cards & containers */
.studio-card {
    background: #151D2C;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

.studio-card-header {
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 10px;
}

/* Badges */
.badge-verified {
    display: inline-flex;
    align-items: center;
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 6px;
}

.badge-repaired {
    display: inline-flex;
    align-items: center;
    background-color: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 6px;
}

.badge-abstain {
    display: inline-flex;
    align-items: center;
    background-color: rgba(148, 163, 184, 0.15);
    color: #94A3B8;
    border: 1px solid rgba(148, 163, 184, 0.35);
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 6px;
}

.badge-priority-critical {
    background-color: rgba(239, 68, 68, 0.2);
    color: #EF4444;
    border: 1px solid #EF4444;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
}

.badge-priority-recommended {
    background-color: rgba(245, 158, 11, 0.2);
    color: #F59E0B;
    border: 1px solid #F59E0B;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
}

.badge-priority-optional {
    background-color: rgba(6, 182, 212, 0.2);
    color: #06B6D4;
    border: 1px solid #06B6D4;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
}

/* Buttons */
.stButton>button {
    border-radius: 6px;
    font-weight: 600;
    transition: all 0.2s ease-in-out;
}

/* Chat bubble aesthetics */
div[data-testid="stChatMessage"] {
    background-color: #111827;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 12px 18px;
    margin-bottom: 12px;
}
</style>
"""

st.markdown(DARK_STUDIO_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# State Initialization
# ---------------------------------------------------------------------------

# Deployment environment detection: True when hosted on Streamlit Community Cloud
IS_CLOUD_DEPLOYMENT = Path("/mount/src").exists()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "engine_mode" not in st.session_state:
    if IS_CLOUD_DEPLOYMENT or not check_ollama_status():
        st.session_state.engine_mode = "☁️ Cloud Demo Mode"
    else:
        st.session_state.engine_mode = "🟢 Local Ollama (Private / Air-Gapped)"

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3.2:3b-instruct-q4_K_M"

if "cloud_provider" not in st.session_state:
    st.session_state.cloud_provider = "gemini"

if "cloud_api_key" not in st.session_state:
    st.session_state.cloud_api_key = ""

if "active_audio_path" not in st.session_state:
    st.session_state.active_audio_path = None

if "active_dataset_path" not in st.session_state:
    st.session_state.active_dataset_path = None

if "audio_metadata" not in st.session_state:
    st.session_state.audio_metadata = None

if "audio_waveform" not in st.session_state:
    st.session_state.audio_waveform = None

if "audio_sr" not in st.session_state:
    st.session_state.audio_sr = 22050

if "messages" not in st.session_state:
    st.session_state.messages = []

# Ensure built-in showcase samples exist
SAMPLE_DIR = ensure_showcase_samples()

# Is this a local private session?
is_local_engine = "Local Ollama" in st.session_state.engine_mode
db = SessionDatabase(enabled=is_local_engine)


# ---------------------------------------------------------------------------
# Agent Factory with Caching
# ---------------------------------------------------------------------------

def get_agent_instance() -> AuralQAgent:
    """Instantiate or return the active AuralQ agent configured for selected engine."""
    if "Local Ollama" in st.session_state.engine_mode:
        client = OllamaClient(model_name=st.session_state.selected_model)
    else:
        client = CloudClient(
            provider=st.session_state.cloud_provider,
            api_key=st.session_state.cloud_api_key,
        )

    # Attach existing session state
    session_state = SessionState(session_id=st.session_state.session_id)
    if st.session_state.audio_metadata and st.session_state.audio_waveform is not None:
        session_state.set_audio(
            metadata=st.session_state.audio_metadata,
            waveform=st.session_state.audio_waveform,
            sr=st.session_state.audio_sr,
        )
    elif st.session_state.active_dataset_path:
        session_state.set_dataset(st.session_state.active_dataset_path)

    return AuralQAgent(client=client, session_state=session_state)


# ---------------------------------------------------------------------------
# Helper: Plot Headless Spectrogram
# ---------------------------------------------------------------------------

def generate_spectrogram_figure(waveform: np.ndarray, sr: int) -> plt.Figure:
    """Generate a clean dark-themed STFT spectrogram visual plot."""
    import librosa
    import librosa.display

    fig, ax = plt.subplots(figsize=(7, 2.2), facecolor="#151D2C")
    ax.set_facecolor("#151D2C")

    # Compute short-time Fourier transform
    stft = librosa.stft(waveform, n_fft=1024, hop_length=256)
    db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    img = librosa.display.specshow(
        db,
        sr=sr,
        hop_length=256,
        x_axis="time",
        y_axis="hz",
        ax=ax,
        cmap="magma",
    )

    ax.set_xlabel("Time (s)", color="#94A3B8", fontsize=8)
    ax.set_ylabel("Frequency (Hz)", color="#94A3B8", fontsize=8)
    ax.tick_params(colors="#94A3B8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(0.8)

    fig.tight_layout()
    return fig


def generate_markdown_audit_report(
    session_id: str,
    messages: List[Dict[str, Any]],
    audio_path: Optional[str],
    dataset_path: Optional[str],
) -> str:
    """Generate a clean, structured Markdown report of the session with DSP evidence."""
    lines = [
        "# 🎙️ AuralQ Acoustic Session Audit Report",
        "",
        f"- **Session ID**: `{session_id}`",
        f"- **Generated At**: `{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}`",
        f"- **Active Audio File**: `{audio_path or 'None'}`",
        f"- **Active Dataset Directory**: `{dataset_path or 'None'}`",
        "",
        "---",
        "## Conversational Turns & Verifiable DSP Citations",
        "",
    ]

    turn_counter = 1
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        turn: Optional[TurnRecord] = msg.get("turn")

        if role == "user":
            lines.append(f"### Turn {turn_counter} (User Query)")
            lines.append(f"> {content}")
            lines.append("")
        elif role == "assistant":
            lines.append(f"### Turn {turn_counter} (AuralQ Grounded Synthesis)")
            lines.append(content)
            lines.append("")

            if turn and turn.grounding_report:
                rep: GroundingReport = turn.grounding_report
                score_pct = rep.grounding_score * 100
                if turn.evidence and turn.evidence.status == "ABSTAIN":
                    lines.append("- **Verification**: `🛑 Abstained: Non-Acoustic Scope`")
                elif rep.was_repaired:
                    lines.append(f"- **Verification**: `🛡️ Grounding Repaired ({len(rep.hallucinated_numbers)} ungrounded claim(s) corrected)`")
                else:
                    lines.append(f"- **Verification**: `✅ 100% Grounded ({len(rep.verified_citations)} verified citations, 0 hallucinations)`")

                if rep.verified_citations:
                    lines.append(f"- **Verified Numerical Claims**: `{', '.join(str(c) for c in rep.verified_citations)}`")

                # Dispatched tools
                if turn.parsed_intent and turn.parsed_intent.selected_tools:
                    tools_used = [t.tool_name for t in turn.parsed_intent.selected_tools]
                    lines.append(f"- **Dispatched DSP Tools**: `{', '.join(tools_used)}`")

                # Latency
                lat = turn.latency_breakdown
                lines.append(f"- **Latency Profile**: Total `{lat.get('total_ms', 0):.0f} ms` (DSP `{lat.get('dsp_ms', 0):.1f} ms`, LLM `{lat.get('synthesis_ms', 0):.1f} ms`)")

                # Mathematical Evidence JSON
                if turn.evidence and turn.evidence.consolidated_metrics:
                    lines.append("")
                    lines.append("```json")
                    lines.append(json.dumps(turn.evidence.consolidated_metrics, indent=2))
                    lines.append("```")
                lines.append("")
            turn_counter += 1
            lines.append("---")
            lines.append("")

    lines.append("### 🛡️ Layer 5 Deterministic Grounding Certification")
    lines.append("This report was generated deterministically by AuralQ. All acoustic assertions are certified against librosa/scipy/numpy mathematical calculations.")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# SIDEBAR: Sessions, Engine Config, Sample Showcase
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎙️ **AuralQ Studio**")
    st.caption("Deterministic Audio Intelligence on Consumer Hardware")

    # 1. Session Management
    col_sess1, col_sess2 = st.columns([1, 1]) if st.session_state.messages else (st.container(), None)

    if col_sess2:
        with col_sess1:
            if st.button("➕ New Session", use_container_width=True, type="primary"):
                st.session_state.session_id = str(uuid.uuid4())[:8]
                st.session_state.active_audio_path = None
                st.session_state.active_dataset_path = None
                st.session_state.audio_metadata = None
                st.session_state.audio_waveform = None
                st.session_state.messages = []
                st.rerun()
        with col_sess2:
            report_md = generate_markdown_audit_report(
                session_id=st.session_state.session_id,
                messages=st.session_state.messages,
                audio_path=st.session_state.active_audio_path,
                dataset_path=st.session_state.active_dataset_path,
            )
            st.download_button(
                label="📥 Export Report",
                data=report_md,
                file_name=f"AuralQ_Audit_{st.session_state.session_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )
    else:
        if st.button("➕ New Analysis Session", use_container_width=True, type="primary"):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.active_audio_path = None
            st.session_state.active_dataset_path = None
            st.session_state.audio_metadata = None
            st.session_state.audio_waveform = None
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # 2. Past Sessions Drawer (Strict Local Privacy Only)
    st.markdown("### 📁 Past Sessions")
    if is_local_engine:
        sessions = db.list_sessions(limit=10)
        if sessions:
            for s in sessions:
                col_title, col_del = st.columns([5, 1])
                is_active = s["session_id"] == st.session_state.session_id
                prefix = "▶ " if is_active else ""
                with col_title:
                    btn_label = f"{prefix}{s['title'][:22]} ({s['turn_count']})"
                    if st.button(btn_label, key=f"sess_{s['session_id']}", use_container_width=True):
                        # Load session turns
                        st.session_state.session_id = s["session_id"]
                        st.session_state.active_audio_path = s["audio_path"]
                        turns = db.get_session_turns(s["session_id"])
                        st.session_state.messages = []
                        for t in turns:
                            st.session_state.messages.append({"role": "user", "content": t.query})
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": t.response_text,
                                "turn": t,
                            })
                        # Reload audio if path exists
                        if s["audio_path"] and Path(s["audio_path"]).exists():
                            if s["is_dataset"]:
                                st.session_state.active_dataset_path = s["audio_path"]
                            else:
                                res = load_audio_file(s["audio_path"])
                                if res.status == "SUCCESS":
                                    st.session_state.audio_metadata = res.metadata
                                    st.session_state.audio_waveform = res.waveform
                                    st.session_state.audio_sr = res.sample_rate
                        st.rerun()
                with col_del:
                    if st.button("🗑️", key=f"del_{s['session_id']}"):
                        db.delete_session(s["session_id"])
                        if is_active:
                            st.session_state.session_id = str(uuid.uuid4())[:8]
                            st.session_state.messages = []
                        st.rerun()
        else:
            st.caption("No saved local sessions yet.")
    else:
        st.markdown(
            """
            <div style="background: rgba(6, 182, 212, 0.1); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 6px; padding: 10px; font-size: 0.8rem; color: #06B6D4;">
                🛡️ <b>Ephemeral Cloud Demo Mode</b><br>Zero data is saved to any database. Conversations exist only in active browser memory.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 3. Engine & Model Configuration
    if IS_CLOUD_DEPLOYMENT:
        st.markdown("### ☁️ Cloud Demo Engine")
        st.caption("Running publicly in ephemeral mode with cloud LLMs.")
        st.session_state.cloud_provider = st.selectbox("Provider:", ["gemini", "groq"])

        # Check if environment or Streamlit secrets has a default key
        env_var_name = "GEMINI_API_KEY" if st.session_state.cloud_provider == "gemini" else "GROQ_API_KEY"
        server_key = os.environ.get(env_var_name)
        try:
            if not server_key and hasattr(st, "secrets"):
                server_key = st.secrets.get(env_var_name)
        except Exception:
            pass

        if server_key:
            st.success(f"🔑 **Hosted `{env_var_name}` Connected**")
            custom_key = st.text_input("Custom Key Override (Optional / Session RAM):", type="password", placeholder="Leave blank to use default host key")
            st.session_state.cloud_api_key = custom_key if custom_key else server_key
        else:
            st.caption(f"🔑 **No `{env_var_name}` configured on host**")
            custom_key = st.text_input(
                f"Enter {st.session_state.cloud_provider.title()} API Key:",
                type="password",
                placeholder="AIzaSy... (Gemini) or gsk_... (Groq)",
            )
            if custom_key:
                st.session_state.cloud_api_key = custom_key
            else:
                st.session_state.cloud_api_key = ""
                st.info(
                    "💡 **API Key Required**\n"
                    "- Get a free key at [Google AI Studio](https://aistudio.google.com/) or [Groq Console](https://console.groq.com/).\n"
                    "- Or clone and run locally for 100% free, private execution with zero keys!"
                )

        # Direct link to run on-premise with local Ollama
        st.markdown("---")
        with st.expander("🔒 Run 100% Air-Gapped with Local Ollama"):
            st.markdown(
                """
                To analyze confidential audio on your machine with **Ollama** and zero cloud dependency:

                1. **Clone the repository**:
                   ```bash
                   git clone https://github.com/Shagun-Thakur/AuralQ.git
                   ```
                2. **One-Click Launch**:
                   - **Windows**: Double-click `run_auralq.bat`
                   - **Linux/macOS**: Run `./run_auralq.sh`

                Automatically connects to your local GPU & Ollama at `http://localhost:8501`.
                """
            )
            st.link_button("🐙 Clone on GitHub", "https://github.com/Shagun-Thakur/AuralQ", use_container_width=True)
    else:
        # Local desktop workstation: Full dual-mode engine selector
        st.markdown("### ⚙️ Engine Selector")
        engine_choice = st.radio(
            "Execution Mode:",
            ["🟢 Local Ollama (Private / Air-Gapped)", "☁️ Cloud Demo Mode"],
            index=0 if is_local_engine else 1,
        )
        if engine_choice != st.session_state.engine_mode:
            st.session_state.engine_mode = engine_choice
            st.rerun()

        if "Local Ollama" in st.session_state.engine_mode:
            ollama_alive = check_ollama_status()
            if ollama_alive:
                st.caption("🟢 **Ollama Daemon**: Connected (`127.0.0.1:11434`)")
                installed_models = get_installed_ollama_models()
                if installed_models:
                    st.session_state.selected_model = st.selectbox(
                        "Active Local Model:",
                        installed_models,
                        index=0,
                    )
                else:
                    st.warning("No local models found.")
                    if st.button("⬇️ Pull `llama3.2:3b` (~2.0 GB)"):
                        with st.spinner("Downloading model via Ollama..."):
                            if pull_ollama_model("llama3.2:3b-instruct-q4_K_M"):
                                st.success("Model downloaded successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to download model. Ensure Ollama is running.")
            else:
                st.error("⚠️ **Ollama is Offline**.\nPlease start Ollama locally (`ollama serve`) or switch to Cloud Demo Mode.")
        else:
            st.caption("☁️ **Cloud Demo**: Runs using high-speed cloud LLM API")
            st.session_state.cloud_provider = st.selectbox("Provider:", ["gemini", "groq"])

            # Check if environment or Streamlit secrets has a default key
            env_var_name = "GEMINI_API_KEY" if st.session_state.cloud_provider == "gemini" else "GROQ_API_KEY"
            server_key = os.environ.get(env_var_name)
            try:
                if not server_key and hasattr(st, "secrets"):
                    server_key = st.secrets.get(env_var_name)
            except Exception:
                pass

            if server_key:
                st.success(f"🔑 **Default `{env_var_name}` Detected**")
                custom_key = st.text_input("Custom Key Override (Optional / Session RAM):", type="password", placeholder="Leave blank to use default key")
                st.session_state.cloud_api_key = custom_key if custom_key else server_key
            else:
                st.caption(f"🔑 **No `{env_var_name}` configured on host**")
                custom_key = st.text_input(
                    f"Enter {st.session_state.cloud_provider.title()} API Key:",
                    type="password",
                    placeholder="AIzaSy... (Gemini) or gsk_... (Groq)",
                )
                if custom_key:
                    st.session_state.cloud_api_key = custom_key
                else:
                    st.session_state.cloud_api_key = ""
                    st.info(
                        "💡 **No API Key?**\n"
                        "- Get a free key at [Google AI Studio](https://aistudio.google.com/) or [Groq Console](https://console.groq.com/).\n"
                        "- Or switch back to **🟢 Local Ollama** above for 100% free, private local execution with zero keys!"
                    )

    st.markdown("---")


    # 4. Sample Showcase Gallery
    st.markdown("### 📦 Quick Demo Presets")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🎵 1kHz Sine", use_container_width=True):
            p = SAMPLE_DIR / "clean_sine_1khz.wav"
            res = load_audio_file(p)
            st.session_state.active_audio_path = str(p.resolve())
            st.session_state.active_dataset_path = None
            st.session_state.audio_metadata = res.metadata
            st.session_state.audio_waveform = res.waveform
            st.session_state.audio_sr = res.sample_rate
            st.rerun()

        if st.button("🥁 Clipped Snare", use_container_width=True):
            p = SAMPLE_DIR / "clipped_snare.wav"
            res = load_audio_file(p)
            st.session_state.active_audio_path = str(p.resolve())
            st.session_state.active_dataset_path = None
            st.session_state.audio_metadata = res.metadata
            st.session_state.audio_waveform = res.waveform
            st.session_state.audio_sr = res.sample_rate
            st.rerun()

    with col_s2:
        if st.button("🌊 Noisy Speech", use_container_width=True):
            p = SAMPLE_DIR / "noisy_audio.wav"
            res = load_audio_file(p)
            st.session_state.active_audio_path = str(p.resolve())
            st.session_state.active_dataset_path = None
            st.session_state.audio_metadata = res.metadata
            st.session_state.audio_waveform = res.waveform
            st.session_state.audio_sr = res.sample_rate
            st.rerun()

        if st.button("📁 Sample Dataset", use_container_width=True):
            p = SAMPLE_DIR / "sample_dataset"
            st.session_state.active_dataset_path = str(p.resolve())
            st.session_state.active_audio_path = None
            st.session_state.audio_metadata = None
            st.session_state.audio_waveform = None
            st.rerun()


# ---------------------------------------------------------------------------
# MAIN STAGE: Top Workspace (Audio / Dataset Viewer)
# ---------------------------------------------------------------------------

st.title("🎙️ AuralQ Studio")
st.caption("Deterministic Natural-Language Audio Intelligence with Verifiable Grounding")

# Context banner
if not is_local_engine:
    st.info("🛡️ **Cloud Demo Mode**: Your uploads and queries are ephemeral and will NOT be stored on any database. For 100% private, on-premise execution, clone the GitHub repository and run locally.")

# Ingestion Tabs
tab_single, tab_dataset = st.tabs(["🎵 Single Audio File", "📁 Audio Dataset Audit"])

with tab_single:
    uploaded_file = st.file_uploader(
        "Upload an audio clip (.wav, .mp3, .flac, .ogg):",
        type=["wav", "mp3", "flac", "ogg"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        # Save to temp location
        temp_upload_path = Path("data/samples/uploaded_active.wav")
        temp_upload_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        res = load_audio_file(temp_upload_path)
        if res.status == "SUCCESS":
            st.session_state.active_audio_path = str(temp_upload_path.resolve())
            st.session_state.active_dataset_path = None
            st.session_state.audio_metadata = res.metadata
            st.session_state.audio_waveform = res.waveform
            st.session_state.audio_sr = res.sample_rate
            st.toast(f"Loaded {uploaded_file.name} successfully!", icon="✅")
        else:
            st.error(f"Ingestion rejected ({res.status}): {res.message}")

    # Display Active Audio Stage
    if st.session_state.active_audio_path and st.session_state.audio_waveform is not None:
        col_player, col_spec, col_metrics = st.columns([2, 3, 2])

        with col_player:
            st.markdown("<div class='studio-card-header'>AUDIO PLAYBACK</div>", unsafe_allow_html=True)
            # Standardize buffer for browser playback
            buf = io.BytesIO()
            sf.write(buf, st.session_state.audio_waveform, st.session_state.audio_sr, format="WAV")
            st.audio(buf.getvalue(), format="audio/wav")
            st.caption(f"File: `{Path(st.session_state.active_audio_path).name}`")

        with col_spec:
            st.markdown("<div class='studio-card-header'>STFT SPECTROGRAM (dBFS)</div>", unsafe_allow_html=True)
            fig = generate_spectrogram_figure(st.session_state.audio_waveform, st.session_state.audio_sr)
            st.pyplot(fig)

        with col_metrics:
            st.markdown("<div class='studio-card-header'>ACOUSTIC PROPERTIES</div>", unsafe_allow_html=True)
            meta = st.session_state.audio_metadata
            if meta:
                st.markdown(f"**Sample Rate**: `{meta.sample_rate} Hz`")
                st.markdown(f"**Duration**: `{meta.duration_sec:.2f} s`")
                st.markdown(f"**Channels**: `{meta.num_channels} ({'Mono' if meta.num_channels==1 else 'Stereo'})`")
                st.markdown(f"**RMS Energy**: `{meta.rms_dbfs:.1f} dBFS`")

                if meta.is_clipped:
                    st.markdown("<span style='color:#EF4444; font-weight:700;'>⚠️ Clipping Detected</span>", unsafe_allow_html=True)

with tab_dataset:
    st.markdown("Audit a directory of audio files for model training hygiene, duration skews, and preprocessing needs.")

    # 1-Click quick load for demo dataset
    demo_samples_dir = Path("data/samples")
    if demo_samples_dir.exists() and demo_samples_dir.is_dir():
        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            st.caption("Quick testing with bundled sample files:")
        with col_d2:
            if st.button("📁 Load Built-in Demo Dataset", use_container_width=True):
                st.session_state.active_dataset_path = str(demo_samples_dir.resolve())
                st.session_state.active_audio_path = None
                st.session_state.audio_metadata = None
                st.session_state.audio_waveform = None
                st.rerun()

    dataset_input = st.text_input(
        "Dataset Directory Path:",
        value=st.session_state.active_dataset_path or "",
        placeholder="C:\\Path\\To\\Audio\\Dataset",
    )
    if st.button("Set Active Dataset"):
        p = Path(dataset_input)
        if p.exists() and p.is_dir():
            st.session_state.active_dataset_path = str(p.resolve())
            st.session_state.active_audio_path = None
            st.session_state.audio_metadata = None
            st.session_state.audio_waveform = None
            st.success(f"Active dataset registered: `{p.resolve()}`")
            st.rerun()
        else:
            st.error("Invalid dataset directory path. Please check the path and try again.")

    if st.session_state.active_dataset_path:
        st.markdown(f"📂 **Active Dataset**: `{st.session_state.active_dataset_path}`")
        if st.button("🔍 Run Quick Acoustic Hygiene Audit", type="primary"):
            agent = get_agent_instance()
            with st.spinner("Streaming through dataset files..."):
                audit_resp = agent.query(
                    "Please audit my audio dataset directory and recommend what preprocessing pipeline I should apply.",
                    dataset_path=st.session_state.active_dataset_path,
                )
                st.session_state.messages.append({"role": "user", "content": f"Audit dataset directory `{st.session_state.active_dataset_path}` and recommend preprocessing pipeline."})
                st.session_state.messages.append({"role": "assistant", "content": audit_resp.response_text, "turn": audit_resp.turn_record})
                if is_local_engine:
                    db.save_turn(st.session_state.session_id, audit_resp.turn_record)
                st.rerun()

st.markdown("---")



# ---------------------------------------------------------------------------
# MAIN STAGE: Conversational Chat Stream
# ---------------------------------------------------------------------------

st.markdown("### 💬 Acoustic Dialogue & Verification Stream")

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display verification badge and tool trace if available
        turn: Optional[TurnRecord] = msg.get("turn")
        if turn and turn.grounding_report:
            rep: GroundingReport = turn.grounding_report
            if turn.evidence and turn.evidence.status == "ABSTAIN":
                st.markdown("<span class='badge-abstain'>🛑 Abstained: Non-Acoustic Scope</span>", unsafe_allow_html=True)
            elif rep.was_repaired:
                st.markdown(f"<span class='badge-repaired'>🛡️ Grounding Repaired: {len(rep.hallucinated_numbers)} unverified claim(s) corrected</span>", unsafe_allow_html=True)
            elif rep.is_fully_grounded:
                st.markdown(f"<span class='badge-verified'>✅ Verified DSP Grounding: {rep.grounding_score*100:.0f}% ({len(rep.verified_citations)} verified numbers)</span>", unsafe_allow_html=True)

            # Expandable Tool Trace Drawer
            with st.expander(f"🔍 Inspect Tool Trace & Evidence ({turn.latency_breakdown.get('total_ms', 0):.0f} ms total)"):
                col_tr1, col_tr2 = st.columns([1, 1])
                with col_tr1:
                    st.markdown("**Dispatched Tools:**")
                    if turn.parsed_intent and turn.parsed_intent.selected_tools:
                        for tc in turn.parsed_intent.selected_tools:
                            st.markdown(f"- `{tc.tool_name}` (Params: `{tc.parameters}`)")
                    else:
                        st.markdown("*(Zero tools dispatched)*")

                    st.markdown("**Latency Breakdown:**")
                    lat = turn.latency_breakdown
                    st.markdown(f"- Intent Parsing: `{lat.get('intent_ms', 0):.1f} ms`")
                    st.markdown(f"- DSP Compute: `{lat.get('dsp_ms', 0):.1f} ms`")
                    st.markdown(f"- Grounded Synthesis: `{lat.get('synthesis_ms', 0):.1f} ms`")
                    st.markdown(f"- Citation Audit: `{lat.get('validation_ms', 0):.1f} ms`")

                with col_tr2:
                    st.markdown("**Calculated Evidence Metrics:**")
                    if turn.evidence and turn.evidence.consolidated_metrics:
                        st.json(turn.evidence.consolidated_metrics)
                    else:
                        st.caption("No numerical evidence extracted.")


# User Query Input
user_query = st.chat_input("Ask an acoustic question about this audio recording or dataset...")

if user_query:
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Execute Agent Loop with Step Indicators
    with st.chat_message("assistant"):
        with st.status("Analyzing audio...", expanded=True) as status_box:
            st.write("⏳ Step 1: Interpreting acoustic intent via LLM...")
            agent = get_agent_instance()

            st.write("⚡ Step 2: Executing deterministic DSP tools...")
            resp: AgentResponse = agent.query(
                question=user_query,
                audio_path=st.session_state.active_audio_path,
                dataset_path=st.session_state.active_dataset_path,
            )

            st.write("🛡️ Step 3: Verifying numerical citations against mathematical evidence...")
            status_box.update(label="Analysis complete & verified!", state="complete", expanded=False)

        st.markdown(resp.response_text)

        # Verification pill
        turn = resp.turn_record
        if turn and turn.grounding_report:
            rep = turn.grounding_report
            if turn.evidence and turn.evidence.status == "ABSTAIN":
                st.markdown("<span class='badge-abstain'>🛑 Abstained: Non-Acoustic Scope</span>", unsafe_allow_html=True)
            elif rep.was_repaired:
                st.markdown(f"<span class='badge-repaired'>🛡️ Grounding Repaired: {len(rep.hallucinated_numbers)} unverified claim(s) corrected</span>", unsafe_allow_html=True)
            elif rep.is_fully_grounded:
                st.markdown(f"<span class='badge-verified'>✅ Verified DSP Grounding: {rep.grounding_score*100:.0f}% ({len(rep.verified_citations)} verified numbers)</span>", unsafe_allow_html=True)

        # Save to state and SQLite (local only)
        st.session_state.messages.append({
            "role": "assistant",
            "content": resp.response_text,
            "turn": resp.turn_record,
        })
        if is_local_engine and resp.turn_record:
            db.save_turn(st.session_state.session_id, resp.turn_record)

        st.rerun()

