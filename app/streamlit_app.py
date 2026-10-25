"""AuralQ Interactive Studio Cockpit (Milestone E7).

A professional, dual-engine audio intelligence cockpit combining local privacy-first
signal processing with a public cloud demo capability. Built with custom Dark Studio
glassmorphism styling, interactive spectrograms, streaming dataset profiling,
and verifiable deterministic citation traces.
"""

import html
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

from app.icons import get_icon
from app.minecraft_theme import (
    apply_minecraft_theme,
    compute_initial_acoustic_profile,
    generate_interactive_spectrogram,
    generate_interactive_waveform,
    render_grounded_answer_html,
    tool_tiles_html,
    ENDER_COLORS,
)
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
# Page Configuration & Minecraft Ender-Acoustic Theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AuralQ — Audio Intelligence Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_minecraft_theme()


def format_relative_time(timestamp: Optional[float]) -> str:
    """Format Unix timestamp as short relative time string (e.g. '2m ago', '1h ago', 'just now')."""
    if not timestamp:
        return ""
    diff_s = max(0, int(time.time() - timestamp))
    if diff_s < 60:
        return "just now"
    elif diff_s < 3600:
        return f"{diff_s // 60}m ago"
    elif diff_s < 86400:
        return f"{diff_s // 3600}h ago"
    else:
        return f"{diff_s // 86400}d ago"


# ---------------------------------------------------------------------------
# State Initialization
# ---------------------------------------------------------------------------

# Deployment environment detection: True when hosted on Streamlit Community Cloud (or forced via AURALQ_FORCE_CLOUD)
IS_CLOUD_DEPLOYMENT = Path("/mount/src").exists() or os.environ.get("AURALQ_FORCE_CLOUD", "").lower() in ("1", "true")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "engine_mode" not in st.session_state:
    if IS_CLOUD_DEPLOYMENT or not check_ollama_status():
        st.session_state.engine_mode = "Cloud Demo Mode"
    else:
        st.session_state.engine_mode = "Local Ollama (Private / Air-Gapped)"

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
# Helper: Session Audit Report Generator
# ---------------------------------------------------------------------------


def generate_markdown_audit_report(
    session_id: str,
    messages: List[Dict[str, Any]],
    audio_path: Optional[str],
    dataset_path: Optional[str],
) -> str:
    """Generate a clean, structured Markdown report of the session with DSP evidence."""
    lines = [
        "# AuralQ Acoustic Session Audit Report",
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
                    tokens_list = [getattr(c, "raw_token", str(c)) for c in rep.verified_citations]
                    lines.append(f"- **Verified Numerical Claims**: `{', '.join(tokens_list)}`")

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
    st.markdown(
        """
        <div class='sidebar-brand-box'>
            <div class='sidebar-title'>AuralQ Studio</div>
            <div class='sidebar-subtitle'>Deterministic Audio Intelligence on Consumer Hardware</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Session Management: Stacked full-width rows (Item 1)
    if st.button("New Session", use_container_width=True, type="primary"):
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.session_state.active_audio_path = None
        st.session_state.active_dataset_path = None
        st.session_state.audio_metadata = None
        st.session_state.audio_waveform = None
        st.session_state.messages = []
        if "_last_uploaded_name" in st.session_state:
            st.session_state._last_uploaded_name = None
        st.rerun()

    report_md = generate_markdown_audit_report(
        session_id=st.session_state.session_id,
        messages=st.session_state.messages,
        audio_path=st.session_state.active_audio_path,
        dataset_path=st.session_state.active_dataset_path,
    )
    st.download_button(
        label="Export Report",
        data=report_md,
        file_name=f"AuralQ_Audit_{st.session_state.session_id}.md",
        mime="text/markdown",
        use_container_width=True,
        type="secondary",
        disabled=not bool(st.session_state.messages),
    )

    st.markdown("---")

    # 2. Past Sessions Drawer (Strict Local Privacy Only - Section 6 Polish)
    st.markdown("### Past Sessions")
    if is_local_engine:
        sessions = db.list_sessions(limit=10)
        if sessions:
            for s in sessions:
                col_title, col_del = st.columns([3.2, 1.1])
                is_active = s["session_id"] == st.session_state.session_id
                prefix = "▶ " if is_active else ""
                
                # Title prioritizes the question title, then audio filename, then session id
                title_text = s.get("title") or (Path(s["audio_path"]).name if s.get("audio_path") else f"Session {s['session_id'][:4]}")
                
                # Two-line button label: line 1 = title, line 2 = relative timestamp (or &nbsp; if none)
                rel_time = format_relative_time(s.get("created_at"))
                if not rel_time and s.get("audio_path"):
                    rel_time = Path(s["audio_path"]).name
                time_line = rel_time if rel_time else "&nbsp;"
                btn_label = f"{prefix}{title_text}\n\n{time_line}"

                del_key = f"del_{s['session_id']}"
                confirm_key = f"confirm_del_{s['session_id']}"
                is_confirming = st.session_state.get(confirm_key, False)

                with col_title:
                    if is_active:
                        st.markdown("<div class='active-sess-indicator'></div>", unsafe_allow_html=True)
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
                    del_label = "Confirm?" if is_confirming else "Del"
                    del_type = "primary" if is_confirming else "secondary"
                    if st.button(del_label, key=del_key, use_container_width=True, type=del_type):
                        if is_confirming:
                            st.session_state[confirm_key] = False
                            db.delete_session(s["session_id"])
                            if is_active:
                                st.session_state.session_id = str(uuid.uuid4())[:8]
                                st.session_state.messages = []
                            st.rerun()
                        else:
                            for k in list(st.session_state.keys()):
                                if k.startswith("confirm_del_"):
                                    st.session_state[k] = False
                            st.session_state[confirm_key] = True
                            st.rerun()
        else:
            st.caption("No saved local sessions yet.")
    else:
        st.caption("Session Memory: Ephemeral (Zero database persistence in Cloud Demo)")

    st.markdown("---")

    # 3. Engine & Model Configuration
    if IS_CLOUD_DEPLOYMENT:
        st.markdown("### Cloud Demo Engine")
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
            st.markdown(f"<div class='cloud-key-badge cloud-key-connected'>✦ Hosted <code>{html.escape(env_var_name)}</code> Connected</div>", unsafe_allow_html=True)
            custom_key = st.text_input("Custom Key Override (Optional):", type="password", placeholder="Leave blank to use default host key")
            st.session_state.cloud_api_key = custom_key if custom_key else server_key
        else:
            st.markdown(f"<div class='cloud-key-badge cloud-key-missing'>✦ No <code>{html.escape(env_var_name)}</code> configured on host</div>", unsafe_allow_html=True)
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
                To analyze confidential audio on your own computer with **Ollama** and zero cloud dependency:

                1. **Get the code on your computer**:
                   - Via Git: `git clone https://github.com/Shagun-Thakur/AuralQ.git`
                   - Or open the GitHub link below and click **Code → Download ZIP**.
                2. **Run it locally on your PC**:
                   - Open the folder in **Windows File Explorer** (`C:\\...\\AuralQ`).
                   - Double-click the file **`run_auralq.bat`** (or run `.\\run_auralq.bat` in Terminal).
                   - *(Linux/macOS: `./run_auralq.sh`)*

                AuralQ will open in your browser at `http://localhost:8501`, connected to your local GPU & Ollama with 100% air-gapped privacy!
                """
            )
            st.link_button("🐙 Open Repository on GitHub", "https://github.com/Shagun-Thakur/AuralQ", use_container_width=True)
    else:
        # Local desktop workstation: Full dual-mode engine selector
        st.markdown("### Engine Selector")
        engine_options = ["Local Ollama (Private / Air-Gapped)", "Cloud Demo Mode"]
        engine_choice = st.radio(
            "Execution Mode:",
            engine_options,
            index=0 if is_local_engine else 1,
        )
        if engine_choice != st.session_state.engine_mode:
            st.session_state.engine_mode = engine_choice
            st.rerun()

        if "Local Ollama" in st.session_state.engine_mode:
            ollama_alive = check_ollama_status()
            if ollama_alive:
                st.caption("Ollama Daemon: Connected (`127.0.0.1:11434`)")
                installed_models = get_installed_ollama_models()
                if installed_models:
                    st.session_state.selected_model = st.selectbox(
                        "Active Local Model:",
                        installed_models,
                        index=0,
                    )
                else:
                    st.warning("No local models found.")
                    if st.button("Pull `llama3.2:3b` (~2.0 GB)"):
                        with st.spinner("Downloading model via Ollama..."):
                            if pull_ollama_model("llama3.2:3b-instruct-q4_K_M"):
                                st.success("Model downloaded successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to download model. Ensure Ollama is running.")
            else:
                st.error("Ollama is Offline.\nPlease start Ollama locally (`ollama serve`) or switch to Cloud Demo Mode.")
        else:
            st.caption("Cloud Demo: Runs using high-speed cloud LLM API")
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
                st.markdown(f"<div class='cloud-key-badge cloud-key-connected'>✦ Default <code>{html.escape(env_var_name)}</code> Detected</div>", unsafe_allow_html=True)
                custom_key = st.text_input("Custom Key Override (Optional):", type="password", placeholder="Leave blank to use default key")
                st.session_state.cloud_api_key = custom_key if custom_key else server_key
            else:
                st.markdown(f"<div class='cloud-key-badge cloud-key-missing'>✦ No <code>{html.escape(env_var_name)}</code> configured on host</div>", unsafe_allow_html=True)
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
                        "**No API Key?**\n"
                        "- Get a free key at [Google AI Studio](https://aistudio.google.com/) or [Groq Console](https://console.groq.com/).\n"
                        "- Or switch back to **Local Ollama** above for 100% free, private local execution with zero keys!"
                    )


# ---------------------------------------------------------------------------
# MAIN STAGE: Top Navigation Bar (Typographic Hero + Subtitle + Status Pill)
# ---------------------------------------------------------------------------

if is_local_engine:
    ollama_alive = check_ollama_status()
    if ollama_alive:
        status_pill_html = "<span class='xp-nameplate-local'>[ LOCAL • AIR-GAPPED ]</span>"
    else:
        status_pill_html = "<span class='xp-nameplate-offline'>[ LOCAL • OFFLINE ]</span>"
else:
    status_pill_html = "<span class='xp-nameplate-cloud'>[ CLOUD • EPHEMERAL ]</span>"

st.markdown(
    f"""
    <div class='top-nav-bar'>
        <div class='top-nav-left'>
            <span class='hero-brand-title'>AuralQ Studio</span>
            <span class='top-nav-sub'>Deterministic Audio Intelligence with Verifiable Grounding</span>
        </div>
        <div class='top-nav-right'>
            {status_pill_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def execute_acoustic_pipeline(query_text: str):
    """Execute the full Layer 1-5 agent grounding pipeline for a query."""
    if not query_text or not query_text.strip():
        return
    q = query_text.strip()
    agent = get_agent_instance()
    with st.spinner("Executing deterministic DSP tools & citation grounding..."):
        resp: AgentResponse = agent.query(
            question=q,
            audio_path=st.session_state.active_audio_path,
            dataset_path=st.session_state.active_dataset_path,
        )
        st.session_state.messages.append({"role": "user", "content": q})
        st.session_state.messages.append({
            "role": "assistant",
            "content": resp.response_text,
            "turn": resp.turn_record,
        })
        if is_local_engine and resp.turn_record:
            db.save_turn(st.session_state.session_id, resp.turn_record)
        st.rerun()


# ---------------------------------------------------------------------------
# ASYMMETRIC STUDIO WORKSPACE (35% : 65%)
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([0.35, 0.65], gap="medium")

# =============================================================================
# LEFT COLUMN: Source Audio & Ingestion (Compact)
# =============================================================================
with col_left:
    st.markdown("<div class='studio-card-header'>HOPPER INGESTION & HOTBAR</div>", unsafe_allow_html=True)

    # 1. Drag & drop uploader (Hopper Slot - Key rotates on New Session for clean state consistency)
    uploader_key = f"audio_file_uploader_{st.session_state.session_id}"
    uploaded_file = st.file_uploader(
        "Upload audio clip (.wav, .mp3, .flac, .ogg):",
        type=["wav", "mp3", "flac", "ogg"],
        key=uploader_key,
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        if getattr(st.session_state, "_last_uploaded_name", None) != uploaded_file.name:
            temp_upload_path = Path("data/samples/uploaded_active.wav")
            temp_upload_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            res = load_audio_file(temp_upload_path)
            if res.status == "SUCCESS":
                st.session_state._last_uploaded_name = uploaded_file.name
                st.session_state.active_audio_path = str(temp_upload_path.resolve())
                st.session_state.active_dataset_path = None
                st.session_state.audio_metadata = res.metadata
                st.session_state.audio_waveform = res.waveform
                st.session_state.audio_sr = res.sample_rate
                st.toast(f"Loaded {uploaded_file.name}!", icon="✅")
                st.rerun()
            else:
                st.error(f"Ingestion rejected: {res.message}")

    # 2. Hotbar Preset Buttons (Square 3D Tiles)
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("1k Sine", use_container_width=True):
            p = SAMPLE_DIR / "clean_sine_1khz.wav"
            res = load_audio_file(p)
            st.session_state._last_uploaded_name = None
            st.session_state.active_audio_path = str(p.resolve())
            st.session_state.active_dataset_path = None
            st.session_state.audio_metadata = res.metadata
            st.session_state.audio_waveform = res.waveform
            st.session_state.audio_sr = res.sample_rate
            st.rerun()
    with col_p2:
        if st.button("Snare", use_container_width=True):
            p = SAMPLE_DIR / "clipped_snare.wav"
            res = load_audio_file(p)
            st.session_state._last_uploaded_name = None
            st.session_state.active_audio_path = str(p.resolve())
            st.session_state.active_dataset_path = None
            st.session_state.audio_metadata = res.metadata
            st.session_state.audio_waveform = res.waveform
            st.session_state.audio_sr = res.sample_rate
            st.rerun()
    with col_p3:
        if st.button("Noise", use_container_width=True):
            p = SAMPLE_DIR / "noisy_audio.wav"
            res = load_audio_file(p)
            st.session_state._last_uploaded_name = None
            st.session_state.active_audio_path = str(p.resolve())
            st.session_state.active_dataset_path = None
            st.session_state.audio_metadata = res.metadata
            st.session_state.audio_waveform = res.waveform
            st.session_state.audio_sr = res.sample_rate
            st.rerun()

    # 3. Audio Playback & 4-Box Ender Stat Grid (When Audio Active)
    if st.session_state.active_audio_path and st.session_state.audio_waveform is not None:
        if getattr(st.session_state, "_last_uploaded_name", None) and "uploaded_active" in Path(st.session_state.active_audio_path).name:
            active_name = st.session_state._last_uploaded_name
        else:
            active_name = Path(st.session_state.active_audio_path).name

        st.markdown(
            f"""
            <div class='jukebox-panel' style='margin: 8px 0 10px 0;'>
                <div class='jukebox-header'>
                    {get_icon('music_disc')} <span style='font-family: var(--mono); font-size: 11px; color: var(--teal); font-weight: 600; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;'>ACTIVE: {html.escape(active_name)}</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        buf = io.BytesIO()
        sf.write(buf, st.session_state.audio_waveform, st.session_state.audio_sr, format="WAV")
        st.audio(buf.getvalue(), format="audio/wav")
        st.markdown("</div>", unsafe_allow_html=True)

        meta = st.session_state.audio_metadata
        if meta:
            sr_khz = meta.sample_rate / 1000.0
            sr_display = f"{sr_khz:.2f} kHz" if meta.sample_rate % 1000 != 0 else f"{int(sr_khz)} kHz"
            dur_display = f"{meta.duration_sec:.2f} s"
            rms_display = f"{meta.rms_dbfs:.1f} dBFS"
            is_clipped = getattr(meta, "is_clipped", False)
            is_silent = getattr(meta, "is_silent", meta.rms_dbfs < -60.0)

            if is_clipped:
                status_html = "<span style='color: var(--redstone);'>🚨 CLIPPED</span>"
            elif is_silent:
                status_html = "<span style='color: var(--warning);'>⚠️ MUTED</span>"
            else:
                status_html = "<span style='color: var(--teal);'>✔ CLEAN</span>"

            st.markdown(
                f"""
                <div class='ender-stat-grid' style='margin-bottom: 10px;'>
                    <div class='ender-stat-box'>
                        <span class='stat-label'>SAMPLE RATE</span>
                        <span class='stat-value'>{sr_display}</span>
                    </div>
                    <div class='ender-stat-box'>
                        <span class='stat-label'>DURATION</span>
                        <span class='stat-value'>{dur_display}</span>
                    </div>
                    <div class='ender-stat-box'>
                        <span class='stat-label'>RMS</span>
                        <span class='stat-value'>{rms_display}</span>
                    </div>
                    <div class='ender-stat-box'>
                        <span class='stat-label'>STATUS</span>
                        <span class='stat-value'>{status_html}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 4. Batch Dataset Audit Expander (Placed Last in Left Column)
    with st.expander("Batch Dataset Audit", expanded=False):
        sample_dataset_dir = SAMPLE_DIR / "sample_dataset"
        if sample_dataset_dir.exists() and sample_dataset_dir.is_dir():
            if st.button("Load Built-in Sample Dataset", use_container_width=True):
                st.session_state.active_dataset_path = str(sample_dataset_dir.resolve())
                st.session_state.active_audio_path = None
                st.session_state.audio_metadata = None
                st.session_state.audio_waveform = None
                st.rerun()

        dataset_input = st.text_input(
            "Directory:",
            value=st.session_state.active_dataset_path or "",
            placeholder="C:\\Path\\To\\Audio\\Dataset",
        )
        if st.button("Set Dataset"):
            p = Path(dataset_input)
            if p.exists() and p.is_dir():
                st.session_state.active_dataset_path = str(p.resolve())
                st.session_state.active_audio_path = None
                st.session_state.audio_metadata = None
                st.session_state.audio_waveform = None
                st.success(f"Active dataset: `{p.resolve()}`")
                st.rerun()

        if st.session_state.active_dataset_path:
            if st.button("Run Dataset Hygiene Audit", type="primary", use_container_width=True):
                execute_acoustic_pipeline("Please audit my audio dataset directory and recommend what preprocessing pipeline I should apply.")


# =============================================================================
# RIGHT COLUMN: Conversation, Terminal & Visualisations
# =============================================================================
with col_right:
    has_audio = bool(st.session_state.active_audio_path or st.session_state.active_dataset_path)

    # 1. Acoustic Query Terminal (Compact ~3 lines, Button directly under)
    st.markdown("<div class='studio-card-header'>ACOUSTIC QUERY TERMINAL</div>", unsafe_allow_html=True)
    with st.form("left_query_form", clear_on_submit=False):
        if has_audio:
            placeholder_text = "Enter acoustic question (e.g. Is there background noise? Calculate spectral centroid)..."
        else:
            placeholder_text = "Select a Hotbar Preset (1k Sine / Snare / Noise) or drop an audio file in the Hopper to ignite the altar..."

        query_input = st.text_area(
            "Query Prompt:",
            placeholder=placeholder_text,
            height=68,
            label_visibility="collapsed",
            disabled=not has_audio,
        )
        run_query_btn = st.form_submit_button(
            "IGNITE ACOUSTIC RUN",
            type="primary",
            use_container_width=True,
            disabled=not has_audio,
        )

    if run_query_btn and query_input.strip() and has_audio:
        execute_acoustic_pipeline(query_input.strip())

    # Check for newest assistant response
    latest_assistant_msg = None
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "assistant":
            latest_assistant_msg = msg
            break

    if not has_audio:
        # EMPTY STATE: ONE unified standby panel only (Item 5)
        st.markdown(
            """
            <div class='profile-card dormant' style='padding: 24px 20px; text-align: center; margin-top: 10px;'>
                <div class='profile-header' style='font-size: 12px; margin-bottom: 8px;'><span style='color: var(--teal);'>✦</span> ACOUSTIC INTELLIGENCE STANDBY</div>
                <div style='color: var(--cream); font-size: 11px; line-height: 1.8; max-width: 520px; margin: 0 auto; font-family: var(--mono);'>
                    Insert an audio record via the <b>Hopper</b> or select a <b>Hotbar Preset</b> to initiate the analysis altar.<br>
                    Awaiting audio signal to render dimensional STFT spectrogram, waveform dynamics, and acoustic telemetry.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # 2. Grounding banner + plain-language verification line + Answer Panel
        if latest_assistant_msg:
            turn: Optional[TurnRecord] = latest_assistant_msg.get("turn")
            box_class = "grounded-answer-box"

            if turn and turn.grounding_report:
                rep: GroundingReport = turn.grounding_report
                if turn.evidence and turn.evidence.status == "ABSTAIN":
                    xp_header_html = "<div class='xp-bar-container abstain'>[ ABSTAINED • NON-ACOUSTIC SCOPE ]</div>"
                    box_class = "grounded-answer-box abstain"
                elif rep.was_repaired:
                    xp_header_html = f"<div class='xp-bar-container repaired'>[ REPAIRED • {len(rep.hallucinated_numbers)} UNVERIFIED REMOVED ]</div>"
                    box_class = "grounded-answer-box repaired"
                elif rep.is_fully_grounded:
                    xp_count = len(rep.verified_citations)
                    xp_header_html = f"<div class='xp-bar-container verified'>[ 100% GROUNDED • {xp_count} XP VERIFIED ]</div>"
                    box_class = "grounded-answer-box verified"
                else:
                    xp_count = len(rep.verified_citations)
                    xp_score = rep.grounding_score * 100
                    xp_header_html = f"<div class='xp-bar-container verified'>[ {xp_score:.0f}% GROUNDED • {xp_count} XP VERIFIED ]</div>"
                    box_class = "grounded-answer-box verified"
            else:
                xp_header_html = "<div class='xp-bar-container verified'>[ GROUNDED SYNTHESIS & AUDIT ]</div>"
                box_class = "grounded-answer-box"

            st.markdown(xp_header_html, unsafe_allow_html=True)

            # Plain-Language Verification Line
            if turn and turn.grounding_report:
                rep = turn.grounding_report
                num_verified = len(rep.verified_citations)
                if rep.is_fully_grounded:
                    plain_txt = f"{num_verified} numeric claim{'s' if num_verified != 1 else ''} verified against deterministic DSP output."
                elif rep.was_repaired:
                    plain_txt = f"{num_verified} claim{'s' if num_verified != 1 else ''} verified • {len(rep.hallucinated_numbers)} ungrounded number{'s' if len(rep.hallucinated_numbers) != 1 else ''} repaired."
                else:
                    plain_txt = f"{num_verified} of {rep.total_numbers_cited} numeric claim{'s' if rep.total_numbers_cited != 1 else ''} verified against DSP evidence."
                st.markdown(f"<div class='verification-plain-line'>{plain_txt}</div>", unsafe_allow_html=True)

            # 3. Answer Box
            answer_html = render_grounded_answer_html(latest_assistant_msg["content"], turn)
            st.markdown(f"<div class='{box_class}'>{answer_html}</div>", unsafe_allow_html=True)

            # 4. Collapsible Expander for Tool Execution Trace
            if turn:
                total_ms = turn.latency_breakdown.get('total_ms', 0) if turn.latency_breakdown else 0
                with st.expander(f"Inspect Tool Trace & Evidence JSON ({total_ms:.0f} ms total)", expanded=False):
                    col_tr1, col_tr2 = st.columns([1, 1])
                    with col_tr1:
                        st.markdown("**Dispatched Tools:**")
                        if turn.parsed_intent and turn.parsed_intent.selected_tools:
                            tool_names = [tc.tool_name for tc in turn.parsed_intent.selected_tools]
                            st.markdown(tool_tiles_html(tool_names), unsafe_allow_html=True)
                            for tc in turn.parsed_intent.selected_tools:
                                st.caption(f"`{tc.tool_name}` (Params: `{tc.parameters}`)")
                        else:
                            st.markdown("*(Zero tools dispatched)*")

                        st.markdown("**Latency Breakdown:**")
                        lat = turn.latency_breakdown or {}
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

        elif st.session_state.audio_waveform is not None:
            # Baseline Profile Card (when audio is loaded but before query execution)
            prof = compute_initial_acoustic_profile(
                st.session_state.audio_waveform,
                st.session_state.audio_sr,
            )
            st.markdown(
                f"""
                <div class='profile-card'>
                    <div class='profile-header'>INITIAL ACOUSTIC BASELINE PROFILE</div>
                    <div class='profile-grid'>
                        <div class='profile-stat-box'>
                            <span class='profile-label'>SPECTRAL CENTROID</span>
                            <span class='profile-val'>{prof['centroid_hz']:.1f} Hz</span>
                            <span class='profile-hint'>{prof['perceptual_band']}</span>
                        </div>
                        <div class='profile-stat-box'>
                            <span class='profile-label'>SPECTRAL FLATNESS</span>
                            <span class='profile-val'>{prof['flatness']:.4f}</span>
                            <span class='profile-hint'>{prof['texture']}</span>
                        </div>
                        <div class='profile-stat-box'>
                            <span class='profile-label'>RMS LEVEL</span>
                            <span class='profile-val'>{prof['rms_dbfs']:.1f} dBFS</span>
                            <span class='profile-hint'>Dynamic Baseline</span>
                        </div>
                        <div class='profile-stat-box'>
                            <span class='profile-label'>GROUNDING STATUS</span>
                            <span class='profile-val' style='color: var(--teal);'>STANDBY</span>
                            <span class='profile-hint'>Ready for Deterministic Audit</span>
                        </div>
                    </div>
                    <div class='profile-footer'>
                        Altar Ready: Enter a question above to ignite the deterministic DSP tools and synthesis pipeline.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 5. Visualisation Tabs (Spectrogram / Waveform / Telemetry) + thresholds legend
        c_tab_spec, c_tab_wave, c_tab_tele = st.tabs(["STFT Spectrogram (Log)", "Waveform", "Telemetry"])

        with c_tab_spec:
            if st.session_state.audio_waveform is not None:
                spec_fig = generate_interactive_spectrogram(
                    st.session_state.audio_waveform,
                    st.session_state.audio_sr,
                )
                st.plotly_chart(spec_fig, use_container_width=True, config={"displayModeBar": False})
                st.markdown(
                    """
                    <div class='spec-legend-box'>
                        <span class='spec-legend-title'>ACOUSTIC PERCEPTUAL THRESHOLDS:</span>
                        <span class='spec-legend-item'><span class='legend-line amethyst-dash'>--</span> <b>1.5 kHz</b>: Warm Midrange / Upper-Mid Boundary (Perceptual Anchor: <code>src/tools/spectral.py:59</code>)</span>
                        <span class='spec-legend-item'><span class='legend-line teal-dash'>--</span> <b>4.0 kHz</b>: Crisp Presence / Treble-Dominant Ceiling (Perceptual Anchor: <code>src/tools/spectral.py:64</code>)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Dataset loaded. Spectrogram displays individual audio files.")

        with c_tab_wave:
            if st.session_state.audio_waveform is not None:
                col_w1, col_w2 = st.columns([1, 3])
                with col_w1:
                    wf_view = st.radio(
                        "Waveform Range:",
                        ["20 ms Zoom", "Full Signal"],
                        horizontal=True,
                        label_visibility="collapsed",
                        key="wf_view_range_select",
                    )
                wave_fig = generate_interactive_waveform(
                    st.session_state.audio_waveform,
                    st.session_state.audio_sr,
                    zoom_ms=20.0 if wf_view == "20 ms Zoom" else None,
                )
                st.plotly_chart(wave_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Dataset loaded. Waveform displays individual audio files.")

        with c_tab_tele:
            if st.session_state.audio_metadata is not None and st.session_state.audio_waveform is not None:
                meta = st.session_state.audio_metadata
                subtype_val = getattr(meta, "subtype", "PCM (Float32)")
                is_clipped = getattr(meta, "is_clipped", False)
                is_silent = getattr(meta, "is_silent", meta.rms_dbfs < -60.0)

                # Silence Ratio calculation
                silence_threshold = 10 ** (-60.0 / 20.0)
                silent_sample_count = np.sum(np.abs(st.session_state.audio_waveform) < silence_threshold)
                silence_ratio = float(silent_sample_count / len(st.session_state.audio_waveform))
                silence_ratio_pct = f"{silence_ratio * 100.0:.1f}%"
                silence_caption = "High silence / muted" if is_silent else "Audible continuous energy"

                st.markdown(
                    f"""
                    <div class='ender-stat-grid' style='grid-template-columns: repeat(2, 1fr); margin-top: 6px;'>
                        <div class='ender-stat-box'>
                            <span class='stat-label'>NYQUIST LIMIT</span>
                            <span class='stat-value'>{meta.sample_rate // 2} Hz</span>
                        </div>
                        <div class='ender-stat-box'>
                            <span class='stat-label'>CHANNELS</span>
                            <span class='stat-value'>{meta.num_channels} ({'Mono' if meta.num_channels == 1 else 'Stereo'})</span>
                        </div>
                        <div class='ender-stat-box'>
                            <span class='stat-label'>BIT DEPTH / FORMAT</span>
                            <span class='stat-value'>{subtype_val}</span>
                        </div>
                        <div class='ender-stat-box'>
                            <span class='stat-label'>TOTAL SAMPLES</span>
                            <span class='stat-value'>{len(st.session_state.audio_waveform):,}</span>
                        </div>
                        <div class='ender-stat-box'>
                            <span class='stat-label'>DYNAMIC HEADROOM</span>
                            <span class='stat-value'>{abs(meta.rms_dbfs):.2f} dB</span>
                        </div>
                        <div class='ender-stat-box'>
                            <span class='stat-label'>CLIPPING STATUS</span>
                            <span class='stat-value'>{'🚨 Detected' if is_clipped else '✔ 0 clipped'}</span>
                        </div>
                        <div class='ender-stat-box' style='grid-column: span 2;'>
                            <span class='stat-label'>SILENCE RATIO</span>
                            <span class='stat-value'>{silence_ratio_pct} <span style='font-size: 11px; color: var(--cream); font-weight: normal;'>({silence_caption})</span></span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Awaiting audio upload for telemetry inspection.")

        # 6. Earlier Session Turns Expander (Placed Below Plots - Item 6)
        if len(st.session_state.messages) > 2:
            with st.expander(f"Earlier Session Turns ({len(st.session_state.messages)//2 - 1})", expanded=False):
                for msg in st.session_state.messages[:-2]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])


