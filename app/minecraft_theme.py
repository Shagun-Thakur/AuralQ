"""Ender-Acoustic Console: Minecraft x Role-Balanced Theme for AuralQ (Streamlit).

Strict Role-Based Color System:
- Neutral Void (~60%): Page & sidebar background (#0D0912), cards/surfaces (#151019),
  inset wells (#100B16), quiet borders (#2A2236). Very low saturation dark stone.
- Cream (End-Stone) (~15%): #DDDFA5 for section headings, labels, axis text, secondary text;
  warm off-white (#EDEAD6) for body text. All contrast ratios >= 11:1 (WCAG AAA).
- Purple (~15-20%): Action and structure only: primary buttons (Ignite, New Session),
  active tab, focus ring, and the app brand title. Chunky bevel strictly on interactive elements.
- Teal (~5-8%): #3DFFC4 for state and truth only: verified citation chips, XP grounding banner,
  active-file strip, CLEAN status, selected radio indicator, waveform trace, 4 kHz threshold.
- Amber / Redstone: Isolated warnings only (#FFAA00 for repaired/muted, #FF5555 for clipped/errors).
"""

import html
import re
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# ROLE-BASED COLOR ARCHITECTURE
# =============================================================================
ENDER_COLORS = {
    # 1. Neutral Void (~60%)
    "void": "#0D0912",          # Page & sidebar background (matching void)
    "sidebar": "#0D0912",       # Sidebar background matching page exactly
    "surface": "#151019",       # Cards, containers, tiles, expanders
    "surface_hover": "#1D1623", # Secondary button hover
    "inset": "#100B16",         # Text input wells, hoppers, radio unselected
    "border_quiet": "#2A2236",  # Passive borders (cards, expanders, answer box)
    "border_subtle": "#1E1728", # Hairlines, dividers, subtle bevel highlights
    "shadow_dark": "#050308",   # 1px text shadow for crisp headings

    # 2. Cream / Warm Off-White (~15%)
    "cream": "#DDDFA5",         # Section headings, labels, axis text, secondary text
    "body_text": "#EDEAD6",     # Warm off-white for body text
    "text": "#EDEAD6",          # Primary text alias
    "muted": "#B5B099",         # Subtle secondary hint / metadata

    # 3. Action & Structure Purple (~15-20%)
    "portal": "#8B3DFF",        # Primary action accent, active tab line, focus ring
    "portal_bright": "#9D4EDD", # Primary button hover / top bevel
    "portal_dark": "#5A189A",   # Primary button bottom shadow / active border
    "amethyst": "#C77DFF",      # Accent highlights, hero title secondary
    "hero_amethyst": "#E0AAFF", # "AuralQ Studio" title (single large purple element)

    # 4. State & Truth Teal (~5-8%)
    "teal": "#3DFFC4",          # State and truth: verified, clean, selected radio, waveform
    "teal_bg": "#0D2C24",       # Verified chip/badge background
    "teal_border": "#1A5E4E",   # Verified chip/badge border

    # 5. Warnings & Alerts (Isolated)
    "warning": "#FFAA00",       # Redstone amber for repaired / muted / warnings
    "warning_bg": "#241400",
    "warning_border": "#663C00",
    "redstone": "#FF5555",      # Redstone alert for clipping / errors / abstention
    "redstone_bg": "#240A0A",
    "redstone_border": "#5E1A1A",

    # Compatibility aliases
    "xp_green": "#3DFFC4",
    "xp_bg": "#0D2C24",
    "xp_border": "#1A5E4E",
    "gold_ingot": "#C77DFF",
    "bevel": "#2A2236",
}

ENDER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    /* 1. Neutral Void (~60%) */
    --void: #0D0912;
    --sidebar: #0D0912;
    --surface: #151019;
    --surface-hover: #1D1623;
    --inset: #100B16;
    --border-quiet: #2A2236;
    --border-subtle: #1E1728;
    --shadow-dark: #050308;

    /* 2. Cream / Warm Off-White (~15%) */
    --cream: #DDDFA5;
    --body-text: #EDEAD6;
    --text: #EDEAD6;
    --muted: #B5B099;

    /* 3. Action & Structure Purple (~15-20%) */
    --portal: #8B3DFF;
    --portal-bright: #9D4EDD;
    --portal-dark: #5A189A;
    --amethyst: #C77DFF;
    --hero-amethyst: #E0AAFF;

    /* 4. State & Truth Teal (~5-8%) */
    --teal: #3DFFC4;
    --teal-bg: #0D2C24;
    --teal-border: #1A5E4E;

    /* 5. Warning & Alerts */
    --warning: #FFAA00;
    --warning-bg: #241400;
    --warning-border: #663C00;
    --redstone: #FF5555;
    --redstone-bg: #240A0A;
    --redstone-border: #5E1A1A;

    /* Strict Font Stacks */
    --pixel: 'Press Start 2P', 'Courier New', monospace;
    --mono: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
}

/* =========================================================================
   1. ZERO RADIUS RULE: Minecraft has strictly NO rounded corners
   ========================================================================= */
*, *::before, *::after,
div, button, input, textarea, select, a, label,
.stButton, .stButton > button,
.stDownloadButton, .stDownloadButton > button,
.stLinkButton, .stLinkButton > a,
.stTextInput, .stTextInput input,
.stTextArea, .stTextArea textarea,
div[data-baseweb="select"], div[data-baseweb="select"] > div,
div[data-baseweb="popover"], div[data-baseweb="menu"],
div[data-testid="stExpander"], div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] [data-testid="stExpanderDetails"],
section[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploaderDropzone"],
.studio-card, .minecraft-box, .profile-card,
.grounded-answer-box, .jukebox-panel,
.xp-nameplate-local, .xp-nameplate-cloud, .xp-nameplate-offline,
.xp-bar-container,
.tool-tile, .citation-verified, .citation-repaired,
div[data-baseweb="tab-list"], div[data-baseweb="tab"],
.stTabs [data-baseweb="tab"],
div[role="tablist"], button[role="tab"], div[data-testid="stTabs"], [data-testid="stTab"],
[data-testid="stChatMessage"],
.stAlert, [data-testid="stAlert"],
.dormant-altar-box,
.ender-stat-box, .active-audio-badge,
audio, code, pre {
    border-radius: 0px !important;
}

/* =========================================================================
   UNIVERSAL MATERIAL ICONS & SYMBOLS PROTECTION
   Prevents font-family rules from overriding icon font ligatures (e.g. "upload")
   ========================================================================= */
[data-testid="stIconMaterial"],
[data-testid="stIconMaterial"] *,
.material-symbols-rounded,
.material-symbols-outlined,
.material-symbols-sharp,
[class*="material-symbols"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Symbols", monospace !important;
    font-feature-settings: "liga" 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
}

/* =========================================================================
   2. NATURAL VIEWPORT FLOW & FIXED HEADER CLEARANCE
   ========================================================================= */
header[data-testid="stHeader"] {
    background: rgba(13, 9, 18, 0.96) !important;
    border-bottom: 1px solid var(--border-quiet) !important;
    z-index: 99 !important;
}

html, body, [data-testid="stAppViewContainer"], .main {
    min-height: 100vh;
    overflow-x: hidden;
}

.main .block-container {
    padding-top: 4.0rem !important; /* Clearance for fixed header */
    padding-bottom: 1.5rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 100% !important;
}

[data-testid="column"] {
    overflow-y: visible;
    padding-right: 6px;
}

/* =========================================================================
   3. NEUTRAL VOID BACKGROUND WITH 16px PIXEL-GRID
   ========================================================================= */
.stApp {
    background-color: var(--void);
    background-image: radial-gradient(#1E1728 1px, transparent 1px);
    background-size: 16px 16px;
    color: var(--body-text);
    font-family: var(--mono);
}

.stApp p, .stApp li, .stApp label, .stApp div[data-testid="stMarkdownContainer"] {
    font-family: var(--mono);
}

/* Subtle Ender particle drift (<2.5% opacity) */
@keyframes enderDrift {
    0% { background-position: 0 0; }
    100% { background-position: 32px 32px; }
}
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.02;
    background-image:
        radial-gradient(circle at 4px 4px, var(--cream) 1px, transparent 1px),
        radial-gradient(circle at 16px 16px, var(--portal) 1px, transparent 1px);
    background-size: 32px 32px;
    animation: enderDrift 40s linear infinite;
}
@media (prefers-reduced-motion: reduce) {
    .stApp::before {
        animation: none !important;
    }
    * {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001ms !important;
    }
}

/* =========================================================================
   4. BRAND HEADER & STATUS PILL (Single Purple Brand Element)
   ========================================================================= */
.top-nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2px 0 8px 0;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border-quiet);
    box-shadow: 0 1px 0px var(--border-subtle);
}

.top-nav-left {
    display: flex;
    align-items: baseline;
    gap: 14px;
    flex-wrap: wrap;
}

.hero-brand-title {
    font-family: var(--pixel) !important;
    font-size: 22px !important;
    color: var(--hero-amethyst) !important;
    text-shadow: 1px 1px 0px #7B2CBF, 2px 2px 0px #3C096C !important;
    letter-spacing: -0.5px;
    display: inline-block;
    line-height: 1.2;
}

.top-nav-sub {
    font-family: var(--mono);
    color: var(--cream);
    font-size: 11px;
    letter-spacing: 0.02em;
    white-space: nowrap;
}

.top-nav-right {
    display: flex;
    align-items: center;
    justify-content: flex-end;
}

/* Single-Line Status Pills (Nowrap) */
.xp-nameplate-local {
    font-family: var(--pixel) !important;
    font-size: 10px !important;
    color: var(--teal) !important;
    background: var(--teal-bg) !important;
    border: 1px solid var(--teal-border) !important;
    box-shadow: inset 1px 1px 0px #081C16, inset -1px -1px 0px #2E8B75 !important;
    padding: 4px 8px !important;
    letter-spacing: 0.04em;
    display: inline-flex !important;
    align-items: center !important;
    white-space: nowrap !important;
}

.xp-nameplate-offline {
    font-family: var(--pixel) !important;
    font-size: 10px !important;
    color: var(--redstone) !important;
    background: var(--redstone-bg) !important;
    border: 1px solid var(--redstone-border) !important;
    box-shadow: inset 1px 1px 0px #120404, inset -1px -1px 0px #8B2E2E !important;
    padding: 4px 8px !important;
    letter-spacing: 0.04em;
    display: inline-flex !important;
    align-items: center !important;
    white-space: nowrap !important;
}

.xp-nameplate-cloud {
    font-family: var(--pixel) !important;
    font-size: 10px !important;
    color: var(--amethyst) !important;
    background: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px #0A0310 !important;
    padding: 4px 8px !important;
    letter-spacing: 0.04em;
    display: inline-flex !important;
    align-items: center !important;
    white-space: nowrap !important;
}

/* =========================================================================
   5. SECTION HEADINGS (Cream with 1px Dark Shadow)
   ========================================================================= */
section[data-testid="stSidebar"] h3,
.main h3,
.ender-section-title,
.studio-card-header,
.profile-header {
    font-family: var(--pixel) !important;
    font-size: 11px !important;
    color: var(--cream) !important;
    text-shadow: 1px 1px 0px var(--shadow-dark) !important;
    letter-spacing: 0.05em !important;
    margin: 10px 0 6px 0 !important;
    line-height: 1.4 !important;
}

.sidebar-brand-box {
    padding: 0 2px 8px 2px !important;
    margin-bottom: 8px !important;
    border-bottom: 1px solid var(--border-quiet) !important;
    max-width: 100% !important;
    min-width: 0 !important;
    overflow-x: hidden !important;
}

.sidebar-title {
    font-family: var(--pixel) !important;
    font-size: 14px !important;
    color: var(--hero-amethyst) !important;
    text-shadow: 1px 1px 0px #3C096C !important;
    letter-spacing: -0.5px;
    line-height: 1.3;
    margin-bottom: 4px;
    max-width: 100% !important;
    min-width: 0 !important;
    word-break: break-word !important;
}

.sidebar-subtitle {
    font-family: var(--mono) !important;
    color: var(--cream) !important;
    font-size: 11px !important;
    line-height: 1.35;
}

/* =========================================================================
   6. PASSIVE CONTAINERS (Quiet Neutral #2A2236 Borders, Zero Bright Purple Bevel)
   ========================================================================= */
.studio-card, .minecraft-box, .profile-card {
    background-color: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px var(--border-subtle), inset -1px -1px 0px var(--shadow-dark) !important;
    border-radius: 0px !important;
    padding: 10px 14px;
    margin-bottom: 10px;
}

/* Single quiet border for expanders: Outer container is transparent, only <details> has border */
div[data-testid="stExpander"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0px !important;
    padding: 0 !important;
    margin-bottom: 8px !important;
}

div[data-testid="stExpander"] details,
div[data-testid="stExpander"] > details {
    background-color: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px var(--border-subtle), inset -1px -1px 0px var(--shadow-dark) !important;
    border-radius: 0px !important;
}

div[data-testid="stExpander"] summary {
    background-color: var(--surface) !important;
    border-bottom: 1px solid var(--border-quiet) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--cream) !important;
    padding: 8px 12px !important;
    border-radius: 0px !important;
    cursor: pointer !important;
}

div[data-testid="stExpander"] summary:hover {
    color: #FFFFFF !important;
    background-color: var(--surface-hover) !important;
}

div[data-testid="stExpander"] summary svg,
div[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
    color: var(--muted) !important;
    fill: var(--muted) !important;
}

div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background-color: var(--surface) !important;
    padding: 10px 12px !important;
    border-radius: 0px !important;
    color: var(--body-text) !important;
    font-size: 11px !important;
    line-height: 1.5 !important;
}

div[data-testid="stExpander"] [data-testid="stExpanderDetails"] p,
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] li {
    font-size: 11px !important;
    line-height: 1.5 !important;
}

/* =========================================================================
   7. INSET WELLS (Hopper Slot, Inputs, Textareas, Zero Native Blue Focus)
   ========================================================================= */
section[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploaderDropzone"] {
    background-color: var(--inset) !important;
    border: 1px dashed var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px var(--shadow-dark) !important;
    border-radius: 0px !important;
    padding: 6px 8px !important;
}

/* Instruction text inside dropzone: Always visible, never hidden by responsive container query */
section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"],
div[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    padding: 0 4px !important;
}

section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] span,
div[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] span,
section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] small,
div[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    font-size: 11px !important;
    line-height: 1.3 !important;
    color: var(--cream) !important;
    font-family: var(--mono) !important;
}

/* Browse files button inside dropzone: Prominent >= 12px with comfortable padding */
section[data-testid="stFileUploaderDropzone"] button,
div[data-testid="stFileUploaderDropzone"] button {
    background-color: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px #2D243A, inset -1px -1px 0px var(--shadow-dark) !important;
    color: var(--cream) !important;
    font-family: var(--pixel) !important;
    font-size: 12px !important;
    border-radius: 0px !important;
    padding: 8px 14px !important;
    min-height: 36px !important;
    margin: 4px 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;
}

/* Hide the material icon ligature inside the uploader button to prevent "uploAload" collision */
section[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"],
div[data-testid="stFileUploaderDropzone"] button [data-testid="stIconMaterial"],
section[data-testid="stFileUploaderDropzone"] button [class*="material"],
div[data-testid="stFileUploaderDropzone"] button [class*="material"] {
    display: none !important;
}

section[data-testid="stFileUploaderDropzone"] button p,
div[data-testid="stFileUploaderDropzone"] button p,
section[data-testid="stFileUploaderDropzone"] button span:not([data-testid="stIconMaterial"]):not([class*="material"]),
div[data-testid="stFileUploaderDropzone"] button span:not([data-testid="stIconMaterial"]):not([class*="material"]) {
    font-family: var(--pixel) !important;
    font-size: 12px !important;
    color: inherit !important;
}

section[data-testid="stFileUploaderDropzone"] button:hover,
div[data-testid="stFileUploaderDropzone"] button:hover,
section[data-testid="stFileUploaderDropzone"] button:focus,
div[data-testid="stFileUploaderDropzone"] button:focus,
section[data-testid="stFileUploaderDropzone"] button:focus-visible,
div[data-testid="stFileUploaderDropzone"] button:focus-visible {
    background-color: var(--surface-hover) !important;
    border-color: var(--portal) !important;
    color: #FFFFFF !important;
    outline: 2px solid var(--portal) !important;
    outline-offset: 1px !important;
    box-shadow: inset 1px 1px 0px #3D2D50, 0 0 6px rgba(139, 61, 255, 0.3) !important;
}

section[data-testid="stFileUploaderDropzone"] button *,
div[data-testid="stFileUploaderDropzone"] button * {
    color: inherit !important;
}

/* Hide Streamlit duplicate file chip so only the Active Jukebox strip shows the filename (Item 3) */
[data-testid="stFileUploaderFile"],
div[data-testid="stFileUploaderFile"],
section[data-testid="stFileUploaderDropzone"] + div:has([data-testid="stFileUploaderFile"]),
section[data-testid="stFileUploaderDropzone"] ~ div [data-testid="stFileUploaderFile"] {
    display: none !important;
}

/* Text Inputs & Textarea Focus Ring: Zero Blue, Portal Purple Focus Ring */
.stTextInput input, .stTextArea textarea, div[data-baseweb="select"] > div,
div[data-baseweb="textarea"], div[data-baseweb="base-input"] {
    background-color: var(--inset) !important;
    color: var(--body-text) !important;
    border: 1px solid var(--border-quiet) !important;
    border-radius: 0px !important;
    box-shadow: inset 2px 2px 0px var(--shadow-dark) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
}

.stTextInput input:focus, .stTextInput input:focus-visible,
.stTextArea textarea:focus, .stTextArea textarea:focus-visible,
div[data-baseweb="textarea"]:focus-within,
div[data-baseweb="base-input"]:focus-within,
div[data-baseweb="textarea"]:has(textarea:focus) {
    border-color: var(--portal) !important;
    outline: 2px solid var(--portal) !important;
    outline-offset: 1px !important;
    box-shadow: inset 1px 1px 0px var(--shadow-dark), 0 0 8px rgba(139, 61, 255, 0.4) !important;
}

.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: var(--muted) !important;
    font-family: var(--mono) !important;
}

/* Password visibility toggle button & eye icon */
div[data-baseweb="base-input"] button,
[data-testid="stTextInput"] button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--muted) !important;
    padding: 0 8px !important;
}
div[data-baseweb="base-input"] button:hover,
[data-testid="stTextInput"] button:hover {
    color: var(--cream) !important;
    background: transparent !important;
}
div[data-baseweb="base-input"] button svg,
[data-testid="stTextInput"] button svg {
    fill: currentColor !important;
    stroke: currentColor !important;
}

/* Selectbox Dropdown Menu & Popover */
div[data-baseweb="select"] svg {
    fill: var(--cream) !important;
    color: var(--cream) !important;
}
div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    border-radius: 0px !important;
    box-shadow: 0 4px 12px var(--shadow-dark) !important;
}
div[data-baseweb="menu"] li, ul[role="listbox"] li, [role="option"] {
    background-color: transparent !important;
    color: var(--cream) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    padding: 6px 10px !important;
}
div[data-baseweb="menu"] li:hover, ul[role="listbox"] li:hover, [role="option"]:hover, [role="option"][aria-selected="true"] {
    background-color: var(--surface-hover) !important;
    color: #FFFFFF !important;
}

/* =========================================================================
   8. BUTTONS: Primary Actions vs Neutral Secondary
   ========================================================================= */
/* Secondary Buttons: Presets, Del, Export Report, Browse */
.stButton > button, .stDownloadButton > button, .stLinkButton > a,
div[data-testid="column"] div[data-testid="stHorizontalBlock"] .stButton > button {
    background-color: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px #2D243A, inset -1px -1px 0px var(--shadow-dark) !important;
    color: var(--cream) !important;
    border-radius: 0px !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    letter-spacing: 0.03em;
    padding: 7px 10px !important;
    transition: none !important;
}

.stButton > button:hover, .stDownloadButton > button:hover, .stLinkButton > a:hover {
    background-color: var(--surface-hover) !important;
    border-color: var(--portal) !important;
    color: #FFFFFF !important;
    box-shadow: inset 1px 1px 0px #3D2D50, inset -1px -1px 0px var(--shadow-dark) !important;
}

.stButton > button:active, .stDownloadButton > button:active, .stLinkButton > a:active {
    box-shadow: inset 1px 1px 0px var(--shadow-dark), inset -1px -1px 0px var(--border-quiet) !important;
    transform: translateY(1px);
}

/* Primary Action Buttons: Ignite Run & New Session (Portal Purple Bevel) */
.stButton > button[kind="primary"]:not(section[data-testid="stSidebar"] div[data-testid="column"] .stButton > button),
div[data-testid="stFormSubmitButton"] > button {
    background: var(--portal) !important;
    border: 2px solid var(--portal-bright) !important;
    box-shadow: inset -2px -2px 0px var(--portal-dark), inset 2px 2px 0px #C77DFF !important;
    color: #FFFFFF !important;
    font-family: var(--pixel) !important;
    font-size: 11px !important;
    padding: 10px 14px !important;
    letter-spacing: 0.05em !important;
    text-shadow: 1px 1px 0px var(--portal-dark) !important;
    transition: none !important;
}

.stButton > button[kind="primary"]:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--portal-bright) !important;
    border-color: #C77DFF !important;
    box-shadow: inset -2px -2px 0px var(--portal-dark), inset 2px 2px 0px #FFFFFF !important;
}

.stButton > button[kind="primary"]:active, div[data-testid="stFormSubmitButton"] > button:active {
    box-shadow: inset 2px 2px 0px var(--portal-dark), inset -2px -2px 0px var(--portal-bright) !important;
    transform: translateY(1px);
}

/* =========================================================================
   9. STAT GRID & ACTIVE FILE STRIP
   ========================================================================= */
.active-audio-badge {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 600;
    color: var(--teal);
    background: var(--teal-bg);
    border: 1px solid var(--teal-border);
    box-shadow: inset 1px 1px 0px #041204;
    padding: 5px 8px;
    margin-bottom: 6px;
    letter-spacing: 0.02em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.ender-stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin: 6px 0 8px 0;
}

.ender-stat-box {
    background: var(--surface);
    border: 1px solid var(--border-quiet);
    box-shadow: inset 1px 1px 0px var(--border-subtle), inset -1px -1px 0px var(--shadow-dark);
    padding: 6px 8px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.stat-label {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    color: var(--cream) !important;
    letter-spacing: 0.02em;
    margin-bottom: 2px;
}

.stat-value {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    color: var(--body-text) !important;
}

/* =========================================================================
   10. TABS (Neutral Inactive, Purple Active Accent)
   ========================================================================= */
/* Tab List Bar: BaseWeb [data-baseweb="tab-list"] & Native div[role="tablist"], [data-testid="stTabs"] */
div[role="tablist"],
div[data-testid="stTabs"] > div:first-child,
div[data-baseweb="tab-list"],
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    border-bottom: 1px solid var(--border-quiet) !important;
    background: transparent !important;
}

/* Inactive Tab Button */
button[role="tab"],
button[data-testid="stTab"],
div[role="tab"],
.stTabs [data-baseweb="tab"],
div[data-baseweb="tab"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    border-bottom: none !important;
    border-radius: 0px !important;
    color: var(--cream) !important;
    padding: 7px 12px !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}

button[role="tab"] p,
button[role="tab"] span:not([data-testid="stIconMaterial"]):not([class*="material"]),
button[data-testid="stTab"] p,
button[data-testid="stTab"] span:not([data-testid="stIconMaterial"]):not([class*="material"]),
div[role="tab"] p,
.stTabs [data-baseweb="tab"] p {
    font-family: var(--pixel) !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    line-height: 1.3 !important;
    margin: 0 !important;
    color: var(--cream) !important;
}

/* Hover Tab */
button[role="tab"]:hover,
button[data-testid="stTab"]:hover,
div[role="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover {
    color: #FFFFFF !important;
    border-color: var(--portal) !important;
    background-color: var(--surface-hover) !important;
}

button[role="tab"]:hover p,
button[role="tab"]:hover span:not([data-testid="stIconMaterial"]):not([class*="material"]),
button[data-testid="stTab"]:hover p,
button[data-testid="stTab"]:hover span:not([data-testid="stIconMaterial"]):not([class*="material"]),
.stTabs [data-baseweb="tab"]:hover p {
    color: #FFFFFF !important;
}

/* Active / Selected Tab */
button[role="tab"][aria-selected="true"],
button[data-testid="stTab"][aria-selected="true"],
div[role="tab"][aria-selected="true"],
.stTabs [aria-selected="true"] {
    background-color: #1F142E !important;
    border-color: var(--portal) !important;
    border-top: 2px solid var(--amethyst) !important;
    border-bottom: 1px solid #1F142E !important;
    margin-bottom: -1px !important;
    color: #FFFFFF !important;
}

button[role="tab"][aria-selected="true"] p,
button[role="tab"][aria-selected="true"] span:not([data-testid="stIconMaterial"]):not([class*="material"]),
button[data-testid="stTab"][aria-selected="true"] p,
button[data-testid="stTab"][aria-selected="true"] span:not([data-testid="stIconMaterial"]):not([class*="material"]),
.stTabs [aria-selected="true"] p {
    color: #FFFFFF !important;
}

/* BaseWeb tab highlight (older Streamlit) */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--portal) !important;
    height: 3px !important;
}

/* Modern Streamlit highlight bar reset */
button[role="tab"]::after,
button[data-testid="stTab"]::after {
    background-color: transparent !important;
    display: none !important;
}

/* =========================================================================
   11. RADIOS: Teal Indicator for Truth & State
   ========================================================================= */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] [data-baseweb="radio"],
div[data-testid="stRadio"] [data-baseweb="radio"] p,
div[data-testid="stRadio"] [data-baseweb="radio"] span,
div[data-testid="stRadio"] [data-baseweb="radio"] div:last-child {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--cream) !important;
    cursor: pointer !important;
    line-height: 1.4 !important;
}

div[data-testid="stRadio"] [data-baseweb="radio"] {
    align-items: center !important;
    gap: 0px !important;
    padding: 2px 4px !important;
}

/* 16px square indicator: dark inset when unselected */
div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
    border-radius: 0px !important;
    width: 16px !important;
    height: 16px !important;
    min-width: 16px !important;
    min-height: 16px !important;
    max-width: 16px !important;
    max-height: 16px !important;
    background-color: var(--inset) !important;
    border: 1px solid var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px var(--shadow-dark) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-right: 8px !important;
    padding: 0 !important;
    box-sizing: border-box !important;
    flex-shrink: 0 !important;
}

/* Unselected: hide inner indicator */
div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child > div {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Selected: Teal indicator (State & Truth) */
div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) > div:first-child,
div[data-testid="stRadio"] [data-baseweb="radio"][aria-checked="true"] > div:first-child {
    border-color: var(--teal) !important;
    background-color: var(--inset) !important;
    box-shadow: inset 1px 1px 0px var(--shadow-dark) !important;
}

div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) > div:first-child > div,
div[data-testid="stRadio"] [data-baseweb="radio"][aria-checked="true"] > div:first-child > div {
    display: block !important;
    width: 8px !important;
    height: 8px !important;
    min-width: 8px !important;
    min-height: 8px !important;
    max-width: 8px !important;
    max-height: 8px !important;
    border-radius: 0px !important;
    background-color: var(--teal) !important;
    box-shadow: 0 0 4px rgba(61, 255, 196, 0.5) !important;
    border: none !important;
    margin: 0 !important;
}

/* Selected text brightness */
div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) p,
div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) span,
div[data-testid="stRadio"] [data-baseweb="radio"][aria-checked="true"] p,
div[data-testid="stRadio"] [data-baseweb="radio"][aria-checked="true"] span {
    color: var(--body-text) !important;
    font-weight: 600 !important;
}

div[data-testid="stRadio"] input:focus-visible + div,
div[data-testid="stRadio"] input:focus-visible ~ div:first-child,
div[data-testid="stRadio"] [data-baseweb="radio"]:focus-within > div:first-child {
    outline: 2px solid var(--portal) !important;
    outline-offset: 2px !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
}

/* =========================================================================
   12. JUKEBOX AUDIO PLAYER PANEL
   ========================================================================= */
.jukebox-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px var(--border-subtle), inset -1px -1px 0px var(--shadow-dark) !important;
    padding: 8px 10px;
    margin: 6px 0;
}

.jukebox-header {
    font-family: var(--pixel);
    font-size: 11px;
    color: var(--cream);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
    letter-spacing: 0.05em;
}

audio {
    color-scheme: dark !important;
    accent-color: var(--portal) !important;
    filter: hue-rotate(265deg) saturate(1.4) contrast(1.1) !important;
    width: 100% !important;
    height: 36px !important;
    background: var(--void) !important;
    border: 1px solid var(--border-quiet) !important;
    border-radius: 0px !important;
}

/* =========================================================================
   13. RUNNING SPINNER & STALE ELEMENT OVERRIDE
   ========================================================================= */
div[data-testid="stSpinner"] {
    background: var(--surface) !important;
    border: 1px solid var(--portal) !important;
    box-shadow: inset 1px 1px 0px var(--shadow-dark), 0 0 8px rgba(139, 61, 255, 0.25) !important;
    padding: 8px 12px !important;
    margin: 8px 0 !important;
}

div[data-testid="stSpinner"] > div {
    border-top-color: var(--portal) !important;
    border-right-color: var(--teal) !important;
    border-bottom-color: var(--border-quiet) !important;
    border-left-color: var(--border-quiet) !important;
}

div[data-testid="stSpinner"] p {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--cream) !important;
}

[data-stale="true"],
div[data-testid="stAppViewContainer"] [data-stale="true"] {
    opacity: 0.6 !important;
}

/* =========================================================================
   14. PROFILE & STANDBY CARDS
   ========================================================================= */
.profile-card {
    background-color: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px var(--border-subtle), inset -1px -1px 0px var(--shadow-dark) !important;
    padding: 10px 14px;
    margin-top: 6px;
    margin-bottom: 8px;
}

.profile-card.dormant {
    border: 1px dashed var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px var(--shadow-dark) !important;
}

.profile-header {
    font-family: var(--pixel);
    font-size: 11px;
    color: var(--cream);
    text-shadow: 1px 1px 0px var(--shadow-dark);
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    text-align: center;
}

.profile-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin-bottom: 8px;
}

.profile-stat-box {
    background: var(--surface);
    border: 1px solid var(--border-quiet);
    box-shadow: inset 1px 1px 0px var(--border-subtle), inset -1px -1px 0px var(--shadow-dark);
    padding: 6px 8px;
    display: flex;
    flex-direction: column;
}

.profile-label {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    color: var(--cream) !important;
    margin-bottom: 2px;
}

.profile-val {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 700;
    color: var(--body-text);
}

.profile-hint {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    margin-top: 1px;
}

.profile-footer {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--cream);
    text-align: center;
    border-top: 1px solid var(--border-subtle);
    padding-top: 6px;
}

/* =========================================================================
   15. GROUNDING XP BAR & VERIFIED CHIPS (Teal for State & Truth)
   ========================================================================= */
.xp-bar-container {
    font-family: var(--pixel) !important;
    font-size: 11px !important;
    letter-spacing: 0.05em !important;
    padding: 7px 12px !important;
    margin-top: 6px !important;
    margin-bottom: 4px !important;
    text-align: center !important;
    display: block !important;
}

.xp-bar-container.verified {
    background: var(--teal-bg) !important;
    border: 1px solid var(--teal-border) !important;
    color: var(--teal) !important;
    box-shadow: inset 1px 1px 0px #081C16, inset -1px -1px 0px #2E8B75 !important;
}

.xp-bar-container.repaired {
    background: var(--warning-bg) !important;
    border: 1px solid var(--warning-border) !important;
    color: var(--warning) !important;
    box-shadow: inset 1px 1px 0px #120A00, inset -1px -1px 0px #8C5200 !important;
}

.xp-bar-container.abstain {
    background: var(--redstone-bg) !important;
    border: 1px solid var(--redstone-border) !important;
    color: var(--redstone) !important;
    box-shadow: inset 1px 1px 0px #120404, inset -1px -1px 0px #8B2E2E !important;
}

.verification-plain-line {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--cream) !important;
    padding: 2px 4px 6px 4px !important;
    letter-spacing: 0.02em !important;
}

.spec-legend-box {
    background: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px var(--shadow-dark) !important;
    padding: 6px 10px !important;
    margin-top: 4px !important;
    margin-bottom: 6px !important;
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 12px !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--cream) !important;
}

.spec-legend-title {
    font-family: var(--pixel) !important;
    font-size: 10px !important;
    color: var(--cream) !important;
    letter-spacing: 0.04em !important;
}

.legend-line {
    font-family: var(--mono) !important;
    font-weight: 700 !important;
}

.legend-line.amethyst-dash {
    color: var(--cream) !important;
}

.legend-line.teal-dash {
    color: var(--teal) !important;
}

.grounded-answer-box {
    background-color: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px var(--border-subtle), inset -1px -1px 0px var(--shadow-dark) !important;
    border-radius: 0px !important;
    padding: 12px 14px;
    margin-bottom: 8px;
    color: var(--body-text) !important;
    font-family: var(--mono);
    font-size: 12px;
    line-height: 1.6;
    max-height: 220px;
    overflow-y: auto;
}

.grounded-answer-box.verified {
    border-color: var(--teal-border) !important;
    box-shadow: inset 1px 1px 0px #081C16, inset -1px -1px 0px var(--shadow-dark) !important;
}

.grounded-answer-box.repaired {
    border-color: var(--warning-border) !important;
    box-shadow: inset 1px 1px 0px #120A00, inset -1px -1px 0px var(--shadow-dark) !important;
}

.grounded-answer-box.abstain {
    border-color: var(--redstone-border) !important;
    box-shadow: inset 1px 1px 0px #120404, inset -1px -1px 0px var(--shadow-dark) !important;
}

/* Citations as glowing Ender-Eye Teal voxel chips */
.citation-verified {
    background: var(--teal-bg) !important;
    color: var(--teal) !important;
    border: 1px solid var(--teal-border) !important;
    box-shadow: inset 1px 1px 0px #081C16, inset -1px -1px 0px #2E8B75 !important;
    padding: 1px 6px !important;
    border-radius: 0px !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    display: inline-block;
    margin: 1px 2px;
}
.citation-verified::before {
    content: "[ ";
    opacity: 0.85;
}
.citation-verified::after {
    content: " ]";
    opacity: 0.85;
}

.citation-repaired {
    background: var(--warning-bg) !important;
    color: var(--warning) !important;
    border: 1px solid var(--warning-border) !important;
    box-shadow: inset 1px 1px 0px #120A00, inset -1px -1px 0px #8C5200 !important;
    padding: 1px 6px !important;
    border-radius: 0px !important;
    font-family: var(--mono) !important;
    font-weight: 700 !important;
    display: inline-block;
    margin: 1px 2px;
}
.citation-repaired::before {
    content: "[ ⚔ ";
    opacity: 0.85;
}
.citation-repaired::after {
    content: " ]";
    opacity: 0.85;
}

.tool-tile {
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border-quiet);
    box-shadow: inset 1px 1px 0px var(--shadow-dark);
    color: var(--amethyst);
    font-family: var(--mono);
    font-weight: 600;
    font-size: 11px;
    padding: 3px 6px;
    margin: 0 4px 4px 0;
    border-radius: 0px !important;
}

/* =========================================================================
   16. SIDEBAR: Same Dark Void Background, Zero Purple Blowout
   ========================================================================= */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar) !important;
    border-right: 1px solid var(--border-quiet) !important;
    box-shadow: none !important;
    overflow-x: hidden !important;
}

section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] .block-container,
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    overflow-x: hidden !important;
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
}

section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    align-items: center !important;
    gap: 4px !important;
    border: none !important;
    box-shadow: none !important;
    margin-bottom: 2px !important;
    max-width: 100% !important;
    min-width: 0 !important;
    overflow-x: hidden !important;
}

section[data-testid="stSidebar"] [data-testid="column"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-width: 0 !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

section[data-testid="stSidebar"] .stButton,
section[data-testid="stSidebar"] .stDownloadButton {
    min-width: 0 !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}

/* Sidebar Top Actions: Stacked Full-Width Rows (New Session & Export Report) */
section[data-testid="stSidebar"] .stButton:not(div[data-testid="column"] .stButton) > button,
section[data-testid="stSidebar"] .stDownloadButton > button {
    min-height: 38px !important;
    height: auto !important;
    max-height: none !important;
    padding: 0 12px !important;
    font-family: var(--pixel) !important;
    font-size: 11px !important;
    letter-spacing: 0.03em !important;
    white-space: nowrap !important;
    overflow-x: hidden !important;
    overflow-y: visible !important;
    text-overflow: ellipsis !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    box-sizing: border-box !important;
    line-height: 1.5 !important;
    margin-bottom: 6px !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

section[data-testid="stSidebar"] .stButton:not(div[data-testid="column"] .stButton) > button [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] .stDownloadButton > button [data-testid="stMarkdownContainer"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1.5 !important;
    overflow-y: visible !important;
    max-width: 100% !important;
}

section[data-testid="stSidebar"] .stButton:not(div[data-testid="column"] .stButton) > button p,
section[data-testid="stSidebar"] .stButton:not(div[data-testid="column"] .stButton) > button span,
section[data-testid="stSidebar"] .stDownloadButton > button p,
section[data-testid="stSidebar"] .stDownloadButton > button span {
    font-family: var(--pixel) !important;
    font-size: 11px !important;
    letter-spacing: 0.03em !important;
    white-space: nowrap !important;
    overflow-x: hidden !important;
    overflow-y: visible !important;
    text-overflow: ellipsis !important;
    margin: 0 !important;
    padding: 2px 0 !important;
    line-height: 1.5 !important;
}

section[data-testid="stSidebar"] .stDownloadButton > button:disabled {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
    background-color: var(--surface) !important;
    border-color: var(--border-quiet) !important;
    color: var(--muted) !important;
}

/* Past Sessions Column Row Container: Zero padding, tight vertical flow */
section[data-testid="stSidebar"] div[data-testid="column"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    min-width: 0 !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
    margin-bottom: 4px !important;
}

section[data-testid="stSidebar"] div[data-testid="column"] [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* Base Session Row & Del Buttons: Matching Two-Line min-height (52px), auto height, JetBrains Mono */
section[data-testid="stSidebar"] div[data-testid="column"] .stButton > button {
    min-height: 52px !important;
    height: auto !important;
    max-height: none !important;
    padding: 8px 10px !important;
    font-family: var(--mono) !important;
    border-radius: 0px !important;
    box-sizing: border-box !important;
    letter-spacing: 0 !important;
    margin-bottom: 0 !important;
    overflow: visible !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

/* Title Button (col_title / first-child): Left-aligned two-line column flexbox */
section[data-testid="stSidebar"] div[data-testid="column"]:first-child .stButton > button {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    justify-content: center !important;
    text-align: left !important;
    width: 100% !important;
    min-width: 0 !important;
    overflow: visible !important;
}

section[data-testid="stSidebar"] div[data-testid="column"]:first-child .stButton > button [data-testid="stMarkdownContainer"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
    justify-content: center !important;
    text-align: left !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    overflow: visible !important;
}

/* First line: Title (12px mono, line-height 1.4, left-aligned, single line ellipsis) */
section[data-testid="stSidebar"] div[data-testid="column"]:first-child .stButton > button [data-testid="stMarkdownContainer"] p:first-child {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    line-height: 1.4 !important;
    color: var(--cream) !important;
    text-align: left !important;
    display: block !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin: 0 !important;
    padding: 0 !important;
    text-transform: none !important;
}

/* Second line: Relative Time (11px mono, line-height 1.3, muted color, left-aligned) */
section[data-testid="stSidebar"] div[data-testid="column"]:first-child .stButton > button [data-testid="stMarkdownContainer"] p:nth-child(2),
section[data-testid="stSidebar"] div[data-testid="column"]:first-child .stButton > button [data-testid="stMarkdownContainer"] p:last-child:not(:first-child) {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    line-height: 1.3 !important;
    color: var(--muted) !important;
    text-align: left !important;
    display: block !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin: 2px 0 0 0 !important;
    padding: 0 !important;
    text-transform: none !important;
}

/* Active Session Row: 3px teal left border, teal color on title, font stays JetBrains Mono 12px regular */
section[data-testid="stSidebar"] div[data-testid="column"]:has(.active-sess-indicator) .stButton > button {
    border-left: 3px solid var(--teal) !important;
    background-color: var(--surface-hover) !important;
}

section[data-testid="stSidebar"] div[data-testid="column"]:has(.active-sess-indicator) .stButton > button [data-testid="stMarkdownContainer"] p:first-child {
    color: var(--teal) !important;
}

/* Active indicator element hiding: takes zero space in DOM */
div[data-testid="element-container"]:has(.active-sess-indicator),
.active-sess-indicator {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    line-height: 0 !important;
    font-size: 0 !important;
}

/* Del Button (col_del / last-child): Center aligned, JetBrains Mono 11px, equal matching 52px min-height */
section[data-testid="stSidebar"] div[data-testid="column"]:last-child .stButton > button {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    min-height: 52px !important;
    height: auto !important;
    max-height: none !important;
    padding: 8px 6px !important;
    width: 100% !important;
    min-width: 0 !important;
    letter-spacing: 0 !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    overflow: visible !important;
    scrollbar-width: none !important;
    text-transform: none !important;
}

section[data-testid="stSidebar"] div[data-testid="column"]:last-child .stButton > button [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] div[data-testid="column"]:last-child .stButton > button p,
section[data-testid="stSidebar"] div[data-testid="column"]:last-child .stButton > button span {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    text-align: center !important;
    line-height: 1.2 !important;
    margin: 0 !important;
    padding: 0 !important;
    text-transform: none !important;
    overflow: visible !important;
}

/* Confirm Del Button (Redstone Danger State): JetBrains Mono 11px, regular weight */
section[data-testid="stSidebar"] div[data-testid="column"]:last-child .stButton > button[kind="primary"] {
    background: var(--redstone) !important;
    border: 2px solid var(--redstone-bright) !important;
    box-shadow: inset -2px -2px 0px var(--redstone-dark), inset 2px 2px 0px #FFAAAA !important;
    color: #FFFFFF !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    text-shadow: none !important;
}

section[data-testid="stSidebar"] div[data-testid="column"]:last-child .stButton > button[kind="primary"] p,
section[data-testid="stSidebar"] div[data-testid="column"]:last-child .stButton > button[kind="primary"] span {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    color: #FFFFFF !important;
}

/* Global Safety Net: Hide internal scrollbars inside any sidebar button */
section[data-testid="stSidebar"] .stButton button,
section[data-testid="stSidebar"] .stButton button *,
section[data-testid="stSidebar"] .stDownloadButton button,
section[data-testid="stSidebar"] .stDownloadButton button * {
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}

section[data-testid="stSidebar"] .stButton button::-webkit-scrollbar,
section[data-testid="stSidebar"] .stButton button *::-webkit-scrollbar,
section[data-testid="stSidebar"] .stDownloadButton button::-webkit-scrollbar,
section[data-testid="stSidebar"] .stDownloadButton button *::-webkit-scrollbar {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
}

/* Cloud Key Badges */
.cloud-key-badge {
    display: block;
    width: 100%;
    padding: 6px 10px;
    margin: 4px 0 8px 0;
    box-sizing: border-box;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.03em;
    border-radius: 0px;
}
.cloud-key-badge.cloud-key-connected {
    background-color: #0D221C;
    border: 1px solid #1A5E4E;
    color: var(--teal);
    box-shadow: inset 1px 1px 0px rgba(61, 255, 196, 0.1);
}
.cloud-key-badge.cloud-key-missing {
    background-color: #1A1215;
    border: 1px solid #5A2630;
    color: var(--redstone);
    box-shadow: inset 1px 1px 0px rgba(255, 77, 109, 0.1);
}
.cloud-key-badge code {
    font-family: var(--mono);
    color: inherit;
    background: transparent;
    padding: 0;
}

/* Alerts restyled to neutral Ender palette */
.stAlert, [data-testid="stAlert"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    color: var(--body-text) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    border-radius: 0px !important;
}
.stAlert [data-testid="stMarkdownContainer"] p,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
.stAlert [data-testid="stMarkdownContainer"] li,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] li {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: var(--body-text) !important;
    line-height: 1.5 !important;
}
.stAlert svg, [data-testid="stAlert"] svg {
    color: var(--amethyst) !important;
    fill: var(--amethyst) !important;
}

/* Sidebar Link Button (GitHub repo button in expander) */
section[data-testid="stSidebar"] .stLinkButton,
section[data-testid="stSidebar"] [data-testid="stLinkButton"] {
    width: 100% !important;
    margin: 8px 0 !important;
}
section[data-testid="stSidebar"] .stLinkButton > a,
section[data-testid="stSidebar"] [data-testid="stLinkButton"] > a {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    min-height: 36px !important;
    box-sizing: border-box !important;
    background-color: var(--surface) !important;
    border: 1px solid var(--border-quiet) !important;
    box-shadow: inset 1px 1px 0px #2D243A, inset -1px -1px 0px var(--shadow-dark) !important;
    color: var(--cream) !important;
    text-decoration: none !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    padding: 8px 10px !important;
    text-align: center !important;
    border-radius: 0px !important;
}
section[data-testid="stSidebar"] .stLinkButton > a:hover,
section[data-testid="stSidebar"] [data-testid="stLinkButton"] > a:hover {
    background-color: var(--surface-hover) !important;
    border-color: var(--portal) !important;
    color: #FFFFFF !important;
    text-decoration: none !important;
}

/* Sidebar Captions */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] .stCaption {
    color: var(--cream) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    line-height: 1.4 !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb { background: var(--border-quiet); border: 1px solid var(--void); }
::-webkit-scrollbar-thumb:hover { background: var(--portal); }
hr { border-color: var(--border-quiet) !important; margin: 6px 0 !important; }
</style>
"""


def apply_minecraft_theme() -> None:
    """Inject the Ender-Acoustic Console CSS."""
    try:
        st.markdown(ENDER_CSS, unsafe_allow_html=True)
    except Exception:
        pass


def tool_tiles_html(tool_names: List[str]) -> str:
    """Render dispatched tools as amethyst inventory-style tiles."""
    return "".join(f"<span class='tool-tile'>⛏ {html.escape(str(n))}</span>" for n in tool_names)


def compute_initial_acoustic_profile(waveform: np.ndarray, sr: int) -> Dict[str, Any]:
    """Compute instantaneous baseline stats (Centroid, Flatness, Perceptual Band)."""
    import librosa
    try:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform)))
        rms = float(np.sqrt(np.mean(waveform**2) + 1e-12))
        rms_dbfs = 20.0 * np.log10(rms) if rms > 0 else -100.0

        if centroid < 1000:
            band = "Bass-Heavy (<1 kHz)"
        elif centroid < 3000:
            band = "Mid-Range / Core (1-3 kHz)"
        elif centroid < 6000:
            band = "Treble-Dominant (3-6 kHz)"
        else:
            band = "Air & Ultra-High (>6 kHz)"

        texture = "Noise-Like Texture" if flatness > 0.08 else "Tonal / Harmonic"
        return {
            "centroid_hz": centroid,
            "flatness": flatness,
            "rms_dbfs": rms_dbfs,
            "perceptual_band": band,
            "texture": texture,
        }
    except Exception:
        return {
            "centroid_hz": 0.0,
            "flatness": 0.0,
            "rms_dbfs": -100.0,
            "perceptual_band": "Audible Range",
            "texture": "Continuous",
        }


def generate_interactive_spectrogram(waveform: np.ndarray, sr: int) -> go.Figure:
    """
    Interactive Plotly Spectrogram with logarithmic frequency scale.
    Colormap: Void (#0D0912) -> Deep Purple (#4B1E78) -> Magenta (#9D2EC5) -> Cream (#DDDFA5).
    Hot end is cream, not teal or white.
    """
    import librosa

    n_fft = 1024
    hop_length = 256
    stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    mag_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    times = librosa.frames_to_time(np.arange(mag_db.shape[1]), sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    min_freq = 40.0
    freq_mask = freqs >= min_freq
    freqs_log = freqs[freq_mask]
    mag_db_log = mag_db[freq_mask, :]

    max_frames = 600
    if mag_db_log.shape[1] > max_frames:
        step = int(np.ceil(mag_db_log.shape[1] / max_frames))
        mag_db_log = mag_db_log[:, ::step]
        times = times[::step]

    # Role-Based Colormap: Void -> Deep Purple -> Magenta -> Cream (hot end)
    ender_colorscale = [
        [0.0, "#0D0912"],
        [0.35, "#4B1E78"],
        [0.70, "#9D2EC5"],
        [1.0, "#DDDFA5"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=mag_db_log,
            x=times,
            y=freqs_log,
            colorscale=ender_colorscale,
            colorbar=dict(
                title=dict(text="dBFS", font=dict(family="JetBrains Mono", size=11, color="#DDDFA5")),
                tickfont=dict(family="JetBrains Mono", size=10, color="#DDDFA5"),
                thickness=12,
                len=0.88,
            ),
            hovertemplate="Time: %{x:.2f}s | Freq: %{y:.0f} Hz | Magnitude: %{z:.1f} dBFS<extra></extra>",
        )
    )

    nyquist = sr / 2
    raw_ticks = [100, 250, 500, 1000, 2000, 4000, 8000]
    tick_vals = [t for t in raw_ticks if min_freq <= t < nyquist]
    tick_text = [f"{t//1000}k" if t >= 1000 else str(t) for t in tick_vals]

    # Reference acoustic threshold lines (src/tools/spectral.py)
    if nyquist > 1500:
        fig.add_hline(
            y=1500,
            line_dash="dash",
            line_color="rgba(221, 223, 165, 0.85)",  # Cream
            line_width=1.5,
            annotation_text="1.5 kHz (Midrange Anchor)",
            annotation_position="top left",
            annotation_font=dict(family="JetBrains Mono", size=10, color="#DDDFA5"),
        )
    if nyquist > 4000:
        fig.add_hline(
            y=4000,
            line_dash="dash",
            line_color="rgba(61, 255, 196, 0.85)",   # Teal
            line_width=1.5,
            annotation_text="4.0 kHz (Presence/Treble Ceiling)",
            annotation_position="top left",
            annotation_font=dict(family="JetBrains Mono", size=10, color="#3DFFC4"),
        )

    fig.update_layout(
        height=310,
        margin=dict(l=48, r=10, t=15, b=30),
        paper_bgcolor="#151019",
        plot_bgcolor="#0D0912",
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(family="JetBrains Mono", size=11, color="#DDDFA5")),
            tickfont=dict(family="JetBrains Mono", size=10, color="#DDDFA5"),
            gridcolor="#1E1728",
            zerolinecolor="#1E1728",
        ),
        yaxis=dict(
            type="log",
            range=[np.log10(min_freq), np.log10(nyquist)],
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            title=dict(text="Freq (Hz - Log)", font=dict(family="JetBrains Mono", size=11, color="#DDDFA5")),
            tickfont=dict(family="JetBrains Mono", size=10, color="#DDDFA5"),
            gridcolor="#1E1728",
            zerolinecolor="#1E1728",
        ),
    )
    return fig


def generate_interactive_waveform(waveform: np.ndarray, sr: int, zoom_ms: Optional[float] = 20.0) -> go.Figure:
    """
    Interactive Plotly Waveform plot with amplitude and time hover tooltips.
    Waveform line: Teal (#3DFFC4) for state and truth.
    """
    if zoom_ms is not None and zoom_ms > 0:
        num_pts = min(len(waveform), int(sr * (zoom_ms / 1000.0)))
        sub_wf = waveform[:num_pts]
        times = np.linspace(0, (num_pts / sr) * 1000.0, len(sub_wf))
        x_title = f"Time (ms) - Zoomed {zoom_ms:.0f} ms Window"
        hover_tmpl = "Time: %{x:.2f} ms<br>Amplitude: %{y:.3f}<extra></extra>"
    else:
        max_pts = 1200
        if len(waveform) > max_pts:
            step = int(np.ceil(len(waveform) / max_pts))
            sub_wf = waveform[::step]
            times = np.linspace(0, len(waveform) / sr, len(sub_wf))
        else:
            sub_wf = waveform
            times = np.linspace(0, len(waveform) / sr, len(sub_wf))
        x_title = "Time (s) - Full Signal Overview"
        hover_tmpl = "Time: %{x:.3f} s<br>Amplitude: %{y:.3f}<extra></extra>"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=sub_wf,
            mode="lines",
            line=dict(color="#3DFFC4", width=1.5),  # Teal for state/truth
            fill="tozeroy",
            fillcolor="rgba(61, 255, 196, 0.15)",
            hovertemplate=hover_tmpl,
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=48, r=10, t=15, b=30),
        paper_bgcolor="#151019",
        plot_bgcolor="#0D0912",
        xaxis=dict(
            title=dict(text=x_title, font=dict(family="JetBrains Mono", size=11, color="#DDDFA5")),
            tickfont=dict(family="JetBrains Mono", size=10, color="#DDDFA5"),
            gridcolor="#1E1728",
            zerolinecolor="#1E1728",
        ),
        yaxis=dict(
            title=dict(text="Amplitude", font=dict(family="JetBrains Mono", size=11, color="#DDDFA5")),
            tickfont=dict(family="JetBrains Mono", size=10, color="#DDDFA5"),
            gridcolor="#1E1728",
            zerolinecolor="#1E1728",
            range=[-1.05, 1.05],
        ),
    )
    return fig


def render_grounded_answer_html(content: str, turn: Optional[object] = None) -> str:
    """Format grounded answer with bolding, code, and glowing citation chips via single-pass regex."""
    safe = html.escape(content)

    if turn and hasattr(turn, "grounding_report") and turn.grounding_report:
        rep = turn.grounding_report
        token_map: Dict[str, str] = {}

        # 1. Collect verified citations (Teal)
        if hasattr(rep, "verified_citations") and rep.verified_citations:
            for cit in rep.verified_citations:
                token = getattr(cit, "raw_token", str(cit))
                if token:
                    s_tok = str(token).strip()
                    if s_tok:
                        token_map[s_tok] = "citation-verified"

        # 2. Collect repaired / hallucinated numbers (Warning Amber)
        if hasattr(rep, "hallucinated_numbers") and rep.hallucinated_numbers:
            for num in rep.hallucinated_numbers:
                token = getattr(num, "raw_token", str(num))
                if token:
                    s_tok = str(token).strip()
                    if s_tok and s_tok not in token_map:
                        token_map[s_tok] = "citation-repaired"

        if token_map:
            sorted_tokens = sorted(token_map.keys(), key=len, reverse=True)
            escaped_tokens = [re.escape(t) for t in sorted_tokens]
            pattern_str = r"(?<![\d.])(" + "|".join(escaped_tokens) + r")(?![\d])"
            pattern = re.compile(pattern_str)

            def _replace_chip(m: re.Match) -> str:
                matched = m.group(1)
                cls_name = token_map.get(matched, "citation-verified")
                return f"<span class='{cls_name}'>{html.escape(matched)}</span>"

            safe = pattern.sub(_replace_chip, safe)

    # Convert bold **text** to <strong>text</strong>
    safe = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", safe)
    # Convert `code` to <code>code</code>
    safe = re.sub(r"`(.*?)`", r"<code>\1</code>", safe)
    # Convert newlines to <br>
    safe = safe.replace("\n", "<br>")

    return safe
