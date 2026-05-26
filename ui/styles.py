"""Streamlit CSS for the KAIROS operational dashboard UI."""

from __future__ import annotations


def dashboard_css() -> str:
    """Return the custom dark dashboard stylesheet."""
    return """
    :root {
        --kairos-bg: #070b10;
        --kairos-bg-soft: #0b1118;
        --kairos-card: #111827;
        --kairos-card-2: #0f1722;
        --kairos-border: rgba(148, 163, 184, 0.24);
        --kairos-border-strong: rgba(34, 211, 238, 0.38);
        --kairos-text: #edf5ff;
        --kairos-muted: #94a3b8;
        --kairos-subtle: #64748b;
        --kairos-accent: #22d3ee;
        --kairos-accent-blue: #38bdf8;
        --kairos-success: #34d399;
        --kairos-warning: #fbbf24;
        --kairos-error: #fb7185;
        --kairos-radius: 8px;
    }

    html, body, [class*="css"] {
        background: var(--kairos-bg) !important;
        color: var(--kairos-text);
        font-size: 16px;
        line-height: 1.55;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(34, 211, 238, 0.12), transparent 30rem),
            linear-gradient(135deg, #070b10 0%, #0b1118 48%, #080c12 100%) !important;
    }

    .block-container {
        padding-top: 1rem;
        padding-right: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 1.2rem;
        max-width: 1760px;
    }

    .kairos-shell,
    .agent-panel,
    .workspace-panel {
        color: var(--kairos-text);
    }

    .agent-panel {
        position: sticky;
        top: 1rem;
        padding: 0.05rem 0 1.5rem;
    }

    .workspace-panel {
        padding-bottom: 2rem;
    }

    .kairos-hero,
    .question-card,
    .control-card,
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.94), rgba(15, 23, 34, 0.94)) !important;
        border: 1px solid var(--kairos-border) !important;
        border-radius: var(--kairos-radius) !important;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.035);
    }

    .kairos-hero {
        padding: 1rem 1rem 0.9rem;
        margin-bottom: 0.85rem;
        border-color: var(--kairos-border-strong) !important;
    }

    .kairos-title {
        font-size: clamp(2.2rem, 4vw, 4rem);
        line-height: 0.95;
        margin: 0 0 0.35rem;
        color: var(--kairos-text);
        font-weight: 780;
        letter-spacing: 0;
    }

    .kairos-subtitle {
        font-size: 0.92rem;
        color: var(--kairos-accent);
        margin: 0 0 0.45rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }

    .kairos-description {
        color: #cbd5e1;
        font-size: 0.96rem;
        margin: 0 0 0.7rem;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.35rem;
    }

    .status-badge,
    .metric-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.28rem 0.55rem;
        border: 1px solid rgba(34, 211, 238, 0.32);
        border-radius: 999px;
        color: #dffaff;
        background: rgba(34, 211, 238, 0.08);
        font-size: 0.78rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .status-badge.success {
        border-color: rgba(52, 211, 153, 0.36);
        color: #d1fae5;
        background: rgba(52, 211, 153, 0.08);
    }

    .status-badge.warning {
        border-color: rgba(251, 191, 36, 0.4);
        color: #fde68a;
        background: rgba(251, 191, 36, 0.08);
    }

    .status-badge.muted {
        border-color: rgba(148, 163, 184, 0.22);
        color: #cbd5e1;
        background: rgba(148, 163, 184, 0.08);
    }

    .section-kicker {
        color: var(--kairos-accent);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.35rem;
    }

    .question-card,
    .control-card {
        padding: 0.9rem;
        margin: 0.75rem 0;
    }

    .workspace-title {
        font-size: clamp(1.65rem, 2.4vw, 2.35rem);
        margin: 0.35rem 0 0.8rem;
        color: var(--kairos-text);
        letter-spacing: 0;
    }

    h1, h2, h3, h4,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3 {
        color: var(--kairos-text) !important;
        letter-spacing: 0;
    }

    h3 {
        font-size: 1.25rem;
        margin-top: 1.45rem;
    }

    p, li,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {
        color: #dbe7f3;
        font-size: 0.98rem;
        line-height: 1.58;
    }

    label, .stTextInput label, .stTextArea label, .stFileUploader label, .stNumberInput label {
        font-size: 0.9rem !important;
        font-weight: 720 !important;
        color: #dbe7f3 !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.96), rgba(8, 13, 20, 0.96));
        border: 1px solid var(--kairos-border);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        min-height: 5.4rem;
    }

    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
        color: var(--kairos-muted) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--kairos-text);
        font-size: 1.65rem;
        font-weight: 760;
    }

    div[data-testid="stFileUploader"],
    div[data-testid="stTextArea"],
    div[data-testid="stNumberInput"] {
        color: var(--kairos-text);
    }

    div[data-testid="stFileUploader"] section {
        background: rgba(15, 23, 42, 0.72);
        border: 1px dashed rgba(34, 211, 238, 0.32);
        border-radius: 8px;
    }

    textarea, input {
        background: rgba(8, 13, 20, 0.88) !important;
        color: var(--kairos-text) !important;
        border: 1px solid rgba(148, 163, 184, 0.26) !important;
        border-radius: 8px !important;
    }

    textarea:focus, input:focus {
        border-color: rgba(34, 211, 238, 0.65) !important;
        box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.3) !important;
    }

    div[data-testid="stButton"] button {
        border-radius: 8px;
        border: 1px solid rgba(34, 211, 238, 0.58);
        background: linear-gradient(135deg, #0891b2, #2563eb);
        color: #f8fbff;
        font-size: 0.95rem;
        font-weight: 760;
        padding: 0.68rem 1rem;
        box-shadow: 0 12px 36px rgba(37, 99, 235, 0.28);
    }

    div[data-testid="stButton"] button:hover {
        border-color: rgba(125, 211, 252, 0.9);
        filter: brightness(1.06);
    }

    div[data-testid="stExpander"] {
        background: rgba(15, 23, 34, 0.78) !important;
        border: 1px solid var(--kairos-border) !important;
        border-radius: 8px !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: rgba(15, 23, 42, 0.86);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--kairos-border);
        border-radius: 8px;
        overflow: hidden;
    }

    .pill {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        margin: 0.16rem 0.2rem 0.16rem 0;
        border: 1px solid rgba(34, 211, 238, 0.28);
        border-radius: 999px;
        background: rgba(34, 211, 238, 0.08);
        color: #dffaff;
        font-size: 0.78rem;
        font-weight: 650;
    }

    code, pre {
        background: rgba(8, 13, 20, 0.9) !important;
        color: #dffaff !important;
        border-radius: 8px !important;
    }
    """
