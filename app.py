import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import base64, os, time

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== SPLASH SCREEN =====================
if 'splash_done' not in st.session_state:
    st.session_state.splash_done = False

if not st.session_state.splash_done:
    splash_placeholder = st.empty()
    with splash_placeholder.container():
        st.markdown("""
        <style>
        .splash-overlay {
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background: radial-gradient(ellipse at 50% 40%, #020614 0%, #000308 100%);
            z-index: 9999; display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            animation: splashFadeOut 0.8s ease 3.5s forwards;
        }
        @keyframes splashFadeOut {
            to { opacity: 0; pointer-events: none; }
        }
        .splash-grid {
            position: absolute; inset: 0;
            background-image:
                linear-gradient(rgba(0,229,255,0.07) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,229,255,0.07) 1px, transparent 1px);
            background-size: 60px 60px;
            animation: gridMove 8s linear infinite;
        }
        @keyframes gridMove {
            0% { transform: perspective(500px) rotateX(20deg) translateY(0); }
            100% { transform: perspective(500px) rotateX(20deg) translateY(60px); }
        }
        .splash-title {
            font-family: 'Orbitron', monospace;
            font-size: clamp(2.5rem, 6vw, 5rem);
            font-weight: 900;
            color: #fff;
            text-align: center;
            letter-spacing: 6px;
            text-shadow:
                0 0 20px rgba(0,229,255,0.9),
                0 0 60px rgba(0,229,255,0.5),
                0 0 120px rgba(0,102,255,0.3);
            animation: titleReveal 1s ease 0.3s both;
            z-index: 2; position: relative;
        }
        @keyframes titleReveal {
            from { opacity: 0; transform: translateY(30px) scale(0.95); letter-spacing: 20px; }
            to   { opacity: 1; transform: translateY(0) scale(1); letter-spacing: 6px; }
        }
        .splash-sub {
            font-family: 'Orbitron', monospace;
            font-size: clamp(0.7rem, 1.5vw, 1rem);
            color: #00e5ff;
            letter-spacing: 5px;
            text-align: center;
            margin-top: 1rem;
            text-transform: uppercase;
            animation: fadeIn 1s ease 1s both;
            z-index: 2; position: relative;
        }
        .splash-line {
            width: 300px; height: 2px; margin-top: 2rem;
            background: linear-gradient(90deg, transparent, #00e5ff, #a855f7, transparent);
            box-shadow: 0 0 20px rgba(0,229,255,0.6);
            animation: lineExpand 1s ease 0.8s both;
            z-index: 2; position: relative;
        }
        @keyframes lineExpand {
            from { width: 0; opacity: 0; }
            to   { width: 300px; opacity: 1; }
        }
        .splash-stats {
            display: flex; gap: 3rem; margin-top: 2.5rem;
            animation: fadeIn 1s ease 1.5s both;
            z-index: 2; position: relative;
        }
        .splash-stat {
            text-align: center;
        }
        .splash-stat-val {
            font-family: 'Orbitron', monospace;
            font-size: 1.6rem; font-weight: 700;
            color: #00e5ff;
            text-shadow: 0 0 15px rgba(0,229,255,0.7);
        }
        .splash-stat-lbl {
            font-size: 0.68rem; color: #8a9bb8;
            letter-spacing: 2px; text-transform: uppercase;
            margin-top: 4px;
        }
        .splash-loading {
            margin-top: 3rem;
            font-family: 'Orbitron', monospace;
            font-size: 0.75rem; color: #8a9bb8;
            letter-spacing: 3px;
            animation: blink 1.2s infinite;
            z-index: 2; position: relative;
        }
        @keyframes blink {
            0%,100% { opacity: 0.4; } 50% { opacity: 1; }
        }
        .splash-bar-wrap {
            width: 260px; height: 3px;
            background: rgba(0,229,255,0.1);
            border-radius: 3px; margin-top: 1rem;
            overflow: hidden; position: relative; z-index: 2;
        }
        .splash-bar {
            height: 100%; width: 0%;
            background: linear-gradient(90deg, #0066ff, #00e5ff);
            border-radius: 3px;
            box-shadow: 0 0 10px rgba(0,229,255,0.5);
            animation: loadBar 3s ease forwards;
        }
        @keyframes loadBar {
            0%   { width: 0%; }
            30%  { width: 45%; }
            70%  { width: 78%; }
            100% { width: 100%; }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        </style>
        <div class="splash-overlay">
            <div class="splash-grid"></div>
            <div class="splash-title">🔒 FraudShield AI</div>
            <div class="splash-sub">Advanced Fraud Detection Platform</div>
            <div class="splash-line"></div>
            <div class="splash-stats">
                <div class="splash-stat">
                    <div class="splash-stat-val">6.3M+</div>
                    <div class="splash-stat-lbl">Transactions</div>
                </div>
                <div class="splash-stat">
                    <div class="splash-stat-val">99.99%</div>
                    <div class="splash-stat-lbl">AUC-ROC</div>
                </div>
                <div class="splash-stat">
                    <div class="splash-stat-val">6</div>
                    <div class="splash-stat-lbl">ML Models</div>
                </div>
            </div>
            <div class="splash-bar-wrap"><div class="splash-bar"></div></div>
            <div class="splash-loading">INITIALIZING SYSTEMS...</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(4.3)
    splash_placeholder.empty()
    st.session_state.splash_done = True
    st.rerun()
# ===================== VIDEO BACKGROUND =====================
def get_video_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# Try multiple video names for compatibility
video_b64 = None
for vid_name in ["fraudshield_background.mp4", "background_blurred.mp4", "background.mp4"]:
    video_b64 = get_video_base64(vid_name)
    if video_b64:
        break

video_html = ""
if video_b64:
    video_html = f"""
    <video autoplay muted loop playsinline
           style="position:fixed;top:0;left:0;width:100%;height:100%;
                  object-fit:cover;z-index:-3;opacity:0.8;">
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
    </video>
    <div style="position:fixed;top:0;left:0;width:100%;height:100%;
                background: linear-gradient(180deg, rgba(2,6,20,0.8) 0%, rgba(2,6,20,0.4) 100%);
                z-index:-2;"></div>
    <div style="position:fixed;top:0;left:0;width:100%;height:100%;
                background: radial-gradient(ellipse at 50% 0%, rgba(59,130,246,0.08) 0%, transparent 60%),
                            radial-gradient(ellipse at 80% 100%, rgba(124,58,237,0.08) 0%, transparent 50%);
                z-index:-1; pointer-events:none;"></div>
    """

# ===================== CSS + NEON EFFECTS =====================
st.markdown(video_html + """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    background: transparent !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: transparent !important;
}

/* App container readability */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: -2;
    pointer-events: none;
    background: repeating-linear-gradient(
        0deg,
        rgba(0,0,0,0.03) 0px,
        rgba(0,0,0,0.03) 1px,
        transparent 1px,
        transparent 4px
    );
}

section[data-testid="stSidebar"] {
    background: rgba(5, 8, 20, 0.75) !important;
    border-right: 1px solid rgba(0, 212, 255, 0.35);
    backdrop-filter: blur(24px) saturate(1.2) !important;
}

/* ── Neon Title ── */
.neon-title {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    color: #ffffff;
    text-shadow:
        0 0 10px rgba(59, 130, 246, 0.8),
        0 0 30px rgba(59, 130, 246, 0.5),
        0 0 60px rgba(37, 99, 235, 0.4),
        0 0 100px rgba(37, 99, 235, 0.2);
    animation: flicker 4s infinite alternate;
    margin-bottom: 0.2rem;
    letter-spacing: 2px;
}
.neon-sub {
    font-family: 'Orbitron', monospace;
    font-size: 0.9rem;
    text-align: center;
    color: #00e5ff;
    letter-spacing: 4px;
    text-transform: uppercase;
    text-shadow:
        0 0 10px rgba(0, 229, 255, 0.6),
        0 0 20px rgba(0, 229, 255, 0.3);
    margin-bottom: 1.5rem;
    font-weight: 600;
}

@keyframes flicker {
    0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {
        text-shadow:
            0 0 10px rgba(59, 130, 246, 0.9),
            0 0 30px rgba(59, 130, 246, 0.6),
            0 0 60px rgba(37, 99, 235, 0.4),
            0 0 100px rgba(37, 99, 235, 0.2);
        color: #ffffff;
    }
    20%, 24%, 55% {
        text-shadow: 0 0 5px rgba(59, 130, 246, 0.4);
        color: rgba(255,255,255,0.85);
    }
}

/* ── Neon Divider ── */
.neon-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #00e5ff, #a855f7, #00e5ff, transparent);
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.5), 0 0 30px rgba(168, 85, 247, 0.3);
    margin: 1rem 0 1.5rem 0;
    border: none;
    border-radius: 2px;
}

/* ── Cards (Glassmorphism v2) ── */
.glass-card {
    background: rgba(5, 8, 22, 0.65) !important;
    border: 1px solid rgba(0, 229, 255, 0.25);
    border-radius: 16px;
    padding: 1.4rem;
    backdrop-filter: blur(20px) saturate(1.3) !important;
    -webkit-backdrop-filter: blur(20px) saturate(1.3) !important;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.4),
        0 0 20px rgba(0, 229, 255, 0.08),
        inset 0 1px 0 rgba(255, 255, 255, 0.06);
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.5), transparent);
}

/* ── Metric Cards ── */
.metric-row { display:flex; gap:12px; margin-bottom:1rem; }
.metric-box {
    flex:1;
    background: rgba(5, 8, 22, 0.75) !important;
    border: 1px solid rgba(0, 229, 255, 0.3);
    border-radius: 14px;
    padding: 18px 12px;
    text-align: center;
    backdrop-filter: blur(16px) !important;
    box-shadow:
        0 4px 20px rgba(0, 0, 0, 0.3),
        0 0 15px rgba(0, 229, 255, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transition: all 0.3s ease;
}
.metric-box:hover {
    transform: translateY(-3px);
    box-shadow:
        0 8px 30px rgba(0, 0, 0, 0.4),
        0 0 25px rgba(0, 229, 255, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.08);
    border-color: rgba(0, 229, 255, 0.5);
}
.metric-val {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00e5ff;
    text-shadow: 0 0 15px rgba(0, 229, 255, 0.6), 0 0 30px rgba(0, 229, 255, 0.2);
}
.metric-lbl {
    font-size: 0.74rem;
    color: #a0b0d0;
    margin-top: 6px;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* ── ANALYZE Button ── */
.stButton > button {
    width: 100% !important;
    height: 60px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    background: linear-gradient(135deg, #0066ff, #7c3aed) !important;
    border: 1px solid rgba(0, 229, 255, 0.5) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    box-shadow:
        0 0 25px rgba(0, 102, 255, 0.4),
        0 0 50px rgba(124, 58, 237, 0.25),
        inset 0 1px 0 rgba(255,255,255,0.1) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    position: relative;
    overflow: hidden;
}
.stButton > button::after {
    content: "";
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: linear-gradient(
        45deg,
        transparent 40%,
        rgba(255,255,255,0.08) 50%,
        transparent 60%
    );
    animation: shimmer 3s infinite;
}
@keyframes shimmer {
    0% { transform: translateX(-100%) rotate(0deg); }
    100% { transform: translateX(100%) rotate(0deg); }
}
.stButton > button:hover {
    box-shadow:
        0 0 35px rgba(0, 102, 255, 0.7),
        0 0 70px rgba(124, 58, 237, 0.4) !important;
    transform: translateY(-2px) !important;
    border-color: rgba(0, 229, 255, 0.8) !important;
}
.stButton > button:active { transform: scale(0.97) !important; }

/* ── Input fields ── */
.stNumberInput input, .stSelectbox select, .stTextInput input {
    background: rgba(5, 8, 22, 0.8) !important;
    border: 1px solid rgba(0, 229, 255, 0.35) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-weight: 500 !important;
    box-shadow: 0 0 10px rgba(0, 229, 255, 0.05), inset 0 1px 0 rgba(255,255,255,0.03) !important;
}
.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: rgba(0, 229, 255, 0.7) !important;
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.15) !important;
}
.stSlider > div > div {
    background: rgba(0, 229, 255, 0.3) !important;
}

/* ── Labels ── */
label, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
    color: #ffffff !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.5) !important;
}
stMarkdown h3 {
    color: #00e5ff !important;
    text-shadow: 0 0 10px rgba(0, 229, 255, 0.3) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(5, 8, 22, 0.6);
    border-radius: 12px;
    padding: 5px;
    border: 1px solid rgba(0, 229, 255, 0.15);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem;
    letter-spacing: 1px;
    color: #8a9bb8 !important;
    border-radius: 10px !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #00e5ff !important;
    background: rgba(0, 229, 255, 0.08) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0, 229, 255, 0.15) !important;
    color: #00e5ff !important;
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.15), inset 0 1px 0 rgba(255,255,255,0.06) !important;
}

/* ── Fraud / Safe alert ── */
.fraud-alert {
    background: rgba(20, 2, 5, 0.7) !important;
    border: 1px solid #ff2d55;
    border-radius: 16px;
    padding: 1.2rem 1.8rem;
    text-align: center;
    box-shadow:
        0 0 30px rgba(255, 45, 85, 0.3),
        0 8px 32px rgba(0,0,0,0.3),
        inset 0 1px 0 rgba(255,255,255,0.05);
    animation: pulse-red 2s infinite;
    backdrop-filter: blur(16px) !important;
    position: relative;
    overflow: hidden;
}
.fraud-alert::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #ff2d55, transparent);
}
.safe-alert {
    background: rgba(2, 20, 8, 0.7) !important;
    border: 1px solid #00e676;
    border-radius: 16px;
    padding: 1.2rem 1.8rem;
    text-align: center;
    box-shadow:
        0 0 30px rgba(0, 230, 118, 0.25),
        0 8px 32px rgba(0,0,0,0.3),
        inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(16px) !important;
    position: relative;
    overflow: hidden;
}
.safe-alert::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00e676, transparent);
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 30px rgba(255, 45, 85, 0.3), 0 8px 32px rgba(0,0,0,0.3); }
    50%      { box-shadow: 0 0 50px rgba(255, 45, 85, 0.6), 0 8px 32px rgba(0,0,0,0.3); }
}

/* ── Sidebar ── */
.sidebar-badge {
    background: rgba(0, 229, 255, 0.1);
    border: 1px solid rgba(0, 229, 255, 0.25);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.84rem;
    color: #e0e8ff;
    backdrop-filter: blur(8px);
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    transition: all 0.2s ease;
}
.sidebar-badge:hover {
    border-color: rgba(0, 229, 255, 0.5);
    background: rgba(0, 229, 255, 0.15);
    transform: translateX(3px);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(5, 8, 22, 0.5); }
::-webkit-scrollbar-thumb {
    background: #00e5ff;
    border-radius: 4px;
    box-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
}
::-webkit-scrollbar-thumb:hover { background: #448aff; }

/* ── DataFrame / Table Styling ── */
.stDataFrame, .stTable {
    background: rgba(5, 8, 22, 0.6) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(0, 229, 255, 0.2) !important;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
}
.stDataFrame th, .stTable th {
    background: rgba(0, 102, 255, 0.2) !important;
    color: #00e5ff !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 12px !important;
    border-bottom: 1px solid rgba(0, 229, 255, 0.3) !important;
}
.stDataFrame td, .stTable td {
    color: #e0e8ff !important;
    font-size: 0.85rem !important;
    padding: 10px 12px !important;
    border-bottom: 1px solid rgba(0, 229, 255, 0.08) !important;
}
.stDataFrame tr:hover td {
    background: rgba(0, 229, 255, 0.06) !important;
    color: #ffffff !important;
}
.stDataFrame tbody tr:nth-child(even) td {
    background: rgba(0, 229, 255, 0.03) !important;
}

/* ── Info / Warning / Error boxes ── */
.stAlert {
    background: rgba(5, 8, 22, 0.75) !important;
    border: 1px solid rgba(0, 229, 255, 0.25) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(16px) !important;
    color: #e0e8ff !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
}
.stAlert p { color: #e0e8ff !important; text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important; }

/* ── Spinner ── */
.stSpinner > div > div {
    border-color: #00e5ff transparent transparent transparent !important;
}

/* ── Selectbox dropdown ── */
ul[data-testid="stSelectboxVirtualDropdown"] {
    background: rgba(5, 8, 22, 0.95) !important;
    border: 1px solid rgba(0, 229, 255, 0.3) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(20px) !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5) !important;
}
ul[data-testid="stSelectboxVirtualDropdown"] li {
    color: #e0e8ff !important;
}
ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
    background: rgba(0, 229, 255, 0.15) !important;
    color: #00e5ff !important;
}

/* ── Expander / Collapse ── */
.st-expander {
    background: rgba(5, 8, 22, 0.6) !important;
    border: 1px solid rgba(0, 229, 255, 0.2) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px) !important;
}

/* ── Plotly chart containers ── */
.js-plotly-plot .plotly {
    border-radius: 14px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
}

/* ── Caption ── */
.caption-text {
    text-align: center;
    color: #8a9bb8;
    font-family: 'Orbitron', monospace;
    font-size: 0.78rem;
    letter-spacing: 2px;
    text-shadow: 0 1px 3px rgba(0,0,0,0.6);
}

/* ── Divider line in sidebar ── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.3), transparent);
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ===================== HEADER =====================
st.markdown('<div class="neon-title">🔒 FraudShield AI</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-sub">Advanced Fraud Detection Platform — Powered by Ensemble ML</div>', unsafe_allow_html=True)
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

# ===================== LOAD MODELS =====================
@st.cache_resource
def load_artifacts():
    model   = joblib.load('best_fraud_model.pkl')
    scaler  = joblib.load('scaler.pkl')
    le      = joblib.load('label_encoder.pkl')
    return model, scaler, le

try:
    model, scaler, le = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.sidebar.error(f"❌ Model Load Error: {e}")

# ===================== SIDEBAR =====================
st.sidebar.markdown("## ⚙️ System Status")
if model_loaded:
    st.sidebar.markdown('<div class="sidebar-badge">🟢 &nbsp; AI Model — Online</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-badge">🟢 &nbsp; Scaler — Loaded</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-badge">🟢 &nbsp; Encoder — Ready</div>', unsafe_allow_html=True)
else:
    st.sidebar.error("❌ Models not found. Check .pkl files.")

st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 Model Leaderboard")

model_scores = {
    "🥇 Voting Ensemble": 0.9999,
    "🥈 LightGBM":        0.9996,
    "🥉 XGBoost":         0.9994,
    "   Random Forest":   0.9996,
    "   Deep Neural Net": 0.9996,
    "   Autoencoder":     0.6512,
}
for name, score in model_scores.items():
    color = "#00e676" if score > 0.95 else "#ffaa00"
    st.sidebar.markdown(
        f'<div class="sidebar-badge">{name} &nbsp; '
        f'<span style="color:{color};font-weight:700;text-shadow:0 0 8px {color}40;">{score:.4f}</span></div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")
st.sidebar.markdown("## 📂 Dataset Info")
st.sidebar.markdown('<div class="sidebar-badge">📌 PaySim Mobile Money</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-badge">📊 6.3M Transactions</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-badge">🎯 Fraud Rate: 0.13%</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-badge">🏆 Best AUC: 0.9999</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("🎓 BS CS — 6th Semester ML Project\nFatima | Fraud Detection using Ensemble ML")

# ===================== TABS =====================
tab1, tab2, tab3 = st.tabs([
    "🔍  Transaction Analyzer",
    "📊  Model Performance",
    "🧠  How It Works"
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — TRANSACTION ANALYZER
# ─────────────────────────────────────────────────────────────
with tab1:
    col_input, col_result = st.columns([1.3, 2.7])

    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Enter Transaction Details")

        # Transaction Type
        type_options = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        txn_type = st.selectbox("Transaction Type", type_options, index=4,
                                help="Fraud occurs ONLY in TRANSFER & CASH_OUT")

        amount = st.number_input("💰 Amount (local currency)",
                                 min_value=0.0, value=50000.0, step=1000.0)

        st.markdown("**Sender Account**")
        old_bal_org = st.number_input("Old Balance (Before)", min_value=0.0,
                                       value=100000.0, step=1000.0)
        new_bal_org = st.number_input("New Balance (After)", min_value=0.0,
                                       value=50000.0, step=1000.0)

        st.markdown("**Receiver Account**")
        old_bal_dest = st.number_input("Old Balance (Before)", min_value=0.0,
                                        value=0.0, step=1000.0, key="dest_old")
        new_bal_dest = st.number_input("New Balance (After)", min_value=0.0,
                                        value=50000.0, step=1000.0, key="dest_new")

        step_val = st.slider("⏱ Step (Hour of Day 1-744)", 1, 744, 100)

        st.markdown("</div>", unsafe_allow_html=True)

        analyze_btn = st.button("🔍 ANALYZE TRANSACTION", use_container_width=True)

    # ── RESULT COLUMN ──
    with col_result:
        if analyze_btn and model_loaded:
            with st.spinner("🤖 Running AI Analysis..."):
                time.sleep(0.6)

                # Encode transaction type
                try:
                    type_encoded = le.transform([txn_type])[0]
                except:
                    type_encoded = type_options.index(txn_type)

                # Feature engineering (match training)
                error_bal_orig = new_bal_org + amount - old_bal_org
                error_bal_dest = old_bal_dest + amount - new_bal_dest
                orig_bal_zero  = int(new_bal_org == 0)

                feature_cols = ['step', 'type', 'amount',
                                'oldbalanceOrg', 'newbalanceOrig',
                                'oldbalanceDest', 'newbalanceDest',
                                'errorBalanceOrig', 'errorBalanceDest',
                                'origBalanceZero']

                input_df = pd.DataFrame([{
                    'step':             step_val,
                    'type':             type_encoded,
                    'amount':           amount,
                    'oldbalanceOrg':    old_bal_org,
                    'newbalanceOrig':   new_bal_org,
                    'oldbalanceDest':   old_bal_dest,
                    'newbalanceDest':   new_bal_dest,
                    'errorBalanceOrig': error_bal_orig,
                    'errorBalanceDest': error_bal_dest,
                    'origBalanceZero':  orig_bal_zero,
                }], columns=feature_cols)

                input_scaled = scaler.transform(input_df)

                try:
                    probability = float(model.predict_proba(input_scaled)[0][1])
                except:
                    probability = float(model.predict(input_scaled)[0])

                is_fraud = probability > 0.5

            # ── Alert Banner ──
            if is_fraud:
                st.markdown(
                    '<div class="fraud-alert">'
                    '<h2 style="color:#ff2d55;margin:0;font-family:Orbitron,monospace;font-size:1.8rem;text-shadow:0 0 20px rgba(255,45,85,0.6);">🚨 FRAUD DETECTED</h2>'
                    '<p style="margin:6px 0 0;color:#ffb3c1;font-size:1rem;text-shadow:0 1px 3px rgba(0,0,0,0.5);">Transaction Blocked by AI</p>'
                    '</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="safe-alert">'
                    '<h2 style="color:#00e676;margin:0;font-family:Orbitron,monospace;font-size:1.8rem;text-shadow:0 0 20px rgba(0,230,118,0.5);">✅ TRANSACTION CLEARED</h2>'
                    '<p style="margin:6px 0 0;color:#a8f5c5;font-size:1rem;text-shadow:0 1px 3px rgba(0,0,0,0.5);">Legitimate Transaction Verified</p>'
                    '</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Gauge ──
            result_color = "#ff2d55" if is_fraud else "#00e676"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(probability * 100, 2),
                title={'text': "Fraud Risk Score (%)",
                       'font': {'size': 18, 'color': '#e0e8ff', 'family': 'Orbitron, monospace'}},
                number={'font': {'size': 44, 'color': result_color, 'family': 'Orbitron, monospace'},
                        'suffix': '%'},
                gauge={
                    'axis': {'range': [0, 100],
                             'tickcolor': '#5a6a8a',
                             'tickfont': {'color': '#8a9bb8', 'size': 12}},
                    'bar': {'color': result_color,
                            'thickness': 0.28,
                            'line': {'color': 'rgba(255,255,255,0.3)', 'width': 1}},
                    'bgcolor': 'rgba(5, 8, 22, 0.5)',
                    'borderwidth': 1,
                    'bordercolor': 'rgba(0, 229, 255, 0.2)',
                    'steps': [
                        {'range': [0, 35],  'color': 'rgba(0, 230, 118, 0.15)'},
                        {'range': [35, 65], 'color': 'rgba(255, 170, 0, 0.15)'},
                        {'range': [65, 100],'color': 'rgba(255, 45, 85, 0.15)'},
                    ],
                    'threshold': {
                        'line': {'color': '#ffffff', 'width': 3},
                        'thickness': 0.85,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(
                height=280,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#e0e8ff'},
                margin=dict(t=50, b=10, l=20, r=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Metric Row ──
            m1, m2, m3, m4 = st.columns(4)
            metrics = [
                ("Risk Score",    f"{probability:.2%}"),
                ("Threshold",     "50%"),
                ("Transaction",   txn_type),
                ("Amount",        f"{amount:,.0f}"),
            ]
            for col, (lbl, val) in zip([m1,m2,m3,m4], metrics):
                col.markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-val">{val}</div>'
                    f'<div class="metric-lbl">{lbl}</div>'
                    f'</div>', unsafe_allow_html=True)

            # ── Feature Insights ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🔎 Transaction Feature Insights")
            insights = {
                'Step (Hour)':        step_val,
                'Amount':             amount,
                'Old Balance Sender': old_bal_org,
                'New Balance Sender': new_bal_org,
                'Balance Error Orig': round(error_bal_orig, 2),
                'Balance Error Dest': round(error_bal_dest, 2),
                'Sender Zero Flag':   orig_bal_zero,
            }
            ins_df = pd.DataFrame(insights.items(), columns=['Feature', 'Value'])
            ins_df['Risk Signal'] = ins_df.apply(lambda r:
                '🔴 High' if (r['Feature'] == 'Sender Zero Flag' and r['Value'] == 1)
                else '🟡 Watch' if r['Feature'] in ['Balance Error Orig','Balance Error Dest']
                else '🟢 Normal', axis=1)
            st.dataframe(ins_df, use_container_width=True, hide_index=True)

        elif analyze_btn and not model_loaded:
            st.error("❌ Models not loaded. Please add .pkl files.")

        else:
            # Default — overview cards
            st.markdown("### 📈 System Overview")
            ov_data = [
                ("6.3M+", "Transactions Trained"),
                ("0.9999", "Best AUC-ROC Score"),
                ("6", "Models Compared"),
                ("0.13%", "Fraud Rate in Dataset"),
            ]
            r1, r2 = st.columns(2), st.columns(2)
            for i, (val, lbl) in enumerate(ov_data):
                row = [r1, r2][i // 2]
                row[i % 2].markdown(
                    f'<div class="metric-box">'
                    f'<div class="metric-val">{val}</div>'
                    f'<div class="metric-lbl">{lbl}</div>'
                    f'</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info("👈 **Enter transaction details** on the left and click **ANALYZE TRANSACTION**")

            st.markdown("### 💡 Demo — Try These Values for Fraud Detection")
            st.markdown("""
            | Scenario | Type | Amount | Old Bal | New Bal |
            |----------|------|--------|---------|---------|
            | 🚨 Likely Fraud | TRANSFER | 500,000 | 500,000 | 0 |
            | ✅ Normal | PAYMENT | 1,500 | 10,000 | 8,500 |
            | 🚨 Fraud | CASH_OUT | 200,000 | 200,000 | 0 |
            """)


with tab2:
    st.markdown("### 📊 Model Comparison Dashboard")

    perf_df = pd.DataFrame({
        'Model':           ['Voting Ensemble','LightGBM','XGBoost',
                            'Random Forest','Deep Neural Net','Autoencoder'],
        'AUC-ROC':         [0.9999, 0.9996, 0.9994, 0.9996, 0.9996, 0.6512],
        'Fraud Precision': [0.97,   0.96,   0.95,   0.97,   0.92,   0.04],
        'Fraud Recall':    [0.99,   0.98,   0.97,   0.97,   0.95,   0.05],
        'Type':            ['Ensemble','Gradient Boost','Gradient Boost',
                            'Ensemble','Deep Learning','Anomaly Detection'],
    })

    colors_list = ['#00e5ff','#a855f7','#00e676','#ffaa00','#ff2d55','#448aff']

    c1, c2 = st.columns(2)

    # ── 3D Bar Chart (was: 2D horizontal bar) ──
    with c1:
        fig_3d_bar = go.Figure()
        for i, row in perf_df.iterrows():
            fig_3d_bar.add_trace(go.Mesh3d(
                x=[i,   i+0.6, i+0.6, i,     i,     i+0.6, i+0.6, i    ],
                y=[0,   0,     row['AUC-ROC'], row['AUC-ROC'],
                   0,   0,     row['AUC-ROC'], row['AUC-ROC']],
                z=[0,   0,     0,     0,     0.5,   0.5,   0.5,   0.5  ],
                i=[0,0,0,1,4,4,4,5,0,3,1,2],
                j=[1,2,4,2,5,6,7,6,3,7,5,6],
                k=[2,3,5,3,6,7,3,7,7,4,2,7],
                color=colors_list[i],
                opacity=0.82,
                name=row['Model'],
                showlegend=True,
                lighting=dict(ambient=0.5, diffuse=0.9, specular=0.4, roughness=0.4),
                lightposition=dict(x=100, y=200, z=150)
            ))

        fig_3d_bar.update_layout(
            title=dict(text='AUC-ROC Score by Model',
                       font=dict(color='#00e5ff', family='Orbitron, monospace', size=15)),
            scene=dict(
                xaxis=dict(
                    title='',
                    ticktext=perf_df['Model'].tolist(),
                    tickvals=list(range(len(perf_df))),
                    tickfont=dict(color='#8a9bb8', size=8),
                    gridcolor='rgba(0,229,255,0.1)',
                    backgroundcolor='rgba(5,8,22,0.85)',
                    showbackground=True,
                    zerolinecolor='rgba(0,229,255,0.15)'
                ),
                yaxis=dict(
                    title='AUC-ROC',
                    tickfont=dict(color='#8a9bb8'),
                    gridcolor='rgba(0,229,255,0.1)',
                    backgroundcolor='rgba(5,8,22,0.85)',
                    showbackground=True,
                    range=[0, 1.05]
                ),
                zaxis=dict(
                    title='',
                    tickfont=dict(color='#8a9bb8'),
                    gridcolor='rgba(0,229,255,0.05)',
                    backgroundcolor='rgba(5,8,22,0.85)',
                    showbackground=True,
                ),
                bgcolor='rgba(5,8,22,0.9)',
                camera=dict(eye=dict(x=1.6, y=-1.8, z=1.1))
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            height=420,
            font_color='#e0e8ff',
            legend=dict(
                font=dict(color='#e0e8ff', size=9),
                bgcolor='rgba(5,8,22,0.7)',
                bordercolor='rgba(0,229,255,0.2)',
                borderwidth=1
            ),
            margin=dict(t=50, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_3d_bar, use_container_width=True)

    # ── 3D Radar → 3D Scatter (was: 2D radar chart) ──
    with c2:
        radar_models = {
            'Voting Ensemble': {'AUC-ROC': 1.00, 'Precision': 0.97, 'Recall': 0.99, 'F1 Score': 0.98, 'Speed': 0.60},
            'LightGBM':        {'AUC-ROC': 0.99, 'Precision': 0.96, 'Recall': 0.98, 'F1 Score': 0.97, 'Speed': 0.95},
            'XGBoost':         {'AUC-ROC': 0.99, 'Precision': 0.95, 'Recall': 0.97, 'F1 Score': 0.96, 'Speed': 0.80},
            'Random Forest':   {'AUC-ROC': 0.99, 'Precision': 0.97, 'Recall': 0.97, 'F1 Score': 0.97, 'Speed': 0.55},
            'Deep Neural Net': {'AUC-ROC': 0.99, 'Precision': 0.92, 'Recall': 0.95, 'F1 Score': 0.93, 'Speed': 0.70},
            'Autoencoder':     {'AUC-ROC': 0.65, 'Precision': 0.04, 'Recall': 0.05, 'F1 Score': 0.04, 'Speed': 0.85},
        }
        radar_colors = ['#00ffff','#cc00ff','#00ff88','#ffcc00','#ff3399','#6699ff']
        categories = ['Precision', 'Recall', 'AUC-ROC', 'F1 Score', 'Speed']

        fig_radar = go.Figure()
        for i, (model_name, scores) in enumerate(radar_models.items()):
            vals = [scores[c] for c in categories]
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=cats_closed,
                fill='toself',
                name=model_name,
                line=dict(color=radar_colors[i], width=2),
                fillcolor=radar_colors[i],
                opacity=0.55,
            ))

        fig_radar.update_layout(
            title=dict(
                text='Radar — Multi-Metric Comparison',
                font=dict(color='#00e5ff', family='Orbitron, monospace', size=15),
                x=0.5
            ),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(color='#8a9bb8', size=9),
                    gridcolor='rgba(0,229,255,0.15)',
                    linecolor='rgba(0,229,255,0.2)',
                ),
                angularaxis=dict(
                    tickfont=dict(color='#00e5ff', size=11, family='Orbitron, monospace'),
                    gridcolor='rgba(0,229,255,0.15)',
                    linecolor='rgba(0,229,255,0.2)',
                ),
                bgcolor='rgba(5,8,22,0.6)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e8ff',
            height=420,
            legend=dict(
                font=dict(color='#e0e8ff', size=10),
                bgcolor='rgba(5,8,22,0.7)',
                bordercolor='rgba(0,229,255,0.2)',
                borderwidth=1,
                x=1.05, y=0.5
            ),
            margin=dict(t=60, b=20, l=40, r=120)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── 3D Scatter (was: 2D bubble/scatter) ──
    fig_3d_scatter = go.Figure(data=[go.Scatter3d(
        x=perf_df['Fraud Precision'],
        y=perf_df['Fraud Recall'],
        z=perf_df['AUC-ROC'],
        mode='markers+text',
        text=perf_df['Model'],
        textposition='top center',
        textfont=dict(color='#e0e8ff', size=9, family='Orbitron, monospace'),
        marker=dict(
            size=perf_df['AUC-ROC'] * 18,
            color=perf_df['AUC-ROC'],
            colorscale=[[0,'#1e3a5f'],[0.4,'#0066ff'],[0.8,'#00e5ff'],[1,'#a855f7']],
            opacity=0.88,
            line=dict(color='rgba(255,255,255,0.3)', width=1),
            showscale=True,
            colorbar=dict(
                title=dict(text='AUC-ROC', font=dict(color='#e0e8ff')),
                tickfont=dict(color='#8a9bb8')
            )
        )
    )])

    fig_3d_scatter.update_layout(
        title=dict(
            text='Fraud Precision vs Recall (bubble size = AUC-ROC)',
            font=dict(color='#00e5ff', family='Orbitron, monospace', size=15)
        ),
        scene=dict(
            xaxis=dict(title='Fraud Precision', tickfont=dict(color='#8a9bb8'),
                       gridcolor='rgba(0,229,255,0.1)',
                       backgroundcolor='rgba(5,8,22,0.85)', showbackground=True),
            yaxis=dict(title='Fraud Recall', tickfont=dict(color='#8a9bb8'),
                       gridcolor='rgba(0,229,255,0.1)',
                       backgroundcolor='rgba(5,8,22,0.85)', showbackground=True),
            zaxis=dict(title='AUC-ROC', tickfont=dict(color='#8a9bb8'),
                       gridcolor='rgba(0,229,255,0.1)',
                       backgroundcolor='rgba(5,8,22,0.85)', showbackground=True,
                       range=[0.6, 1.01]),
            bgcolor='rgba(5,8,22,0.9)',
            camera=dict(eye=dict(x=1.4, y=-1.6, z=1.1))
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=480,
        font_color='#e0e8ff',
        margin=dict(t=50, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_3d_scatter, use_container_width=True)

    st.markdown("### 📋 Full Results Table")
    styled_df = perf_df.sort_values('AUC-ROC', ascending=False).reset_index(drop=True)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
# ─────────────────────────────────────────────────────────────
# TAB 3 — HOW IT WORKS
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🧠 How FraudShield Works")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
**🔄 ML Pipeline**

1. 📥 **Data** → PaySim 6.3M transactions
2. 🧹 **Preprocessing** → Label encoding, feature engineering
3. ⚖️ **SMOTE** → Balance fraud/normal classes
4. 🤖 **Training** → 6 models trained & compared
5. 🏆 **Best Model** → Voting Ensemble (AUC: 0.9999)
6. 🎯 **Prediction** → Risk probability score
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
**🔑 Key Insight — Fraud Pattern**

Fraud in PaySim ONLY happens in:
- 🔴 **TRANSFER** transactions
- 🔴 **CASH_OUT** transactions

Fraud signature:
- Sender balance drops to **zero** after transaction
- `errorBalanceOrig` is very high
- Large amount relative to balance
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
**🤖 Models Used**

| Model | Type |
|-------|------|
| 🥇 Voting Ensemble | RF + XGB + LGB combined |
| 🥈 LightGBM | Fast gradient boosting |
| 🥉 XGBoost | Extreme gradient boosting |
| Random Forest | 200 decision trees |
| Deep Neural Net | 4-layer with Dropout |
| Autoencoder | Anomaly detection |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
**📐 Engineered Features**

| Feature | Meaning |
|---------|---------|
| `errorBalanceOrig` | Sender balance mismatch |
| `errorBalanceDest` | Receiver balance mismatch |
| `origBalanceZero` | Sender went to zero? |

These 3 features are the **most powerful predictors** of fraud — confirmed by SHAP analysis!
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p class="caption-text">'
        '🎓 BS COMPUTER SCIENCE — 6TH SEMESTER ML PROJECT<br>'
        'FRAUD DETECTION IN FINANCIAL TRANSACTIONS USING ENSEMBLE ML'
        '</p>', unsafe_allow_html=True)


