import os
import subprocess
import sys
import html
import logging
import warnings
import tempfile

# Always define project root
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add project root to path when running directly
if __package__ is None or __package__ == '':
    sys.path.insert(0, _project_root)
#=======================================
#=======================================
#=======================================

# Suppress Streamlit warnings
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*to view this Streamlit app.*')

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import streamlit as st
import threading
from queue import Queue

# ════════════════════════════════════════════════════════════════════════════════
#                              PAGE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CoLLM • ML Toolbox",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import project modules AFTER page config to avoid Streamlit issues
from source.utils.requirements_check import ensure_packages 
from source.runs.run_preselection_GUI import run_LLM
LLM_RUNNER_AVAILABLE = True

# ════════════════════════════════════════════════════════════════════════════════
#                              CUSTOM CSS STYLING
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&family=Sora:wght@300;400;500;600&display=swap');
    
    /* ═══════════════════ ROOT VARIABLES ═══════════════════ */
    :root {
        --bg-primary: #121220;
        --bg-secondary: #1a1a2e;
        --bg-card: #1e1e30;
        --bg-hover: #282840;
        --accent-primary: #6366f1;
        --accent-secondary: #8b5cf6;
        --accent-tertiary: #a855f7;
        --accent-glow: rgba(99, 102, 241, 0.3);
        --text-primary: #f1f5f9;
        --text-secondary: #fff;
        --text-title: #94a3b8;
        --text-muted: #64748b;
        --border-color: #3a3a4a;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --gradient-1: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        --gradient-2: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    }
    
    /* ═══════════════════ GLOBAL STYLES ═══════════════════ */
    .stApp {
        background: var(--bg-primary);
        font-family: 'Sora', sans-serif;
    }
    
    .stApp > header {
        background: transparent;
    }
    
    /* ═══════════════════ SIDEBAR STYLING ═══════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #121220 100%);
        border-right: 1px solid var(--border-color);
        width: 340px !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: var(--text-secondary);
    }
    
    /* ═══════════════════ MAIN CONTENT ═══════════════════ */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* ═══════════════════ TYPOGRAPHY ═══════════════════ */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    p, span, label, div {
        color: var(--text-secondary);
    }
    
    /* ═══════════════════ CUSTOM HERO SECTION ═══════════════════ */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-1);
    }
    
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-family: 'Sora', sans-serif;
        font-size: 1.1rem;
        color: var(--text-title);
        max-width: 700px;
        line-height: 1.7;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: var(--accent-primary);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    /* ═══════════════════ SECTION HEADERS ═══════════════════ */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .section-icon {
        width: 44px;
        height: 44px;
        background: var(--gradient-1);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
    }
    
    .section-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .section-desc {
        font-size: 0.9rem;
        color: var(--text-muted);
        margin: 0;
    }
    
    /* ═══════════════════ CARD COMPONENTS ═══════════════════ */
    .custom-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        border-color: var(--accent-primary);
        box-shadow: 0 0 30px var(--accent-glow);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-color);
    }
    
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    /* ═══════════════════ INPUT STYLING ═══════════════════ */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }
    
    .stTextInput label,
    .stNumberInput label,
    .stTextArea label,
    .stSelectbox label,
    .stRadio label {
        font-family: 'Sora', sans-serif !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
    }
    
    /* ═══════════════════ SELECT BOX STYLING ═══════════════════ */
    .stSelectbox > div > div {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important;
    }
    
    /* ═══════════════════ BUTTON STYLING ═══════════════════ */
    .stButton > button {
        background: var(--gradient-1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 1.5rem !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px var(--accent-glow) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px var(--accent-glow) !important;
    }
    
    /* ═══════════════════ RADIO BUTTONS ═══════════════════ */
    .stRadio > div {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stRadio > div > div > label {
        color: var(--text-secondary) !important;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stRadio > div > div > label:hover {
        background: var(--bg-hover);
    }
    
    /* ═══════════════════ CHECKBOX STYLING ═══════════════════ */
    .stCheckbox > label {
        color: var(--text-secondary) !important;
    }
    
    /* ═══════════════════ EXPANDER STYLING ═══════════════════ */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* ═══════════════════ TABS STYLING ═══════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 24px;
        padding: 18px;
        gap: 20px;
        border: 2px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 18px !important;
        color: var(--text-secondary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        padding: 32px 60px !important;
        transition: all 0.2s ease !important;
        min-height: 90px !important;

    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-hover) !important;
        color: var(--text-primary) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--gradient-1) !important;
        color: white !important;
    }
    
    
    /* ═══════════════════ PROGRESS BAR ═══════════════════ */
    .stProgress > div > div > div {
        background: var(--gradient-1) !important;
    }
    
    /* ═══════════════════ SUCCESS/ERROR MESSAGES ═══════════════════ */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 10px !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 10px !important;
    }
    
    /* ═══════════════════ DIVIDER ═══════════════════ */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2rem 0;
    }
    
    /* ═══════════════════ METRIC CARDS ═══════════════════ */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        background: var(--gradient-1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }
    
    /* ═══════════════════ AUTHOR CARDS ═══════════════════ */
    .author-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    
    .author-card:hover {
        border-color: var(--accent-primary);
    }
    
    .author-name {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.95rem;
    }
    
    .author-email {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    
    /* ═══════════════════ SCROLLBAR ═══════════════════ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-primary);
    }
    
    /* ═══════════════════ ANIMATION KEYFRAMES ═══════════════════ */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes terminalPulse {
        0%, 100% { 
            box-shadow: 0 0 5px rgba(99, 102, 241, 0.3);
        }
        50% { 
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.6), 0 0 30px rgba(168, 85, 247, 0.3);
        }
    }
    
    @keyframes cursorBlink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .terminal-output {
        animation: terminalPulse 1.5s ease-in-out infinite;
        border: 1px solid var(--accent-primary) !important;
        border-radius: 10px;
    }
    
    .terminal-cursor {
        display: inline-block;
        width: 8px;
        height: 16px;
        background: var(--accent-primary);
        animation: cursorBlink 1s step-end infinite;
        margin-left: 2px;
        vertical-align: middle;
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* ═══════════════════ TOOLTIP STYLING ═══════════════════ */
    .stTooltipIcon {
        color: var(--text-muted) !important;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#                              SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo placeholder
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
        <div style="
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            border-radius: 20px;
            margin: 0 auto 1rem auto;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
            box-shadow: 0 10px 40px rgba(99, 102, 241, 0.3);
        ">🔬</div>
        <h2 style="
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        ">CoLLM</h2>
        <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.3rem;">ML Toolbox for Collider Physics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Authors Section
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <p style="
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.75rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.75rem;
        ">👥 AUTHORS</p>
    </div>
    """, unsafe_allow_html=True)
    
    authors = [
        {"name": "Ahmed Hammad", "email": "ahammad115566@gmail.com"},
        {"name": "Waleed Esamil", "email": "waleed.physics@gmail.com"},
        {"name": "Mihoko Nojiri", "email": "mihoko.nojiri@gmail.com"}
    ]
    
    for author in authors:
        st.markdown(f"""
        <div class="author-card">
            <div class="author-name">{author['name']}</div>
            <div class="author-email">📧 {author['email']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # User Manual Section
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <p style="
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.75rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.75rem;
        ">📖 USER MANUAL</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🚀 Getting Started", expanded=False):
        st.markdown("""
        <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.6;">
            <p><strong>1.</strong> Upload your signal and background files</p>
            <p><strong>2.</strong> Configure preselection cuts using natural language</p>
            <p><strong>3.</strong> Select and configure your ML model</p>
            <p><strong>4.</strong> Run training and view results</p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("🔧 Configuration Tips", expanded=False):
        st.markdown("""
        <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.6;">
            <p>• Start with <strong>2-3 hidden layers</strong> for MLPs</p>
            <p>• Use <strong>ReLU</strong> activation for hidden layers</p>
            <p>• Begin with a learning rate of <strong>0.001</strong></p>
            <p>• Set early stopping patience to <strong>5-10 epochs</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📊 Supported Formats", expanded=False):
        st.markdown("""
        <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.6;">
            <p>• <code>.root</code> - ROOT files</p>
            <p>• <code>.h5</code> - HDF5 files</p>
            <p>• <code>.csv</code> - CSV files</p>
            <p>• <code>.parquet</code> - Parquet files</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Version info
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <p style="color: #475569; font-size: 0.75rem; font-family: 'JetBrains Mono', monospace;">
            v1.0.0 • Built with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#                              MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════════

# Hero Section
st.markdown("""
<div class="hero-container animate-fade-in">
    <div class="">
    </div>
    <h1 class="hero-title">CoLLM</h1>
    <p class="hero-subtitle">
        A next-generation automated machine learning toolbox designed for high-energy physics 
        and collider analyses. Leverage the power of large language models to generate analysis 
        code and train sophisticated deep learning models with an intuitive interface.
    </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#                              MAIN TABS
# ════════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "🎯 Preselection Analysis",
    "🧠 Deep Learning",
    "📊 Results"
])

# ════════════════════════════════════════════════════════════════════════════════
#                          TAB 1: PRESELECTION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

 #   st.markdown("""
 #   <div class="section-header">
 #    </div>
#    """, unsafe_allow_html=True)
#    
#    # Signal Files Section
#    with st.container():
#        col1, col2 = st.columns([1, 1])
#        
#        with col1:
#            st.markdown("""
#            <div class="custom-card">
#                <div class="card-header">
#                    <span style="font-size: 1.2rem;">⚡</span>
#                    <h4 class="card-title">Signal Files</h4>
#                </div>
#            """, unsafe_allow_html=True)
 #           
 #           num_sig = st.number_input(
 #               label="Number of signal types",
#                min_value=1,
#                max_value=100,
#                step=1,
#                value=1,
#                key="num_signals",
#                help="Specify the number of different signal samples"
#            )
 #           
 #           st.markdown("</div>", unsafe_allow_html=True)
 #           
 #           sig_dirs = ['sig_' + str(i) for i in range(int(num_sig))]
 #           sigma_sig = []
 #           
 #           for i, item in enumerate(sig_dirs):
 #               with st.expander(f"📂 Signal {i+1} Configuration", expanded=(i==0)):
 #                   sig_dirs[i] = st.text_input(
 #                       label=f"Path to signal-{i+1} directory",
 #                       placeholder=f"/path/to/signal_{i+1}/files",
 #                       key=item
 #                   )
 #                   sigma_1 = st.number_input(
 #                       label=f"Cross section (pb)",
 #                       min_value=1.0e-08,
 #                       step=1.0e-08,
 #                       format="%.8f",
 #                       value=1.0,
 #                       key=f"sigma_sig_{i}",
 #                       help="Cross section value in picobarns"
 #                   )
 #                   sigma_sig.append(sigma_1)
 #       
 #       with col2:
 #           st.markdown("""
 #           <div class="custom-card">
 #               <div class="card-header">
 #                   <span style="font-size: 1.2rem;">🌫️</span>
 #                   <h4 class="card-title">Background Files</h4>
 #               </div>
 #           """, unsafe_allow_html=True)
 #           
 #           num_bkg = st.number_input(
  #              label="Number of background types",
 #               min_value=1,
 #               max_value=100,
 #               step=1,
 #               value=1,
 #               key="num_backgrounds",
 #               help="Specify the number of different background samples"
 #           )
  #          
 #           st.markdown("</div>", unsafe_allow_html=True)
 #           
 #           bkg_dirs = ['bkg_' + str(i) for i in range(int(num_bkg))]
#            sigma_bkg = []
            
#            for i, item in enumerate(bkg_dirs):
#                with st.expander(f"📂 Background {i+1} Configuration", expanded=(i==0)):
#                    bkg_dirs[i] = st.text_input(
#                        label=f"Path to background-{i+1} directory",
 #                       placeholder=f"/path/to/background_{i+1}/files",
 #                       key=item
 #                   )
 #                   sigma_ = st.number_input(
 #                       label=f"Cross section (pb)",
 #                       min_value=1.0e-08,
 #                       step=1.0e-08,
 #                       format="%.8f",
 #                       value=1.0,
 #                       key=f"sigma_bkg_{i}",
 #                       help="Cross section value in picobarns"
 #                   )
 #                   sigma_bkg.append(sigma_)
    
    # File Validation
  #  st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
 #   col_val1, col_val2, col_val3 = st.columns([1, 1, 2])
 #   with col_val1:
 #       check_files = st.button(" Validate Files", use_container_width=True)
 #   
 #   if check_files:
#        with st.spinner("Validating file paths..."):
#            time.sleep(0.5)
#            all_valid = True
#            
#            for path_ in sig_dirs:
#                if path_ and not os.path.exists(path_):
#                    st.error(f" Signal path not found: `{path_}`")
#                    all_valid = False
#                elif path_:
#                    st.success(f" Signal files verified: `{path_}`")
 #           
#            for path_ in bkg_dirs:
 #               if path_ and not os.path.exists(path_):
  #                  st.error(f" Background path not found: `{path_}`")
 #                   all_valid = False
 #               elif path_:
 #                   st.success(f" Background files verified: `{path_}`")
with tab1:    
    # LLM Code Generation Section
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🤖</div>
        <div>
            <h3 class="section-title">LLM  Analysis Generation</h3>
            <p class="section-desc">Describe your analysis in natural language</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    HEADERS = [
        "### SELECTION CUTS",
        "### PLOTS FOR VALIDATION",
        "### OUTPUT STRUCTURE"
    ]
    
    template = "\n\n".join([h + "\n" for h in HEADERS])
    
    col_llm1, col_llm2 = st.columns([2, 1])
    
    with col_llm1:
        text = st.text_area(
            "Analysis Specification",
            value=template,
            height=350,
            key="preselection_analysis_specification",
            help="Describe your analysis cuts, validation plots, and output format"
        )
        
        # Restore missing headers
        for h in HEADERS:
            if h not in text:
                text = h + "\n\n" + text
    
    with col_llm2:
        st.markdown("""
        <div class="custom-card" style="height: 100%;">
            <div class="card-header">
                <span style="font-size: 1.2rem;">💡</span>
                <h4 class="card-title">Example Template</h4>
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; line-height: 1.7; font-family: 'JetBrains Mono', monospace;">
                <p><strong style="color: #a855f7;">### SELECTION CUTS</strong></p>
                <p>• Leptons: PT > 20 GeV, |Eta| < 2.4</p>
                <p>• At least two leptons</p>
                <p>• Require at least two b jets</p>
                <br>
                <p><strong style="color: #a855f7;">### PLOTS FOR VALIDATION</strong></p>
                <p>• Dielectron mass: 60 bins, 60–120 GeV</p>
                <p>• Plot the missing energy distribution</p>
                <br>
                <p><strong style="color: #a855f7;">### OUTPUT STRUCTURE</strong></p>
                <p>• Save plots as PNG (150 dpi)</p>
                <p>• print summary statistics </p>
                 <p>• save the following  in a single  csv file for MLP analysis: </p>
                 <p>1- pt of the leading letpton </p>
                 <p>2- pt of the leading jet </p>
                 <p>3- delta R between the leading  and subleading b jet </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # LLM Configuration Section
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">⚙️</div>
        <div>
            <h3 class="section-title">LLM Configuration</h3>
            <p class="section-desc">Configure the language model and execution settings</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_cfg1, col_cfg2 = st.columns(2)
    
    with col_cfg1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span style="font-size: 1.2rem;">📂</span>
                <h4 class="card-title">Paths Configuration</h4>
            </div>
        """, unsafe_allow_html=True)
        
        output_dir = st.text_input(
            "Output Directory (Please enter the full path)",
            value="Enter path to the ouput directory",
            help="Directory where generated analysis and plots will be saved"
        )
        
        input_file = st.text_input(
            "Input LHCO File (Please enter the full path)",
            value="data/signal_1.lhco",
            help="Path to the LHCO file for testing the generated analysis"
        )
        
       # user_input = st.text_input(
        #    "User Input Template",
         #   value="/Users/hammad/work/CoLLM/templates/user_input_1.txt",
          #  help="Path to save the user input template"
       # )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col_cfg2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span style="font-size: 1.2rem;">🤖</span>
                <h4 class="card-title">Model Settings</h4>
            </div>
        """, unsafe_allow_html=True)
        
        default_model = st.selectbox(
            "LLM Model",
            options=[
          "Qwen/Qwen2.5-Coder-7B-Instruct",      # Best balance of speed/quality
          "Qwen/Qwen2.5-Coder-32B-Instruct",     # Higher quality
          "Qwen/Qwen3-Coder-30B-A3B-Instruct",   # Latest MoE coder
         # General purpose (Good at code too)
         "meta-llama/Llama-3.1-8B-Instruct",
         "meta-llama/Llama-3.3-70B-Instruct",
         "Qwen/Qwen2.5-72B-Instruct",
   
        # Lightweight/Fast options
        "Qwen/Qwen2.5-Coder-3B-Instruct",
       "meta-llama/Llama-3.2-3B-Instruct",
       # Reasoning models (good for complex code)
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
         "Qwen/QwQ-32B",         
            ],
            index=4,
            help="Select the Hugging Face model for code generation"
        )
        
        max_retries = st.number_input(
            "Max Retries",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum number of attempts to fix generated code"
        )
        
        use_api = st.checkbox(
            "Use Hugging Face API",
            value=False,
            help="Use Hugging Face Inference API instead of local model. If not, LLM will be downloaded and decoded locally."
        )
        
        api_key = st.text_input(
            "API Key",
            value="",
            type="password",
            help="Your Hugging Face API key (required if using API)",
            disabled=not use_api
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col_run1, col_run2, col_run3 = st.columns([1, 1, 2])
    
    with col_run1:
        run_analysis = st.button("🚀 Run Preselection Analysis", use_container_width=True)
    
    #with col_run2:
     #   save_config = st.button("💾 Save Configuration", use_container_width=True)
    
     
    if run_analysis:
        # Convert text format from GUI to expected format
        user_input_text = text.replace("### SELECTION CUTS", "[SELECTION_CUTS]")
        user_input_text = user_input_text.replace("### PLOTS FOR VALIDATION", "[PLOTS_FOR_VALIDATION]")
        user_input_text = user_input_text.replace("### OUTPUT STRUCTURE", "[OUTPUT_STRUCTURE]")
        
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📟</div>
            <div>
                <h3 class="section-title">Terminal Output</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a placeholder for terminal output
        terminal_output = st.empty()
        output_messages = []
        
        def update_terminal(message: str, status_type: str = 'info'):
            """Callback to update terminal output."""
            output_messages.append(message)
            output_text = '\n'.join(output_messages)
            escaped_output = html.escape(output_text)
            
            # Color based on status type
            color_map = {
                'info': '#10b981',
                'success': '#22c55e', 
                'warning': '#f59e0b',
                'error': '#ef4444'
            }
            color = color_map.get(status_type, '#10b981')
            
            terminal_output.markdown(f"""
            <div class="terminal-output" style="
                background: #0a0a12;
                padding: 1rem;
                border-radius: 10px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                max-height: 400px;
                overflow-y: auto;
            ">
                <pre style="margin: 0; color: {color}; white-space: pre-wrap;">{escaped_output}<span class="terminal-cursor"></span></pre>
            </div>
            """, unsafe_allow_html=True)
        
        with st.spinner("🔄 Running analysis pipeline..."):
            status_placeholder = st.empty()
            
            try:
                # Create a temporary Python script to run the analysis
                
                script_content = f'''
import sys
import os
sys.path.insert(0, {repr(_project_root)})
os.chdir({repr(_project_root)})

from source.runs.run_preselection_GUI import run_LLM

run_LLM(
    {repr(output_dir)},
    {repr(default_model)},
    {repr(input_file)},
    {repr(user_input_text)},
    {repr(output_dir + "generated_lhco_analysis.py")},
    {max_retries},
    {use_api},
    {repr(api_key)}
)
'''
                
                # Write the script to a temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(script_content)
                    temp_script = f.name
                
                # Run the script via subprocess and capture output in real-time
                process = subprocess.Popen(
                    [sys.executable, '-u', temp_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=_project_root,
                    env={**os.environ, 'PYTHONUNBUFFERED': '1'}
                )
                
                # Stream output line by line
                for line in iter(process.stdout.readline, ''):
                    if line:
                        line = line.rstrip('\n')
                        update_terminal(line, 'info')
                
                process.wait()
                
                # Clean up temp file
                try:
                    os.unlink(temp_script)
                except:
                    pass
                
                if process.returncode == 0:
                    update_terminal("Analysis completed successfully!", "success")
                    status_placeholder.success("Analysis completed successfully!")
                else:
                    update_terminal(f"Process exited with code {process.returncode}", "error")
                    status_placeholder.error("Analysis failed")
                    
            except Exception as e:
                status_placeholder.error(f"❌ Error occurred: {e}")
                st.error(f"Error running analysis: {e}")
                import traceback
                st.code(traceback.format_exc(), language="bash")
            
            # Display generated files if output directory exists
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                if files:
                    st.markdown("### 📁 Generated Files")
                    for f in files:
                        file_path = os.path.join(output_dir, f)
                        if f.endswith('.png'):
                            st.image(file_path, caption=f)
                        elif f.endswith('.py'):
                            st.markdown(f"📄 **{f}**")
                            with st.expander("View generated code"):
                                with open(file_path, 'r') as code_file:
                                    st.code(code_file.read(), language="python")

# ════════════════════════════════════════════════════════════════════════════════
#                          TAB 2: DEEP LEARNING
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">🧠</div>
        <div>
            <h3 class="section-title">Model Architecture</h3>
            <p class="section-desc">Select and configure your deep learning model</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Selection
    ML_type = st.radio(
        "Select Model Type",
        options=[
            "Multi-Layer Perceptron (MLP)",
            "Graph Neural Network (GNN)",
            "Transformer"
        ],
        index=0,
        horizontal=True
    )
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # MLP Configuration
    if ML_type == "Multi-Layer Perceptron (MLP)":
        
        # MLP Architecture Builder
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span style="font-size: 1.2rem;">🏗️</span>
                <h4 class="card-title">Network Architecture Builder</h4>
            </div>
        """, unsafe_allow_html=True)
        
        col_arch1, col_arch2 = st.columns([1, 3])
        
        with col_arch1:
            num_layers = st.number_input(
                "Number of layers",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                help="Total number of layers including input, hidden, and output"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        mlp_config = {"model_type": "MLP", "layers": []}
        
        for i in range(int(num_layers)):
            with st.expander(f"🔧 Layer {i+1} Configuration", expanded=(i < 2)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if i == int(num_layers) - 1:
                        layer_options = ["Output Layer"]
                    else:
                        layer_options = ["Dense Layer", "Dropout Layer", "BatchNorm Layer", "Flatten Layer"]
                    
                    layer_type = st.selectbox(
                        "Layer Type",
                        options=layer_options,
                        key=f"{i}_type"
                    )
                
                layer_dict = {"type": layer_type}
                
                if layer_type in ["Dense Layer", "Output Layer"]:
                    with col2:
                        neurons = st.number_input(
                            "Neurons",
                            min_value=1,
                            max_value=10000 if layer_type == "Dense Layer" else 10,
                            step=1,
                            value=128 if layer_type == "Dense Layer" else 1,
                            key=f"{i}_neurons"
                        )
                        layer_dict["neurons"] = neurons
                    
                    with col3:
                        if layer_type == "Output Layer":
                            activation_options = ["Sigmoid"] if neurons == 1 else ["Softmax"]
                        else:
                            activation_options = [
                                "ReLU", "LeakyReLU", "PReLU", "ELU", "SELU", "GELU",
                                "Tanh", "Softplus", "Linear"
                            ]
                        
                        activation = st.selectbox(
                            "Activation",
                            options=activation_options,
                            key=f"{i}_activation"
                        )
                        layer_dict["activation"] = activation
                
                elif layer_type == "Dropout Layer":
                    with col2:
                        rate = st.slider(
                            "Dropout Rate",
                            min_value=0.0,
                            max_value=0.9,
                            value=0.2,
                            step=0.05,
                            key=f"{i}_drop_rate"
                        )
                        layer_dict["rate"] = rate
                
                mlp_config["layers"].append(layer_dict)
        
        col_cfg1, col_cfg2 = st.columns([1, 3])
        with col_cfg1:
            if st.button("📋 Show Configuration", use_container_width=True):
                st.json(mlp_config)
    
    elif ML_type == "Graph Neural Network (GNN)":
        st.info("🚧 GNN configuration coming soon! This feature is under development.")
    
    else:
        st.info("🚧 Transformer configuration coming soon! This feature is under development.")
    
    # Training Parameters Section
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">⚙️</div>
        <div>
            <h3 class="section-title">Training Configuration</h3>
            <p class="section-desc">Configure training hyperparameters and resources</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span>📂</span>
                <h4 class="card-title">Data Settings</h4>
            </div>
        """, unsafe_allow_html=True)
        
        sig_events = st.text_input(
            "Signal Events File",
            placeholder="/path/to/signal_events.h5",
            help="Full path to the signal dataset file"
        )
        
        if sig_events:
            if os.path.exists(sig_events):
                st.success("✅ File found")
            else:
                st.error("❌ File not found")
        
        train_size = st.number_input(
            "Training Size",
            min_value=1000,
            max_value=5000000,
            value=100000,
            step=1000,
            help="Number of events for training"
        )
        
        test_size = st.number_input(
            "Test Size",
            min_value=1000,
            max_value=5000000,
            value=20000,
            step=1000,
            help="Number of events for testing"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span>🎛️</span>
                <h4 class="card-title">Training Parameters</h4>
            </div>
        """, unsafe_allow_html=True)
        
        bkg_events = st.text_input(
            "Background Events File",
            placeholder="/path/to/background_events.h5",
            help="Full path to the background dataset file"
        )
        
        if bkg_events:
            if os.path.exists(bkg_events):
                st.success("✅ File found")
            else:
                st.error("❌ File not found")
        
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=1000,
            value=50,
            help="Number of training epochs"
        )
        
        batch_size = st.select_slider(
            "Batch Size",
            options=np.arange(32,1024,1),#[32, 64, 128, 256, 512, 1024, 2048],
            value=256,
            help="Samples per gradient update"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="custom-card">
            <div class="card-header">
                <span>🖥️</span>
                <h4 class="card-title">Resources & Optimization</h4>
            </div>
        """, unsafe_allow_html=True)
        
        device = st.selectbox(
            "Hardware Device",
            options=["CPU", "GPU"],
            help="Select computing device"
        )
        
        if device == "GPU":
            gpu_count = st.number_input(
                "Number of GPUs",
                min_value=1,
                max_value=8,
                value=1
            )
        
        lr = st.select_slider(
            "Learning Rate",
            options=np.arange(1e-5,1e-2,1e-5),
            value=1e-3,
            format_func=lambda x: f"{x:.0e}"
        )
        
        scheduler = st.selectbox(
            "LR Scheduler",
            options=["StepLR", "ReduceLROnPlateau", "CosineAnnealing", "OneCycleLR"]
        )
        
        precision = st.selectbox(
            "Training Precision",
            options=["float32", "float16", "mixed"],
            help="Numerical precision for training"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional Settings Row
    col_add1, col_add2, col_add3 = st.columns(3)
    
    with col_add1:
        eval_metric = st.selectbox(
            "📊 Evaluation Metric",
            options=["Accuracy", "AUC", "Precision", "Recall", "F1 Score"]
        )
    
    with col_add2:
        patience = st.number_input(
            "⏱️ Early Stopping Patience",
            min_value=1,
            max_value=50,
            value=5,
            help="Epochs to wait before stopping"
        )
    
    with col_add3:
        seed = st.number_input(
            "🎲 Random Seed",
            min_value=1,
            max_value=9999,
            value=42,
            help="For reproducibility"
        )
    
    validation_ratio = st.slider(
        "Validation Split Ratio",
        min_value=0.05,
        max_value=0.3,
        value=0.15,
        step=0.005,
        help="Fraction of training data for validation"
    )
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    col_train1, col_train2, col_train3 = st.columns([1, 1, 2])
    
    with col_train1:
        if st.button("🚀 Start Training", use_container_width=True):
            st.info("🔄 Training would start here...")

# ════════════════════════════════════════════════════════════════════════════════
#                          TAB 3: RESULTS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class="section-icon">📊</div>
        <div>
            <h3 class="section-title">Analysis Results</h3>
            <p class="section-desc">View training metrics and model performance</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Placeholder metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">--</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">--</div>
            <div class="metric-label">AUC Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">--</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">--</div>
            <div class="metric-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%);
        border: 1px dashed #3a3a4a;
        border-radius: 16px;
        padding: 4rem;
        text-align: center;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📈</div>
        <h3 style="
            font-family: 'Space Grotesk', sans-serif;
            color: #f1f5f9;
            margin-bottom: 0.5rem;
        ">No Results Yet</h3>
        <p style="color: #64748b; max-width: 400px; margin: 0 auto;">
            Run a training job to see performance metrics, learning curves, and model evaluation results here.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#                              FOOTER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="
    text-align: center;
    padding: 2rem 0;
    margin-top: 3rem;
    border-top: 1px solid #3a3a4a;
">
    <p style="color: #475569; font-size: 0.8rem;">
        Built  for the High Energy Physics community
    </p>
</div>
""", unsafe_allow_html=True)
