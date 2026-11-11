"""
🧬 CURE Customer Intelligence — Enterprise Streamlit Application
Dataset: UCI Online Retail II
Big Data Analytics Pipeline:
- PySpark distributed transaction cleaning + customer-level RFM aggregation (with vectorized fallback)
- Log1p transformation + outlier winsorization + feature standardization
- Sample-based CURE (Clustering Using REpresentatives) with Guha et al. outlier pruning
- Multi-metric clustering evaluation (Silhouette, Calinski-Harabasz, Davies-Bouldin)
- Interactive Plotly visualizations and comprehensive export suite
"""

import sys
import os
import time
import ctypes
from pathlib import Path

# Configure PySpark worker interpreters before importing Spark.
# Use 8.3 short path on Windows to prevent path-with-space issues in batch scripts.
def _get_short_path(path_str: str) -> str:
    try:
        buf = ctypes.create_unicode_buffer(500)
        if ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, 500):
            if buf.value and os.path.exists(buf.value):
                return buf.value
    except Exception:
        pass
    return path_str

_safe_py = _get_short_path(sys.executable)
os.environ["PYSPARK_PYTHON"] = _safe_py
os.environ["PYSPARK_DRIVER_PYTHON"] = _safe_py
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
try:
    import pyspark
    os.environ["SPARK_HOME"] = _get_short_path(os.path.dirname(pyspark.__file__))
except Exception:
    pass

import numpy as np
import pandas as pd
import streamlit as st

# Internal modules
from data_processing import clean_and_audit_transactions, load_dataset_from_source
from rfm import compute_rfm, prepare_rfm_features, initialize_spark_session
from cure import run_cure
from evaluation import evaluate_clustering, generate_cluster_profiles
from visualizations import (
    plot_customer_distribution,
    plot_customer_percentage_donut,
    plot_rfm_boxplots,
    plot_3d_cluster_scatter,
    plot_cluster_comparison_radar,
    plot_silhouette_scores,
    CLUSTER_PALETTE,
)
from utils import render_kpi_card, generate_executive_report

# ----------------------------- Streamlit Page Config -----------------------------
st.set_page_config(
    page_title="CURE | Customer Intelligence Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Professional Dual-Tone SaaS Styling -----------------------------
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp {
    background: radial-gradient(circle at 50% 0%, #0d1527 0%, #070b14 100%);
    color: #f1f5f9;
    font-family: 'Inter', -apple-system, sans-serif;
}
[data-testid="stHeader"] {
    background: rgba(7, 11, 20, 0.85);
    backdrop-filter: blur(12px);
}

/* Light-Themed Modern Sidebar */
[data-testid="stSidebar"] {
    background-color: #f8fafc !important;
    border-right: 1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] * {
    color: #1e293b !important;
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4 {
    color: #0f172a !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #334155 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #0f172a !important;
    font-weight: 650 !important;
    font-size: 0.88rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: #e2e8f0 !important;
    margin: 14px 0 !important;
}
[data-testid="stSidebar"] .stRadio label span {
    color: #1e293b !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}

.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
}
.hero-title {
    font-size: 2.0rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 0.92rem;
    margin: 0;
    font-weight: 400;
}
.pill-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.3);
}
.segment-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.segment-card:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 650;
    padding: 10px 20px;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4);
    transition: all 0.2s ease;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    transform: translateY(-1px);
}
div.stButton > button[kind="secondary"] {
    background: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 10px;
    font-weight: 600;
}
div.stDownloadButton > button {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 650 !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 14px rgba(239, 68, 68, 0.35) !important;
    transition: all 0.2s ease !important;
}
div.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.55) !important;
    transform: translateY(-1px) !important;
}
div.stDownloadButton > button:focus,
div.stDownloadButton > button:active {
    background: #b91c1c !important;
    color: #ffffff !important;
    border: none !important;
    outline: none !important;
}
div.stDownloadButton > button p,
div.stDownloadButton > button span {
    color: #ffffff !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
    font-weight: 600;
    color: #94a3b8;
}
.stTabs [aria-selected="true"] {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------- Spark Resource Cache -----------------------------
@st.cache_resource(show_spinner=False)
def get_cached_spark():
    try:
        spark = initialize_spark_session(driver_memory="4g")
        return spark
    except Exception:
        return None


# ----------------------------- Data Caching -----------------------------
@st.cache_data(show_spinner=False)
def load_cached_raw_dataset(source_choice: str, uploaded_file=None):
    return load_dataset_from_source(source_choice, uploaded_file=uploaded_file)


# ----------------------------- Session State Setup -----------------------------
DEFAULT_STATE = {
    "raw_df": None,
    "clean_df": None,
    "audit_report": None,
    "rfm_df": None,
    "X": None,
    "feature_meta": None,
    "rfm_meta": None,
    "cure_meta": None,
    "labels": None,
    "summary_df": None,
    "segment_info": None,
    "eval_meta": None,
    "dataset_ready": False,
    "clustering_done": False,
    "step1_msg": None,
    "step2_msg": None,
}

for key, val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ----------------------------- Sidebar Controls -----------------------------
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid #e2e8f0;">
    <span style="font-size:1.8rem;">🧬</span>
    <div>
        <div style="font-size:1.18rem; font-weight:800; color:#0f172a;">CURE Studio</div>
        <div style="font-size:0.75rem; color:#64748b; font-weight:600;">Customer Analytics Platform</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Dataset source selector
st.sidebar.markdown("### 1. Data Ingestion")
source_option = st.sidebar.radio(
    "Dataset Source",
    [
        "Packaged UCI Dataset (Local)",
        "Upload Custom File (.zip, .xlsx, .csv)",
    ],
    help="Select the source for the Online Retail II dataset."
)

uploaded_file = None
source_key = "local"
if source_option == "Upload Custom File (.zip, .xlsx, .csv)":
    source_key = "upload"
    uploaded_file = st.sidebar.file_uploader(
        "Upload dataset archive or spreadsheet",
        type=["zip", "xlsx", "xls", "csv"],
        help="Upload the official UCI zip archive or extracted Excel/CSV file."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. CURE Parameters")

k_clusters = st.sidebar.slider(
    "Target Clusters (k)", 
    min_value=2, max_value=8, value=4, step=1,
    help="Number of target customer segments to produce."
)

sample_size = st.sidebar.slider(
    "Sample Size (s)",
    min_value=300, max_value=2000, value=1000, step=100,
    help="Number of customers sampled for initial hierarchical clustering."
)

reps_per_cluster = st.sidebar.slider(
    "Representatives per Cluster (c)",
    min_value=2, max_value=15, value=8, step=1,
    help="Number of well-scattered boundary points chosen to define each cluster's shape."
)

shrink_factor = st.sidebar.slider(
    "Shrink Factor (alpha)",
    min_value=0.10, max_value=0.90, value=0.30, step=0.05,
    help="Fraction by which representative points are shrunk towards the cluster centroid."
)

# Fixed optimal hyperparameters under the hood
random_seed = 42
guha_outlier_pruning = True
winsorize_outliers = True
eval_sample = 1500

# PySpark engine indicator
spark_inst = get_cached_spark()
engine_status = "⚡ PySpark 3.5 Active" if spark_inst is not None else "🐍 Pandas Vectorized Active"
engine_color = "#059669" if spark_inst is not None else "#0284c7"
st.sidebar.markdown(f"""
<div style="margin-top:16px; padding:11px 13px; background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
    <div style="font-size:0.72rem; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.04em;">Big Data Engine</div>
    <div style="font-size:0.86rem; font-weight:750; color:{engine_color}; margin-top:2px;">{engine_status}</div>
</div>
""", unsafe_allow_html=True)


# ----------------------------- Hero Banner -----------------------------
st.markdown("""
<div class="hero-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div class="hero-title">🧬 CURE Customer Segmentation</div>
            <p class="hero-subtitle">Customer Analytics & Behavioral Segmentation Platform</p>
        </div>
        <div>
            <span class="pill-tag">UCI Online Retail II</span>
            <span class="pill-tag">PySpark 3.5</span>
            <span class="pill-tag">CURE Algorithm</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------- Step 1 & 2 Execution Buttons -----------------------------
col_step1, col_step2 = st.columns([1.2, 1.0])

with col_step1:
    st.markdown("#### **Step 1: Ingest Data & Build RFM Matrix**")
    load_btn = st.button("🚀 Load Dataset & Compute RFM", type="primary" if not st.session_state.dataset_ready else "secondary", use_container_width=True)

with col_step2:
    st.markdown("#### **Step 2: Run CURE Clustering**")
    run_cure_btn = st.button(
        "🔮 Execute CURE Clustering", 
        type="primary",
        disabled=False, 
        use_container_width=True
    )



# ----------------------------- Step 1: Data Ingestion & RFM Pipeline -----------------------------
if load_btn:
    try:
        pbar = st.progress(0, text="Initializing dataset loader...")
        status_box = st.empty()
        
        def update_pbar(frac, msg):
            pbar.progress(min(max(float(frac), 0.0), 1.0), text=msg)

        status_box.info("Accessing Online Retail II transactions...")
        raw_df = load_cached_raw_dataset(source_key, uploaded_file=uploaded_file)
        update_pbar(0.20, f"Loaded {len(raw_df):,} raw transaction records. Cleaning and validating schema...")
        
        clean_df, audit_rep = clean_and_audit_transactions(raw_df)
        update_pbar(0.40, f"Retained {len(clean_df):,} valid transactions. Aggregating customer RFM features...")
        
        spark_session = get_cached_spark()
        engine_to_use = "spark" if spark_session is not None else "pandas"
        
        rfm_df, rfm_meta = compute_rfm(
            clean_df=clean_df,
            engine=engine_to_use,
            spark_session=spark_session,
            progress_callback=update_pbar
        )
        
        update_pbar(0.85, "Applying log1p transformation and feature standardization...")
        prepared_rfm, X, feature_meta = prepare_rfm_features(
            rfm_df=rfm_df,
            winsorize=winsorize_outliers,
            winsorize_percentile=0.99,
            scaler_type="standard"
        )
        
        # Save to session state
        st.session_state.raw_df = raw_df
        st.session_state.clean_df = clean_df
        st.session_state.audit_report = audit_rep
        st.session_state.rfm_df = prepared_rfm
        st.session_state.X = X
        st.session_state.feature_meta = feature_meta
        st.session_state.rfm_meta = rfm_meta
        st.session_state.dataset_ready = True
        
        # Reset previous clustering results
        st.session_state.clustering_done = False
        st.session_state.labels = None
        st.session_state.summary_df = None
        st.session_state.eval_meta = None
        st.session_state.step2_msg = None
        
        st.session_state.step1_msg = (
            f"✅ Dataset Ingestion Complete: {len(prepared_rfm):,} valid customers extracted from {audit_rep['clean_transactions']:,} transactions "
            f"(Total Revenue: £{audit_rep['total_gross_revenue']:,.2f}, Engine: {rfm_meta['engine']}). Ready for Step 2!"
        )
        st.rerun()
    except Exception as exc:
        st.error(f"❌ Data ingestion or RFM computation failed: {exc}")


# ----------------------------- Step 2: CURE Clustering Pipeline -----------------------------
if run_cure_btn:
    try:
        # If RFM is not prepared yet, run Step 1 automatically first
        if st.session_state.X is None:
            with st.spinner("Preparing dataset and customer RFM features (Step 1 auto-prep)..."):
                raw_df = load_cached_raw_dataset(source_key, uploaded_file=uploaded_file)
                clean_df, audit_rep = clean_and_audit_transactions(raw_df)
                spark_session = get_cached_spark()
                engine_to_use = "spark" if spark_session is not None else "pandas"
                rfm_df, rfm_meta = compute_rfm(clean_df=clean_df, engine=engine_to_use, spark_session=spark_session)
                prepared_rfm, X, feature_meta = prepare_rfm_features(
                    rfm_df=rfm_df,
                    winsorize=winsorize_outliers,
                    winsorize_percentile=0.99,
                    scaler_type="standard"
                )
                st.session_state.raw_df = raw_df
                st.session_state.clean_df = clean_df
                st.session_state.audit_report = audit_rep
                st.session_state.rfm_df = prepared_rfm
                st.session_state.X = X
                st.session_state.feature_meta = feature_meta
                st.session_state.rfm_meta = rfm_meta
                st.session_state.dataset_ready = True
                st.session_state.step1_msg = (
                    f"✅ Dataset Ingestion Complete: {len(prepared_rfm):,} valid customers extracted from {audit_rep['clean_transactions']:,} transactions "
                    f"(Total Revenue: £{audit_rep['total_gross_revenue']:,.2f}, Engine: {rfm_meta['engine']})."
                )

        cure_pbar = st.progress(0, text="Initializing CURE clustering...")
        cure_status = st.empty()
        
        def cure_pbar_update(frac, msg):
            cure_pbar.progress(min(max(float(frac), 0.0), 1.0), text=msg)

        cure_results = run_cure(
            X=st.session_state.X,
            k=k_clusters,
            sample_size=sample_size,
            representatives_per_cluster=reps_per_cluster,
            shrink_factor=shrink_factor,
            outlier_handling=guha_outlier_pruning,
            random_seed=int(random_seed),
            progress_callback=cure_pbar_update,
        )
        
        cure_pbar_update(0.90, "Evaluating clustering quality and computing silhouette metrics...")
        eval_results = evaluate_clustering(
            X=st.session_state.X,
            labels=cure_results["labels"],
            max_eval_samples=int(eval_sample),
            seed=int(random_seed),
        )
        
        cure_pbar_update(0.96, "Generating statistical profiles and customer segment insights...")
        summary_df, segment_info = generate_cluster_profiles(
            rfm_df=st.session_state.rfm_df,
            labels=cure_results["labels"]
        )
        
        # Save results to session state
        st.session_state.cure_meta = cure_results
        st.session_state.labels = cure_results["labels"]
        st.session_state.eval_meta = eval_results
        st.session_state.summary_df = summary_df
        st.session_state.segment_info = segment_info
        st.session_state.clustering_done = True
        
        # Augment RFM DataFrame with cluster and segment label
        st.session_state.rfm_df["Cluster"] = cure_results["labels"]
        st.session_state.rfm_df["Segment_Name"] = [
            segment_info[c]["label"] for c in cure_results["labels"]
        ]
        
        st.session_state.step2_msg = (
            f"🎉 CURE Clustering Succeeded! Formed {k_clusters} customer segments across {len(st.session_state.rfm_df):,} customers. "
            f"Runtime: {cure_results['runtime_seconds']}s | Overall Silhouette: {eval_results['silhouette_overall']}."
        )
        st.rerun()
    except Exception as exc:
        st.error(f"❌ CURE clustering execution failed: {exc}")


# ----------------------------- Dashboard & Insights Display -----------------------------
if st.session_state.clustering_done and st.session_state.summary_df is not None:
    st.markdown("---")
    
    cure_meta = st.session_state.cure_meta
    eval_meta = st.session_state.eval_meta
    summary_df = st.session_state.summary_df
    segment_info = st.session_state.segment_info
    rfm_df = st.session_state.rfm_df
    audit_rep = st.session_state.audit_report
    
    # ----------------------------- Executive KPI Cards -----------------------------
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    
    with kpi1:
        st.markdown(render_kpi_card(
            title="Total Customers",
            value=f"{len(rfm_df):,}",
            subtitle="Unique Buyers",
            badge_text="Audited",
            badge_color="#38bdf8",
            icon="👥"
        ), unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(render_kpi_card(
            title="Clean Transactions",
            value=f"{audit_rep['clean_transactions']:,}",
            subtitle="Valid Invoices",
            badge_text="Filtered",
            badge_color="#34d399",
            icon="🧾"
        ), unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(render_kpi_card(
            title="Total Revenue",
            value=f"£{audit_rep['total_gross_revenue']/1e6:.2f}M",
            subtitle="Gross Sales",
            badge_text="GBP (£)",
            badge_color="#fbbf24",
            icon="💳"
        ), unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(render_kpi_card(
            title="Clusters (k)",
            value=str(cure_meta["k"]),
            subtitle="Customer Cohorts",
            badge_text=f"k = {cure_meta['k']}",
            badge_color="#a855f7",
            icon="🎯"
        ), unsafe_allow_html=True)
        
    with kpi5:
        sil_val = eval_meta.get("silhouette_overall")
        sil_display = f"{sil_val:.4f}" if sil_val is not None else "N/A"
        sil_rating = "Good" if (sil_val or 0) >= 0.25 else "Moderate"
        sil_badge_color = "#34d399" if (sil_val or 0) >= 0.25 else "#f59e0b"
        
        st.markdown(render_kpi_card(
            title="Silhouette Score",
            value=sil_display,
            subtitle="Cluster Quality",
            badge_text=sil_rating,
            badge_color=sil_badge_color,
            icon="📊"
        ), unsafe_allow_html=True)
        
    with kpi6:
        st.markdown(render_kpi_card(
            title="CURE Runtime",
            value=f"{cure_meta['runtime_seconds']}s",
            subtitle="Execution Time",
            badge_text="Fast",
            badge_color="#ec4899",
            icon="⚡"
        ), unsafe_allow_html=True)

    # Ingestion Audit Expander
    with st.expander("🔍 View Data Ingestion & Transaction Cleaning Audit Breakdown"):
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Raw Rows Loaded", f"{audit_rep['initial_rows']:,}")
        a2.metric("Null Customer IDs", f"-{audit_rep['null_customer_ids']:,}")
        a3.metric("Cancellations ('C')", f"-{audit_rep['cancellations']:,}")
        a4.metric("Non-Positive Qty/Price", f"-{audit_rep['non_positive_quantity'] + audit_rep['non_positive_price']:,}")
        a5.metric("Final Retained Rows", f"{audit_rep['clean_transactions']:,}")
        st.caption(f"📅 Active Date Range: **{audit_rep['date_range_start']}** to **{audit_rep['date_range_end']}** | RFM Reference Date: **{st.session_state.rfm_meta['reference_date']}**")

    # ----------------------------- Dashboard Tabs -----------------------------
    tab_overview, tab_rfm, tab_3d, tab_explorer, tab_export = st.tabs([
        "📊 Executive Overview",
        "📈 RFM Analytics",
        "🌐 3D Cluster Space",
        "🔎 Customer Explorer",
        "📥 Export Data",
    ])

    # ---------------- Tab 1: Executive Overview ----------------
    with tab_overview:
        c_left, c_right = st.columns([1.1, 1.0])
        with c_left:
            dist_fig = plot_customer_distribution(summary_df)
            st.plotly_chart(dist_fig, use_container_width=True)
        with c_right:
            donut_fig = plot_customer_percentage_donut(summary_df)
            st.plotly_chart(donut_fig, use_container_width=True)
            
        st.markdown("### 🎯 Data-Driven Customer Segments & Strategic Playbooks")
        
        seg_cols = st.columns(len(summary_df))
        for idx, c_id in enumerate(summary_df.index):
            info = segment_info[c_id]
            with seg_cols[idx]:
                st.markdown(f"""
                <div class="segment-card" style="border-top: 4px solid {info['color']};">
                    <div style="font-size:0.8rem; font-weight:700; color:{info['color']}; text-transform:uppercase;">Cluster {c_id}</div>
                    <div style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin:4px 0 8px 0;">{info['label']}</div>
                    <div style="font-size:0.85rem; color:#94a3b8; margin-bottom:12px;">{info['persona']}</div>
                    <div style="background:#1e293b; border-radius:8px; padding:10px; margin-bottom:12px;">
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#cbd5e1; margin-bottom:4px;">
                            <span>Volume:</span> <b>{info['count']:,} ({info['percentage']}%)</b>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#cbd5e1; margin-bottom:4px;">
                            <span>Med. Recency:</span> <b>{info['recency_med']:.0f} days</b>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#cbd5e1; margin-bottom:4px;">
                            <span>Med. Orders:</span> <b>{info['frequency_med']:.0f}</b>
                        </div>
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#cbd5e1;">
                            <span>Med. Spend:</span> <b>£{info['monetary_med']:,.2f}</b>
                        </div>
                    </div>
                    <div style="font-size:0.75rem; color:#64748b; font-weight:600; text-transform:uppercase; margin-bottom:4px;">Recommended Strategy:</div>
                    <div style="font-size:0.8rem; color:#e2e8f0; line-height:1.35;">{info['strategy']}</div>
                </div>
                """, unsafe_allow_html=True)

    # ---------------- Tab 2: RFM Analytics ----------------
    with tab_rfm:
        st.markdown("#### **RFM Feature Distribution by Cluster**")
        
        box_figs = plot_rfm_boxplots(rfm_df)
        b1, b2, b3 = st.columns(3)
        with b1:
            st.plotly_chart(box_figs["Recency"], use_container_width=True)
        with b2:
            st.plotly_chart(box_figs["Frequency"], use_container_width=True)
        with b3:
            st.plotly_chart(box_figs["Monetary"], use_container_width=True)
            
        st.markdown("#### **Segment Statistical Profiles**")
        st.dataframe(
            summary_df[[
                "Segment_Name", "Customer_Count", "Customer_Pct",
                "Recency_Median", "Frequency_Median", "Monetary_Median",
                "Monetary_Total", "Revenue_Pct"
            ]],
            use_container_width=True,
            height=320
        )

    # ---------------- Tab 3: 3D Cluster Space ----------------
    with tab_3d:
        st.markdown("#### **Interactive 3D Cluster Representation in Standardized RFM Space**")
        
        scatter_3d_fig = plot_3d_cluster_scatter(st.session_state.X, rfm_df, max_points=None, seed=int(random_seed))
        st.plotly_chart(scatter_3d_fig, use_container_width=True)

    # ---------------- Tab 4: Customer Explorer ----------------
    with tab_explorer:
        st.markdown("#### **Customer Drill-Down & Advanced Filtering**")
        
        f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.0, 1.0])
        all_clusters = sorted(rfm_df["Cluster"].unique().tolist())
        with f1:
            selected_clusters = st.multiselect("Filter Clusters", all_clusters, default=all_clusters)
        with f2:
            search_query = st.text_input("Search Customer ID", "", placeholder="Enter ID...")
        with f3:
            min_orders = st.number_input("Min Frequency (Orders)", min_value=1, value=1)
        with f4:
            min_spend = st.number_input("Min Monetary (£)", min_value=0.0, value=0.0, step=100.0)
            
        filtered_customers = rfm_df[
            (rfm_df["Cluster"].isin(selected_clusters)) &
            (rfm_df["Frequency"] >= min_orders) &
            (rfm_df["Monetary"] >= min_spend)
        ].copy()
        
        if search_query.strip():
            filtered_customers = filtered_customers[
                filtered_customers["Customer ID"].astype(str).str.contains(search_query.strip(), regex=False)
            ]
            
        st.markdown(f"**Showing {len(filtered_customers):,} matching customers:**")
        st.dataframe(
            filtered_customers[["Customer ID", "Segment_Name", "Cluster", "Recency", "Frequency", "Monetary"]],
            use_container_width=True,
            height=420
        )

    # ---------------- Tab 5: Export Data ----------------
    with tab_export:
        st.markdown("#### **Data Downloads**")
        
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            cust_csv = rfm_df[["Customer ID", "Segment_Name", "Cluster", "Recency", "Frequency", "Monetary"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Customer Segments (CSV)",
                data=cust_csv,
                file_name="CURE_Customer_Segments.csv",
                mime="text/csv",
                use_container_width=True,
            )
            
        with exp_col2:
            sum_csv = summary_df.to_csv().encode("utf-8")
            st.download_button(
                "📥 Cluster Summary (CSV)",
                data=sum_csv,
                file_name="CURE_Cluster_Summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    # Initial landing screen before Step 1 / Step 2 execution
    st.markdown("---")
    st.markdown("""
    <div style="background:#0f172a; border:1px solid #1e293b; border-radius:14px; padding:20px 24px; margin-top:8px; text-align:center;">
        <h4 style="color:#f8fafc; margin:0 0 8px 0;">🚀 Ready for Customer Segmentation</h4>
        <p style="color:#94a3b8; font-size:0.92rem; margin:0 auto; max-width:640px;">
            Click <b>"🚀 Load Dataset & Compute RFM"</b> in Step 1, then click <b>"🔮 Execute CURE Clustering"</b> in Step 2 to generate interactive customer insights and visualizations.
        </p>
    </div>
    """, unsafe_allow_html=True)
