"""
Modern Interactive Plotly Visualizations for CURE Customer Segmentation
Features a dark SaaS theme matching modern BI and analytics dashboards.
"""

from typing import Dict, Any, Optional
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Standardized palette for consistent cluster identity across all charts
CLUSTER_PALETTE = [
    "#6366f1",  # Indigo / Royal Blue
    "#06b6d4",  # Cyan / Teal
    "#10b981",  # Emerald Green
    "#f59e0b",  # Amber / Gold
    "#ec4899",  # Magenta / Pink
    "#8b5cf6",  # Vibrant Purple
    "#3b82f6",  # Electric Blue
    "#14b8a6",  # Mint / Teal
]

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.6)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=40, r=30, t=50, b=40),
)


def plot_customer_distribution(summary_df: pd.DataFrame) -> go.Figure:
    """Horizontal/vertical bar chart showing customer count and percentage per cluster."""
    df = summary_df.reset_index().copy()
    df["Cluster_Label"] = "Cluster " + df["Cluster"].astype(str) + " (" + df["Segment_Name"] + ")"
    
    colors = [CLUSTER_PALETTE[(int(c) - 1) % len(CLUSTER_PALETTE)] for c in df["Cluster"]]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Cluster"].astype(str),
        y=df["Customer_Count"],
        text=[f"{count:,} ({pct}%)" for count, pct in zip(df["Customer_Count"], df["Customer_Pct"])],
        textposition="outside",
        marker=dict(color=colors, line=dict(color="#1e293b", width=1.5)),
        hovertemplate="<b>%{x}</b><br>Customers: %{y:,}<br><extra></extra>",
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Customer Volume Distribution Across Clusters</b>",
        xaxis=dict(title="Cluster ID", tickmode="linear"),
        yaxis=dict(title="Number of Customers", gridcolor="#1e293b"),
        showlegend=False,
        height=380,
    )
    return fig


def plot_customer_percentage_donut(summary_df: pd.DataFrame) -> go.Figure:
    """Donut chart illustrating customer segment proportions."""
    df = summary_df.reset_index().copy()
    labels = ["C" + str(c) + ": " + name for c, name in zip(df["Cluster"], df["Segment_Name"])]
    colors = [CLUSTER_PALETTE[(int(c) - 1) % len(CLUSTER_PALETTE)] for c in df["Cluster"]]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=df["Customer_Count"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="percent+label",
        hoverinfo="label+value+percent",
    )])
    
    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Customer Share by Segment</b>",
        showlegend=False,
        height=380,
    )
    return fig


def plot_rfm_boxplots(result_df: pd.DataFrame) -> Dict[str, go.Figure]:
    """Generates box plots for Recency, Frequency, and Monetary with outlier visualization."""
    figs = {}
    metrics = [
        ("Recency", "Recency (Days Elapsed)", "#38bdf8"),
        ("Frequency", "Frequency (Distinct Invoices)", "#a78bfa"),
        ("Monetary", "Monetary Value (£)", "#34d399"),
    ]
    
    for col, title, accent in metrics:
        fig = px.box(
            result_df,
            x="Cluster",
            y=col,
            color="Cluster",
            points="outliers",
            color_discrete_sequence=CLUSTER_PALETTE,
        )
        fig.update_layout(
            **DARK_LAYOUT,
            title=f"<b>{title}</b>",
            xaxis=dict(title="Cluster ID", tickmode="linear"),
            yaxis=dict(title=col, gridcolor="#1e293b"),
            showlegend=False,
            height=340,
        )
        figs[col] = fig
        
    return figs


def plot_3d_cluster_scatter(
    X: np.ndarray,
    result_df: pd.DataFrame,
    max_points: Optional[int] = None,
    seed: int = 42
) -> go.Figure:
    """
    Interactive 3D scatter plot in standardized log-RFM feature space.
    Renders all customers without unnecessary downsampling text.
    """
    n = len(X)
    if max_points is not None and max_points < n:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, size=max_points, replace=False)
    else:
        indices = np.arange(n)
    
    view_df = pd.DataFrame(
        X[indices],
        columns=["Recency_Scaled", "Frequency_Scaled", "Monetary_Scaled"]
    )
    view_df["Customer ID"] = result_df["Customer ID"].to_numpy()[indices]
    view_df["Cluster"] = result_df["Cluster"].to_numpy()[indices].astype(str)
    view_df["Recency"] = result_df["Recency"].to_numpy()[indices]
    view_df["Frequency"] = result_df["Frequency"].to_numpy()[indices]
    view_df["Monetary"] = result_df["Monetary"].to_numpy()[indices]
    
    if "Segment_Name" in result_df.columns:
        view_df["Segment"] = result_df["Segment_Name"].to_numpy()[indices]
    else:
        view_df["Segment"] = "Cluster " + view_df["Cluster"]
        
    fig = px.scatter_3d(
        view_df,
        x="Recency_Scaled",
        y="Frequency_Scaled",
        z="Monetary_Scaled",
        color="Cluster",
        color_discrete_sequence=CLUSTER_PALETTE,
        opacity=0.75,
        hover_data={
            "Customer ID": True,
            "Segment": True,
            "Recency": True,
            "Frequency": True,
            "Monetary": ":.2f",
            "Recency_Scaled": False,
            "Frequency_Scaled": False,
            "Monetary_Scaled": False,
        },
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=650,
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(
            xaxis=dict(title="Recency (Log-Scaled)", backgroundcolor="rgba(15,23,42,0.8)", gridcolor="#334155"),
            yaxis=dict(title="Frequency (Log-Scaled)", backgroundcolor="rgba(15,23,42,0.8)", gridcolor="#334155"),
            zaxis=dict(title="Monetary (Log-Scaled)", backgroundcolor="rgba(15,23,42,0.8)", gridcolor="#334155"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(15,23,42,0.6)"
        )
    )
    return fig


def plot_cluster_comparison_radar(summary_df: pd.DataFrame) -> go.Figure:
    """Radar chart comparing normalized mean RFM characteristics of clusters."""
    df = summary_df.copy()
    categories = ["Recency (Days)", "Frequency (Orders)", "Monetary (£)"]
    
    # Min-max normalize the means across clusters for balanced radar representation
    r_vals = df["Recency_Mean"].to_numpy()
    f_vals = df["Frequency_Mean"].to_numpy()
    m_vals = df["Monetary_Mean"].to_numpy()
    
    def norm_feature(arr):
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return np.full_like(arr, 0.5)
        return (arr - mn) / (mx - mn)
        
    r_norm = norm_feature(r_vals)
    f_norm = norm_feature(f_vals)
    m_norm = norm_feature(m_vals)
    
    fig = go.Figure()
    
    for idx, c_id in enumerate(df.index):
        vals = [r_norm[idx], f_norm[idx], m_norm[idx], r_norm[idx]]
        color = CLUSTER_PALETTE[(int(c_id) - 1) % len(CLUSTER_PALETTE)]
        name = f"Cluster {c_id}: {df.loc[c_id, 'Segment_Name']}"
        
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            name=name,
            line=dict(color=color, width=2),
            fill="toself",
            fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}",
        ))
        
    fig.update_layout(
        **DARK_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#334155"),
            bgcolor="rgba(15,23,42,0.6)",
        ),
        title="<b>Relative RFM Profile Comparison (Min-Max Normalized)</b>",
        height=450,
        showlegend=True,
    )
    return fig


def plot_silhouette_scores(silhouette_detail_df: pd.DataFrame) -> go.Figure:
    """Bar chart displaying mean silhouette coefficient per cluster."""
    df = silhouette_detail_df.copy()
    colors = [CLUSTER_PALETTE[(int(c) - 1) % len(CLUSTER_PALETTE)] for c in df["Cluster"]]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Cluster"].astype(str),
        y=df["Mean"],
        text=[f"{v:.3f}" for v in df["Mean"]],
        textposition="outside",
        marker=dict(color=colors, line=dict(color="#1e293b", width=1.5)),
        hovertemplate="Cluster %{x}<br>Mean Silhouette: %{y:.4f}<extra></extra>",
    ))
    
    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Silhouette Coefficient by Cluster</b>",
        xaxis=dict(title="Cluster ID", tickmode="linear"),
        yaxis=dict(title="Average Silhouette Score", gridcolor="#1e293b", range=[-0.2, 1.0]),
        height=350,
        showlegend=False,
    )
    return fig
