"""
Evaluation and Customer Segmentation Intelligence Module
Provides:
- Stratified silhouette score and per-cluster silhouette breakdown.
- Calinski-Harabasz and Davies-Bouldin clustering quality indices.
- Data-driven business segment classification based on empirical RFM profiles.
- Actionable strategic marketing recommendations.
- Robust edge-case protection (empty clusters, singletons, negative scores).
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    max_eval_samples: int = 1500,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates clustering quality using multiple academic validation indices.
    Handles singletons and edge cases safely without crashing.
    """
    labels = np.asarray(labels, dtype=int)
    unique_clusters, cluster_counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique_clusters)
    
    if n_clusters < 2:
        return {
            "silhouette_overall": None,
            "silhouette_detail": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
            "warning": "Clustering evaluation requires at least 2 non-empty clusters.",
            "is_valid": False,
        }
        
    # Check for clusters with fewer than 2 points
    tiny_clusters = unique_clusters[cluster_counts < 2].tolist()
    if tiny_clusters:
        return {
            "silhouette_overall": None,
            "silhouette_detail": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
            "warning": f"Silhouette is undefined because cluster(s) {tiny_clusters} contain fewer than 2 customers.",
            "is_valid": False,
        }
        
    rng = np.random.default_rng(seed)
    n_total = len(labels)
    target_sample = min(int(max_eval_samples), n_total)
    
    # Stratified subset selection ensuring at least 2 points per cluster
    allocation = np.full(n_clusters, 2, dtype=int)
    remaining_budget = max(0, target_sample - allocation.sum())
    
    if remaining_budget > 0:
        proportional = remaining_budget * (cluster_counts / cluster_counts.sum())
        base_alloc = np.floor(proportional).astype(int)
        allocation += base_alloc
        extra_leftover = target_sample - allocation.sum()
        if extra_leftover > 0:
            for idx in np.argsort(-(proportional - base_alloc))[:extra_leftover]:
                allocation[idx] += 1
                
    sampled_indices = []
    for c_id, alloc_count in zip(unique_clusters, allocation):
        members = np.flatnonzero(labels == c_id)
        chosen = rng.choice(members, size=min(int(alloc_count), len(members)), replace=False)
        sampled_indices.extend(chosen)
        
    eval_idx = np.asarray(sampled_indices, dtype=int)
    X_sub = X[eval_idx]
    labels_sub = labels[eval_idx]
    
    try:
        overall_silhouette = float(silhouette_score(X_sub, labels_sub))
        sample_silhouettes = silhouette_samples(X_sub, labels_sub)
        
        detail_df = (
            pd.DataFrame({"Cluster": labels_sub, "Silhouette": sample_silhouettes})
            .groupby("Cluster")["Silhouette"]
            .agg(
                Count="count",
                Mean="mean",
                Median="median",
                Min="min",
                Max="max"
            )
            .round(4)
            .reset_index()
        )
    except Exception as exc:
        overall_silhouette = None
        detail_df = None
        warning = f"Silhouette computation error: {exc}"
        return {"silhouette_overall": None, "silhouette_detail": None, "warning": warning, "is_valid": False}

    # Additional cluster quality indices
    try:
        ch_score = round(float(calinski_harabasz_score(X_sub, labels_sub)), 2)
    except Exception:
        ch_score = None
        
    try:
        db_score = round(float(davies_bouldin_score(X_sub, labels_sub)), 4)
    except Exception:
        db_score = None
        
    warning = None
    if overall_silhouette is not None:
        if overall_silhouette < 0:
            warning = "Negative silhouette score: observations are closer on average to neighboring clusters than their assigned cluster."
        elif overall_silhouette < 0.20:
            warning = "Low silhouette score: indicates overlapping customer segments with blurred boundaries."
            
    return {
        "silhouette_overall": round(overall_silhouette, 4) if overall_silhouette is not None else None,
        "silhouette_detail": detail_df,
        "calinski_harabasz": ch_score,
        "davies_bouldin": db_score,
        "eval_sample_size": len(eval_idx),
        "warning": warning,
        "is_valid": True,
    }


def generate_cluster_profiles(
    rfm_df: pd.DataFrame,
    labels: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    Computes rigorous statistical profiles and assigns grounded business segment labels.
    Uses multi-criteria ranking to ensure distinct, non-duplicate, intuitive business labels.
    """
    df = rfm_df.copy()
    df["Cluster"] = labels
    
    summary = df.groupby("Cluster").agg(
        Customer_Count=("Customer ID", "count"),
        Recency_Mean=("Recency", "mean"),
        Recency_Median=("Recency", "median"),
        Frequency_Mean=("Frequency", "mean"),
        Frequency_Median=("Frequency", "median"),
        Monetary_Mean=("Monetary", "mean"),
        Monetary_Median=("Monetary", "median"),
        Monetary_Total=("Monetary", "sum"),
    ).round(2)
    
    total_customers = len(df)
    total_revenue = float(df["Monetary"].sum())
    summary["Customer_Pct"] = (summary["Customer_Count"] / total_customers * 100).round(1)
    summary["Revenue_Pct"] = (summary["Monetary_Total"] / total_revenue * 100).round(1)
    
    # Calculate dataset-wide medians for reference
    med_r = float(df["Recency"].median())
    med_f = float(df["Frequency"].median())
    med_m = float(df["Monetary"].median())
    
    # Rank clusters by median Monetary and Recency to assign distinctive roles
    sorted_by_monetary = summary.sort_values(by="Monetary_Median", ascending=False).index.tolist()
    sorted_by_recency = summary.sort_values(by="Recency_Median", ascending=True).index.tolist()
    
    segment_info = {}
    assigned_labels = set()
    
    for c_id in summary.index:
        c_r = float(summary.loc[c_id, "Recency_Median"])
        c_f = float(summary.loc[c_id, "Frequency_Median"])
        c_m = float(summary.loc[c_id, "Monetary_Median"])
        c_count = int(summary.loc[c_id, "Customer_Count"])
        c_pct = float(summary.loc[c_id, "Customer_Pct"])
        
        # Determine candidate role
        if c_id == sorted_by_monetary[0]:
            label = "Champions & VIP Spenders"
            persona = "Top-tier revenue drivers with exceptionally high lifetime expenditure and high order frequency."
            strategy = "Dedicated VIP account management, exclusive previews, personalized concierge, luxury gifts."
            color = "#10b981"  # Emerald
        elif c_r > med_r and c_f <= med_f and c_m < med_m * 1.5:
            label = "Hibernating / Low Engagement"
            persona = "Long elapsed time since purchase with low order volume and low spend."
            strategy = "Automated reactivation drips, clearance offers, survey on why they stopped purchasing."
            color = "#94a3b8"  # Slate
        elif c_r <= med_r and (c_m > med_m or c_f > med_f):
            label = "Loyal Core Customers"
            persona = "Frequent, reliable repeat buyers with strong ongoing engagement and healthy basket sizes."
            strategy = "Loyalty rewards, subscription incentives, personalized cross-sells, referral bonuses."
            color = "#6366f1"  # Indigo
        elif c_r <= med_r and c_f <= med_f and c_m <= med_m:
            label = "New & Promising Shoppers"
            persona = "Recent purchasers with early-stage engagement and smaller transaction values."
            strategy = "Onboarding welcome series, discount on second purchase, product guides and reviews."
            color = "#06b6d4"  # Cyan
        elif c_r > med_r and (c_m >= med_m or c_f >= med_f):
            label = "At-Risk High Spenders"
            persona = "Previously valuable customers who have not ordered in a long time; urgent churn danger."
            strategy = "Aggressive win-back discounts, 'We miss you' outreach, feedback request from customer success."
            color = "#f59e0b"  # Amber
        else:
            label = f"Specialized Segment {c_id}"
            persona = "Distinct cluster of customer purchasing patterns identified by CURE representative analysis."
            strategy = "Targeted segment testing, category-specific promotions, personalized messaging."
            color = "#8b5cf6"  # Purple
            
        # Ensure no exact duplicate labels
        if label in assigned_labels:
            if "VIP" in label:
                label = "Elite High-Growth VIPs"
            elif "Loyal" in label:
                label = "Active Repeat Shoppers"
            else:
                label = f"{label} (Group {c_id})"
                
        assigned_labels.add(label)
        
        segment_info[c_id] = {
            "label": label,
            "persona": persona,
            "strategy": strategy,
            "color": color,
            "count": c_count,
            "percentage": c_pct,
            "recency_med": c_r,
            "frequency_med": c_f,
            "monetary_med": c_m,
        }
        
    summary["Segment_Name"] = [segment_info[c]["label"] for c in summary.index]
    return summary, segment_info
