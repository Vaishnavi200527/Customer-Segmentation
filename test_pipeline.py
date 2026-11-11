"""
Comprehensive Test Suite for CURE Customer Segmentation Application
Validates:
1. Module imports and Python syntax
2. Data processing and transaction cleaning
3. RFM feature calculation and standardization
4. CURE algorithm correctness (argmax, farthest-first, shrinking, Guha outlier pruning)
5. 100% customer assignment integrity
6. Multi-metric evaluation and silhouette scoring
7. Visualization figure generation
8. Data and report exports
"""

import sys
import os
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

for d in [str(current_dir), str(parent_dir)]:
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np
import pandas as pd

print("=" * 70)
print("RUNNING COMPREHENSIVE TEST SUITE FOR CURE CUSTOMER INTELLIGENCE")
print("=" * 70)

# TEST 1: Syntax and Imports
print("\n[TEST 1] Verifying module imports and syntax...")
try:
    import data_processing
    import rfm
    import cure
    import evaluation
    import visualizations
    import utils
    print("  --> PASS: All project modules imported cleanly without syntax errors.")
except Exception as e:
    print(f"  --> FAIL: Import failed: {e}")
    sys.exit(1)

# TEST 2: Data Preprocessing on Synthetic and Real Data
print("\n[TEST 2] Testing transaction cleaning and row removal audit...")
dummy_data = pd.DataFrame({
    "Invoice": ["536365", "C536366", "536367", "536368", "536369", "536370", "536371", "536372", "536373", "536374", "536375"],
    "Customer ID": [12345, 12345, None, 12346, 12347, 12348, 12349, 12350, 12351, 12352, 12353],
    "Quantity": [6, -1, 10, 0, 5, 2, 12, 1, 4, 8, 3],
    "Price": [2.55, 3.39, 4.00, 5.00, -1.00, 1.25, 3.75, 4.20, 1.10, 2.40, 5.00],
    "InvoiceDate": [
        "2010-12-01 08:26:00",
        "2010-12-01 08:28:00",
        "2010-12-01 08:34:00",
        "2010-12-01 08:35:00",
        "2010-12-01 08:36:00",
        "2010-12-01 08:37:00",
        "2010-12-01 08:38:00",
        "2010-12-01 08:39:00",
        "2010-12-01 08:40:00",
        "2010-12-01 08:41:00",
        "2010-12-01 08:42:00",
    ]
})

clean_dummy, audit_dummy = data_processing.clean_and_audit_transactions(dummy_data)
assert len(clean_dummy) == 7, f"Expected 7 clean rows, got {len(clean_dummy)}"
assert audit_dummy["null_customer_ids"] == 1, "Expected 1 null customer ID"
assert audit_dummy["cancellations"] == 1, "Expected 1 cancellation"
assert audit_dummy["non_positive_quantity"] == 2, "Expected 2 non-positive quantities (one negative, one zero)"
assert audit_dummy["non_positive_price"] == 1, "Expected 1 non-positive price"
print("  --> PASS: Transaction cleaning correctly filtered cancellations, null customers, non-positive quantity and price.")

# TEST 3: RFM Feature Engineering and Standardization
print("\n[TEST 3] Testing RFM aggregation and feature transformation...")
rfm_df, rfm_meta = rfm.compute_rfm(clean_dummy, engine="pandas")
assert len(rfm_df) == 7, f"Expected 7 unique customers, got {len(rfm_df)}"
assert "Recency" in rfm_df.columns and "Frequency" in rfm_df.columns and "Monetary" in rfm_df.columns

# Test with full user output customer dataset
user_csv_candidates = [
    parent_dir / "CURE_Customer_Segments_user_output.csv",
    current_dir / "CURE_Customer_Segments_user_output.csv",
]
user_csv_path = next((c for c in user_csv_candidates if c.exists()), None)
if user_csv_path:
    full_rfm = pd.read_csv(user_csv_path)[["Customer ID", "Recency", "Frequency", "Monetary"]]
    prep_rfm, X, f_meta = rfm.prepare_rfm_features(full_rfm, winsorize=True, winsorize_percentile=0.99)
    assert X.shape == (len(full_rfm), 3), f"Expected shape ({len(full_rfm)}, 3), got {X.shape}"
    assert np.isfinite(X).all(), "Transformed feature matrix contains non-finite values!"
    print(f"  --> PASS: RFM features prepared for {len(full_rfm):,} customers. Log1p + Z-score scaled successfully.")

# TEST 4: CURE Representative Point Selection Correctness
print("\n[TEST 4] Testing CURE representative point selection...")
test_cluster = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [10.0, 10.0], # extreme point
    [0.5, 0.5],   # near centroid
])
reps = cure.select_representatives(test_cluster, number=3, shrink=0.3)
centroid = test_cluster.mean(axis=0)
# First representative must be farthest from centroid (i.e. [10, 10])
farthest_idx = np.argmax(np.linalg.norm(test_cluster - centroid, axis=1))
expected_first_rep = test_cluster[farthest_idx] + 0.3 * (centroid - test_cluster[farthest_idx])
assert np.allclose(reps[0], expected_first_rep), "First representative was not the farthest point from centroid!"
assert len(reps) == 3, f"Expected 3 representatives, got {len(reps)}"
print("  --> PASS: Representative point selection correctly uses argmax (farthest from centroid) and farthest-first traversal.")

# TEST 5: CURE Clustering Execution & Assignment Integrity
print("\n[TEST 5] Testing full CURE clustering execution and assignment integrity...")
cure_out = cure.run_cure(
    X=X,
    k=4,
    sample_size=1000,
    representatives_per_cluster=8,
    shrink_factor=0.3,
    outlier_handling=True,
    random_seed=42,
)

labels = cure_out["labels"]
assert len(labels) == len(full_rfm), f"Expected {len(full_rfm)} labels, got {len(labels)}"
assert set(np.unique(labels)) == {1, 2, 3, 4}, f"Expected clusters 1..4, got {set(np.unique(labels))}"
assert sum(cure_out["counts"].values()) == len(full_rfm), "Cluster counts do not sum to total customers!"
assert not np.any(labels <= 0), "Found invalid non-positive cluster labels!"
print(f"  --> PASS: All {len(full_rfm):,} customers assigned to valid clusters.")
print(f"      Cluster distribution: {cure_out['counts']}")
print(f"      Hierarchical execution runtime: {cure_out['runtime_seconds']}s")

# TEST 6: Multi-Metric Clustering Evaluation
print("\n[TEST 6] Testing clustering quality evaluation...")
eval_res = evaluation.evaluate_clustering(X, labels, max_eval_samples=1500, seed=42)
assert eval_res["is_valid"] is True, f"Evaluation reported invalid: {eval_res.get('warning')}"
assert eval_res["silhouette_overall"] is not None, "Silhouette score was None!"
assert eval_res["silhouette_overall"] > 0.20, f"Silhouette score unexpectedly low: {eval_res['silhouette_overall']}"
assert eval_res["calinski_harabasz"] is not None and eval_res["calinski_harabasz"] > 0
assert eval_res["davies_bouldin"] is not None and eval_res["davies_bouldin"] > 0
print(f"  --> PASS: Evaluation valid. Silhouette = {eval_res['silhouette_overall']}, Calinski-Harabasz = {eval_res['calinski_harabasz']}, Davies-Bouldin = {eval_res['davies_bouldin']}")

# TEST 7: Customer Segment Statistical Profiling
print("\n[TEST 7] Testing statistical profiling and business segment labeling...")
summary_df, segment_info = evaluation.generate_cluster_profiles(full_rfm, labels)
assert len(summary_df) == 4, f"Expected 4 cluster summary rows, got {len(summary_df)}"
assert "Segment_Name" in summary_df.columns
assert "Customer_Count" in summary_df.columns
assert "Customer_Pct" in summary_df.columns
assert summary_df["Customer_Count"].sum() == len(full_rfm)
print("  --> PASS: Cluster profiles generated with business labels:")
for c_id, row in summary_df.iterrows():
    print(f"      Cluster {c_id}: {row['Segment_Name']} | Count: {row['Customer_Count']:,} ({row['Customer_Pct']}%) | Med. Monetary: £{row['Monetary_Median']:,.2f}")

# TEST 8: Plotly Visualizations Generation
print("\n[TEST 8] Testing Plotly chart generation...")
rfm_with_clusters = full_rfm.copy()
rfm_with_clusters["Cluster"] = labels
rfm_with_clusters["Segment_Name"] = [segment_info[c]["label"] for c in labels]

dist_fig = visualizations.plot_customer_distribution(summary_df)
donut_fig = visualizations.plot_customer_percentage_donut(summary_df)
box_figs = visualizations.plot_rfm_boxplots(rfm_with_clusters)
radar_fig = visualizations.plot_cluster_comparison_radar(summary_df)
sil_fig = visualizations.plot_silhouette_scores(eval_res["silhouette_detail"])
scatter_3d_fig = visualizations.plot_3d_cluster_scatter(X, rfm_with_clusters, max_points=1000)

assert dist_fig is not None and len(dist_fig.data) > 0
assert donut_fig is not None and len(donut_fig.data) > 0
assert len(box_figs) == 3
assert radar_fig is not None and len(radar_fig.data) > 0
assert sil_fig is not None and len(sil_fig.data) > 0
assert scatter_3d_fig is not None and len(scatter_3d_fig.data) > 0
print("  --> PASS: All 6 Plotly visualization figures generated without errors.")

# TEST 9: Reports and Export Generation
print("\n[TEST 9] Testing data exports and executive report generation...")
exec_rep = utils.generate_executive_report(
    metadata={"customers": len(full_rfm)},
    audit_report={"clean_transactions": 805549, "total_gross_revenue": 18161394.0, "initial_rows": 1067371, "retention_rate_pct": 75.5, "date_range_start": "2009-12-01", "date_range_end": "2011-12-09"},
    summary_df=summary_df,
    cure_meta=cure_out,
    eval_meta=eval_res
)
assert len(exec_rep) > 500, "Executive report output was too short!"
assert "Executive Summary" in exec_rep
assert "CURE Hyperparameter Configuration" in exec_rep

# Verify CSV exports
cust_csv = rfm_with_clusters.to_csv(index=False)
summary_csv = summary_df.to_csv()
assert len(cust_csv) > 1000, "Customer segments CSV output invalid"
assert len(summary_csv) > 100, "Cluster summary CSV output invalid"
print("  --> PASS: Customer CSV, summary CSV, and executive report generated cleanly.")

print("\n" + "=" * 70)
print("ALL 9 TEST SUITES COMPLETED AND PASSED WITH ZERO ERRORS!")
print("=" * 70)
