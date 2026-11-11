"""
CURE (Clustering Using REpresentatives) Algorithm Implementation
Reference: Guha, Rastogi, Shim (1998) "CURE: An Efficient Clustering Algorithm for Large Databases"

Key Algorithmic Components:
1. Well-scattered representative point selection:
   - 1st point: argmax distance to cluster centroid (farthest from mean).
   - Remaining c-1 points: farthest-point traversal (maximizing minimum distance to chosen points).
2. Shrinking:
   - Representative points are shrunk towards cluster centroid by factor alpha (0 <= alpha <= 1).
3. Distance metric:
   - Distance between two clusters is the minimum Euclidean distance between their representative sets.
4. Outlier Handling (Guha et al. Section 4.2):
   - Phase 1: Pruning slow-growing clusters (size <= 2) at intermediate active cluster count.
5. Assignment of remaining points:
   - Fast chunked batch assignment to the nearest representative set across all clusters.
"""

import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def select_representatives(points: np.ndarray, number: int = 8, shrink: float = 0.3) -> np.ndarray:
    """
    Select well-scattered representative points for a cluster and shrink them towards the centroid.
    
    Guha et al. (1998) specification:
    - If points <= number, all points are representatives.
    - 1st representative: Point in cluster farthest from centroid (argmax norm(point - centroid)).
    - Next representatives: Point in cluster farthest from previously chosen points.
    - Shrinking: p_rep = p + alpha * (centroid - p) = (1 - alpha) * p + alpha * centroid.
    """
    points = np.asarray(points, dtype=float)
    n_pts = len(points)
    if n_pts == 0:
        raise ValueError("Cannot select representatives from an empty cluster.")
    
    centroid = points.mean(axis=0)
    
    if n_pts <= number:
        reps = points.copy()
        return reps + shrink * (centroid - reps)
    
    # 1. First representative: Farthest point from centroid (Guha et al.)
    dists_from_centroid = np.linalg.norm(points - centroid, axis=1)
    first_idx = int(np.argmax(dists_from_centroid))
    
    selected_indices = [first_idx]
    min_dist = cdist(points, points[[first_idx]]).ravel()
    min_dist[first_idx] = -np.inf
    
    # 2. Remaining representatives: Farthest-point traversal
    for _ in range(1, number):
        nxt_idx = int(np.argmax(min_dist))
        selected_indices.append(nxt_idx)
        min_dist = np.minimum(min_dist, cdist(points, points[[nxt_idx]]).ravel())
        min_dist[nxt_idx] = -np.inf
        
    reps = points[selected_indices].copy()
    
    # 3. Shrink representative points towards the cluster centroid
    shrunk_reps = reps + shrink * (centroid - reps)
    return shrunk_reps


def cure_fit_sample(
    X: np.ndarray,
    sample_indices: np.ndarray,
    k: int = 4,
    number: int = 8,
    shrink: float = 0.3,
    outlier_handling: bool = True,
    prune_ratio: float = 0.12,
    dist_metric: str = "min",
    progress_callback = None,
):
    """
    Hierarchical representative-point clustering on sample data.
    
    Uses an active pairwise distance matrix for O(1) distance lookups and vector-accelerated
    updates, maintaining O(N_sample) active state.
    """
    sample_indices = np.asarray(sample_indices, dtype=int)
    n_sample = len(sample_indices)
    
    if not (2 <= k < n_sample):
        raise ValueError(f"k must be between 2 and sample size {n_sample}.")
    if not (0.0 <= shrink <= 1.0):
        raise ValueError("Shrink factor alpha must be between 0.0 and 1.0.")
        
    # Initialize clusters as singletons
    clusters = {i: [int(sample_indices[i])] for i in range(n_sample)}
    reps = {i: X[[int(sample_indices[i])]].copy() for i in range(n_sample)}
    active_ids = list(range(n_sample))
    
    # Initial pairwise distance matrix between singletons
    sample_points = X[sample_indices]
    dist_matrix = cdist(sample_points, sample_points)
    np.fill_diagonal(dist_matrix, np.inf)
    
    outliers = []
    pruned_phase1_done = False
    prune_threshold = max(k * 4, int(n_sample * prune_ratio))
    
    def calc_rep_distance(r1: np.ndarray, r2: np.ndarray) -> float:
        if dist_metric == "mean":
            return float(cdist(r1, r2).mean())
        # Default CURE: minimum distance between representative points
        return float(cdist(r1, r2).min())
        
    total_merges = n_sample - k
    merges_done = 0
    
    while len(active_ids) > k:
        # Phase 1 Outlier Elimination (Guha et al.)
        if outlier_handling and not pruned_phase1_done and len(active_ids) <= prune_threshold:
            keep_mask = np.ones(len(active_ids), dtype=bool)
            new_active = []
            for idx, c_id in enumerate(active_ids):
                # Clusters with size <= 2 that failed to grow are marked as outliers
                if len(clusters[c_id]) <= 2:
                    outliers.extend(clusters[c_id])
                    keep_mask[idx] = False
                else:
                    new_active.append(c_id)
                    
            if len(new_active) >= k:
                active_ids = new_active
                dist_matrix = dist_matrix[np.ix_(keep_mask, keep_mask)]
                pruned_phase1_done = True
                if len(active_ids) <= k:
                    break
            else:
                # If pruning leaves fewer than k, retain all active
                pruned_phase1_done = True
                
        # Find closest pair of active clusters
        min_flat_idx = np.argmin(dist_matrix)
        i_idx, j_idx = np.unravel_index(min_flat_idx, dist_matrix.shape)
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
            
        c_a = active_ids[i_idx]
        c_b = active_ids[j_idx]
        
        # Merge cluster b into cluster a
        merged_members = clusters[c_a] + clusters[c_b]
        clusters[c_a] = merged_members
        reps[c_a] = select_representatives(X[merged_members], number=number, shrink=shrink)
        
        # Remove cluster b
        del clusters[c_b]
        del reps[c_b]
        active_ids.pop(j_idx)
        dist_matrix = np.delete(dist_matrix, j_idx, axis=0)
        dist_matrix = np.delete(dist_matrix, j_idx, axis=1)
        
        # Recompute distances from merged cluster c_a to all other active clusters
        for other_idx, other_id in enumerate(active_ids):
            if other_id == c_a:
                dist_matrix[i_idx, other_idx] = np.inf
                dist_matrix[other_idx, i_idx] = np.inf
            else:
                d = calc_rep_distance(reps[c_a], reps[other_id])
                dist_matrix[i_idx, other_idx] = d
                dist_matrix[other_idx, i_idx] = d
                
        merges_done += 1
        if progress_callback and (merges_done % 20 == 0 or len(active_ids) == k):
            frac = min(1.0, merges_done / max(1, total_merges))
            progress_callback(frac, f"CURE hierarchical merging: {len(active_ids)} clusters remaining...")
            
    final_clusters = [clusters[c_id] for c_id in active_ids]
    final_reps = [reps[c_id] for c_id in active_ids]
    
    return final_clusters, final_reps, outliers


def assign_customers_to_clusters(
    X: np.ndarray,
    representative_sets: list,
    chunk_size: int = 4000,
) -> np.ndarray:
    """
    Assign every customer to the cluster whose representative set has the minimum distance.
    Processed in vectorized chunks for memory efficiency and high speed.
    """
    n = len(X)
    labels = np.empty(n, dtype=int)
    
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        chunk = X[start:stop]
        # Distance from each customer in chunk to closest rep of each cluster
        cluster_distances = [
            cdist(chunk, rep_set).min(axis=1)
            for rep_set in representative_sets
        ]
        # Assign 1-indexed cluster label
        labels[start:stop] = np.column_stack(cluster_distances).argmin(axis=1) + 1
        
    return labels


def run_cure(
    X: np.ndarray,
    k: int = 4,
    sample_size: int = 1000,
    representatives_per_cluster: int = 8,
    shrink_factor: float = 0.3,
    outlier_handling: bool = True,
    random_seed: int = 42,
    progress_callback = None,
) -> dict:
    """
    Complete CURE clustering workflow:
    1. Sample extraction.
    2. Hierarchical clustering with representative points, shrinking, and Guha outlier pruning.
    3. Full-dataset assignment to nearest representative sets.
    4. Validation and metadata generation.
    """
    t_start = time.time()
    n = len(X)
    
    sample_size = min(int(sample_size), n)
    if sample_size <= k:
        raise ValueError(f"Sample size ({sample_size}) must be greater than number of clusters ({k}).")
        
    rng = np.random.default_rng(random_seed)
    sample_indices = rng.choice(n, size=sample_size, replace=False)
    
    if progress_callback:
        progress_callback(0.05, f"Selected random sample of {sample_size:,} customers (seed={random_seed})...")
        
    sample_clusters, rep_sets, sample_outliers = cure_fit_sample(
        X=X,
        sample_indices=sample_indices,
        k=k,
        number=representatives_per_cluster,
        shrink=shrink_factor,
        outlier_handling=outlier_handling,
        progress_callback=progress_callback,
    )
    
    if progress_callback:
        progress_callback(0.85, "Assigning all customers to nearest CURE representative sets...")
        
    # Assign entire dataset to nearest representative sets
    labels = assign_customers_to_clusters(X, rep_sets)
    
    # Integrity check: verify that all customers received a valid cluster assignment
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise RuntimeError("CURE produced fewer than 2 non-empty clusters.")
    if np.any(labels <= 0) or np.any(labels > k):
        raise RuntimeError(f"Invalid cluster labels detected: {unique_labels}")
        
    t_elapsed = time.time() - t_start
    
    # Compute cluster centroids in scaled space
    cluster_centroids = [
        X[labels == c_id].mean(axis=0) for c_id in range(1, k + 1)
    ]
    
    # Cluster counts
    counts = pd.Series(labels).value_counts().sort_index().to_dict()
    
    return {
        "labels": labels,
        "representative_sets": rep_sets,
        "sample_clusters": sample_clusters,
        "sample_indices": sample_indices,
        "sample_outliers_count": len(sample_outliers),
        "cluster_centroids": cluster_centroids,
        "counts": counts,
        "k": k,
        "sample_size": sample_size,
        "representatives": representatives_per_cluster,
        "shrink": shrink_factor,
        "outlier_handling": outlier_handling,
        "seed": random_seed,
        "runtime_seconds": round(t_elapsed, 2),
    }
