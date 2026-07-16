"""
This module provides evaluation metrics for clustering algorithms.

It implements various metrics to assess clustering quality, including:
- Normalized Intra-cluster Variance (NICV)
- Between-Cluster Sum of Squares (BCSS)
- Empty cluster detection
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Dunn Index
- Mean Cosine Similarity (average cosine similarity of each point to its centroid)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def safe_metric(func, default, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Warning: {func.__name__} couldn't be calculated: {e}")
        return default


def evaluate(centroids, values, gt_centroids, metrics="nicv"):
    """
    Evaluates the quality of a clustering solution using multiple metrics.

    This function computes several clustering evaluation metrics:
    1. Normalized Intra-cluster Variance (NICV): Measures the average variance within clusters
    2. Between-Cluster Sum of Squares (BCSS): Measures the separation between clusters
    3. Empty Clusters: Counts clusters with no assigned points
    4. Silhouette Score: Measures how well each object lies within its cluster
    5. Davies-Bouldin Index: Ratio of within-cluster and between-cluster distances
    6. Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion
    7. Dunn Index: Ratio of min between-cluster distance to max within-cluster distance
    8. Mean Squared Error: Average squared distance between assigned centroids and ground truth centroids

    Parameters
    ----------
    centroids : numpy.ndarray
        Array of cluster centroids, shape (n_clusters, n_features)
    values : numpy.ndarray
        Array of data points to be clustered, shape (n_samples, n_features)
    gt_centroids : numpy.ndarray
        Array of ground truth centroids, shape (n_clusters, n_features)
    metrics : str or list, optional
        Metrics to compute. Default is "nicv". If a list, can include:
        - "nicv": Normalized Intra-cluster Variance
        - "bcss": Between-Cluster Sum of Squares
        - "empty_clusters": Count of empty clusters
        - "silhouette": Silhouette Score
        - "davies_bouldin": Davies-Bouldin Index
        - "calinski_harabasz": Calinski-Harabasz Index
        - "dunn_index": Dunn Index
        - "mse": Mean Squared Error
        - "cosine_similarity": Mean cosine similarity of each point to its
          assigned centroid (averaged over all points)
        - "all": Compute all metrics

    Returns
    -------
    dict
        Dictionary containing the computed metrics

    Raises
    ------
    ValueError
        If no non-empty clusters are detected in the solution
    """
    associations = _assign_nearest(values, centroids)
    non_empty_clusters = np.unique(associations).size
    empty_clusters = centroids.shape[0] - non_empty_clusters

    if non_empty_clusters == 0:
        raise ValueError("No non-empty clusters detected.")

    # Define metric functions and their configurations
    metric_config = {
        "nicv": {
            "name": "Normalized Intra-cluster Variance (NICV)",
            "func": lambda: evaluate_NICV(associations, centroids, values),
            "default": 0,
            "requires_multi_cluster": False
        },
        "bcss": {
            "name": "Between-Cluster Sum of Squares (BCSS)",
            "func": lambda: evaluate_BCSS(associations, centroids, values),
            "default": 0,
            "requires_multi_cluster": False
        },
        "empty_clusters": {
            "name": "Empty Clusters",
            "func": lambda: empty_clusters,
            "default": 0,
            "requires_multi_cluster": False
        },
        "mse": {
            "name": "Mean Squared Error",
            "func": lambda: evaluate_MSE(centroids, gt_centroids),
            "default": 0,
            "requires_multi_cluster": False
        },
        "cosine_similarity": {
            "name": "Mean Cosine Similarity",
            "func": lambda: evaluate_mean_cosine_similarity(associations, centroids, values),
            "default": 0,
            "requires_multi_cluster": False
        },
        "silhouette": {
            "name": "Silhouette Score",
            "func": lambda: safe_metric(silhouette_score, -1, values, associations),
            "default": -1,
            "requires_multi_cluster": True
        },
        "davies_bouldin": {
            "name": "Davies-Bouldin Index",
            "func": lambda: safe_metric(davies_bouldin_score, np.inf, values, associations),
            "default": np.inf,
            "requires_multi_cluster": True
        },
        "calinski_harabasz": {
            "name": "Calinski-Harabasz Index",
            "func": lambda: safe_metric(calinski_harabasz_score, 0, values, associations),
            "default": 0,
            "requires_multi_cluster": True
        },
        "dunn_index": {
            "name": "Dunn Index",
            "func": lambda: safe_metric(evaluate_dunn_index, 0, associations, values),
            "default": 0,
            "requires_multi_cluster": True
        }
    }

    # Determine which metrics to compute
    if metrics == "all":
        metrics_list = list(metric_config.keys())
    elif isinstance(metrics, str):
        metrics_list = [metrics]
    else:
        metrics_list = metrics

    # Compute requested metrics
    results = {}
    for metric in metrics_list:
        if metric in metric_config:
            config = metric_config[metric]
            if config["requires_multi_cluster"] and non_empty_clusters < 2:
                results[config["name"]] = config["default"]
            else:
                results[config["name"]] = config["func"]()

    return results


def _assign_nearest(values, centroids, mem_budget=256_000_000):
    """Index of the nearest centroid for each point, in row-chunks.

    Never materializes the full (n_samples, n_centroids) distance matrix (which
    OOMs when there are many leaf-centroids, e.g. ~5k leaves x 400k points = 16 GB).
    Uses ||v-c||^2 = ||v||^2 + ||c||^2 - 2 v.c^T and drops the per-row ||v||^2
    constant (irrelevant for argmin). Peak memory is bounded by `mem_budget`.
    """
    centroids = np.asarray(centroids, dtype=float)
    n_centroids = centroids.shape[0]
    c2 = np.einsum("ij,ij->i", centroids, centroids)          # ||c||^2, (n_centroids,)
    n = values.shape[0]
    chunk = max(1, int(mem_budget // (8 * max(n_centroids, 1))))
    assoc = np.empty(n, dtype=np.int64)
    for s in range(0, n, chunk):
        v = values[s:s + chunk]
        d2 = c2[None, :] - 2.0 * (v @ centroids.T)            # (chunk, n_centroids)
        assoc[s:s + chunk] = np.argmin(d2, axis=1)
    return assoc


def evaluate_NICV(associations, centroids, values):
    """
    Calculates the Normalized Intra-cluster Variance (NICV).

    NICV is the Within-Cluster Sum of Squares (WCSS) normalized by the number
    of data points. It represents the average variance of points within their
    clusters, with lower values indicating more compact clusters.

    Parameters
    ----------
    associations : numpy.ndarray
        Array of cluster assignments for each point, shape (n_samples,)
    centroids : numpy.ndarray
        Array of cluster centroids, shape (n_clusters, n_features)
    values : numpy.ndarray
        Array of data points, shape (n_samples, n_features)

    Returns
    -------
    float
        The NICV value (WCSS divided by number of samples)
    """
    return evaluate_WCSS(associations, centroids, values) / values.shape[0]


def evaluate_WCSS(associations, centroids, values):
    """
    Calculates the Within-Cluster Sum of Squares (WCSS).

    WCSS measures the compactness of clusters by summing the squared distances
    between each point and its assigned cluster centroid. Lower values indicate
    more compact clusters.

    Parameters
    ----------
    associations : numpy.ndarray
        Array of cluster assignments for each point, shape (n_samples,)
    centroids : numpy.ndarray
        Array of cluster centroids, shape (n_clusters, n_features)
    values : numpy.ndarray
        Array of data points, shape (n_samples, n_features)

    Returns
    -------
    float
        The WCSS value - sum of squared distances between points and their centroids
    """
    return sum([np.sum((values[associations == cluster] - centroids[cluster]) ** 2) for cluster in
                range(centroids.shape[0]) if np.sum(associations == cluster) > 0])


def evaluate_mean_cosine_similarity(associations, centroids, values, mem_budget=256_000_000):
    """
    Calculates the mean cosine similarity of each point to its assigned centroid.

    For every data point, computes the cosine similarity between the point and
    the centroid of the cluster it was assigned to, then averages over all
    points. Values lie in [-1, 1]; higher values indicate points that are more
    tightly aligned (in angle) with their cluster centroid, i.e. a better
    clustering under cosine geometry.

    A point (or centroid) with zero norm has no defined direction; its cosine
    similarity is treated as 0 (neither aligned nor opposed).

    Computed in row-chunks so the full (n_samples, n_features) gathered-centroid
    matrix is never materialized (matches the memory bound used elsewhere in
    this module for large datasets, e.g. glove100 with ~400k points).

    Parameters
    ----------
    associations : numpy.ndarray
        Array of cluster assignments for each point, shape (n_samples,)
    centroids : numpy.ndarray
        Array of cluster centroids, shape (n_clusters, n_features)
    values : numpy.ndarray
        Array of data points, shape (n_samples, n_features)
    mem_budget : int, optional
        Approximate peak-memory budget in bytes for the chunked computation.

    Returns
    -------
    float
        Mean cosine similarity over all points, in [-1, 1]. Returns 0.0 when
        there are no points.
    """
    values = np.asarray(values, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    n, d = values.shape
    if n == 0:
        return 0.0

    c_norms = np.linalg.norm(centroids, axis=1)                  # ||c||, (n_clusters,)
    chunk = max(1, int(mem_budget // (8 * max(d, 1))))
    total = 0.0
    for s in range(0, n, chunk):
        v = values[s:s + chunk]
        assigned = centroids[associations[s:s + chunk]]          # (chunk, d)
        dots = np.einsum("ij,ij->i", v, assigned)                # v . c
        denom = np.linalg.norm(v, axis=1) * c_norms[associations[s:s + chunk]]
        # Zero-norm point or centroid -> undefined direction -> similarity 0.
        cos = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)
        total += cos.sum()
    return total / n


def evaluate_BCSS(associations, centroids, values):
    """
    Calculates the Between-Cluster Sum of Squares (BCSS).

    BCSS measures the separation between clusters by summing the weighted squared
    distances between each cluster centroid and the overall data centroid. Higher
    values indicate better-separated clusters.

    Parameters
    ----------
    associations : numpy.ndarray
        Array of cluster assignments for each point, shape (n_samples,)
    centroids : numpy.ndarray
        Array of cluster centroids, shape (n_clusters, n_features)
    values : numpy.ndarray
        Array of data points, shape (n_samples, n_features)

    Returns
    -------
    float
        The BCSS value - weighted sum of squared distances between centroids
        and the overall centroid
    """
    overall_centroid = np.mean(values, axis=0)
    return sum(
        [(np.linalg.norm(centroids[cluster] - overall_centroid) ** 2) * np.sum(associations == cluster) for cluster in
         range(centroids.shape[0])])


def evaluate_dunn_index(associations, values):
    """
    Calculates the Dunn Index for a clustering.

    The Dunn index is the ratio of the minimum inter-cluster distance to the
    maximum intra-cluster distance. Higher values indicate better clustering.

    Parameters
    ----------
    associations : numpy.ndarray
        Array of cluster assignments for each point, shape (n_samples,)
    values : numpy.ndarray
        Array of data points, shape (n_samples, n_features)

    Returns
    -------
    float
        The Dunn index
    """
    unique_clusters = np.unique(associations)
    n_clusters = unique_clusters.size

    # Check if we have more than one cluster
    if n_clusters < 2:
        return np.nan

    # Calculate minimum inter-cluster distance
    min_inter_dist = float('inf')
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = values[associations == unique_clusters[i]]
            cluster_j = values[associations == unique_clusters[j]]

            # Calculate minimum distance between points in cluster i and j
            if len(cluster_i) > 0 and len(cluster_j) > 0:
                inter_dist = np.min(cdist(cluster_i, cluster_j))
                min_inter_dist = min(min_inter_dist, inter_dist)

    # Calculate maximum intra-cluster distance
    max_intra_dist = 0
    for i in range(n_clusters):
        cluster_i = values[associations == unique_clusters[i]]

        # Skip empty clusters
        if len(cluster_i) <= 1:
            continue

        # Calculate maximum distance between points in the same cluster
        intra_dist = np.max(cdist(cluster_i, cluster_i))
        max_intra_dist = max(max_intra_dist, intra_dist)

    # Handle edge cases
    if max_intra_dist == 0 or min_inter_dist == float('inf'):
        return np.nan

    return min_inter_dist / max_intra_dist


def evaluate_MSE(centroids, gt_centroids):
    """
    Calculates the Mean Squared Error between predicted and ground truth centroids.
    
    Uses the Hungarian algorithm to find the optimal assignment between predicted
    and ground truth centroids, then computes the average squared distance.

    Parameters
    ----------
    centroids : numpy.ndarray
        Array of predicted cluster centroids, shape (n_clusters, n_features)
    gt_centroids : numpy.ndarray
        Array of ground truth centroids, shape (n_clusters, n_features)

    Returns
    -------
    float
        The Mean Squared Error between optimally matched centroids
    """
    cost_matrix = cdist(centroids, gt_centroids)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_sq_dists = cost_matrix[row_ind, col_ind]
    return matched_sq_dists.mean()
