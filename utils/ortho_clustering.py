import numpy as np


def orthogonalize_svd(R):
    """
    Orthogonalize a matrix via economy SVD.

    Args:
        R: (d, k) matrix to orthogonalize

    Returns:
        (d, k) matrix with orthonormal columns
    """
    Q, _, _ = np.linalg.svd(R, full_matrices=False)
    Q = Q / np.linalg.norm(Q, axis=0)
    return Q


def random_orthogonal_basis(d, d_prime, seed=42, orthogonalize=None):
    """
    Generate a random orthonormal basis.

    Draws a random Gaussian matrix and orthogonalizes it using the
    provided method (defaults to SVD).

    Args:
        d: ambient dimensionality (number of rows)
        d_prime: desired number of orthogonal basis vectors
        seed: random seed for reproducibility
        orthogonalize: callable (R) -> Q that takes a (d, k) matrix and
                       returns a (d, k) orthonormal matrix. Defaults to
                       orthogonalize_svd.

    Returns:
        (d, d_eff) array with orthonormal columns, where d_eff = min(d_prime, d)
    """
    if orthogonalize is None:
        orthogonalize = orthogonalize_svd
    d_eff = min(d_prime, d)
    rng = np.random.RandomState(seed)
    R = rng.randn(d, d_eff)
    return orthogonalize(R)


def ortho_assign(values, d_prime, seed=42, basis=None):
    """
    Assign points to clusters via orthogonal projection quadrants.

    Projects each point onto an orthonormal basis and assigns it to a
    quadrant based on the sign of each projection. Produces up to
    2^d_prime clusters.

    Args:
        values: (n, d) array of data points
        d_prime: number of orthogonal basis vectors
        seed: random seed for reproducibility (ignored when basis is provided)
        basis: optional (d, d_prime) orthonormal matrix to use instead of
               generating one via SVD. When provided, seed is ignored.

    Returns:
        (n,) integer array of cluster labels in [0, 2^d_eff - 1]
    """
    n, d = values.shape
    d_eff = min(d_prime, d)

    if basis is not None:
        Q = basis
    else:
        Q = random_orthogonal_basis(d, d_prime, seed=seed)

    # Project points onto orthogonal basis
    projections = values @ Q  # (n, d_eff)

    # Convert sign pattern to cluster ID
    signs = (projections >= 0).astype(int)  # (n, d_eff) binary
    labels = signs @ (2 ** np.arange(Q.shape[1]))  # (n,) integer

    return labels


def noisy_cluster_centers(values, labels, sigma, seed=42):
    """
    Compute cluster centroids with Gaussian noise added to sums.

    Adds isotropic Gaussian noise N(0, sigma^2 I) to each cluster's
    sum vector before dividing by count, following the DP mechanism
    from "Improved Private DP Clustering via Projections".

    Args:
        values: (n, d) array of data points
        labels: (n,) integer cluster labels
        sigma: standard deviation of Gaussian noise added to each cluster sum
        seed: random seed for noise generation

    Returns:
        (k, d) array of noisy cluster centroids
        unique_labels: (k,) sorted array of the label ids
    """
    unique_labels = np.unique(labels)
    d = values.shape[1]
    rng = np.random.RandomState(seed)
    centers = np.empty((len(unique_labels), d))
    for i, lab in enumerate(unique_labels):
        cluster_points = values[labels == lab]
        cluster_sum = cluster_points.sum(axis=0)
        noise = rng.normal(0, sigma, size=d)
        centers[i] = (cluster_sum + noise) / cluster_points.shape[0]
    return centers, unique_labels


def cluster_counts(labels):
    """
    Compute the number of points in each cluster.

    Args:
        labels: (n,) integer cluster labels

    Returns:
        counts: (k,) array of point counts per cluster
        unique_labels: (k,) sorted array of the label ids
    """
    unique_labels = np.unique(labels)
    counts = np.array([np.sum(labels == lab) for lab in unique_labels])
    return counts, unique_labels


def cluster_centers(values, labels):
    """
    Compute the centroid of each cluster.

    Args:
        values: (n, d) array of data points
        labels: (n,) integer cluster labels

    Returns:
        (k, d) array of cluster centroids, where k = number of unique labels.
              Row i is the mean of all points with label unique_labels[i].
        unique_labels: (k,) sorted array of the label ids
    """
    unique_labels = np.unique(labels)
    centers = np.empty((len(unique_labels), values.shape[1]))
    for i, lab in enumerate(unique_labels):
        centers[i] = values[labels == lab].mean(axis=0)
    return centers, unique_labels
