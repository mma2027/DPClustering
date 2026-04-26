import numpy as np


def _find_sigma_autodp(epsilon, delta, n, batch_size, epochs):
    """Binary-search for the Gaussian noise multiplier that achieves (epsilon, delta)-DP.

    Uses Rényi DP accounting (autodp) for the subsampled Gaussian mechanism composed
    over T = epochs * floor(n / batch_size) steps with sampling rate q = batch_size / n.

    Args:
        epsilon: target privacy budget
        delta: target delta
        n: dataset size
        batch_size: mini-batch size
        epochs: number of SGD epochs

    Returns:
        float: smallest sigma s.t. the total privacy cost ≤ (epsilon, delta)
    """
    from autodp import rdp_bank

    T = epochs * max(1, n // batch_size)
    q = min(1.0, batch_size / n)
    alphas = list(range(2, 256))

    def dp_cost(sigma):
        min_eps = float("inf")
        for alpha in alphas:
            try:
                rdp = T * rdp_bank.RDP_gaussian_subsampled({"prob": q, "sigma": sigma}, alpha)
            except Exception:
                rdp = T * q * rdp_bank.RDP_gaussian({"sigma": sigma}, alpha)
            eps = rdp + np.log(1.0 / delta) / (alpha - 1)
            if eps < min_eps:
                min_eps = eps
        return min_eps

    lo, hi = 0.01, 1000.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if dp_cost(mid) > epsilon:
            lo = mid
        else:
            hi = mid
    return hi


def dpsgd_pca_basis(X, d_prime, epsilon, delta, clip_norm, epochs=10, lr=0.01, batch_size=256):
    """Compute a differentially private PCA basis via DP-SGD.

    Runs stochastic gradient descent on the variance-maximization objective
    -trace((XW)^T (XW)) with per-sample gradient clipping and calibrated
    Gaussian noise, then re-orthonormalizes W via QR after each step.

    Privacy accounting is done via the Rényi DP moments accountant (autodp),
    giving tight (epsilon, delta)-DP guarantees for the subsampled Gaussian
    mechanism composed over all SGD steps.

    Args:
        X: (n, d) data matrix
        d_prime: number of private principal components to return
        epsilon: privacy budget for this computation
        delta: privacy delta for this computation
        clip_norm: per-sample gradient clipping threshold (Frobenius norm)
        epochs: number of passes over the data (default: 10)
        lr: SGD learning rate (default: 0.01)
        batch_size: mini-batch size (default: 256)

    Returns:
        (d, d_eff) orthonormal matrix of private principal components,
        where d_eff = min(d_prime, d)
    """
    n, d = X.shape
    d_eff = min(d_prime, d)

    # Center the data
    X_c = X - X.mean(axis=0)

    # Initialize W as a random orthonormal matrix
    W = random_orthogonal_basis(d, d_eff)

    # Find noise multiplier via moments accountant
    sigma = _find_sigma_autodp(epsilon, delta, n, batch_size, epochs)

    rng = np.random.RandomState(None)
    for _ in range(epochs):
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = X_c[indices[start: start + batch_size]]
            b = len(batch)
            if b == 0:
                continue

            # Accumulate per-sample clipped gradients
            agg_grad = np.zeros_like(W)
            for xi in batch:
                proj = xi @ W                          # (d_eff,)
                g_i = -2.0 * np.outer(xi, proj)        # (d, d_eff)
                g_norm = np.linalg.norm(g_i)
                g_i *= min(1.0, clip_norm / (g_norm + 1e-8))
                agg_grad += g_i

            # Add calibrated Gaussian noise (sensitivity = clip_norm)
            noise = rng.normal(0.0, sigma * clip_norm, size=W.shape)
            noisy_grad = (agg_grad + noise) / b

            # Gradient descent step
            W = W - lr * noisy_grad

            # Re-orthonormalize so sign-based assignment stays meaningful
            W, _ = np.linalg.qr(W)

    return W[:, :d_eff]


def orthogonal_basis(X, d_prime, method="random", seed=42, **kwargs):
    """Dispatcher for orthonormal basis generation.

    Args:
        X: (n, d) data matrix (used for dpsgd_pca; only shape used for random)
        d_prime: desired number of basis vectors
        method: "random" (default) or "dpsgd_pca"
        seed: random seed (used for random; ignored for dpsgd_pca)
        **kwargs: for dpsgd_pca — epsilon, delta, clip_norm required

    Returns:
        (d, d_eff) orthonormal matrix, d_eff = min(d_prime, d)
    """
    if method == "random":
        return random_orthogonal_basis(X.shape[1], d_prime, seed=seed)
    elif method == "dpsgd_pca":
        return dpsgd_pca_basis(
            X, d_prime,
            epsilon=kwargs["epsilon"],
            delta=kwargs["delta"],
            clip_norm=kwargs["clip_norm"],
        )
    else:
        raise ValueError(f"Unknown basis method: {method!r}. Choose 'random' or 'dpsgd_pca'.")


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
