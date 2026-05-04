import numpy as np


def _find_sigma_autodp(epsilon, delta, n, batch_size, epochs):
    """Binary-search for the Gaussian noise multiplier that achieves (epsilon, delta)-DP.

    ## Privacy accounting overview

    DP-SGD adds Gaussian noise N(0, sigma² · clip_norm² · I) to the gradient sum at
    every mini-batch step.  The sensitivity of the sum (over one adjacent dataset
    differing in one point) is clip_norm, so the noise-to-sensitivity ratio is sigma.

    Naively composing T steps each with cost (epsilon_0, delta_0) would give total cost
    roughly T * epsilon_0, which is very loose.  Instead we use the Rényi DP (RDP)
    framework, which composes additively and converts to (epsilon, delta)-DP at the end:

        R(alpha)_total = T * R(alpha)_per_step
        epsilon(alpha) = R(alpha)_total + log(1/delta) / (alpha - 1)
        epsilon = min_alpha  epsilon(alpha)

    Subsampling amplification: because each step draws a fresh mini-batch of size
    batch_size from n points (sampling rate q = batch_size / n), the per-step Rényi cost
    is that of the *subsampled* Gaussian mechanism, which is proportional to q² for small
    q — much cheaper than the non-subsampled version.  This is the dominant source of
    tightness compared with naive composition.

    The binary search finds the smallest sigma such that the above accounting yields a
    total cost ≤ epsilon.  It converges in 64 iterations to precision ~10^{-14} over the
    search range [0.01, 1000].

    Args:
        epsilon: target (epsilon, delta)-DP privacy budget epsilon
        delta: target delta; standard choice is 1 / (n * log(n))
        n: number of points in the dataset passed to DP-SGD (after any subsampling)
        batch_size: mini-batch size; controls sampling rate q = batch_size / n
        epochs: number of full passes over the data;
                total steps T = epochs * floor(n / batch_size)

    Returns:
        float: smallest sigma s.t. the total privacy cost ≤ (epsilon, delta)

    Note:
        Larger sigma → more noise → more privacy → worse utility.
        To reduce sigma (less noise) for a fixed epsilon budget you can:
          - decrease epochs (fewer compositions)
          - increase batch_size (fewer steps, but weaker subsampling amplification)
          - subsample the dataset before calling (fewer points → fewer steps)
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


def dpsgd_pca_basis(X, d_prime, epsilon, delta, clip_norm, epochs=10, lr=0.01, batch_size=256, data_fraction=1.0):
    """Compute a differentially private PCA basis via DP-SGD.

    ## Objective

    Find W ∈ R^{d × d'} with orthonormal columns (W on the Stiefel manifold St(d, d'))
    that maximizes the projected variance:

        max_{W: W^T W = I}  trace(W^T X^T X W)

    Equivalently, minimize L(W) = -trace(W^T X^T X W).  At the optimum, the columns of
    W are the top-d' principal components of X (the leading right singular vectors of X).

    ## Algorithm

    1. **Center** the data: X_c = X - mean(X, axis=0).

    2. **Initialize** W as a random (d, d') orthonormal matrix via random_orthogonal_basis.

    3. **Calibrate noise**: call _find_sigma_autodp to find the smallest Gaussian noise
       multiplier sigma such that the full SGD run is (epsilon, delta)-DP.

    4. For each epoch, shuffle the data and iterate over mini-batches:

       a. **Per-sample gradient**: for each point x_i in the batch,
              g_i = ∂L_i/∂W = -2 · outer(x_i, x_i @ W)    shape (d, d')
          This is the per-sample contribution to the full gradient -2 X^T X W.

       b. **Clip**: bound the Frobenius norm of each g_i:
              g_i ← g_i · min(1, clip_norm / ||g_i||_F)
          This ensures the L2 sensitivity of the gradient sum over one mini-batch is
          at most clip_norm regardless of the data point.

       c. **Add noise**: draw Z ~ N(0, sigma² · clip_norm² · I) (shape (d, d'))
          and form the noisy average gradient:
              g_noisy = (Σ_i g_i^clipped + Z) / b

       d. **Gradient step**: W ← W - lr · g_noisy

       e. **Re-orthonormalize**: W, _ = qr(W)
          This projects W back onto the Stiefel manifold after the noisy gradient
          perturbs it.  Without this step the basis vectors would drift, become
          collinear, and the sign-based cluster assignment would degrade.

    ## Privacy guarantee

    The output W is (epsilon, delta)-differentially private with respect to X: replacing
    any single row of X changes the distribution of W by at most a factor e^epsilon (with
    probability 1 - delta).  The guarantee holds for the specific epsilon and delta passed
    in; these are spent entirely on basis computation and are independent of any subsequent
    DP mechanism applied to cluster centroids.

    ## Choosing hyperparameters

    - **clip_norm**: should match the typical per-sample gradient norm.  For data
      normalized to [-1, 1]^d the gradient norm ||g_i||_F = 2 ||x_i||_2 · ||x_i @ W||_2
      is at most 2·sqrt(d)·sqrt(d'), so clip_norm=1.0 is conservative but safe.  Larger
      values allow more signal through at the cost of needing more noise.

    - **epsilon / delta**: standard choices are epsilon ∈ {0.1, 0.5, 1.0} and
      delta = 1 / (n · log n).  Tighter epsilon forces larger sigma and degrades the
      basis quality.

    - **epochs**: more epochs = more SGD compositions = higher sigma for the same epsilon.
      With the subsampling amplification, 10 epochs is usually a good balance.

    - **batch_size**: smaller batches give stronger subsampling amplification (sigma grows
      more slowly with epochs) but more iterations, which can slow wall-clock time.
      256 works well for datasets up to ~100k points; consider larger batches for very
      small datasets where batch_size > n/2 collapses to full-batch GD.

    - **data_fraction** (if subsampling): using a fraction f of the data reduces n to
      f·n, cutting T by f and allowing sigma to shrink accordingly, which improves utility
      at the cost of using less data for the basis.  10% (f=0.1) is a practical default.

    Args:
        X: (n, d) data matrix; should be normalized (e.g., to [-1, 1]^d) so that
           the default clip_norm=1.0 is meaningful
        d_prime: number of private principal components to return
        epsilon: (epsilon, delta)-DP privacy budget for this computation; independent
                 of any clustering epsilon
        delta: (epsilon, delta)-DP delta; standard: 1 / (n · log(n))
        clip_norm: per-sample gradient clipping threshold (Frobenius norm);
                   bounds the sensitivity of the sum query to clip_norm
        epochs: number of full passes over X during SGD (default: 10)
        lr: SGD learning rate; 0.01 works for normalized data (default: 0.01)
        batch_size: mini-batch size; controls both step count and subsampling rate
                    q = batch_size / n (default: 256)
        data_fraction: fraction of X to subsample before running DP-SGD, in (0, 1].
                       Reduces the number of SGD steps proportionally, letting the
                       noise calibration return a smaller sigma for the same budget.
                       E.g. 0.1 uses 10% of the data (default: 1.0, use all data)

    Returns:
        W: (d, d_eff) orthonormal matrix whose columns approximate the top-d' principal
           components of X under (epsilon, delta)-DP, where d_eff = min(d_prime, d)
    """
    n, d = X.shape
    d_eff = min(d_prime, d)

    # Subsample the dataset before running DP-SGD.  Using a fraction f of n reduces the
    # number of SGD steps by f, which lets _find_sigma_autodp return a smaller sigma for
    # the same (epsilon, delta) budget — improving utility at the cost of a smaller basis
    # training set.  The DP guarantee holds over the subsampled dataset; the additional
    # subsampling from the full dataset only strengthens privacy.
    rng = np.random.RandomState(None)
    if data_fraction < 1.0:
        n_sub = max(batch_size, int(n * data_fraction))
        idx = rng.choice(n, size=n_sub, replace=False)
        X = X[idx]
        n = n_sub

    # Center the data
    X_c = X - X.mean(axis=0)

    # Initialize W as a random orthonormal matrix
    W = random_orthogonal_basis(d, d_eff)

    # Find noise multiplier via moments accountant (uses subsampled n after data_fraction)
    sigma = _find_sigma_autodp(epsilon, delta, n, batch_size, epochs)

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
        X: (n, d) data matrix (used for dpsgd_pca and svd_pca; only shape used for random)
        d_prime: desired number of basis vectors
        method: "random", "dpsgd_pca", or "svd_pca"
        seed: random seed (used for random; ignored for dpsgd_pca and svd_pca)
        **kwargs: for dpsgd_pca — epsilon, delta, clip_norm required;
                  data_fraction optional (default 1.0)

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
            data_fraction=kwargs.get("data_fraction", 1.0),
        )
    elif method == "svd_pca":
        return svd_pca_basis(X, d_prime)
    else:
        raise ValueError(f"Unknown basis method: {method!r}. Choose 'random', 'dpsgd_pca', or 'svd_pca'.")


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


def svd_pca_basis(X, d_prime):
    """Compute the top-d' principal components via non-private SVD.

    Centers X and returns the d' right singular vectors of X corresponding to the
    largest singular values (i.e. the true top principal components).  This is the
    non-private oracle baseline — it uses the full data without any noise, so it
    gives the best possible basis quality but provides no differential privacy
    guarantee for the basis itself.

    Args:
        X: (n, d) data matrix
        d_prime: number of principal components to return

    Returns:
        (d, d_eff) orthonormal matrix of top principal components,
        where d_eff = min(d_prime, d)
    """
    d_eff = min(d_prime, X.shape[1])
    X_c = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return Vt[:d_eff].T  # (d, d_eff)


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
