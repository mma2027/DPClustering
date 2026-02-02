import numpy as np


def ortho_assign(values, d_prime, seed=42):
    """
    Assign points to clusters via orthogonal projection quadrants.

    Generates d' orthogonal basis vectors via SVD of a random matrix,
    then assigns each point to a quadrant based on the sign of its
    dot product with each basis vector. Produces up to 2^d_prime clusters.

    Args:
        values: (n, d) array of data points
        d_prime: number of orthogonal basis vectors
        seed: random seed for reproducibility

    Returns:
        (n,) integer array of cluster labels in [0, 2^d_prime - 1]
    """
    n, d = values.shape

    # Can't have more orthogonal vectors than dimensions
    d_eff = min(d_prime, d)

    rng = np.random.RandomState(seed)

    # Generate random matrix and extract orthogonal columns via SVD
    R = rng.randn(d, d_eff)
    Q, _, _ = np.linalg.svd(R, full_matrices=False)
    # Q is (d, d_eff) with orthonormal columns after economy SVD

    # Normalize columns explicitly
    Q = Q / np.linalg.norm(Q, axis=0)

    # Project points onto orthogonal basis
    projections = values @ Q  # (n, d_eff)

    # Convert sign pattern to cluster ID
    signs = (projections >= 0).astype(int)  # (n, d_eff) binary
    labels = signs @ (2 ** np.arange(d_eff))  # (n,) integer

    return labels
