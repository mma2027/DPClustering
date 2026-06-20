import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import Iterable, Tuple

plt.rcParams.update({'font.size': 16})


def set_seed(seed):
    np.random.seed(seed)


def mean_confidence_interval(vals: Iterable[float], confidence: float = 0.95) -> Tuple[float, float]:
    a = np.array(vals, dtype=float)
    a = a[~np.isnan(a)]
    n = a.size
    if n == 0:
        raise ValueError("mean_confidence_interval requires at least one non-NaN value.")
    m = float(np.mean(a))
    if n < 2:
        return m, 0.0
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    return m, h


def plot_clusters(centroids, values):
    plt.clf()
    distances = distance_matrix_squared(values, centroids)
    associations = np.argmin(distances, axis=1)
    colors = ['r', 'g', 'b', 'y', 'darkgray', 'cyan', 'pink', 'orange', 'purple', 'olive', 'gray', 'brown', 'teal',
              'yellowgreen', 'lightcoral', 'lightpink', 'peru', 'tomato', 'gold', 'magenta']
    k = centroids.shape[0]
    for cluster in range(k):
        vals = values[associations == cluster]
        plt.scatter(vals[:, 0], vals[:, 1], color=colors[cluster % len(colors)])
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black')


def distance_matrix_squared(X, Y):
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y, computed without the (n, k, d)
    # broadcast intermediate (which OOMs for large n / high d). Result is (n, k).
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    d2 = (np.sum(X * X, axis=1)[:, None]
          + np.sum(Y * Y, axis=1)[None, :]
          - 2.0 * (X @ Y.T))
    np.maximum(d2, 0, out=d2)   # clip tiny negatives from floating-point error
    return d2


def nearest_centroid_sq(X, Y, mem_budget=256_000_000):
    """Per-row nearest centroid index and its squared distance, in row-chunks.

    Returns (associations, min_sq_dist) without ever materializing the full
    (n, k) matrix -- which is ~6 GB at glove100 n/clients x k=4000 and OOM-kills
    the node. Peak memory is bounded by `mem_budget`. Equivalent to
    argmin / min over distance_matrix_squared(X, Y).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, k = X.shape[0], Y.shape[0]
    y2 = np.sum(Y * Y, axis=1)
    chunk = max(1, int(mem_budget // (8 * max(k, 1))))
    assoc = np.empty(n, dtype=np.int64)
    min_d2 = np.empty(n, dtype=float)
    for s in range(0, n, chunk):
        Xc = X[s:s + chunk]
        d2 = np.sum(Xc * Xc, axis=1)[:, None] + y2[None, :] - 2.0 * (Xc @ Y.T)
        a = np.argmin(d2, axis=1)
        assoc[s:s + chunk] = a
        m = d2[np.arange(d2.shape[0]), a]
        np.maximum(m, 0, out=m)
        min_d2[s:s + chunk] = m
    return assoc, min_d2
