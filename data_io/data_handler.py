"""
Utilities for loading, processing, and normalizing data for distributed computing applications.

This module provides functions for loading data from text files, splitting data among clients,
and normalizing values, with optional fixed-point conversion support.
"""

import os

import numpy as np

from data_io import to_fixed


def load_txt(path: str):
    """Load numerical data from a text file, skipping lines containing 'x'.

    Memory-lean: streams the file into a pre-allocated array instead of the old
    ``readlines()`` + list-of-lists-of-Python-floats, whose peak RSS was ~7x the final
    array (readlines buffer + boxed floats + array all live at once). On a 400k x 300
    dataset the peak drops from ~7 GB to ~1 GB, which is what lets several MPI ranks load
    the same file on one host without OOM.

    An **opt-in** ``<path>.npy`` cache (set ``LOAD_TXT_CACHE=1``, invalidated by mtime) can
    make repeat loads near-instant on a fast local disk. It is OFF by default because on a
    slow shared filesystem the ~1 GB binary read can be *slower* than re-parsing and many
    concurrent readers contend; the lean parse already caps memory, which is what fixes the
    OOM. When enabled, the cache is written by a single process (an ``O_EXCL`` lock elects
    the writer) via a temp file + atomic rename, so concurrent jobs neither race nor storm
    the disk. All caching is best-effort: any failure falls back to a plain parse.

    Args:
        path (str): Path to the text file containing numerical data.

    Returns:
        np.ndarray: (n, d) float64 array of the loaded values (identical to the old loader).
    """
    use_cache = bool(os.environ.get("LOAD_TXT_CACHE"))
    cache = path + ".npy"
    if use_cache:
        try:
            if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
                return np.load(cache)
        except Exception:
            pass  # unreadable/stale cache -> just parse

    def usable(line):
        return ("x" not in line) and line.strip()

    # Pass 1: count usable rows and detect the column count (no large buffers held).
    n_rows, n_cols = 0, None
    with open(path, "r") as f:
        for line in f:
            if not usable(line):
                continue
            if n_cols is None:
                n_cols = len(line.split())
            n_rows += 1
    if not n_rows or not n_cols:
        return np.empty((0, 0))

    # Pass 2: fill a pre-allocated array row by row; peak ~= the final array.
    arr = np.empty((n_rows, n_cols), dtype=np.float64)
    i = 0
    with open(path, "r") as f:
        for line in f:
            if not usable(line):
                continue
            row = np.array(line.split(), dtype=np.float64)
            if row.shape[0] != n_cols:      # skip ragged/short lines defensively
                continue
            arr[i] = row
            i += 1
    arr = arr[:i]

    if use_cache:
        _save_npy_cache(cache, arr)
    return arr


def txt_shape(path: str):
    """(#usable rows, #cols) of a ``load_txt`` file WITHOUT materialising the array.

    Streams the file (skipping 'x'/blank lines, like ``load_txt``) so the MPI server rank
    can obtain the dataset's (n, d) -- needed for ``params.data_size``/``dim`` (hence delta
    and noise calibration) -- without holding the full array in memory. Peak RSS ~ one line.
    """
    n, d = 0, None
    with open(path, "r") as f:
        for line in f:
            if ("x" in line) or (not line.strip()):
                continue
            if d is None:
                d = len(line.split())
            n += 1
    return n, (d or 0)


def _save_npy_cache(cache: str, arr: np.ndarray):
    """Best-effort, single-writer, atomic .npy cache write. A concurrent-safe O_EXCL lock
    elects one writer so many jobs sharing a filesystem don't all rewrite the same cache."""
    lock = cache + ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # atomic; only one winner
    except FileExistsError:
        return  # another process is (or was) writing it
    except Exception:
        return
    try:
        tmp = f"{cache}.tmp.{os.getpid()}"
        with open(tmp, "wb") as fh:      # file handle -> np.save does NOT append ".npy"
            np.save(fh, arr)
        os.replace(tmp, cache)           # atomic publish
    except Exception:
        try:
            os.path.exists(tmp) and os.remove(tmp)
        except Exception:
            pass
    finally:
        try:
            os.remove(lock)
        except Exception:
            pass


def shuffle_and_split(values, clients, proportions=None):
    """
    Randomly shuffle data and split it among multiple clients.

    Args:
        values (np.ndarray): Input data array to be split
        clients (int): Number of clients to split the data among
        proportions (list of float, optional): Relative proportions for splitting data.
            If None, data is split equally. Defaults to None.

    Returns:
        list of np.ndarray: List of data arrays, one for each client

    Example:
        >>> data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> splits = shuffle_and_split(data, clients=2)
        >>> print([split.shape for split in splits])
        [(2, 2), (2, 2)]  # Each client gets 2 samples
    """
    if proportions is None:
        size = len(values) // clients
        sizes = [size for _ in range(clients - 1)]
    else:
        prop_sum = sum(proportions)
        total = values.shape[0]
        sizes = [int(proportions[i] / prop_sum * total) for i in range(clients - 1)]
    np.random.shuffle(values)
    st = 0
    value_lists = []
    for client in range(clients - 1):
        size = sizes[client]
        value_lists.append(values[st: st + size, :])
        st += size
    value_lists.append(values[st:, :])
    return value_lists


def normalize(values, fixed=False):
    """
    Normalize data to [-1, 1] range, with optional fixed-point conversion.

    This function applies min-max normalization to map values to [-1, 1].
    For columns with all equal values, the normalized value is set to 0.

    Args:
        values (np.ndarray): Input array to normalize
        fixed (bool, optional): Whether to convert to fixed-point representation.
            Defaults to False.

    Returns:
        np.ndarray: Normalized values, optionally in fixed-point representation

    Example:
        >>> data = np.array([[1, 2], [3, 4]])
        >>> normalized = normalize(data)
        >>> print(normalized)
        [[-1. -1.]
         [ 1.  1.]]
    """
    mx = values.max(axis=0)
    mn = values.min(axis=0)
    normalized = np.zeros_like(values)
    empty = mx - mn == 0
    normalized[:, empty] = 0.5
    normalized[:, ~empty] = (values[:, ~empty] - mn[~empty]) / (mx[~empty] - mn[~empty])
    normalized = normalized * 2 - 1
    if fixed:
        return to_fixed(normalized)
    else:
        return normalized


def ensure_unit_norm(values, atol=1e-6):
    """Safeguard: guarantee every data point lies on (or inside) the unit L2 ball.

    The DP aggregation calibrates its noise assuming each contributed point has
    L2 norm <= 1 (``utils.ortho_clustering.compute_dp_sigmas_zcdp`` derives
    ``sigma_centers`` from sensitivity 1). Per-feature min-max scaling only bounds
    points to ``[-1, 1]^d``, whose norm can reach ``sqrt(d)`` -- which would make
    the aggregation under-noise. This safeguard enforces the missing precondition:

      - rows already at unit norm (within ``atol``) pass through unchanged
        ("proceed as normal"), so the call is idempotent / a no-op on prepared data;
      - any other non-zero row is rescaled to unit L2 norm;
      - a zero row cannot be put on the unit sphere, so it is left at the origin
        (norm 0 <= 1, still inside the unit ball, so the sensitivity bound holds).

    This is a data-preparation step and MUST run outside any timed/measured region
    so benchmarks reflect the algorithm, not the normalization.

    Args:
        values (np.ndarray): (n, d) data, one point per row.
        atol (float): tolerance for treating a row as already unit-norm.

    Returns:
        np.ndarray: float64 data with every row's L2 norm in {0} U {1}.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D (n, d) array, got shape {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    already_ok = (np.abs(norms - 1.0) <= atol) | (norms <= atol)
    if np.all(already_ok):
        return values                       # already unit-norm: proceed as normal
    safe = np.where(norms == 0.0, 1.0, norms)   # leave zero rows at the origin
    return values / safe
