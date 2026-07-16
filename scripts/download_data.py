"""
Download and prepare datasets for FastLloyd experiments using scikit-learn.

Generates .txt files in the data/ directory in the format expected by data_handler.py:
  - Space-separated floating point values
  - One sample per row
  - Features only (no labels)

Datasets prepared:
  - iris (3 clusters, 4 dims, 150 samples)
  - wine (3 clusters, 13 dims, 178 samples)
  - breast (2 clusters, 30 dims, 569 samples)
  - house (3 clusters, 8 dims, ~20k samples)
  - yeast (10 clusters, 8 dims) - from UCI via URL
  - adult (3 clusters, 14 dims) - from UCI via URL
  - mnist (10 clusters, 64 dims, 1797 samples) - sklearn digits
  - s1 (15 clusters, 2 dims) - synthetic approximation
  - lsun (3 clusters, 2 dims) - synthetic approximation
  - birch2 (100 clusters, 2 dims) - synthetic approximation

Usage:
  python scripts/download_data.py
"""

import os
import sys
import numpy as np
from pathlib import Path

# Ensure we can import from project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data_io import normalize, ensure_unit_norm


def save_dataset(name: str, data: np.ndarray, data_dir: Path):
    """Normalize to final form and save as a space-separated text file.

    All preprocessing lives here, in data preparation, so the experiment pipeline
    can load the files as-is (and so the cost is OUTSIDE any timed/measured region):

      1. per-feature min-max to ``[-1, 1]`` (puts heterogeneous feature scales on a
         common range), then
      2. ``ensure_unit_norm`` -- per-point L2 normalization onto the unit sphere.

    Step 2 is the privacy safeguard: the DP aggregation calibrates its noise for
    points of L2 norm <= 1, but min-max alone only bounds points to ``[-1, 1]^d``
    (norm up to ``sqrt(d)``). Writing unit-norm data here guarantees the assumption
    holds at aggregation time without any per-query normalization.
    """
    data = ensure_unit_norm(normalize(np.asarray(data, dtype=float), fixed=False))
    path = data_dir / f"{name}.txt"
    np.savetxt(path, data, fmt="%.6f", delimiter=" ")
    print(f"  {name}.txt: {data.shape[0]} samples, {data.shape[1]} features (unit-norm)")


def prepare_sklearn_datasets(data_dir: Path):
    """Prepare datasets available directly from scikit-learn."""
    from sklearn.datasets import (
        load_iris, load_wine, load_breast_cancer,
        load_digits, fetch_california_housing
    )

    # iris: 150 samples, 4 features, 3 clusters
    iris = load_iris()
    save_dataset("iris", iris.data, data_dir)

    # wine: 178 samples, 13 features, 3 clusters
    wine = load_wine()
    save_dataset("wine", wine.data, data_dir)

    # breast: 569 samples, 30 features, 2 clusters
    breast = load_breast_cancer()
    save_dataset("breast", breast.data, data_dir)

    # mnist (using sklearn digits: 8x8 images, 1797 samples, 10 classes)
    digits = load_digits()
    save_dataset("mnist", digits.data, data_dir)

    # house: California Housing, ~20k samples, 8 features, 3 clusters
    housing = fetch_california_housing()
    save_dataset("house", housing.data, data_dir)


def prepare_uci_datasets(data_dir: Path):
    """Prepare UCI datasets (adult, yeast) by downloading from the web."""
    import urllib.request

    # --- yeast dataset ---
    # 10 clusters, 8 numeric features
    print("  Downloading yeast dataset from UCI...")
    yeast_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    try:
        response = urllib.request.urlopen(yeast_url)
        lines = response.read().decode("utf-8").strip().split("\n")
        data = []
        for line in lines:
            parts = line.split()
            # First column is sequence name (string), last column is label
            # Columns 1-8 are numeric features
            if len(parts) >= 9:
                data.append([float(x) for x in parts[1:9]])
        yeast_data = np.array(data)
        save_dataset("yeast", yeast_data, data_dir)
    except Exception as e:
        print(f"  WARNING: Could not download yeast dataset: {e}")
        print("  Generating synthetic replacement...")
        generate_synthetic_replacement("yeast", 10, 8, 1484, data_dir)

    # --- adult dataset ---
    # 3 clusters, numeric features only
    print("  Downloading adult dataset from UCI...")
    adult_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    try:
        response = urllib.request.urlopen(adult_url)
        lines = response.read().decode("utf-8").strip().split("\n")
        data = []
        # Numeric column indices: 0(age), 2(fnlwgt), 4(education-num),
        # 10(capital-gain), 11(capital-loss), 12(hours-per-week)
        numeric_cols = [0, 2, 4, 10, 11, 12]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 15:
                try:
                    row = [float(parts[i]) for i in numeric_cols]
                    data.append(row)
                except ValueError:
                    continue
        adult_data = np.array(data)
        save_dataset("adult", adult_data, data_dir)
    except Exception as e:
        print(f"  WARNING: Could not download adult dataset: {e}")
        print("  Generating synthetic replacement...")
        generate_synthetic_replacement("adult", 3, 6, 30000, data_dir)


def generate_synthetic_replacement(name: str, k: int, dim: int, n: int, data_dir: Path):
    """Generate a synthetic dataset as a fallback replacement."""
    from sklearn.datasets import make_blobs
    data, _ = make_blobs(n_samples=n, n_features=dim, centers=k, random_state=42)
    save_dataset(name, data, data_dir)


def prepare_sipu_synthetic(data_dir: Path):
    """
    Generate synthetic approximations of SIPU clustering benchmark datasets.
    The originals (s1, lsun, birch2) come from cs.joensuu.fi/sipu/datasets/
    but aren't freely downloadable in the needed format. We generate
    synthetic versions with matching properties.
    """
    from sklearn.datasets import make_blobs

    # --- s1: 15 clusters, 2D, 5000 samples ---
    # The original S1 dataset has 15 Gaussian clusters in 2D
    s1_data, _ = make_blobs(
        n_samples=5000, n_features=2, centers=15,
        cluster_std=1.5, random_state=42
    )
    save_dataset("s1", s1_data, data_dir)

    # --- lsun: 3 clusters, 2D ---
    # The LSUN shape dataset has 3 clusters in an L-shape arrangement
    rng = np.random.RandomState(42)
    n_per = 500
    # Cluster 1: horizontal bar
    c1 = np.column_stack([rng.uniform(0, 4, n_per), rng.uniform(0, 1, n_per)])
    # Cluster 2: vertical bar
    c2 = np.column_stack([rng.uniform(0, 1, n_per), rng.uniform(1, 4, n_per)])
    # Cluster 3: separate blob
    c3 = np.column_stack([rng.uniform(5, 7, n_per), rng.uniform(5, 7, n_per)])
    lsun_data = np.vstack([c1, c2, c3])
    save_dataset("lsun", lsun_data, data_dir)

    # --- birch2: 100 clusters, 2D ---
    # BIRCH2 is a large dataset with 100 clusters arranged in a grid
    birch2_data, _ = make_blobs(
        n_samples=100000, n_features=2, centers=100,
        cluster_std=0.5, random_state=42
    )
    save_dataset("birch2", birch2_data, data_dir)


def prepare_mnist784(data_dir: Path):
    """Full MNIST: 70,000 samples x 784 features (OpenML).

    Large, dense, high-dimensional — the regime where LSH's dimension-independent
    DP noise should beat FastLloyd. Saved as `mnist784` (distinct from the small
    sklearn-digits `mnist`, 1797 x 64). k = 10.
    """
    from sklearn.datasets import fetch_openml
    print("  Fetching mnist_784 from OpenML (~70k x 784, ~15 MB download)...")
    X, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    save_dataset("mnist784", np.asarray(X, dtype=np.float32), data_dir)


def prepare_glove100(data_dir: Path):
    """GloVe-6B 100d word embeddings: ~400,000 vectors x 100 (Stanford NLP).

    Dense, high-dimensional, cosine-natural — the canonical embedding workload
    for SimHash/LSH. Saved as `glove100`.

    Source resolution (in order):
      1. GLOVE_TXT  — path to a pre-downloaded glove.6B.100d.txt (no network)
      2. GLOVE_URL  — zip to download (default: Stanford glove.6B.zip, ~822 MB)
    Set GLOVE_MAX_ROWS to cap the number of vectors (default: all).
    """
    import io
    import tempfile
    import urllib.request
    import zipfile

    max_rows = int(os.environ.get("GLOVE_MAX_ROWS", "0")) or None
    glove_txt = os.environ.get("GLOVE_TXT")

    def parse(fobj):
        vecs = []
        for i, line in enumerate(fobj):
            if max_rows and i >= max_rows:
                break
            parts = line.rstrip().split(" ")
            vecs.append([float(x) for x in parts[1:]])   # drop leading word token
        return np.asarray(vecs, dtype=np.float32)

    if glove_txt and os.path.exists(glove_txt):
        print(f"  Parsing local GloVe file: {glove_txt}")
        with open(glove_txt, "r", encoding="utf-8") as f:
            data = parse(f)
    else:
        url = os.environ.get("GLOVE_URL", "https://nlp.stanford.edu/data/glove.6B.zip")
        print(f"  Downloading GloVe from {url} (~822 MB)...")
        with tempfile.TemporaryDirectory() as td:
            zpath = os.path.join(td, "glove.6B.zip")
            urllib.request.urlretrieve(url, zpath)
            with zipfile.ZipFile(zpath) as z, z.open("glove.6B.100d.txt") as f:
                data = parse(io.TextIOWrapper(f, encoding="utf-8"))
    save_dataset("glove100", data, data_dir)


def prepare_glove300(data_dir: Path):
    """GloVe-6B 300d word embeddings: ~400,000 vectors x 300 (Stanford NLP).

    Same corpus/vocabulary as `glove100` (GloVe-6B) but the 300-dim variant -- 3x the
    dimensionality at the same 400k points (n*d ~ 1.2e8). A higher-d member of the LARGE
    tier that stays well within limits. Saved as `glove300`. Comes from the same
    glove.6B.zip as glove100 (just a different member file).

    Source resolution (in order):
      1. GLOVE300_TXT — path to a pre-downloaded glove.6B.300d.txt (no network)
      2. GLOVE_URL    — zip to download (default: Stanford glove.6B.zip, ~822 MB; shared
                        with glove100)
    Set GLOVE300_MAX_ROWS to cap the number of vectors (default: all).
    """
    import io
    import tempfile
    import urllib.request
    import zipfile

    max_rows = int(os.environ.get("GLOVE300_MAX_ROWS", "0")) or None
    glove_txt = os.environ.get("GLOVE300_TXT")

    def parse(fobj):
        vecs = []
        for i, line in enumerate(fobj):
            if max_rows and i >= max_rows:
                break
            parts = line.rstrip().split(" ")
            vecs.append([float(x) for x in parts[1:]])   # 6B tokens are single (no spaces)
        return np.asarray(vecs, dtype=np.float32)

    if glove_txt and os.path.exists(glove_txt):
        print(f"  Parsing local GloVe file: {glove_txt}")
        with open(glove_txt, "r", encoding="utf-8") as f:
            data = parse(f)
    else:
        url = os.environ.get("GLOVE_URL", "https://nlp.stanford.edu/data/glove.6B.zip")
        print(f"  Downloading GloVe from {url} (~822 MB)...")
        with tempfile.TemporaryDirectory() as td:
            zpath = os.path.join(td, "glove.6B.zip")
            urllib.request.urlretrieve(url, zpath)
            with zipfile.ZipFile(zpath) as z, z.open("glove.6B.300d.txt") as f:
                data = parse(io.TextIOWrapper(f, encoding="utf-8"))
    save_dataset("glove300", data, data_dir)


def prepare_glove840b(data_dir: Path):
    """GloVe-840B 300d word embeddings: ~2,196,017 vectors x 300 (Stanford NLP).

    The large sibling of `glove100` (GloVe-6B-100d): ~5.5x the vectors and 3x the
    dimensionality (n*d ~ 6.6e8), chosen to push the FastLloyd baseline close to
    its practical limit while the LSH method stays cheap. Saved as `glove840b`.

    Source resolution (in order):
      1. GLOVE840B_TXT — path to a pre-downloaded glove.840B.300d.txt (no network)
      2. GLOVE840B_URL — zip to download (default: Stanford glove.840B.300d.zip, ~2.0 GB)
    Set GLOVE840B_MAX_ROWS to cap the number of vectors (default: all).

    Parsing note: unlike the 6B files, some 840B tokens contain spaces, so the word
    is NOT always a single leading field. We take the vector as the LAST 300
    whitespace-separated fields (word = everything before) and skip any malformed
    line. Rows are kept as small float32 arrays (not Python lists) so 2.2M x 300
    parses in ~2.6 GB rather than exploding on boxed-float overhead.
    """
    import io
    import os
    import tempfile
    import urllib.request
    import zipfile

    DIM = 300
    max_rows = int(os.environ.get("GLOVE840B_MAX_ROWS", "0")) or None
    glove_txt = os.environ.get("GLOVE840B_TXT")

    def parse(fobj):
        rows = []
        kept = skipped = 0
        for line in fobj:
            if max_rows and kept >= max_rows:
                break
            parts = line.rstrip().split(" ")
            if len(parts) < DIM + 1:
                skipped += 1
                continue
            try:
                vec = np.asarray(parts[-DIM:], dtype=np.float32)   # last 300 = vector
            except ValueError:
                skipped += 1
                continue
            rows.append(vec)
            kept += 1
        if skipped:
            print(f"  ({skipped} malformed lines skipped)")
        return np.vstack(rows) if rows else np.empty((0, DIM), dtype=np.float32)

    if glove_txt and os.path.exists(glove_txt):
        print(f"  Parsing local GloVe file: {glove_txt}")
        with open(glove_txt, "r", encoding="utf-8") as f:
            data = parse(f)
    else:
        url = os.environ.get("GLOVE840B_URL",
                             "https://nlp.stanford.edu/data/glove.840B.300d.zip")
        print(f"  Downloading GloVe-840B from {url} (~2.0 GB)...")
        with tempfile.TemporaryDirectory() as td:
            zpath = os.path.join(td, "glove.840B.300d.zip")
            urllib.request.urlretrieve(url, zpath)
            with zipfile.ZipFile(zpath) as z, z.open("glove.840B.300d.txt") as f:
                data = parse(io.TextIOWrapper(f, encoding="utf-8"))
    print(f"  Parsed {data.shape[0]} vectors x {data.shape[1]}")
    save_dataset("glove840b", data, data_dir)


def prepare_timing_datasets(data_dir: Path):
    """Generate synthetic datasets used by timing experiments."""
    from sklearn.datasets import make_blobs

    for k in [2, 5]:
        for d in [2, 5]:
            for n in [10000, 100000]:
                name = f"timesynth_{k}_{d}_{n}"
                data, _ = make_blobs(
                    n_samples=n, n_features=d, centers=k,
                    random_state=42
                )
                save_dataset(name, data, data_dir)


# LARGE tier: the default high-d datasets (fit comfortably; FastLloyd baseline feasible).
LARGE_BUILDERS = {
    "mnist784": prepare_mnist784,
    "glove100": prepare_glove100,
    "glove300": prepare_glove300,
}
# HUGE tier: near the FastLloyd practical limit; opt-in, heavy download/compute.
HUGE_BUILDERS = {
    "glove840b": prepare_glove840b,   # ~2.2M x 300, ~2 GB download
}
# Combined registry for --only (build any single dataset by name).
OPT_IN_BUILDERS = {**LARGE_BUILDERS, **HUGE_BUILDERS}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Prepare FastLloyd / LSH datasets")
    ap.add_argument("--large", action="store_true",
                    help="also prepare the large high-d datasets (mnist784, glove100, glove300)")
    ap.add_argument("--huge", action="store_true",
                    help="also prepare the HUGE tier near FastLloyd's limit (glove840b, ~2 GB dl)")
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME",
                    help="prepare ONLY these datasets, e.g. --only mnist784 glove100 glove300 glove840b")
    args = ap.parse_args()

    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("FastLloyd Dataset Preparation")
    print("=" * 60)

    # --only: build just the requested datasets (large or huge tier) and stop.
    if args.only:
        for name in args.only:
            if name in OPT_IN_BUILDERS:
                print(f"\nPreparing {name}...")
                OPT_IN_BUILDERS[name](data_dir)
            else:
                print(f"  (skipping unknown --only target: {name})")
        print("\nDone:", data_dir)
        return

    print("\n[1/4] Preparing scikit-learn datasets...")
    prepare_sklearn_datasets(data_dir)

    print("\n[2/4] Preparing UCI datasets...")
    prepare_uci_datasets(data_dir)

    print("\n[3/4] Preparing SIPU-style synthetic datasets...")
    prepare_sipu_synthetic(data_dir)

    print("\n[4/4] Preparing timing experiment datasets...")
    prepare_timing_datasets(data_dir)

    if args.large:
        print("\n[5/5] Preparing large high-d datasets (mnist784, glove100, glove300)...")
        for build in LARGE_BUILDERS.values():
            build(data_dir)

    if args.huge:
        print("\n[huge] Preparing the HUGE tier near FastLloyd's limit (glove840b)...")
        for build in HUGE_BUILDERS.values():
            build(data_dir)

    print("\n" + "=" * 60)
    print("Done! All datasets saved to:", data_dir)
    print("=" * 60)

    # List what's available
    txt_files = sorted(data_dir.glob("*.txt"))
    print(f"\n{len(txt_files)} dataset files created:")
    for f in txt_files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
