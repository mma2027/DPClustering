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


def save_dataset(name: str, data: np.ndarray, data_dir: Path):
    """Save dataset as space-separated text file."""
    path = data_dir / f"{name}.txt"
    np.savetxt(path, data, fmt="%.6f", delimiter=" ")
    print(f"  {name}.txt: {data.shape[0]} samples, {data.shape[1]} features")


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


def main():
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("FastLloyd Dataset Preparation")
    print("=" * 60)

    print("\n[1/4] Preparing scikit-learn datasets...")
    prepare_sklearn_datasets(data_dir)

    print("\n[2/4] Preparing UCI datasets...")
    prepare_uci_datasets(data_dir)

    print("\n[3/4] Preparing SIPU-style synthetic datasets...")
    prepare_sipu_synthetic(data_dir)

    print("\n[4/4] Preparing timing experiment datasets...")
    prepare_timing_datasets(data_dir)

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
