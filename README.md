# FastLloyd: Federated, Accurate, Secure, and Tunable k-Means Clustering with Differential Privacy

FastLloyd addresses the challenge of performing $k$-means clustering in a horizontally federated setting while preserving data privacy. Existing methods suffer from either high computational overhead or significantly degrade clustering utility. FastLloyd overcomes these limitations through a novel federated protocol that enhances both the differential privacy mechanism and the secure computation components. 

A new DP clustering algorithm is introduced, which incorporates a radius constraint on clusters and uses relative updates to improve utility. This algorithm is integrated into a federated setting using a lightweight, secure aggregation protocol (Masked Secure Aggregation - MSA) that leverages the computational DP model. This allows intermediate differentially private updates to be published, significantly reducing the overhead of secure computation by performing expensive operations like assignments and divisions locally. 

FastLloyd significantly outperforms previous work; It achieves up to a five-orders-of-magnitude speed-up in runtime compared to state-of-the-art secure federated $k$-means approaches, while also reducing communication by up to seven orders of magnitude. Furthermore, FastLloyd not only matches the utility of central DP models but improves upon state-of-the-art DP $k$-means algorithms, especially in higher dimensions and for a larger number of clusters, demonstrating up to an 88% reduction in clustering error.

## Overview

This repository implements the FastLloyd protocol described in the paper "FastLloyd: Federated, Accurate, Secure, and
Tunable k-Means Clustering with Differential Privacy". FastLloyd addresses the challenging problem of collaborative
clustering across multiple data owners without compromising privacy, through:

1. A novel differentially private k-means algorithm with radius constraints
2. A lightweight secure aggregation protocol for federated settings

## Installation

### Requirements

- Python 3.8 or higher
- Open MPI (for multiparty communication)
- Required Python packages listed in `env.yml`

### Setup

1. Clone the repository:

```bash
git clone https://github.com/D-Diaa/FastLloyd.git
cd FastLloyd
```

2. Create and activate the conda environment:

```bash
conda env create -f env.yml
conda activate fastlloyd
```

3. Download and prepare datasets:

```bash
python scripts/download_data.py
```

This creates the `data/` directory and populates it with all required datasets. The script fetches:

| Source | Datasets |
|--------|----------|
| scikit-learn | `iris`, `wine`, `breast`, `house`, `mnist` |
| UCI ML Repository | `adult`, `yeast` |
| Synthetic (generated) | `s1`, `lsun`, `birch2`, `timesynth_*` |

> **Note:** The `s1`, `lsun`, and `birch2` datasets are synthetic approximations of the
> original SIPU clustering benchmarks. The `mnist` dataset uses sklearn's 8x8 digits
> (1797 samples) rather than the full 28x28 MNIST. Results may differ from the original paper.

For synthetic scale/ablation datasets, you can additionally run the R generator:

```bash
Rscript scripts/generator.R
```

## Repository Structure

```
├── configs/                                                              
│   ├── defaults.py       # Default configuration settings, dataset definitions
│   └── params.py         # Parameter class for clustering and privacy settings
│                                                                                                                                                    
├── data_io/                                                                                                                                          
│   ├── comm.py           # MPI communication wrapper with delay simulation
│   ├── data_handler.py   # Functions for loading and processing datasets
│   └── fixed.py          # Fixed-point arithmetic implementation
│                                                                                                                                                    
├── parties/                                                                                                                                          
│   ├── client.py         # Client implementations (masked and unmasked)
│   └── server.py         # Server implementation with DP mechanisms
│                                                                                                                                                    
├── plots/                                                                                                                                           
│   ├── ablation_plots.py      # Visualization for ablation studies
│   ├── compare_basis.py       # Compare random / SVD PCA / DP-SGD PCA basis methods
│   ├── compare_protocols.py   # Compare ortho vs original algorithm results
│   ├── per_dataset.py         # Dataset-specific result visualization
│   ├── scale_heatmap.py  # Heatmap generation for scalability results
│   ├── synthetic_bar.py  # Bar charts for synthetic dataset results
│   └── timing_analysis.py # Analysis of timing experiments
│                                                                                                                                                    
├── scripts/
│   ├── download_data.py                  # Download and prepare datasets from sklearn/UCI
│   ├── generator.R                       # R script for generating synthetic datasets
│   ├── setup.sh                          # Extract data archives and create conda environment
│   ├── run_accuracy_scale_experiments.sh # Run accuracy and scale experiments in parallel
│   ├── run_timing_experiments.sh         # Run timing experiments with 2/4/8 clients
│   ├── run_experiments.sh                # Orchestrates accuracy, scale, and timing runs
│   ├── generate_plots.sh                 # Generate all plots and analysis from results
│   ├── no_setup.sh                       # Run experiments + plots (environment already active)
│   └── end_to_end.sh                     # Full pipeline: setup → experiments → plots
│                                                                                                                                                    
├── utils/                                                                                                                                           
│   ├── evaluations.py    # Clustering quality evaluation metrics
│   ├── ortho_clustering.py # Orthogonal projection clustering (basis generation, assignment, centroids)
│   ├── protocols.py      # Clustering protocols (local, MPI, ortho)
│   └── utils.py          # General utility functions
│                                                                                                                                                    
├── experiments.py        # Main experiment runner
├── ortho_cluster_test.py # Standalone test for orthogonal projection clustering
├── env.yml               # Conda environment specification
└── README.md             # Project documentation
```

## Usage

### Running Experiments

FastLloyd supports multiple experiment types:

1. **Accuracy**: Evaluate clustering quality across different privacy settings

```bash
python experiments.py --exp_type "accuracy"
```

2. **Scale**: Analyze scalability with dataset size, dimensions, and number of clusters

```bash
python experiments.py --exp_type "scale"
```

3. **Timing**: Measure communication and computation time

```bash
mpirun -np 3 python experiments.py --exp_type "timing"
```

You can also use the provided scripts to run multiple experiment types:

```bash
bash scripts/run_accuracy_scale_experiments.sh  # Accuracy and scale experiments in parallel
bash scripts/run_timing_experiments.sh          # Timing experiments with 2, 4, and 8 clients
bash scripts/run_experiments.sh                 # All of the above
bash scripts/generate_plots.sh                  # Generate all plots from results
bash scripts/end_to_end.sh                      # Full pipeline from scratch (setup + experiments + plots)
```

### Visualization

The repository includes several visualization tools in the `plots` directory:

- `per_dataset.py`: Creates performance visualizations for individual datasets
- `compare_protocols.py`: Compares ortho vs original algorithm results (bar charts + summary CSV)
- `compare_basis.py`: Compares the three ortho basis methods (random, SVD PCA, DP-SGD PCA) across d' values for each dataset
- `scale_heatmap.py`: Generates heatmaps to analyze scalability
- `synthetic_bar.py`: Creates bar plots comparing performance on synthetic datasets
- `ablation_plots.py`: Creates plots for ablation studies
- `timing_analysis.py`: Analyzes and reports execution timing data

## Customization

You can customize various aspects of the experiments through the argument parser in `experiments.py`:

```bash
python experiments.py --exp_type "test" --datasets "mnist" "adult" --method "diagonal_then_frac" --alpha 0.8 --post "fold" --results_folder "my_results"
```

Key parameters include:

- `--exp_type`: Type of experiment to run (accuracy, scale, timing, test)
- `--protocol`: Clustering protocol to use (`local` for FastLloyd, `ortho` for orthogonal projection)
- `--datasets`: Datasets to use for the experiment
- `--method`: Maximum distance method to use
- `--alpha`: Maximum distance parameter
- `--post`: Post-processing method for centroids
- `--d_primes`: d' values to sweep when using `--protocol ortho` (default: 1 2 3 4 5)
- `--sigma`: Gaussian noise std dev(s) for ortho DP centroids (default sweeps 0.0, 0.1, 0.5, 1.0, 5.0)
- `--basis_data_fraction`: fraction of data to subsample before DP-SGD PCA (default: 0.1; see [data subsampling](#data-subsampling-for-dp-sgd))
- `--results_folder`: Folder to store results

## Orthogonal Projection Clustering

`utils/ortho_clustering.py` implements a fast clustering method based on orthogonal projections.

**How it works:** Given `n` points in `d` dimensions, the algorithm projects each point onto `d'` orthonormal basis vectors and assigns cluster membership by the sign pattern of the projections, partitioning the space into up to `2^d'` quadrants. Three basis methods are supported:

| Method | `--basis_method` | Privacy cost | Description |
|--------|-----------------|-------------|-------------|
| Random | `random` | None | Random Gaussian matrix orthogonalized via SVD; no data used |
| SVD PCA | `svd_pca` | None (non-private) | True top-`d'` principal components via standard SVD; oracle baseline |
| DP-SGD PCA | `dpsgd_pca` | `(basis_epsilon, basis_delta)` | Differentially private PCA via DP-SGD; default method |

### DP-SGD PCA Basis

`dpsgd_pca_basis(X, d_prime, epsilon, delta, clip_norm, ...)` computes a differentially private orthonormal basis by running SGD on the variance-maximization objective.

#### Algorithm

1. **Subsample** (optional): draw `data_fraction × n` rows of `X` uniformly at random. This reduces the number of SGD steps, allowing the noise calibration to return a smaller `sigma` for the same privacy budget.
2. **Center** the (sub)sampled data: `X_c = X - mean(X, axis=0)`.
3. **Initialize** `W` as a random `(d, d')` orthonormal matrix.
4. **Calibrate noise**: find the smallest Gaussian noise multiplier `sigma` such that the full SGD run is `(epsilon, delta)`-DP, using Rényi DP accounting over the subsampled Gaussian mechanism (see [Privacy Accounting](#privacy-accounting) below).
5. **SGD loop** — for each epoch, shuffle the data and iterate over mini-batches:
   - **Per-sample gradient**: for each point `x_i`, compute `g_i = -2 · outer(x_i, x_i @ W)` (shape `(d, d')`).
   - **Clip**: scale `g_i` so its Frobenius norm is at most `clip_norm`, bounding the sensitivity of the sum to `clip_norm` regardless of the data.
   - **Add noise**: draw `Z ~ N(0, sigma² · clip_norm² · I)` and form the noisy average gradient `(Σ g_i + Z) / batch_size`.
   - **Gradient step**: `W ← W - lr · noisy_gradient`.
   - **Re-orthonormalize**: `W, _ = qr(W)`. This projects `W` back onto the Stiefel manifold after the noisy step perturbs it. Without this, basis vectors drift and become collinear, which degrades the sign-based cluster assignment.
6. **Return** the first `d'` columns of the final `W`.

#### Privacy Accounting

Noise calibration (`_find_sigma_autodp`) uses the **Rényi DP (RDP)** framework rather than naive composition:

- Each mini-batch step uses the **subsampled Gaussian mechanism** at sampling rate `q = batch_size / n`. For small `q`, the per-step RDP cost scales as `~q²`, much cheaper than the non-subsampled version.
- RDP composes additively over `T = epochs × ⌊n / batch_size⌋` steps: `R(α)_total = T · R(α)_per_step`.
- Convert to `(ε, δ)`-DP at the end: `ε(α) = R(α)_total + log(1/δ) / (α - 1)`, minimized over `α ∈ {2, …, 255}`.

This gives a much tighter bound than naive composition (`T × ε_per_step`) due to the subsampling amplification. The binary search converges in 64 iterations to find the smallest `sigma` satisfying the target `(epsilon, delta)`.

**Privacy budget split:** `basis_epsilon` / `basis_delta` are spent entirely on the basis computation. They are independent of the clustering step's budget (`--eps`).

#### Parameter Reference

| Parameter | Default | Effect |
|-----------|---------|--------|
| `epsilon` | `0.5` | Privacy budget for the basis. Larger → less noise → better basis quality, weaker privacy. |
| `delta` | `1e-5` | Failure probability. Standard choice: `1 / (n · log(n))`. |
| `clip_norm` | `1.0` | Per-sample gradient clipping threshold (Frobenius norm). Controls the sensitivity of the gradient sum. Set to match the typical gradient magnitude; `1.0` is appropriate for data normalized to `[-1, 1]^d`. Larger values let more signal through but require proportionally more noise. |
| `epochs` | `10` | Number of full passes over the data. More epochs → more SGD compositions → higher required `sigma` for the same budget. |
| `lr` | `0.01` | SGD learning rate. `0.01` works well for normalized data. |
| `batch_size` | `256` | Mini-batch size. Smaller batches give stronger subsampling amplification (cheaper per step) but more steps overall. |
| `data_fraction` | `0.1` | Fraction of `X` to subsample before running DP-SGD. Reduces `n` to `fraction × n`, cutting `T` proportionally, which lets `sigma` shrink — improving basis quality for the same budget. `0.1` (10%) is the default. See [Data Subsampling](#data-subsampling-for-dp-sgd). |

#### Data Subsampling for DP-SGD

By default, `dpsgd_pca_basis` trains on only **10% of the data** (`--basis_data_fraction 0.1`). This works because:

- Fewer data points → fewer SGD steps `T` → the Rényi accountant requires less noise `sigma` for the same `(epsilon, delta)` budget.
- The DP guarantee holds over the subsampled dataset. The additional subsampling from the full dataset only strengthens the overall privacy guarantee.
- In practice, 10% of data is usually sufficient to find good principal directions, while the noise reduction meaningfully improves basis quality.

To use the full dataset set `--basis_data_fraction 1.0`.

### SVD Non-Uniqueness and Sign Ambiguity

> **Research note** — this is an important property of any SVD/PCA-based basis that directly affects ortho clustering.

#### The mathematical issue

SVD is not unique. For any valid decomposition `X = U Σ V^T`, negating any column pair `(u_i, v_i)` produces another equally valid decomposition:

```
X = (U · diag(±1)) · Σ · (V · diag(±1))^T
```

For `d'` basis vectors this gives **2^d' mathematically equivalent solutions**, differing only in the signs of the columns of `V` (the basis `W`). The singular values and the subspace spanned by `V` are unique — the individual column directions are not.

The same ambiguity appears in DP-SGD PCA: after each QR re-orthonormalization step, the sign convention of each column of `W` is determined by the numerical algorithm's path, not by any canonical rule.

#### Why this matters for sign-pattern clustering

The ortho algorithm assigns clusters *entirely* by sign pattern:

```python
signs = (projections >= 0).astype(int)   # +1 or 0 per basis direction
labels = signs @ (2 ** np.arange(d_eff)) # → cluster ID in [0, 2^d' - 1]
```

If one basis vector is negated, every projection onto that direction flips sign, which **moves every data point between two cluster halves**. Two SVD solutions spanning the same subspace can produce completely different labelings.

Concretely:
- Permuting the rows of `X` before calling `np.linalg.svd` does not change the column space of `V`, but the numerical algorithm may converge to a different sign convention.
- Different random seeds in DP-SGD (`rng.permutation(n)` inside each epoch) lead to different sign conventions in the returned `W`.
- Different platforms, NumPy versions, or BLAS implementations may produce different signs even on the same input.

#### Practical implications

1. **Cross-seed comparison**: NICV scores are invariant to sign flips (cluster compactness doesn't depend on labeling), but the raw cluster IDs are not comparable across runs.
2. **Reproducibility**: results from `svd_pca_basis` are reproducible (same `X` → same `np.linalg.svd` → same `V`), but `dpsgd_pca_basis` is stochastic by design.
3. **Canonicalization**: a deterministic sign convention can be imposed after computing the basis by ensuring each column's largest-magnitude element is positive:
   ```python
   W *= np.sign(W[np.abs(W).argmax(axis=0), np.arange(W.shape[1])])
   ```
   This makes cross-run comparison of cluster IDs meaningful. It is not currently applied by default.
4. **Exhaustive sign search**: since there are only `2^d'` sign patterns, it is cheap to evaluate all of them and keep the one minimizing NICV — a free improvement over a single random sign convention.

### API

```python
from utils.ortho_clustering import (
    orthogonal_basis, svd_pca_basis, dpsgd_pca_basis,
    orthogonalize_svd, random_orthogonal_basis,
    ortho_assign, cluster_centers, cluster_counts, noisy_cluster_centers,
)
```

**Basis generation**

- `orthogonal_basis(X, d_prime, method="dpsgd_pca", seed=42, **kwargs)` — dispatcher returning a `(d, d')` orthonormal basis. `method` is one of `"random"`, `"svd_pca"`, or `"dpsgd_pca"`. For `dpsgd_pca`, pass `epsilon`, `delta`, `clip_norm`, and optionally `data_fraction` as keyword arguments.
- `svd_pca_basis(X, d_prime)` — non-private PCA: centers `X` and returns the top-`d'` right singular vectors. Oracle baseline; no privacy guarantee.
- `dpsgd_pca_basis(X, d_prime, epsilon, delta, clip_norm, epochs=10, lr=0.01, batch_size=256, data_fraction=0.1)` — private PCA via DP-SGD. See [DP-SGD PCA Basis](#dp-sgd-pca-basis) for full parameter docs.
- `random_orthogonal_basis(d, d_prime, seed=42, orthogonalize=None)` — random Gaussian matrix orthogonalized via SVD.
- `orthogonalize_svd(R)` — orthogonalize a `(d, k)` matrix via economy SVD; used internally.

**Assignment and centroids**

- `ortho_assign(values, d_prime, seed=42, basis=None)` — project `values` onto `basis` (or generate a fresh random basis if `None`) and return integer cluster labels in `[0, 2^d' - 1]` based on the sign pattern of each projection.
- `cluster_centers(values, labels)` — compute the exact centroid of each cluster. Returns `(centers, unique_labels)`.
- `cluster_counts(labels)` — return `(counts, unique_labels)` giving the number of points per cluster.
- `noisy_cluster_centers(values, labels, sigma, seed=42)` — compute centroids with Gaussian noise `N(0, sigma²)` added to each cluster *sum* before dividing by count. The effective per-dimension noise on the centroid is `sigma / count`. This is the DP mechanism for sum queries applied to centroids.

### Running with the experiment framework

The ortho algorithm is integrated as a protocol, so it can be run with any experiment type:

```bash
# Accuracy experiments (DP-SGD PCA basis, default settings)
python experiments.py --exp_type accuracy --protocol ortho

# Scale experiments
python experiments.py --exp_type scale --protocol ortho

# Custom d' sweep and datasets
python experiments.py --exp_type accuracy --protocol ortho --d_primes 2 3 4 --datasets iris mnist

# Custom sigma sweep (noise added to cluster sums for DP centroids)
python experiments.py --exp_type accuracy --protocol ortho --sigma 0.0 1.0 10.0

# Random basis (no DP for basis computation)
python experiments.py --exp_type accuracy --protocol ortho --basis_method random --d_primes 1 2 3

# Non-private SVD PCA basis (oracle baseline)
python experiments.py --exp_type accuracy --protocol ortho --basis_method svd_pca --d_primes 1 2 3 4 5

# DP-SGD PCA basis with custom privacy budget and data fraction
python experiments.py --exp_type accuracy --protocol ortho \
    --basis_method dpsgd_pca --d_prime 3 \
    --basis_epsilon 0.5 --basis_delta 1e-5 --basis_clip_norm 1.0 \
    --basis_data_fraction 0.1 \
    --datasets iris mnist --num_runs 5
```

When `--protocol ortho` is used, DP/method/post parameters are automatically set to `"none"` (since they don't apply), and `d_prime` is swept over the values given by `--d_primes` (default: 1 2 3 4 5). `sigma` controls Gaussian noise added to cluster sums before computing centroids; passing `--sigma` with no values defaults to `0.0`. Results are saved to the same CSV format and evaluated with the same metrics as other protocols.

**Basis parameters (ortho protocol only):**

| Argument | Default | Description |
|----------|---------|-------------|
| `--basis_method` | `dpsgd_pca` | Basis generation method: `dpsgd_pca` (private PCA via DP-SGD), `svd_pca` (non-private SVD PCA, oracle), or `random` (random orthonormal, no data used) |
| `--d_prime` | `1 2 3 4 5` | d' value(s) to sweep (space-separated); number of basis vectors and log₂ of max clusters |
| `--basis_epsilon` | `0.5` | Privacy budget ε for the DP-SGD basis step |
| `--basis_delta` | `1e-5` | Privacy δ for the DP-SGD basis step (standard: `1/(n·log(n))`) |
| `--basis_clip_norm` | `1.0` | Per-sample gradient clipping norm; bounds sensitivity to `clip_norm` |
| `--basis_data_fraction` | `0.1` | Fraction of data used for DP-SGD training; reduces noise by cutting the number of SGD steps |

### Running the standalone test

```bash
python ortho_cluster_test.py
```

This runs verification tests followed by benchmarks across the accuracy datasets for `d' = 1..5` with 10 random seeds each, saving results to `ortho_results/results.csv`.

### Comparing protocols

After running both local and ortho experiments, compare results with:

```bash
# Compare accuracy results (default)
python -m plots.compare_protocols

# Custom results folder
python -m plots.compare_protocols my_results

# Compare other experiment types (e.g., scale)
python -m plots.compare_protocols submission --exp_type scale
```

This generates:
- **Per-dataset bar charts** (`NICV.pdf`, `Silhouette.pdf`, etc.) in each dataset folder, showing side-by-side metric values with confidence intervals for Lloyd, FastLloyd (best epsilon), and Ortho (each d' value).
- **Summary CSV** (`submission/accuracy/comparison_summary.csv`) with one row per (dataset, method) and all metric columns, for easy analysis in pandas or a spreadsheet.

### Comparing basis methods

After running the ortho protocol with two or more `--basis_method` values, compare them with:

```bash
# Compare all basis methods found in results (sigma=0 by default)
python -m plots.compare_basis

# Custom results folder
python -m plots.compare_basis my_results

# Filter to a specific sigma value
python -m plots.compare_basis my_results --sigma 0.5
```

This generates one PDF per metric per dataset (e.g., `basis_compare_NICV.pdf`) showing grouped bars over d' values, with one bar group per basis method. Lloyd and FastLloyd baselines are drawn as horizontal reference lines. A summary CSV (`basis_comparison_summary.csv`) is also written to the experiment folder.

### Results file layout

```
submission/accuracy/<dataset>/
├── variances.csv                  # Local protocol (Lloyd, FastLloyd, etc.)
├── variances_ortho.csv            # Ortho protocol results (all basis methods)
├── NICV.pdf                       # Protocol comparison: NICV
├── Silhouette.pdf                 # Protocol comparison: Silhouette Score
├── basis_compare_NICV.pdf         # Basis method comparison: NICV across d'
├── basis_compare_Silhouette.pdf   # Basis method comparison: Silhouette across d'
└── ...                            # One chart per metric for each comparison
```

## Citation

If you use FastLloyd in your research, please cite the paper:

```
@article{diaa2024fastlloyd,
  title={FastLloyd: Federated, Accurate, Secure, and Tunable $ k $-Means Clustering with Differential Privacy},
  author={Diaa, Abdulrahman and Humphries, Thomas and Kerschbaum, Florian},
  journal={arXiv preprint arXiv:2405.02437},
  year={2024}
}
```
