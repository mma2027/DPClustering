# DP-LSH Clustering: Scalable Federated $k$-Means via SimHash Trees

This project studies a **differentially private LSH prefix-tree** clustering method and
its **scalability**. Each point is hashed by a SimHash basis into a prefix tree; small
branches are pruned under differential privacy and every surviving leaf yields one
private centroid. It is a *one-pass* alternative to iterative DP $k$-means, designed for
**large, high-dimensional, dense** data (e.g. embeddings): its DP noise and per-round
communication are independent of the dimension $d$, so it should scale where iterative
methods struggle.

The code is built on the **FastLloyd** federated DP-clustering framework, which is used
throughout as the **baseline**. The repository provides both a centralized and an MPI
(federated) implementation of the LSH method, plus tooling to compare its **accuracy**
and **scalability** against FastLloyd.

## Overview

- **LSH prefix-tree clustering** — the focus: one-pass SimHash hashing + DP pruning, with
  a centralized (`lsh_proto`) and a federated MPI (`mpi_lsh_proto`) variant. See
  [LSH Prefix-Tree Clustering](#lsh-prefix-tree-clustering).
- **FastLloyd baseline** — the framework this builds on: an iterative, radius-constrained
  DP $k$-means with a lightweight masked secure-aggregation protocol, run locally
  (`local_proto`) or over MPI (`mpi_proto`). See the [paper](#citation) for details.
- **Comparison tooling** — accuracy (clustering quality vs $\varepsilon$) and scalability
  (communication rounds / bytes / wall-time vs number of clients), with ready-to-run
  experiment packages in `small/`, `medium/`, and `large/`.

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
│   ├── compare_basis.py       # Compare random / SVD PCA / DP-SGD PCA basis methods (LSH)
│   ├── compare_methods.py     # Accuracy: LSH vs baselines, line charts over epsilon
│   ├── compare_timing.py      # Scalability: LSH vs baselines (rounds / bytes / wall-time)
│   ├── per_dataset.py         # Dataset-specific result visualization
│   ├── scale_heatmap.py  # Heatmap generation for scalability results
│   ├── synthetic_bar.py  # Bar charts for synthetic dataset results
│   └── timing_analysis.py # Analysis of timing experiments (FastLloyd)
│                                                                                                                                                    
├── scripts/
│   ├── download_data.py                  # Download and prepare datasets from sklearn/UCI
│   ├── generator.R                       # R script for generating synthetic datasets
│   ├── setup.sh                          # Extract data archives and create conda environment
│   ├── run_lsh.sh                        # Accuracy: baselines + LSH (3 bases) + plots
│   ├── run_lsh_timing.sh                 # Scalability: LSH vs FastLloyd over MPI + plots
│   ├── run_accuracy_scale_experiments.sh # Run accuracy and scale experiments in parallel
│   ├── run_timing_experiments.sh         # Run timing experiments with 2/4/8 clients
│   ├── run_experiments.sh                # Orchestrates accuracy, scale, and timing runs
│   ├── generate_plots.sh                 # Generate all plots and analysis from results
│   ├── no_setup.sh                       # Run experiments + plots (environment already active)
│   └── end_to_end.sh                     # Full pipeline: setup → experiments → plots
│                                                                                                                                                    
├── parties/lsh_server.py # LSH server: aggregate sparse counts/sums, add noise, prune
├── parties/lsh_client.py # LSH client shards: local hashing + sparse counts/sums
├── utils/                                                                                                                                           
│   ├── evaluations.py    # Clustering quality evaluation metrics
│   ├── LSHTree.py        # LSH prefix tree: SimHash hashing, pruning, centroids
│   ├── ortho_clustering.py # SimHash basis generation (random/SVD/DP-SGD PCA) + zCDP noise
│   ├── protocols.py      # Clustering protocols (local, MPI FastLloyd, LSH, MPI LSH)
│   └── utils.py          # General utility functions
│                                                                                                                                                    
├── experiments.py        # Main experiment runner
├── test_lshtree.py       # Unit tests: LSH tree + pruning
├── test_lsh_federated.py # Unit tests: federated LSH == centralized (in-process)
├── test_mpi_lsh.py       # MPI equivalence test (run via mpirun)
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
- `compare_methods.py`: Accuracy comparison — LSH (each basis) vs baselines (Lloyd/FastLloyd) as line charts over epsilon, one subplot per d'
- `compare_basis.py`: Compares the three LSH basis methods (random, SVD PCA, DP-SGD PCA) across d' values at a fixed epsilon
- `compare_timing.py`: Scalability comparison — LSH vs FastLloyd wall-time and communication vs number of clients
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
- `--protocol`: Clustering protocol (`local` for FastLloyd/baselines, `lsh` for the LSH prefix tree). Under `--exp_type timing` these run over MPI (`mpi_proto` and `mpi_lsh_proto`).
- `--datasets`: Datasets to use for the experiment
- `--method`, `--alpha`, `--post`: Baseline (FastLloyd) max-distance method, parameter, and centroid post-processing (ignored by `lsh`)
- `--d_primes`: d' values to sweep with `--protocol lsh` (basis width / max tree depth; default: 1 2 3 4 5)
- `--basis_method`: LSH SimHash basis, one or more of `random svd_pca dpsgd_pca` (default: `dpsgd_pca`)
- `--tree_min_count`: noisy-count pruning threshold for `lsh`; branches below it are pruned (default: 0 = no pruning)
- `--tree_max_depth`: max LSH tree depth (default: 0 = use `d_prime`)
- `--basis_epsilon`: fraction in (0,1) of the total `(ε, δ)` budget spent on the DP-SGD-PCA basis (default: 0.1; only `dpsgd_pca` spends basis budget; the other `1 - basis_epsilon` goes to clustering)
- `--basis_clip_norm`, `--basis_data_fraction`: DP-SGD-PCA gradient clip norm and data subsample fraction (defaults: 1.0, 0.1; see [data subsampling](#data-subsampling-for-dp-sgd))
- `--num_runs`: random seeds per configuration (averaged; default 10)
- `--results_folder`: Folder to store results
- `--export_centroids`: dump the final centroids of every run to `<results_folder>/<exp_type>/<dataset>/centroids/` — one `.npy` (the `(n_clusters, dim)` centroid array) plus a matching `.json` sidecar recording the full parameter set (protocol, basis, `d'`, `eps`, `seed`, …), keyed by a short hash so stems are unique. Off by default. Centroids are otherwise computed in memory and discarded after evaluation, so adding a **new point-to-centroid metric** later (e.g. cosine similarity) would require re-running the whole sweep; with them saved, any such metric can be recomputed **offline** by reloading the dataset and calling `utils.evaluate` on the saved centroids — no clustering re-run needed. In an MPI (`timing`) run only the first client rank writes (the global centroids are identical across ranks). See [Recomputing metrics offline](#recomputing-metrics-offline).

The privacy budget ε for `lsh` is **not** a CLI flag: it comes from the experiment's `eps_budgets` (`configs/defaults.py`, e.g. `0.5 1 2 4` for accuracy) and is split between leaf centroids and per-node counts by `sigma_fraction` (a `Params` default of 10).

### Recomputing metrics offline

The result CSVs store only **aggregated scalar metrics** (e.g. NICV, cosine similarity, and their confidence-interval half-widths `_h`) — not the centroids. So a *new* point-to-centroid metric normally can't be derived from past runs and would need the whole sweep re-run. Run with `--export_centroids` to avoid this: the final centroids of every run are saved, and any point-to-centroid metric can then be recomputed **without re-clustering**. Each run writes two same-stem files under `<results_folder>/<exp_type>/<dataset>/centroids/`:

- `<stem>.npy` — the `(n_clusters, dim)` final centroid array.
- `<stem>.json` — the full parameter set for that run (`protocol`, `basis_method`, `eps`, `seed`, `d_prime` for LSH, …), matching the columns of the corresponding `variances*.csv` row.

To recompute a metric, reload the dataset **exactly as the pipeline does** (unit-norm + fixed-point round-trip), then call `utils.evaluate` (or a metric function directly) on each saved centroid array, grouping by config and averaging over seeds to reproduce the CSV aggregate:

```python
import glob, json, numpy as np
from collections import defaultdict
from data_io import load_txt, ensure_unit_norm, to_fixed, unscale
from utils.evaluations import evaluate_mean_cosine_similarity, _assign_nearest

values = unscale(to_fixed(ensure_unit_norm(load_txt("data/iris.txt"))))  # fixed=True path
groups = defaultdict(list)
for jf in glob.glob("results/accuracy/iris/centroids/*.json"):
    meta = json.load(open(jf))
    C = np.load(jf[:-5] + ".npy")
    cos = evaluate_mean_cosine_similarity(_assign_nearest(values, C), C, values)
    key = tuple(sorted((k, str(v)) for k, v in meta.items() if k != "seed"))  # config = params minus seed
    groups[key].append(cos)
per_config_mean = {k: float(np.mean(v)) for k, v in groups.items()}  # matches CSV "Mean Cosine Similarity"
```

(The `cosine_similarity` metric is already computed inline on every run, so this offline path is needed only for metrics added *after* a sweep was run.)

## LSH Prefix-Tree Clustering

`utils/LSHTree.py` (tree + pruning) and `utils/ortho_clustering.py` (basis + noise) implement a one-pass, differentially private clustering method based on locality-sensitive hashing (SimHash), evaluated here as an alternative to the iterative FastLloyd baselines.

**How it works:** each of `n` points in `d` dimensions is hashed bit-by-bit by projecting it onto the columns of a `(d, d')` SimHash basis and taking the sign of each projection (bit `j` = `sign(x · basis[:, j])`). Points are grouped into a **prefix tree**: the root holds everything, and a node at depth `j` splits its points on bit `j`. Every node carries a **noisy** count, and a branch is **pruned** when that noisy count falls below `--tree_min_count`. Each surviving leaf yields one differentially private centroid (`noisy_sum / noisy_count`), so the number of clusters is data-dependent (up to `2^d'`) rather than fixed. Three SimHash basis methods are supported:

| Method | `--basis_method` | Privacy cost | Description |
|--------|-----------------|-------------|-------------|
| Random | `random` | None (no data used) | Random Gaussian matrix orthogonalized via SVD — the canonical angular-LSH basis |
| SVD PCA | `svd_pca` | None (non-private) | True top-`d'` principal components via standard SVD; oracle baseline |
| DP-SGD PCA | `dpsgd_pca` | `basis_epsilon · ε` | Differentially private PCA via DP-SGD; default method |

**Why LSH (vs. FastLloyd) for large, high-dimensional, dense data:** the per-leaf sum and per-node count queries have L2 sensitivity 1 on unit-norm data, so the centroid noise is **independent of `d`** (FastLloyd's scales with `√d`); the count rounds are also `d`-independent in size. Combined with a fixed, small number of communication rounds (vs. one per Lloyd iteration), this makes LSH attractive in the high-`d`/many-cluster regime — at the cost of a coarser, projection-based partition.

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

**Privacy budget split:** at the CLI, `--basis_epsilon` is a *fraction* in (0,1) of the run's total ε (from `eps_budgets`) spent on the DP-SGD-PCA basis; the remainder goes to the clustering step (leaf sums + node counts). The `epsilon`/`delta` in the table below are the absolute values the underlying `dpsgd_pca_basis` function receives after the split. `random` and `svd_pca` spend no basis budget, so the whole ε is left for clustering.

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

> **Research note** — this is an important property of any SVD/PCA-based basis that directly affects sign-pattern (SimHash) hashing.

#### The mathematical issue

SVD is not unique. For any valid decomposition `X = U Σ V^T`, negating any column pair `(u_i, v_i)` produces another equally valid decomposition:

```
X = (U · diag(±1)) · Σ · (V · diag(±1))^T
```

For `d'` basis vectors this gives **2^d' mathematically equivalent solutions**, differing only in the signs of the columns of `V` (the basis `W`). The singular values and the subspace spanned by `V` are unique — the individual column directions are not.

The same ambiguity appears in DP-SGD PCA: after each QR re-orthonormalization step, the sign convention of each column of `W` is determined by the numerical algorithm's path, not by any canonical rule.

#### Why this matters for sign-pattern clustering

LSH assigns each point's hash *entirely* by the sign of its projections (bit `j` = `sign(x · basis[:, j])`), so the prefix — and therefore the leaf — is a pure function of those signs. If one basis vector is negated, every projection onto that direction flips sign, which **moves every data point to the sibling branch** at that level. Two SVD solutions spanning the same subspace can produce completely different trees.

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
    orthogonal_basis, svd_pca_basis, dpsgd_pca_basis, random_orthogonal_basis,
    compute_dp_sigmas_zcdp, zcdp_rho_from_epsilon,
)
from utils.LSHTree import hash_leaf_ids, build_lsh_tree, prune_to_leaves
from utils.protocols import lsh_proto, mpi_lsh_proto
```

**Basis generation** (`utils/ortho_clustering.py`)

- `orthogonal_basis(X, d_prime, method="dpsgd_pca", seed=42, **kwargs)` — dispatcher returning a `(d, d')` orthonormal SimHash basis (columns = projection vectors). `method ∈ {"random", "svd_pca", "dpsgd_pca"}`. For `dpsgd_pca`, pass `epsilon`, `delta`, `clip_norm`, optionally `data_fraction`.
- `svd_pca_basis(X, d_prime)` — non-private PCA (oracle baseline).
- `dpsgd_pca_basis(X, d_prime, epsilon, delta, clip_norm, epochs=10, lr=0.01, batch_size=256, data_fraction=0.1)` — private PCA via DP-SGD. See [DP-SGD PCA Basis](#dp-sgd-pca-basis).
- `random_orthogonal_basis(d, d_prime, seed=42)` — random Gaussian matrix orthogonalized via SVD.

**LSH tree & DP noise** (`utils/LSHTree.py`, `utils/ortho_clustering.py`)

- `hash_leaf_ids(X, basis)` — vectorized: every point's `d'` sign bits packed into one integer leaf id in `[0, 2^d')` (MSB-first, so a length-`L` prefix `v` covers the leaf range `[v << (d'-L), (v+1) << (d'-L))`).
- `build_lsh_tree(points, basis, max_depth, min_count_to_branch, min_count_in_node, count_sigma, base_seed=0)` — bucket points by leaf id and grow the pruned prefix tree; `tree.private_centers(center_sigma)` returns one noisy centroid per surviving leaf.
- `prune_to_leaves(get_count, d_prime, min_count_to_branch, min_count_in_node)` — pure pruning: turn a per-node (noisy) count oracle into the surviving-leaf set. Shared by the centralized tree and the federated server so they always agree.
- `compute_dp_sigmas_zcdp(epsilon, delta, sigma_fraction, count_levels)` — **rigorous zCDP** split of the aggregation budget into `(sigma_centers, sigma_count)`. One leaf-sum release plus `count_levels = max_depth + 1` sequential count releases compose in zCDP; `sigma_count / sigma_centers == sigma_fraction`. Sensitivity 1 on unit-norm data (so noise is dimension-independent).

**Protocols** (`utils/protocols.py`)

- `lsh_proto(value_lists, params)` — centralized LSH (used by `--exp_type accuracy`/`scale`).
- `mpi_lsh_proto(value_lists, params)` — federated LSH over MPI (rank 0 = server, ranks 1..N = client shards), used by `--exp_type timing --protocol lsh`. Produces results identical to `lsh_proto` (up to float summation order) while measuring communication.

### Implementation (centralized and federated)

Both share the exact same math and the same prefix-seeded noise, so they give the
same result; they differ only in *where* the data lives.

**Centralized — `lsh_proto` → `build_lsh_tree` (`utils/LSHTree.py`).** Single process:
1. Hash every point to one integer **leaf id** (`hash_leaf_ids`) and sort the ids.
2. A node's count = how many ids fall in its leaf-id range (a binary search on the
   sorted ids); add Gaussian count noise (`sigma_count`) and **prune** with
   `prune_to_leaves`.
3. For each surviving leaf, centroid = `(sum of its points + Gaussian noise) / noisy_count`.

Memory is **O(n·d)**: points are bucketed by leaf id, never copied once per tree
level — this is what lets it run at large `n` and high `d'`.

**Federated — `mpi_lsh_proto` → `LshClient` / `LshServer` (`parties/`).** Distributed
over MPI exactly like FastLloyd (rank 0 = server, ranks 1..N = client shards holding
the data), in three short rounds:
1. **Basis** — clients send a small data subsample; the server builds the SimHash
   basis and broadcasts it.
2. **Counts** — each client hashes its shard and sends a **sparse** histogram (only
   its *occupied* leaf ids + counts). The server merges them, adds count noise,
   prunes, and broadcasts the surviving leaf-id ranges.
3. **Sums** — each client returns its per-leaf sums (vectorized scatter-add); the
   server adds the centroid noise, forms the centroids, and broadcasts them.

Communication is **O(occupied leaves)** for counts and **O(leaves·d)** for sums —
never the dense `2^d'` — and per-rank memory is O(n·d). The server only ever sees
*aggregated, noisy* counts and sums (the same trust model as FastLloyd's server);
the DP noise is added by the server before anything is released.

### Testing accuracy vs. baselines

This compares LSH (each basis) against the FastLloyd-family baselines (Lloyd, FastLloyd, …) on clustering-quality metrics across the privacy sweep.

**One command** — run baselines + LSH (all three bases) and generate the plots:

```bash
bash scripts/run_lsh.sh                 # datasets/bases/d' configured at the top of the script
```

**Or step by step:**

```bash
# 1. Baselines (Lloyd / FastLloyd / …) -> variances.csv
python experiments.py --exp_type accuracy --protocol local --datasets mnist

# 2. LSH with all three bases -> variances_lsh.csv
python experiments.py --exp_type accuracy --protocol lsh \
    --basis_method random svd_pca dpsgd_pca \
    --d_primes 1 2 3 4 5 \
    --tree_min_count 30 \
    --basis_epsilon 0.1 --basis_clip_norm 1.0 --basis_data_fraction 0.1 \
    --datasets mnist --num_runs 10 --results_folder submission

# 3a. Accuracy: LSH vs baselines (line charts over epsilon, one subplot per d')
python -m plots.compare_methods submission --ignore SuLloyd

# 3b. Basis comparison at a fixed epsilon (grouped bars over d')
python -m plots.compare_basis submission --eps 1.0
```

With `--protocol lsh`, the DP/method/post knobs are forced to `"none"`; `d'` is swept over `--d_primes`; ε is the experiment's `eps_budgets` (e.g. `0.5 1 2 4`); and the centroid/count noise is calibrated per ε via `compute_dp_sigmas_zcdp`. Results use the same CSV format and metrics (NICV, Silhouette, …) as the baselines.

### Testing scalability vs. baselines

This compares federated LSH (`mpi_lsh_proto`) against FastLloyd (`mpi_proto`) on **communication rounds, bytes, and wall-time** (with a simulated per-round network delay), sweeping the number of clients and dataset size `n`.

**One command** — sweep clients for both protocols and generate the plots:

```bash
bash scripts/run_lsh_timing.sh          # NCLIENTS / DATASETS / NUM_RUNS / … via env vars
```

It writes FastLloyd to `$RESULT_FOLDER/baselines/` and LSH to `$RESULT_FOLDER/lsh/` (separate folders — the timing harness names every protocol's per-rank output `variances_<rank>.csv`, so they must not share a folder), then runs `compare_timing` into `$RESULT_FOLDER/timing_compare/`.

**Or step by step** (one client count shown; `-np` = clients + 1 for the server):

```bash
mpirun -np 5 python experiments.py --exp_type timing --protocol local \
    --datasets timesynth_2_2_10000 timesynth_2_2_100000 \
    --results_folder submission_timing/baselines

mpirun -np 5 python experiments.py --exp_type timing --protocol lsh \
    --basis_method random --d_primes 2 --tree_min_count 30 \
    --datasets timesynth_2_2_10000 timesynth_2_2_100000 \
    --results_folder submission_timing/lsh

python -m plots.compare_timing submission_timing/baselines submission_timing/lsh \
    --out submission_timing/timing_compare
```

> Note: under `--exp_type timing`, `--protocol local` selects the MPI baseline `mpi_proto` (FastLloyd over MPI), not the single-process `local_proto`. `compare_timing` emits `time_*.pdf`, `comm_*.pdf`, and `timing_compare.csv` (wall-time and total bytes vs. number of clients, one line per protocol). The `n`-axis is swept by passing `timesynth_<k>_<d>_<n>` datasets of different sizes; `d`-scaling is not yet wired up.

### Results file layout

```
submission/accuracy/<dataset>/
├── variances.csv                       # Baselines (Lloyd, FastLloyd, …)
├── variances_lsh.csv                   # LSH results (all basis methods)
├── NICV.pdf, Silhouette.pdf, …         # compare_methods: LSH vs baselines per metric
└── basis_compare_NICV_eps1.0.pdf, …    # compare_basis: bases across d' at fixed eps
submission/accuracy/
├── comparison_summary.csv              # compare_methods summary
└── basis_comparison_summary.csv        # compare_basis summary

submission_timing/                      # scalability (RESULT_FOLDER in run_lsh_timing.sh)
├── baselines/timing_<n>/<dataset>/variances_<rank>.csv   # FastLloyd (mpi_proto)
├── lsh/timing_<n>/<dataset>/variances_<rank>.csv         # LSH (mpi_lsh_proto)
└── timing_compare/                     # time_*.pdf, comm_*.pdf, timing_compare.csv
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
