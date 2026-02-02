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
│   ├── ablation_plots.py # Visualization for ablation studies
│   ├── per_dataset.py    # Dataset-specific result visualization
│   ├── scale_heatmap.py  # Heatmap generation for scalability results
│   ├── synthetic_bar.py  # Bar charts for synthetic dataset results
│   └── timing_analysis.py # Analysis of timing experiments
│                                                                                                                                                    
├── scripts/
│   ├── download_data.py   # Download and prepare datasets from sklearn/UCI
│   ├── generator.R        # R script for generating synthetic datasets
│   ├── experiment_runner.sh # Script for running accuracy and scale experiments
│   └── timing_runner.sh   # Script for running timing experiments
│                                                                                                                                                    
├── utils/                                                                                                                                           
│   ├── evaluations.py    # Clustering quality evaluation metrics
│   ├── ortho_clustering.py # Orthogonal projection clustering assignment
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
bash scripts/experiment_runner.sh  # For accuracy and scale experiments
bash scripts/timing_runner.sh      # For timing experiments with varying numbers of clients
```

### Visualization

The repository includes several visualization tools in the `plots` directory:

- `per_dataset.py`: Creates performance visualizations for individual datasets
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
- `--datasets`: Datasets to use for the experiment
- `--method`: Maximum distance method to use
- `--alpha`: Maximum distance parameter
- `--post`: Post-processing method for centroids
- `--results_folder`: Folder to store results

## Orthogonal Projection Clustering

`utils/ortho_clustering.py` implements a fast clustering assignment based on orthogonal projections:

```python
from utils.ortho_clustering import ortho_assign

labels = ortho_assign(values, d_prime, seed=42)
```

**How it works:** Given `n` points in `d` dimensions, the function generates `d'` random orthonormal basis vectors via SVD of a random matrix. Each point is projected onto this basis, and cluster membership is determined by the sign pattern of the projections — effectively assigning points to quadrants in the projected space. This produces up to `2^d'` clusters.

- `values`: `(n, d)` numpy array of data points
- `d_prime`: number of orthogonal basis vectors (capped at `d` automatically)
- `seed`: random seed for reproducibility
- Returns: `(n,)` integer array of cluster labels in `[0, 2^d' - 1]`

### Running the standalone test

```bash
python ortho_cluster_test.py
```

This runs `ortho_assign` across the accuracy datasets for `d' = 1..5` with 10 random seeds each, and saves results to `ortho_results/results.csv` with columns: `dataset`, `n`, `d`, `d_prime`, `seed`, `num_clusters`, `num_occupied`, `elapsed`, `cluster_size_min`, `cluster_size_max`, `cluster_size_std`.

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
