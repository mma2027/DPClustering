"""
Compare clustering results between ortho protocol and original algorithms.

Loads variances.csv (local protocol) and variances_ortho.csv (ortho protocol)
from each dataset folder. For each dataset, generates one PDF per metric where:
  - rows  = d' values
  - bars  = baseline algorithms first, then ortho sigma variants for that d'

Output: submission/accuracy/<dataset>/<MetricName>.pdf

Usage:
    python -m plots.compare_protocols                          # default
    python -m plots.compare_protocols submission               # custom folder
    python -m plots.compare_protocols submission --exp_type scale
"""

import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from configs.defaults import accuracy_datasets

# Standard plotting configuration (matches per_dataset.py)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 18})

# Metric full names -> short labels
METRICS_DICT = {
    "Normalized Intra-cluster Variance (NICV)": "NICV",
    "Between-Cluster Sum of Squares (BCSS)": "BCSS",
    "Silhouette Score": "Silhouette",
    "Davies-Bouldin Index": "Davies-Bouldin",
    "Calinski-Harabasz Index": "Calinski-Harabasz",
    "Dunn Index": "Dunn",
    "Mean Squared Error": "MSE",
}

# Method key -> display name (for local protocol rows)
METHOD_NAMES = {
    ("none", "none", "none"): "Lloyd",
    ("none", "laplace", "none"): "SuLloyd",
    ("none", "gaussiananalytic", "none"): "GLloyd",
    ("diagonal_then_frac", "gaussiananalytic", "fold"): "FastLloyd",
}

# Colors per label prefix
COLORS = {
    "Lloyd": "black",
    "SuLloyd": "red",
    "GLloyd": "orange",
    "FastLloyd": "green",
    "Ortho-DP": "mediumpurple",  # check Ortho-DP before Ortho so prefix match is correct
    "Ortho": "royalblue",
}


def _label_color(label):
    """Return bar color based on method label prefix."""
    for prefix, color in COLORS.items():
        if label.startswith(prefix):
            return color
    return "gray"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_local_rows(df):
    """Extract representative baseline rows from local protocol results.

    Returns rows tagged with sigma=None, d_prime=None.
    Picks best epsilon (lowest NICV) for each DP method.
    """
    nicv_col = "Normalized Intra-cluster Variance (NICV)"
    rows = []

    for (method, dp, post), group in df.groupby(["method", "dp", "post"]):
        key = (method, dp, post)
        if key not in METHOD_NAMES:
            continue
        name = METHOD_NAMES[key]

        if dp == "none":
            best = group.loc[group[nicv_col].idxmin()] if len(group) > 1 else group.iloc[0]
            row = {"label": name, "sigma": None, "d_prime": None}
        else:
            best = group.loc[group[nicv_col].idxmin()]
            row = {"label": f"{name} (e={best['eps']})", "sigma": None, "d_prime": None}

        for metric in METRICS_DICT:
            if metric in best.index:
                row[metric] = best[metric]
                row[f"{metric}_h"] = best.get(f"{metric}_h", 0)
        rows.append(row)

    return rows


def load_ortho_rows(df):
    """Extract ortho rows tagged with d_prime and sigma.

    Label shows only sigma (d' is shown as the row/subplot label instead).
    """
    rows = []
    for _, r in df.iterrows():
        d_prime = int(r.get("d_prime", 0))
        sigma = float(r.get("sigma", 0.0))
        # Label shows sigma only — d' is the row header in the plot
        label = f"Ortho-DP (σ={sigma})" if sigma > 0 else "Ortho (σ=0)"
        row = {"label": label, "sigma": sigma, "d_prime": d_prime}
        for metric in METRICS_DICT:
            if metric in r.index:
                row[metric] = r[metric]
                row[f"{metric}_h"] = r.get(f"{metric}_h", 0)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Plotting: one PDF per dataset per metric
# ---------------------------------------------------------------------------

def plot_dataset_metric(rows, metric, dataset, d_primes, sigmas, out_folder):
    """
    For one dataset + one metric, generate a PDF with:
      rows  = d' values (one subplot each)
      bars  = [baselines...,  Ortho(d', σ=0),  Ortho-DP(d', σ=0.1), ...]

    Saved as <MetricShortName>.pdf inside out_folder.
    """
    baseline_rows = [r for r in rows if r["d_prime"] is None]
    n_bars = len(baseline_rows) + len(sigmas)
    cell_w = max(8, n_bars * 1.2)

    fig, axes = plt.subplots(len(d_primes), 1,
                             figsize=(cell_w, len(d_primes) * 3.5),
                             squeeze=False)

    for row_i, d_prime in enumerate(d_primes):
        ax = axes[row_i][0]

        # Ortho rows for this d', ordered by sigma
        ortho_rows = sorted(
            [r for r in rows if r["d_prime"] == d_prime],
            key=lambda r: r["sigma"]
        )

        display_rows = baseline_rows + ortho_rows
        if not display_rows:
            ax.set_visible(False)
            continue

        labels = [r["label"] for r in display_rows]
        vals = np.array([r.get(metric, np.nan) for r in display_rows], dtype=float)
        errs = np.array([r.get(f"{metric}_h", 0) for r in display_rows], dtype=float)
        colors = [_label_color(l) for l in labels]

        x = np.arange(len(labels))
        ax.bar(x, vals, yerr=errs, color=colors, capsize=4, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
        ax.set_ylabel(f"d' = {d_prime}", fontsize=12, labelpad=8)
        ax.grid(axis="y", alpha=0.3)

    short = METRICS_DICT.get(metric, metric)
    fig.suptitle(f"{dataset}  —  {short}", fontsize=16, y=1.01)
    fig.tight_layout()

    safe = short.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(out_folder, f"{safe}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_datasets(results_folder, exp_type):
    """Load all datasets, generate one PDF per metric per dataset."""
    base = Path(results_folder) / exp_type
    if not base.is_dir():
        print(f"Results folder not found: {base}")
        return

    all_rows = []  # for summary CSV
    datasets = sorted([d for d in os.listdir(base) if (base / d).is_dir()])

    for dataset in tqdm(datasets, desc="datasets"):
        folder = base / dataset
        local_path = folder / "variances.csv"
        ortho_path = folder / "variances_ortho.csv"

        has_local = local_path.exists()
        has_ortho = ortho_path.exists()

        if not has_local and not has_ortho:
            continue

        rows = []
        if has_local:
            rows.extend(load_local_rows(pd.read_csv(local_path)))
        if has_ortho:
            rows.extend(load_ortho_rows(pd.read_csv(ortho_path)))

        if not rows:
            continue

        # Accumulate for summary CSV
        for r in rows:
            all_rows.append({**r, "dataset": dataset})

        # d' values and sigma values present in this dataset's ortho data
        d_primes = sorted(set(r["d_prime"] for r in rows if r["d_prime"] is not None))
        sigmas = sorted(set(r["sigma"] for r in rows if r["sigma"] is not None))

        if not d_primes:
            # No ortho data yet — skip plotting for now
            print(f"  {dataset}: no ortho data, skipping plots")
            continue

        # One PDF per metric
        for metric in METRICS_DICT:
            plot_dataset_metric(rows, metric, dataset, d_primes, sigmas, str(folder))

        print(f"  {dataset}: {len(rows)} entries, {len(d_primes)} d' values, {len(sigmas)} sigmas")

    # Summary CSV
    if all_rows:
        summary = pd.DataFrame(all_rows)
        cols = ["dataset", "label"] + [c for c in summary.columns if c not in ("dataset", "label")]
        summary[cols].to_csv(base / "comparison_summary.csv", index=False)
        print(f"\nSummary saved to {base / 'comparison_summary.csv'}")


def parse_args():
    parser = ArgumentParser(description="Compare ortho vs original protocol results")
    parser.add_argument("results_folder", nargs="?", default="submission",
                        help="root results folder (default: submission)")
    parser.add_argument("--exp_type", default="accuracy",
                        help="experiment type subfolder (default: accuracy)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_datasets(args.results_folder, args.exp_type)
