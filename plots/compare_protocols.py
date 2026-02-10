"""
Compare clustering results between ortho protocol and original algorithms.

Loads variances.csv (local protocol) and variances_ortho.csv (ortho protocol)
from each dataset folder, builds a side-by-side comparison, and generates
per-dataset bar charts and a summary CSV.

Usage:
    python -m plots.compare_protocols                          # default
    python -m plots.compare_protocols submission               # custom folder
    python -m plots.compare_protocols submission --exp_type scale
"""

import os
import sys
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

# Metric full names -> short labels (shared with per_dataset.py)
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
    """Extract representative rows from local protocol results.

    Returns a list of dicts with keys: label, and each metric (value + _h).
    - Lloyd (dp=none) baseline
    - Best FastLloyd row per epsilon (lowest NICV)
    """
    nicv_col = "Normalized Intra-cluster Variance (NICV)"
    rows = []

    for (method, dp, post), group in df.groupby(["method", "dp", "post"]):
        key = (method, dp, post)
        if key not in METHOD_NAMES:
            continue
        name = METHOD_NAMES[key]

        if dp == "none":
            # Non-private baseline — single row
            best = group.loc[group[nicv_col].idxmin()] if len(group) > 1 else group.iloc[0]
            row = {"label": name}
            for metric in METRICS_DICT:
                if metric in best.index:
                    row[metric] = best[metric]
                    row[f"{metric}_h"] = best.get(f"{metric}_h", 0)
            rows.append(row)
        else:
            # DP methods — pick best epsilon
            best = group.loc[group[nicv_col].idxmin()]
            eps_val = best["eps"]
            label = f"{name} (e={eps_val})"
            row = {"label": label}
            for metric in METRICS_DICT:
                if metric in best.index:
                    row[metric] = best[metric]
                    row[f"{metric}_h"] = best.get(f"{metric}_h", 0)
            rows.append(row)

    return rows


def load_ortho_rows(df):
    """Extract rows from ortho protocol results (one per d_prime)."""
    rows = []
    for _, r in df.iterrows():
        d_prime = int(r.get("d_prime", 0))
        label = f"Ortho (d'={d_prime})"
        row = {"label": label}
        for metric in METRICS_DICT:
            if metric in r.index:
                row[metric] = r[metric]
                row[f"{metric}_h"] = r.get(f"{metric}_h", 0)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(comp_df, dataset, folder, metrics):
    """Generate one bar chart per metric for a dataset."""
    labels = comp_df["label"].tolist()
    colors = [_label_color(l) for l in labels]
    x = np.arange(len(labels))

    for metric in metrics:
        if metric not in comp_df.columns:
            continue
        short = METRICS_DICT[metric]

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
        vals = comp_df[metric].values.astype(float)
        errs = comp_df.get(f"{metric}_h", pd.Series(np.zeros(len(vals)))).values.astype(float)

        ax.bar(x, vals, yerr=errs, color=colors, capsize=4, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=12)
        ax.set_ylabel(short)
        ax.set_title(f"{dataset} — {short}")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"compare_{short}.pdf"), bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_datasets(results_folder, exp_type):
    """Process all datasets and generate comparisons."""
    base = Path(results_folder) / exp_type
    if not base.is_dir():
        print(f"Results folder not found: {base}")
        return

    all_rows = []
    datasets = [d for d in os.listdir(base) if (base / d).is_dir()]

    for dataset in tqdm(sorted(datasets), desc="datasets"):
        folder = base / dataset
        local_path = folder / "variances.csv"
        ortho_path = folder / "variances_ortho.csv"

        has_local = local_path.exists()
        has_ortho = ortho_path.exists()

        if not has_local and not has_ortho:
            continue

        # Collect comparison rows
        comp_rows = []

        if has_local:
            local_df = pd.read_csv(local_path)
            comp_rows.extend(load_local_rows(local_df))

        if has_ortho:
            ortho_df = pd.read_csv(ortho_path)
            comp_rows.extend(load_ortho_rows(ortho_df))

        if not comp_rows:
            continue

        comp_df = pd.DataFrame(comp_rows)

        # Available metrics for this dataset
        available_metrics = [m for m in METRICS_DICT if m in comp_df.columns]

        # Per-dataset bar charts
        plot_comparison(comp_df, dataset, str(folder), available_metrics)

        # Accumulate for summary
        for row in comp_rows:
            row["dataset"] = dataset
        all_rows.extend(comp_rows)

        print(f"  {dataset}: {len(comp_rows)} methods compared")

    # Summary CSV
    if all_rows:
        summary = pd.DataFrame(all_rows)
        # Reorder columns: dataset, label first
        cols = ["dataset", "label"] + [c for c in summary.columns if c not in ("dataset", "label")]
        summary = summary[cols]
        summary_path = base / "comparison_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")


def parse_args():
    parser = ArgumentParser(description="Compare ortho vs original protocol results")
    parser.add_argument("results_folder", nargs="?", default="submission",
                        help="root results folder (default: submission)")
    parser.add_argument("--exp_type", default="accuracy",
                        help="experiment type to compare (default: accuracy)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_datasets(args.results_folder, args.exp_type)
