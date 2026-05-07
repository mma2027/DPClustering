"""
Compare clustering quality across three orthogonal-projection basis methods:
  random     — random orthonormal basis (no DP cost, no data-dependent direction)
  svd_pca    — non-private PCA via SVD (oracle: best possible basis, no privacy)
  dpsgd_pca  — DP-SGD PCA (private basis, our method)

For each dataset, generates one PDF per metric showing NICV (and other metrics)
as grouped bars over d' values, one group per basis method.

Assumes all three methods have been run with `--protocol ortho` and their results
are stored in the same variances_ortho.csv (distinguished by the basis_method column).

Usage:
    python -m plots.compare_basis                         # submission/, accuracy
    python -m plots.compare_basis my_results              # custom folder
    python -m plots.compare_basis my_results --exp_type scale
    python -m plots.compare_basis my_results --sigma 0.0  # filter to one sigma
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

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 14})

METRICS_DICT = {
    "Normalized Intra-cluster Variance (NICV)": "NICV",
    "Between-Cluster Sum of Squares (BCSS)": "BCSS",
    "Silhouette Score": "Silhouette",
    "Davies-Bouldin Index": "Davies-Bouldin",
    "Calinski-Harabasz Index": "Calinski-Harabasz",
    "Dunn Index": "Dunn",
    "Mean Squared Error": "MSE",
}

BASIS_DISPLAY = {
    "random":    "Random",
    "svd_pca":   "SVD PCA",
    "dpsgd_pca": "DP-SGD PCA",
}

BASIS_COLORS = {
    "random":    "steelblue",
    "svd_pca":   "seagreen",
    "dpsgd_pca": "mediumpurple",
}

# Within a basis group, show bars for local baselines too (from variances.csv)
LOCAL_BASELINES = {
    ("none", "none", "none"):                           ("Lloyd",     "black"),
    ("diagonal_then_frac", "gaussiananalytic", "fold"): ("FastLloyd", "darkorange"),
}


def _load_local_baselines(csv_path):
    """Return {label: {metric: (mean, half_ci)}} for representative local rows."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    nicv = "Normalized Intra-cluster Variance (NICV)"
    out = {}
    for (method, dp, post), (name, _) in LOCAL_BASELINES.items():
        sub = df[(df["method"] == method) & (df["dp"] == dp) & (df["post"] == post)]
        if sub.empty:
            continue
        best = sub.loc[sub[nicv].idxmin()]
        label = name if dp == "none" else f"{name} (e={best['eps']})"
        out[label] = {}
        for metric in METRICS_DICT:
            if metric in best.index:
                out[label][metric] = (best[metric], best.get(f"{metric}_h", 0))
    return out


def _best_dpsgd_row(group):
    """From a group of dpsgd_pca rows for a fixed d', pick the one with lowest NICV."""
    nicv = "Normalized Intra-cluster Variance (NICV)"
    if nicv in group.columns:
        return group.loc[group[nicv].idxmin()]
    return group.iloc[0]


def plot_basis_comparison(ortho_df, local_baselines, metric, dataset, sigma_filter, out_folder):
    """Plot grouped bars comparing basis methods across d' values for one metric.

    Layout: one bar group per d'.  Within each group: [random, svd_pca, dpsgd_pca].
    Local baselines (Lloyd, FastLloyd) are drawn as horizontal reference lines.

    Args:
        ortho_df: DataFrame from variances_ortho.csv, already filtered to one sigma
        local_baselines: dict from _load_local_baselines
        metric: full metric column name
        dataset: dataset name (for title)
        sigma_filter: the sigma value that was used to filter (for subtitle)
        out_folder: directory to save the PDF
    """
    if metric not in ortho_df.columns:
        return

    basis_methods = [m for m in ["random", "svd_pca", "dpsgd_pca"] if m in ortho_df["basis_method"].values]
    if not basis_methods:
        return

    d_primes = sorted(ortho_df["d_prime"].dropna().unique().astype(int))
    if not d_primes:
        return

    n_groups = len(d_primes)
    n_per_group = len(basis_methods)
    bar_width = 0.7 / n_per_group
    group_positions = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(6, n_groups * n_per_group * 1.0), 5))

    for i, basis in enumerate(basis_methods):
        sub = ortho_df[ortho_df["basis_method"] == basis]
        vals, errs = [], []
        for d_prime in d_primes:
            rows = sub[sub["d_prime"] == d_prime]
            if rows.empty:
                vals.append(np.nan)
                errs.append(0)
                continue
            # For dpsgd_pca pick best epsilon; for others take first (should be unique)
            row = _best_dpsgd_row(rows) if basis == "dpsgd_pca" else rows.iloc[0]
            vals.append(row[metric])
            errs.append(row.get(f"{metric}_h", 0))

        offset = (i - (n_per_group - 1) / 2) * bar_width
        errs_arr = np.array(errs, dtype=float)
        yerr = errs_arr if np.any(np.isfinite(errs_arr) & (errs_arr > 0)) else None
        ax.bar(
            group_positions + offset,
            vals, bar_width,
            yerr=yerr,
            label=BASIS_DISPLAY.get(basis, basis),
            color=BASIS_COLORS.get(basis, "gray"),
            capsize=3, edgecolor="white", alpha=0.9,
        )

    # Draw local baselines as horizontal lines
    line_styles = ["--", ":"]
    for j, (label, metrics) in enumerate(local_baselines.items()):
        if metric not in metrics:
            continue
        val, _ = metrics[metric]
        color = dict(LOCAL_BASELINES.values()).get(label.split(" ")[0], "gray")
        ax.axhline(val, color=color, linestyle=line_styles[j % 2],
                   linewidth=1.5, label=label, alpha=0.8)

    short = METRICS_DICT.get(metric, metric)
    sigma_str = f"  (sigma={sigma_filter})" if sigma_filter is not None else ""
    ax.set_title(f"{dataset} — {short}{sigma_str}", fontsize=14)
    ax.set_xlabel("d'")
    ax.set_ylabel(short)
    ax.set_xticks(group_positions)
    ax.set_xticklabels([str(d) for d in d_primes])
    ax.legend(fontsize=10, loc="best")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    safe = short.replace(" ", "_").replace("/", "_")
    sigma_tag = f"_sigma{sigma_filter}" if sigma_filter is not None else ""
    out_path = os.path.join(out_folder, f"basis_compare_{safe}{sigma_tag}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def process_datasets(results_folder, exp_type, sigma_filter):
    base = Path(results_folder) / exp_type
    if not base.is_dir():
        print(f"Results folder not found: {base}")
        return

    datasets = sorted(d for d in os.listdir(base) if (base / d).is_dir())
    summary_rows = []

    for dataset in tqdm(datasets, desc="datasets"):
        folder = base / dataset
        ortho_path = folder / "variances_ortho.csv"
        if not ortho_path.exists():
            continue

        ortho_df = pd.read_csv(ortho_path)
        if "basis_method" not in ortho_df.columns:
            print(f"  {dataset}: variances_ortho.csv has no basis_method column, skipping")
            continue

        # Filter to requested sigma (or use all sigmas if None)
        if sigma_filter is not None:
            ortho_df = ortho_df[np.isclose(ortho_df["sigma"].fillna(0.0), sigma_filter)]
        else:
            # Default: use sigma=0 if available, otherwise all
            if 0.0 in ortho_df["sigma"].values:
                ortho_df = ortho_df[np.isclose(ortho_df["sigma"].fillna(0.0), 0.0)]

        if ortho_df.empty:
            print(f"  {dataset}: no rows after sigma filter, skipping")
            continue

        local_baselines = _load_local_baselines(folder / "variances.csv")

        for metric in METRICS_DICT:
            plot_basis_comparison(ortho_df, local_baselines, metric, dataset, sigma_filter, str(folder))

        # Collect summary: best NICV per basis_method per d'
        nicv = "Normalized Intra-cluster Variance (NICV)"
        for basis, grp in ortho_df.groupby("basis_method"):
            for d_prime, sub in grp.groupby("d_prime"):
                row = _best_dpsgd_row(sub) if basis == "dpsgd_pca" else sub.iloc[0]
                summary_rows.append({
                    "dataset": dataset,
                    "basis_method": basis,
                    "d_prime": int(d_prime),
                    "sigma": row.get("sigma", 0.0),
                    **{short: row.get(full, np.nan) for full, short in METRICS_DICT.items()},
                })

        basis_present = ortho_df["basis_method"].unique().tolist()
        d_primes = sorted(ortho_df["d_prime"].dropna().unique().astype(int))
        print(f"  {dataset}: basis={basis_present}, d'={d_primes}")

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        out_csv = base / "basis_comparison_summary.csv"
        summary.to_csv(out_csv, index=False)
        print(f"\nSummary saved to {out_csv}")


def parse_args():
    parser = ArgumentParser(description="Compare ortho clustering across basis methods")
    parser.add_argument("results_folder", nargs="?", default="submission",
                        help="root results folder (default: submission)")
    parser.add_argument("--exp_type", default="accuracy",
                        help="experiment type subfolder (default: accuracy)")
    parser.add_argument("--sigma", type=float, default=None,
                        help="filter to a specific sigma value (default: use sigma=0 if present)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_datasets(args.results_folder, args.exp_type, args.sigma)
