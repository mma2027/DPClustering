"""
Compare clustering results between the LSH protocol and original algorithms.

Loads variances.csv (local protocol) and variances_lsh.csv (LSH protocol)
from each dataset folder. For each dataset, generates one PDF per metric where:
  - rows    = d' values  (one subplot per d'; for LSH d' is the basis width /
              max tree depth)
  - x-axis  = epsilon (privacy budget)
  - y-axis  = metric value
  - lines   = methods (baselines + LSH variants), one colored line each

Within a subplot, every method that depends on epsilon (SuLloyd, GLloyd,
FastLloyd, and the LSH-* variants for that d') is drawn as a line over the
swept epsilon values. The non-private baseline (Lloyd) has no epsilon
dependence and is drawn as a horizontal reference line. Baseline lines are
repeated in every d' subplot so the LSH results for that d' can be read
against them.

Output: submission/accuracy/<dataset>/<MetricName>.pdf

Usage:
    python -m plots.compare_methods                          # default
    python -m plots.compare_methods submission               # custom folder
    python -m plots.compare_methods submission --exp_type scale
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

# Preferred legend / line ordering
METHOD_ORDER = [
    "Lloyd", "SuLloyd", "GLloyd", "FastLloyd",
    "LSH-Rand", "LSH-SVD", "LSH-DP-SGD",
]

# Colors per label prefix
COLORS = {
    "Lloyd": "black",
    "SuLloyd": "red",
    "GLloyd": "orange",
    "FastLloyd": "green",
    "LSH-DP-SGD": "mediumpurple",
    "LSH-SVD": "seagreen",
    "LSH-Rand": "steelblue",
    "LSH": "royalblue",  # fallback for unlabelled LSH rows
}


def _label_color(label):
    """Return line color based on method label prefix."""
    for prefix, color in COLORS.items():
        if label.startswith(prefix):
            return color
    return "gray"


# Method labels to exclude entirely (no line, no legend entry, no summary row).
# Matching is by prefix, mirroring _label_color, so:
#   "LSH"      -> drops every LSH-* variant
#   "SuLloyd"  -> drops only SuLloyd
# Leave empty to keep everything. Can also be extended at runtime via --ignore.
IGNORE_METHODS = set()  # e.g. {"SuLloyd", "LSH-Rand"}


def _is_ignored(label, ignore):
    """True if the method label matches any ignore-list prefix."""
    return any(label.startswith(prefix) for prefix in ignore)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_local_rows(df):
    """Extract baseline rows from local protocol results.

    Loads ALL epsilon values (not only the best one):
      - DP methods (SuLloyd, GLloyd, FastLloyd) yield one row per epsilon,
        each tagged with that eps (averaged over any repeated runs at the
        same eps).
      - The non-private baseline (Lloyd) has no epsilon dependence, so it
        yields a single row with eps=None.

    All baseline rows are tagged with d_prime=None so they can be drawn as
    reference lines in every d' subplot.
    """
    rows = []

    for (method, dp, post), group in df.groupby(["method", "dp", "post"]):
        key = (method, dp, post)
        if key not in METHOD_NAMES:
            continue
        name = METHOD_NAMES[key]

        if dp == "none":
            # Non-private baseline: averaged over any repeated runs, eps-free.
            row = {"label": name, "eps": None, "d_prime": None}
            for metric in METRICS_DICT:
                if metric in group.columns:
                    row[metric] = float(group[metric].mean())
                    hcol = f"{metric}_h"
                    row[hcol] = float(group[hcol].mean()) if hcol in group.columns else 0.0
            rows.append(row)
        else:
            # DP method: keep every epsilon as its own point on the line.
            for eps_val, eg in group.groupby("eps"):
                row = {"label": name, "eps": float(eps_val), "d_prime": None}
                for metric in METRICS_DICT:
                    if metric in eg.columns:
                        row[metric] = float(eg[metric].mean())
                        hcol = f"{metric}_h"
                        row[hcol] = float(eg[hcol].mean()) if hcol in eg.columns else 0.0
                rows.append(row)

    return rows


def load_lsh_rows(df):
    """Extract LSH rows tagged with d_prime and eps.

    The label encodes only the method (basis), since eps is now the x-axis
    and d' is the subplot. One row per CSV row.
    """
    rows = []
    for _, r in df.iterrows():
        d_prime = int(r.get("d_prime", 0))
        eps = float(r.get("eps", 0.0))
        basis = r.get("basis_method", "")
        basis_short = {"random": "Rand", "svd_pca": "SVD", "dpsgd_pca": "DP-SGD"}.get(basis, basis)
        label = f"LSH-{basis_short}"
        row = {"label": label, "eps": eps, "d_prime": d_prime, "basis_method": basis}
        for metric in METRICS_DICT:
            if metric in r.index:
                row[metric] = r[metric]
                row[f"{metric}_h"] = r.get(f"{metric}_h", 0)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Plotting: one PDF per dataset per metric
# ---------------------------------------------------------------------------

def plot_dataset_metric(rows, metric, dataset, d_primes, epss, out_folder):
    """
    For one dataset + one metric, generate a PDF with:
      rows  = d' values (one subplot each)
      x     = epsilon, y = metric value
      lines = methods. Baselines are repeated in every subplot (DP baselines
              as dashed lines over eps, the non-private Lloyd as a dotted
              horizontal reference). LSH variants for that d' are solid
              lines over eps.

    Saved as <MetricShortName>.pdf inside out_folder.
    """
    baseline_rows = [r for r in rows if r["d_prime"] is None]

    # sharey keeps the (identical) baselines at the same height across d'
    # subplots, which makes the comparison easy to read. Flip to False if the
    # metric scale differs wildly between d' values.
    fig, axes = plt.subplots(len(d_primes), 1,
                             figsize=(10, max(3.5, len(d_primes) * 3.2)),
                             squeeze=False, sharex=True, sharey=True)

    for row_i, d_prime in enumerate(d_primes):
        ax = axes[row_i, 0]

        lsh_here = [r for r in rows if r["d_prime"] == d_prime]
        subplot_rows = baseline_rows + lsh_here
        if not subplot_rows:
            ax.set_visible(False)
            continue

        # Stable, readable method ordering
        methods = []
        for r in subplot_rows:
            if r["label"] not in methods:
                methods.append(r["label"])
        methods.sort(key=lambda m: METHOD_ORDER.index(m)
                     if m in METHOD_ORDER else len(METHOD_ORDER))

        for label in methods:
            series = [r for r in subplot_rows if r["label"] == label]
            color = _label_color(label)
            is_baseline = series[0]["d_prime"] is None

            const = [r for r in series if r["eps"] is None]
            # eps > 0 only: a log axis cannot place eps == 0
            varying = sorted([r for r in series if r["eps"] is not None and r["eps"] > 0],
                             key=lambda r: r["eps"])

            labeled = False

            # Non-private baseline -> horizontal reference line
            if const:
                val = float(np.nanmean([r.get(metric, np.nan) for r in const]))
                if not np.isnan(val):
                    ax.axhline(val, color=color, linestyle=":",
                               linewidth=1.6, alpha=0.9, label=label)
                    labeled = True

            # eps-dependent methods -> line over epsilon
            if varying:
                xs = np.array([r["eps"] for r in varying], dtype=float)
                ys = np.array([r.get(metric, np.nan) for r in varying], dtype=float)
                es = np.array([r.get(f"{metric}_h", 0) for r in varying], dtype=float)
                ls = "--" if is_baseline else "-"
                ax.errorbar(xs, ys, yerr=es, color=color, linestyle=ls,
                            marker="o", markersize=5, capsize=3, linewidth=1.8,
                            label=(None if labeled else label))

        # --- log-scale x-axis (epsilon values are powers of 2) ---
        ax.set_xscale("log", base=2)
        pos_eps = sorted(e for e in epss if e > 0)
        if pos_eps:
            ax.set_xticks(pos_eps)
            ax.set_xticklabels([f"{e:g}" for e in pos_eps])  # 0.25, 0.5, 1, 2, 4...
            ax.minorticks_off()                              # hide dense log minor ticks
        ax.set_ylabel(f"d' = {d_prime}\n{METRICS_DICT.get(metric, metric)}",
                      fontsize=11, labelpad=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("ε (privacy budget)", fontsize=13)

    # One combined legend (collected across subplots so nothing is missed),
    # placed just outside the top subplot to avoid covering data.
    handles, labels = [], []
    for ax in axes[:, 0]:
        if not ax.get_visible():
            continue
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                labels.append(l)
                handles.append(h)
    if handles:
        axes[0, 0].legend(handles, labels, fontsize=8, title="Method",
                          loc="upper left", bbox_to_anchor=(1.02, 1.0),
                          borderaxespad=0.0)

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

def process_datasets(results_folder, exp_type, ignore=IGNORE_METHODS):
    """Load all datasets, generate one PDF per metric per dataset.

    `ignore` is a set of method-label prefixes to drop (see IGNORE_METHODS).
    """
    base = Path(results_folder) / exp_type
    if not base.is_dir():
        print(f"Results folder not found: {base}")
        return

    all_rows = []  # for summary CSV
    datasets = sorted([d for d in os.listdir(base) if (base / d).is_dir()])

    for dataset in tqdm(datasets, desc="datasets"):
        folder = base / dataset
        local_path = folder / "variances.csv"
        lsh_path = folder / "variances_lsh.csv"

        has_local = local_path.exists()
        has_lsh = lsh_path.exists()

        if not has_local and not has_lsh:
            continue

        rows = []
        if has_local:
            rows.extend(load_local_rows(pd.read_csv(local_path)))
        if has_lsh:
            rows.extend(load_lsh_rows(pd.read_csv(lsh_path)))

        # Drop ignored methods before anything else looks at `rows`.
        if ignore:
            rows = [r for r in rows if not _is_ignored(r["label"], ignore)]

        if not rows:
            continue

        # Accumulate for summary CSV
        for r in rows:
            all_rows.append({**r, "dataset": dataset})

        # d' values (from LSH data) and eps values present in this dataset.
        # epss now spans both local DP sweeps and LSH sweeps -> shared ticks.
        d_primes = sorted(set(r["d_prime"] for r in rows if r["d_prime"] is not None))
        epss = sorted(set(r["eps"] for r in rows if r["eps"] is not None))

        if not d_primes:
            # No LSH data yet — no d' rows to draw against.
            print(f"  {dataset}: no LSH data, skipping plots")
            continue

        # One PDF per metric
        for metric in METRICS_DICT:
            plot_dataset_metric(rows, metric, dataset, d_primes, epss, str(folder))

        print(f"  {dataset}: {len(rows)} entries, {len(d_primes)} d' values, {len(epss)} epss")

    # Summary CSV
    if all_rows:
        summary = pd.DataFrame(all_rows)
        front = [c for c in ["dataset", "label", "d_prime", "eps"] if c in summary.columns]
        cols = front + [c for c in summary.columns if c not in front]
        summary[cols].to_csv(base / "comparison_summary.csv", index=False)
        print(f"\nSummary saved to {base / 'comparison_summary.csv'}")


def parse_args():
    parser = ArgumentParser(description="Compare LSH vs original protocol results")
    parser.add_argument("results_folder", nargs="?", default="submission",
                        help="root results folder (default: submission)")
    parser.add_argument("--exp_type", default="accuracy",
                        help="experiment type subfolder (default: accuracy)")
    parser.add_argument("--ignore", nargs="*", default=None, metavar="PREFIX",
                        help="method label prefixes to skip, e.g. "
                             "--ignore SuLloyd LSH-Rand  (adds to IGNORE_METHODS)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ignore = set(IGNORE_METHODS)
    if args.ignore:
        ignore |= set(args.ignore)
    process_datasets(args.results_folder, args.exp_type, ignore)
