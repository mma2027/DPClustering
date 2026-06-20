"""
Compare federated LSH vs FastLloyd scalability from MPI timing runs.

Reads `timing_<n_clients>/<dataset>/variances_<rank>.csv` produced by
`experiments.py --exp_type timing` from one or more results folders (e.g.
`<root>/baselines` for FastLloyd and `<root>/lsh` for LSH). For each dataset,
produces one PDF per timing metric laid out as a GRID:
  - rows    = d' values (basis width / max tree depth)
  - columns = epsilon (privacy budget)
  - x-axis  = number of clients (log2: 2, 4, 8, 16, 32, ...)
  - y-axis  = the timing metric (wall-time, communication bytes, or rounds)
  - lines   = methods: FastLloyd (baseline, repeated across all d' rows at its
              epsilon column) plus one line per LSH basis (LSH-Rand/SVD/DP-SGD)

Wall-time depends on the simulated network delay, so it gets one PDF per delay;
communication bytes and rounds are delay-independent (one PDF each).

Usage:
    python -m plots.compare_timing submission_timing/baselines submission_timing/lsh
    python -m plots.compare_timing folderA folderB --out submission_timing/timing_compare
"""

import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 11})

BASIS_SHORT = {"random": "Rand", "svd_pca": "SVD", "dpsgd_pca": "DP-SGD"}

# Preferred legend / line ordering and colors (match compare_methods.py)
METHOD_ORDER = ["FastLloyd", "LSH-Rand", "LSH-SVD", "LSH-DP-SGD"]
COLORS = {
    "FastLloyd":   "green",
    "LSH-Rand":    "steelblue",
    "LSH-SVD":     "seagreen",
    "LSH-DP-SGD":  "mediumpurple",
}

# Timing metrics -> (y-axis label, CI column or None, delay-dependent?)
METRICS = {
    "elapsed_ms": ("Wall-time (ms)", "elapsed_h_ms", True),
    "comm_bytes": ("Communication (bytes)", None, False),
    "rounds":     ("Communication rounds", None, False),
}


def _label_color(label):
    return COLORS.get(label, "gray")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_dataset_dir(ds_dir):
    """Merge the server CSV (elapsed/rounds) with total comm summed over ranks."""
    server = pd.read_csv(ds_dir / "variances_0.csv")
    rank_files = list(ds_dir.glob("variances_*.csv"))
    allranks = pd.concat([pd.read_csv(f) for f in rank_files], ignore_index=True)

    key = [c for c in ["protocol", "eps", "delay", "d_prime", "basis_method"]
           if c in server.columns]
    comm = (allranks.groupby(key, dropna=False)["comm_size"].sum()
            .reset_index().rename(columns={"comm_size": "comm_bytes"}))
    return server.merge(comm, on=key, how="left")


def collect(folders):
    """Tidy table: one row per (protocol, n_clients, dataset, delay, eps, d')."""
    recs = []
    for folder in folders:
        base = Path(folder)
        if not base.is_dir():
            continue
        for tdir in sorted(base.glob("timing_*")):
            m = re.match(r"timing_(\d+)$", tdir.name)
            if not m:
                continue
            n_clients = int(m.group(1))
            for ds in sorted(p for p in tdir.iterdir() if p.is_dir()):
                if not (ds / "variances_0.csv").exists():
                    continue
                df = _load_dataset_dir(ds)
                for _, row in df.iterrows():
                    proto = row["protocol"]
                    if proto == "mpi_proto":
                        label, d_prime = "FastLloyd", None
                    elif proto == "mpi_lsh_proto":
                        b = row.get("basis_method", "")
                        label = f"LSH-{BASIS_SHORT.get(b, b)}"
                        d_prime = int(row["d_prime"])
                    else:
                        continue
                    recs.append({
                        "n_clients": n_clients,
                        "dataset": ds.name,
                        "label": label,
                        "d_prime": d_prime,
                        "eps": float(row["eps"]),
                        "delay": float(row["delay"]),
                        "elapsed_ms": float(row["elapsed"]) * 1000,
                        "elapsed_h_ms": float(row.get("elapsed_h", 0)) * 1000,
                        "comm_bytes": float(row["comm_bytes"]),
                        "rounds": float(row["num_comm_rounds"]),
                    })
    return pd.DataFrame(recs)


def _rows_for(g, value_col, ci_col):
    """List of dicts the grid plotter consumes (one per row of g)."""
    rows = []
    for _, r in g.iterrows():
        d_prime = None if pd.isna(r["d_prime"]) else int(r["d_prime"])
        rows.append({
            "label": r["label"], "d_prime": d_prime, "eps": float(r["eps"]),
            "n_clients": int(r["n_clients"]), "value": float(r[value_col]),
            "ci": float(r[ci_col]) if ci_col else None,
        })
    return rows


# ---------------------------------------------------------------------------
# Grid plot: rows = d', columns = epsilon, x = #clients, y = metric
# ---------------------------------------------------------------------------

def _plot_grid(rows, ylabel, d_primes, epss, title, out_path):
    nrows, ncols = len(d_primes), len(epss)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(max(4.0, ncols * 3.0), max(3.0, nrows * 2.6)),
                             squeeze=False, sharex=True, sharey=True)
    clients = sorted({r["n_clients"] for r in rows})

    for i, d_prime in enumerate(d_primes):
        for j, eps in enumerate(epss):
            ax = axes[i][j]
            # FastLloyd (d_prime is None) is shown in every row at its eps column.
            cell = [r for r in rows if r["eps"] == eps
                    and (r["d_prime"] is None or r["d_prime"] == d_prime)]

            methods = []
            for r in cell:
                if r["label"] not in methods:
                    methods.append(r["label"])
            methods.sort(key=lambda m: METHOD_ORDER.index(m)
                         if m in METHOD_ORDER else len(METHOD_ORDER))

            for label in methods:
                series = sorted([r for r in cell if r["label"] == label],
                                key=lambda r: r["n_clients"])
                if not series:
                    continue
                xs = np.array([r["n_clients"] for r in series], dtype=float)
                ys = np.array([r["value"] for r in series], dtype=float)
                es = np.array([r["ci"] or 0 for r in series], dtype=float)
                is_baseline = series[0]["d_prime"] is None
                ax.errorbar(xs, ys, yerr=es if np.any(es > 0) else None,
                            color=_label_color(label),
                            linestyle="--" if is_baseline else "-",
                            marker="o", markersize=4, capsize=3, linewidth=1.6,
                            label=label)

            ax.set_xscale("log", base=2)
            if clients:
                ax.set_xticks(clients)
                ax.set_xticklabels([str(c) for c in clients])
                ax.minorticks_off()
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"ε = {eps:g}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"d' = {d_prime}\n{ylabel}", fontsize=10, labelpad=8)
            if i == nrows - 1:
                ax.set_xlabel("clients", fontsize=10)

    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                labels.append(l)
                handles.append(h)
    if handles:
        fig.legend(handles, labels, fontsize=9, title="Method",
                   loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = ArgumentParser(description="Compare LSH vs FastLloyd timing (d' x eps grid, x=clients)")
    ap.add_argument("folders", nargs="+", help="results folders to scan")
    ap.add_argument("--out", default="timing_compare", help="output folder")
    args = ap.parse_args()

    df = collect(args.folders)
    if df.empty:
        print("No timing data found in:", args.folders)
        return 1

    os.makedirs(args.out, exist_ok=True)
    df.sort_values(["dataset", "label", "d_prime", "eps", "n_clients", "delay"]).to_csv(
        os.path.join(args.out, "timing_compare.csv"), index=False)

    for dataset, g in df.groupby("dataset"):
        d_primes = sorted(int(x) for x in g["d_prime"].dropna().unique())
        epss = sorted(g["eps"].unique())
        if not d_primes or not epss:
            print(f"  {dataset}: missing LSH d' or eps sweep, skipping")
            continue
        out_ds = os.path.join(args.out, dataset)
        os.makedirs(out_ds, exist_ok=True)

        for metric, (ylabel, ci_col, delay_dep) in METRICS.items():
            if delay_dep:
                for delay, gd in g.groupby("delay"):
                    _plot_grid(_rows_for(gd, metric, ci_col), ylabel, d_primes, epss,
                               f"{dataset}  —  {ylabel}  (delay={delay}s)",
                               os.path.join(out_ds, f"{metric}_delay{delay}.pdf"))
            else:
                gd = g[np.isclose(g["delay"], g["delay"].min())]
                _plot_grid(_rows_for(gd, metric, ci_col), ylabel, d_primes, epss,
                           f"{dataset}  —  {ylabel}",
                           os.path.join(out_ds, f"{metric}.pdf"))

        n_clients = sorted(g["n_clients"].unique())
        print(f"  {dataset}: d'={d_primes}, eps={epss}, clients={n_clients}")

    print(f"\nWrote grid plots + timing_compare.csv to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
