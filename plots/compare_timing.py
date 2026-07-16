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

# SVD is the non-private oracle -- excluded from the timing story (its basis is free but
# undefined as a private method). We keep FastLloyd (baseline) and the two end-to-end
# private LSH bases, split into "Clustering" (basis-free) and "+ basis" lines.
BASIS_SHORT = {"random": "Rand", "dpsgd_pca": "DP-SGD"}

# Persistent palette shared with plots/accuracy_grid.py:
#   LSH-DP-SGD = green solid, LSH-Rand = red solid, FastLloyd = black dotted,
#   LSH-PCA = purple dashed (accuracy only). "Clustering" is timing-only (the basis-free
#   LSH rounds) -> its own orange dashed so it doesn't collide with the shared four.
METHOD_ORDER = ["FastLloyd", "Clustering", "LSH-Rand", "LSH-DP-SGD"]
COLORS = {
    "FastLloyd":   "black",
    "Clustering":  "darkorange",
    "LSH-Rand":    "tab:red",
    "LSH-DP-SGD":  "green",
    "LSH-PCA":     "purple",
}
LINESTYLE = {
    "FastLloyd":   ":",
    "Clustering":  "--",
    "LSH-Rand":    "-",
    "LSH-DP-SGD":  "-",
    "LSH-PCA":     "--",
}

# Timing metrics -> (y-axis label, CI column or None, delay-dependent?, log-y?)
# Wall-time gets a log y-axis: the dpsgd sigma-calibration offset is orders of magnitude
# above the clustering, so a linear axis would flatten every other line to zero.
METRICS = {
    "elapsed_ms": ("Wall-time (ms)", "elapsed_h_ms", True, True),
    "comm_bytes": ("Communication (bytes)", None, False, False),
    "rounds":     ("Communication rounds", None, False, False),
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
    """Tidy table: one row per (protocol, n_clients, dataset, delay, eps, d', basis).

    Carries the phase split for LSH rows: cluster_ms (clustering rounds), basis_build_ms
    (basis construction + basis comm round), basis_calib_ms (dpsgd sigma calibration).
    Older CSVs without these columns fall back to the total `elapsed` for cluster_ms and 0
    for the basis parts, so the plot still renders.
    """
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
                    elapsed_ms = float(row["elapsed"]) * 1000
                    if proto == "mpi_proto":
                        basis, d_prime = None, None
                    elif proto == "mpi_lsh_proto":
                        basis = row.get("basis_method", "")
                        if basis not in BASIS_SHORT:      # drop svd_pca (non-private oracle)
                            continue
                        d_prime = int(row["d_prime"])
                    else:
                        continue
                    # phase split (fall back to total for legacy CSVs)
                    cluster_ms = float(row.get("cluster_ms", elapsed_ms) or elapsed_ms)
                    recs.append({
                        "n_clients": n_clients,
                        "dataset": ds.name,
                        "proto": proto,
                        "basis": basis,
                        "d_prime": d_prime,
                        "eps": float(row["eps"]),
                        "delay": float(row["delay"]),
                        "elapsed_ms": elapsed_ms,
                        "elapsed_h_ms": float(row.get("elapsed_h", 0)) * 1000,
                        "cluster_ms": cluster_ms,
                        "cluster_h_ms": float(row.get("cluster_ms_h", 0) or 0),
                        "basis_build_ms": float(row.get("basis_build_ms", 0) or 0),
                        "basis_calib_ms": float(row.get("basis_calib_ms", 0) or 0),
                        "comm_bytes": float(row["comm_bytes"]),
                        "rounds": float(row["num_comm_rounds"]),
                    })
    return pd.DataFrame(recs)


def _label_for(rec):
    """Legend label for a raw record (FastLloyd baseline or one of the private LSH bases)."""
    if rec["proto"] == "mpi_proto":
        return "FastLloyd"
    return f"LSH-{BASIS_SHORT[rec['basis']]}"


def _rows_for(g, value_col, ci_col):
    """Pass-through rows (comm_bytes / rounds): one line per method, SVD already dropped."""
    rows = []
    for _, r in g.iterrows():
        d_prime = None if pd.isna(r["d_prime"]) else int(r["d_prime"])
        rows.append({
            "label": _label_for(r), "d_prime": d_prime, "eps": float(r["eps"]),
            "n_clients": int(r["n_clients"]), "value": float(r[value_col]),
            "ci": float(r[ci_col]) if ci_col else None,
        })
    return rows


def _walltime_rows(g):
    """Derive the three wall-time lines (+ FastLloyd baseline) the grid plotter consumes:

      - FastLloyd    : baseline total wall-time (d'=None, shown in every d' row).
      - Clustering   : the basis-free LSH clustering rounds (mean over the private bases).
      - LSH-Rand     : Clustering + random basis build (basis ~ free).
      - LSH-DP-SGD   : Clustering + dpsgd basis build + sigma calibration.

    Values are in ms; the wall-time PDF uses a log y-axis so the (large) dpsgd calibration
    offset doesn't flatten the other lines.
    """
    rows = []
    # FastLloyd baseline (one line, repeated across d' rows by the plotter).
    for _, r in g[g["proto"] == "mpi_proto"].iterrows():
        rows.append({"label": "FastLloyd", "d_prime": None, "eps": float(r["eps"]),
                     "n_clients": int(r["n_clients"]),
                     "value": float(r["elapsed_ms"]), "ci": float(r["elapsed_h_ms"])})

    lsh = g[g["proto"] == "mpi_lsh_proto"]
    for (dp, eps, n), cell in lsh.groupby(["d_prime", "eps", "n_clients"]):
        by_basis = {row["basis"]: row for _, row in cell.iterrows()}
        # Clustering: basis-free, so average whatever private bases are present (they share
        # the same clustering mechanism; small differences are basis-induced tree changes).
        clus = float(cell["cluster_ms"].mean())
        clus_ci = float(np.sqrt((cell["cluster_h_ms"] ** 2).mean()))
        rows.append({"label": "Clustering", "d_prime": int(dp), "eps": float(eps),
                     "n_clients": int(n), "value": clus, "ci": clus_ci})
        for basis, label in (("random", "LSH-Rand"), ("dpsgd_pca", "LSH-DP-SGD")):
            if basis not in by_basis:
                continue
            r = by_basis[basis]
            value = float(r["cluster_ms"] + r["basis_build_ms"] + r["basis_calib_ms"])
            ci = float(r["cluster_h_ms"])   # dominant uncertainty is the clustering timing
            rows.append({"label": label, "d_prime": int(dp), "eps": float(eps),
                         "n_clients": int(n), "value": value, "ci": ci})
    return rows


# ---------------------------------------------------------------------------
# Grid plot: rows = d', columns = epsilon, x = #clients, y = metric
# ---------------------------------------------------------------------------

def _plot_grid(rows, ylabel, d_primes, epss, title, out_path, logy=False):
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
                ax.errorbar(xs, ys, yerr=es if np.any(es > 0) else None,
                            color=_label_color(label),
                            linestyle=LINESTYLE.get(label, "-"),
                            marker="o", markersize=4, capsize=3, linewidth=1.6,
                            label=label)

            ax.set_xscale("log", base=2)
            if logy:
                ax.set_yscale("log")
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
    ap.add_argument("folders", nargs="*", help="results folders to scan")
    ap.add_argument("--out", default="timing_compare", help="output folder")
    ap.add_argument("--from-csv", default=None,
                    help="replot from an existing timing_compare.csv instead of re-scanning "
                         "the results folders")
    ap.add_argument("--max-clients", type=int, default=None,
                    help="only include rows with n_clients <= this (e.g. 4 to drop n=8)")
    args = ap.parse_args()

    if args.from_csv:
        df = pd.read_csv(args.from_csv)
    else:
        df = collect(args.folders)
    if df.empty:
        print("No timing data found in:", args.from_csv or args.folders)
        return 1
    if args.max_clients is not None:
        df = df[df["n_clients"] <= args.max_clients].copy()

    os.makedirs(args.out, exist_ok=True)
    # Dump the raw per-phase table (cluster_ms / basis_build_ms / basis_calib_ms) so both
    # decompositions -- clustering-only and clustering+basis, with or without the sigma
    # calibration -- can be replotted without re-running the experiments.
    if "label" not in df.columns:
        df["label"] = df.apply(_label_for, axis=1)
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

        for metric, (ylabel, ci_col, delay_dep, logy) in METRICS.items():
            is_walltime = metric == "elapsed_ms"
            build_rows = (lambda gd: _walltime_rows(gd)) if is_walltime \
                else (lambda gd, m=metric, c=ci_col: _rows_for(gd, m, c))
            if delay_dep:
                for delay, gd in g.groupby("delay"):
                    _plot_grid(build_rows(gd), ylabel, d_primes, epss,
                               f"{dataset}  —  {ylabel}  (delay={delay}s)",
                               os.path.join(out_ds, f"{metric}_delay{delay}.pdf"), logy=logy)
            else:
                gd = g[np.isclose(g["delay"], g["delay"].min())]
                _plot_grid(build_rows(gd), ylabel, d_primes, epss,
                           f"{dataset}  —  {ylabel}",
                           os.path.join(out_ds, f"{metric}.pdf"), logy=logy)

        n_clients = sorted(g["n_clients"].unique())
        print(f"  {dataset}: d'={d_primes}, eps={epss}, clients={n_clients}")

    print(f"\nWrote grid plots + timing_compare.csv to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
