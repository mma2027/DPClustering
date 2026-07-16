"""Runtime scaling: wall-time vs #clients, one panel per dataset, at the knee d'.

FastLloyd wall = elapsed; LSH-* wall = sigma calibration + matrix build + clustering.
Averaged over epsilon, LAN delay. Only client counts with complete data for all three
methods are plotted (all datasets now sweep n in {2,4,8}).

Persistent palette (shared with accuracy_grid.py / compare_timing.py):
  LSH-DP-SGD green solid, LSH-Rand red solid, FastLloyd black dotted.

Usage: python -m plots.runtime_scaling [timing_compare.csv] [--out out.pdf]
"""
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

KNEE = {"mnist784": 10, "glove100": 12, "glove300": 12}
# label -> (color, linestyle, marker)
STYLE = {
    "LSH-DP-SGD": ("green",   "-", "s"),
    "LSH-Rand":   ("tab:red", "-", "^"),
    "FastLloyd":  ("black",   ":", "o"),
}
ORDER = ["LSH-DP-SGD", "LSH-Rand", "FastLloyd"]


def wall_by_clients(df, dataset, label):
    """{n_clients: mean wall-time (s)} at the knee d', LAN, averaged over eps."""
    lan = df["delay"] < 1e-3
    if label == "FastLloyd":
        g = df[lan & (df["proto"] == "mpi_proto") & (df["dataset"] == dataset)].copy()
        g["wall"] = g["elapsed_ms"] / 1000.0
    else:
        basis = "dpsgd_pca" if label == "LSH-DP-SGD" else "random"
        g = df[lan & (df["proto"] == "mpi_lsh_proto") & (df["dataset"] == dataset)
               & (df["basis"] == basis) & (df["d_prime"] == KNEE[dataset])].copy()
        g["wall"] = (g["basis_calib_ms"] + g["basis_build_ms"] + g["cluster_ms"]) / 1000.0
    return g.groupby("n_clients")["wall"].mean().to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", nargs="?",
                    default="large/timing/timing_compare/timing_compare.csv")
    ap.add_argument("--out", default="large/timing/runtime_scaling.pdf")
    ap.add_argument("--datasets", nargs="+",
                    default=["mnist784", "glove100", "glove300"])
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    for c in ["delay", "elapsed_ms", "basis_calib_ms", "basis_build_ms",
              "cluster_ms", "d_prime", "n_clients"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    datasets = [d for d in args.datasets if d in set(df["dataset"])]
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(3.7 * len(datasets), 3.4), squeeze=False)

    for j, ds in enumerate(datasets):
        ax = axes[0][j]
        walls = {lbl: wall_by_clients(df, ds, lbl) for lbl in ORDER}
        # clients complete for ALL methods (currently n in {2,4,8} for every dataset)
        clients = sorted(set.intersection(*[set(w) for w in walls.values()]))
        for lbl in ORDER:
            color, ls, mk = STYLE[lbl]
            ys = [walls[lbl][n] for n in clients]
            ax.plot(clients, ys, color=color, ls=ls, marker=mk, ms=6,
                    lw=2.2 if lbl != "LSH-Rand" else 1.8, label=lbl)
        ax.set_title(ds, fontsize=14, fontweight="bold")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(clients)
        ax.set_xticklabels([str(n) for n in clients])
        ax.minorticks_off()
        ax.set_xlabel("clients", fontsize=13)
        if j == 0:
            ax.set_ylabel("wall-time (s)", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(True, which="both", alpha=0.25, lw=0.6)

    handles = [plt.Line2D([0], [0], color=STYLE[l][0], ls=STYLE[l][1],
                          marker=STYLE[l][2], ms=6, lw=2.2, label=l) for l in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(ORDER),
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
