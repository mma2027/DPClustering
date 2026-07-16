"""Combined accuracy grid: datasets by column, d' by row, NICV vs epsilon.

One figure for all large datasets. Each sub-plot autoscales its own y-axis (a per-cell
"zoom") so the LSH-DP-SGD vs FastLloyd gap is visible even where the absolute NICV values
are close. FastLloyd is d'-independent, so it is repeated (as the fixed baseline) in every
row of its column.

Highlighted pair (bold, solid): FastLloyd and LSH-DP-SGD.
Muted references (thin, dashed/dotted): LSH-Rand and LSH-PCA (non-private).

Usage:
    python -m plots.accuracy_grid [summary.csv] [--out out.pdf] [--datasets mnist784 glove100]
"""
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

NICV = "Normalized Intra-cluster Variance (NICV)"

# Fixed categorical order + style. Identity is never color-alone: each series also has a
# distinct linestyle and marker. The two entities we want to compare are the bold solid
# lines; the two reference series are thin and dashed/dotted.
# Persistent palette shared with plots/compare_timing.py:
#   LSH-DP-SGD = green solid, LSH-Rand = red solid, FastLloyd = black dotted,
#   LSH-PCA = purple dashed.
STYLE = {
    # label        color       ls     lw   marker  ms  z    band
    "LSH-DP-SGD":  ("green",   "-",  2.4, "s",    6,  5,  True),
    "LSH-Rand":    ("tab:red", "-",  1.8, "^",    5,  3,  False),
    "FastLloyd":   ("black",   ":",  2.0, "o",    5,  4,  True),
    "LSH-PCA":     ("purple",  "--", 1.8, "v",    5,  3,  False),
}
ORDER = ["LSH-DP-SGD", "LSH-Rand", "FastLloyd", "LSH-PCA"]
# csv label -> plotted label (LSH-SVD is the non-private PCA basis)
RENAME = {"LSH-SVD": "LSH-PCA"}

DPRIME_FS, EPS_FS, COLTITLE_FS, LEG_FS, TICK_FS = 15, 14, 16, 13, 11


def load(path):
    df = pd.read_csv(path)
    df = df[df["label"] != "label"].copy()             # drop any stray repeated header
    df["label"] = df["label"].replace(RENAME)
    df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    df[NICV] = pd.to_numeric(df[NICV], errors="coerce")
    df[NICV + "_h"] = pd.to_numeric(df[NICV + "_h"], errors="coerce").fillna(0.0)
    df["d_prime"] = pd.to_numeric(df["d_prime"], errors="coerce")
    return df


def series(df, dataset, label, d_prime=None):
    """(eps, nicv, ci) sorted by eps for a method; FastLloyd ignores d_prime."""
    g = df[(df["dataset"] == dataset) & (df["label"] == label)]
    if label != "FastLloyd":
        g = g[g["d_prime"] == d_prime]
    g = g.dropna(subset=["eps", NICV]).sort_values("eps")
    return g["eps"].to_numpy(), g[NICV].to_numpy(), g[NICV + "_h"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", nargs="?",
                    default="large/accuracy/comparison_summary.csv")
    ap.add_argument("--out", default="large/accuracy/accuracy_grid.pdf")
    ap.add_argument("--datasets", nargs="+",
                    default=["mnist784", "glove100", "glove300"])
    args = ap.parse_args()

    df = load(args.summary)
    datasets = [d for d in args.datasets if d in set(df["dataset"])]
    # d' rows per dataset (from the private LSH-DP-SGD sweep)
    dprimes = {ds: sorted(int(x) for x in
                          df[(df["dataset"] == ds) & (df["label"] == "LSH-DP-SGD")]
                          ["d_prime"].dropna().unique())
               for ds in datasets}
    nrows = max(len(v) for v in dprimes.values())
    ncols = len(datasets)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 2.5 * nrows),
                             sharex="col", squeeze=False)

    for c, ds in enumerate(datasets):
        axes[0][c].set_title(ds, fontsize=COLTITLE_FS, fontweight="bold", pad=8)
        dps = dprimes[ds]
        for r in range(nrows):
            ax = axes[r][c]
            if r >= len(dps):
                ax.axis("off")
                continue
            dp = dps[r]
            for label in ORDER:
                color, ls, lw, mk, ms, z, band = STYLE[label]
                x, y, ci = series(df, ds, label, dp)
                if len(x) == 0:
                    continue
                ax.plot(x, y, color=color, ls=ls, lw=lw, marker=mk, ms=ms,
                        zorder=z, label=label)
                if band and ci.any():
                    ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.15,
                                    lw=0, zorder=z - 1)
            # per-cell zoom: autoscale y to this cell's data with a small margin
            ax.margins(y=0.12)
            ax.set_xscale("log", base=2)
            ax.set_xticks([0.5, 1, 2, 4])
            ax.set_xticklabels(["0.5", "1", "2", "4"])
            ax.tick_params(labelsize=TICK_FS)
            ax.grid(True, which="major", alpha=0.25, lw=0.6)
            # d' label (big), per cell since d' differs per dataset
            ax.text(0.04, 0.93, f"$d'={dp}$", transform=ax.transAxes,
                    fontsize=DPRIME_FS, fontweight="bold", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
            if r == len(dps) - 1:
                ax.set_xlabel(r"$\varepsilon$ (privacy budget)", fontsize=EPS_FS)

    # one horizontal legend at the bottom
    handles = [plt.Line2D([0], [0], color=STYLE[l][0], ls=STYLE[l][1],
                          lw=STYLE[l][2], marker=STYLE[l][3], ms=STYLE[l][4], label=l)
               for l in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(ORDER),
               fontsize=LEG_FS, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("NICV (lower is better)", fontsize=COLTITLE_FS + 1, y=0.995)
    fig.tight_layout(rect=[0, 0.035, 1, 0.98])
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
