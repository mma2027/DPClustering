"""Generate figures for the DP-SGD PCA investigation from the step CSVs."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import RESULTS_DIR

DS = ["mnist784", "glove100"]


def _csv(name):
    return pd.read_csv(os.path.join(RESULTS_DIR, name))


def plot_dprime():
    df = _csv("step3_dprime.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name in zip(axes, DS):
        d = df[df.dataset == name].sort_values("d_prime")
        ax.plot(d.d_prime, d.evr_pca, "o-", color="seagreen", label="PCA (optimal)")
        ax.plot(d.d_prime, d.evr, "s-", color="mediumpurple", label="DP-SGD PCA (ε=0.25)")
        ax.plot(d.d_prime, d.evr_random, "^--", color="steelblue", label="random (≈d'/d)")
        ax2 = ax.twinx()
        ax2.plot(d.d_prime, d.captured_gain, "x:", color="darkorange", label="captured gain")
        ax2.set_ylabel("captured gain (frac of PCA−random gap)", color="darkorange")
        ax2.set_ylim(0, 1)
        ax.set_title(f"{name}: EVR vs d'  (ε=0.25 on basis)")
        ax.set_xlabel("d'"); ax.set_ylabel("explained variance ratio")
        ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_dprime.png"), dpi=130)
    plt.close(fig)


def plot_sweep(csv, xcol, title, fname, logx=False):
    df = _csv(csv)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name in zip(axes, DS):
        d = df[df.dataset == name].sort_values(xcol)
        ax.errorbar(d[xcol], d.evr, yerr=d.evr_std, marker="o", color="mediumpurple",
                    capsize=3, label="DP-SGD PCA")
        ax.axhline(d.evr_random.iloc[0], color="steelblue", ls="--", label="random")
        ax.axhline(d.evr_pca.iloc[0], color="seagreen", ls=":", label="PCA optimal")
        if logx:
            ax.set_xscale("log")
        ax.set_title(f"{name}: {title}")
        ax.set_xlabel(xcol); ax.set_ylabel("EVR (d'=10)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    plot_dprime()
    plot_sweep("step3_clip.csv", "clip", "EVR vs clip_norm (ε=0.25)", "fig_clip.png", logx=True)
    plot_sweep("step3_epochs.csv", "epochs", "EVR vs epochs (=#steps; ε=0.25)", "fig_epochs.png")
    plot_sweep("step3_batch.csv", "batch", "EVR vs batch size (ε=0.25)", "fig_batch.png", logx=True)
    print("Wrote fig_dprime.png, fig_clip.png, fig_epochs.png, fig_batch.png to results/")
