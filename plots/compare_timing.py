"""
Compare federated LSH vs FastLloyd scalability from MPI timing runs.

Reads `timing_<n_clients>/<dataset>/variances_<rank>.csv` produced by
`experiments.py --exp_type timing` from one or more results folders (e.g.
`submission/` for FastLloyd and `submission_lsh/` for LSH), and plots total
wall-time and total communication vs the number of clients, one line per
protocol. Protocol is read from the CSV `protocol` column, so folders may be
passed in any order.

Usage:
    python -m plots.compare_timing submission submission_lsh
    python -m plots.compare_timing folderA folderB --out timing_compare
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
plt.rcParams.update({'font.size': 12})

PROTO_LABEL = {"mpi_proto": "FastLloyd", "mpi_lsh_proto": "LSH"}
PROTO_COLOR = {"FastLloyd": "green", "LSH": "mediumpurple"}


def collect(folders):
    """Tidy table: one row per (protocol, n_clients, dataset, delay)."""
    rows = []
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
                server = ds / "variances_0.csv"
                if not server.exists():
                    continue
                sdf = pd.read_csv(server)
                rank_files = list(ds.glob("variances_*.csv"))
                for delay, g in sdf.groupby("delay"):
                    proto = g["protocol"].iloc[0]
                    # total bytes on the wire = sum over all ranks at this delay
                    total_comm = 0
                    for f in rank_files:
                        df = pd.read_csv(f)
                        total_comm += df[df["delay"] == delay]["comm_size"].sum()
                    rows.append({
                        "protocol": PROTO_LABEL.get(proto, proto),
                        "n_clients": n_clients,
                        "dataset": ds.name,
                        "delay": delay,
                        "elapsed_ms": g["elapsed"].mean() * 1000,
                        "elapsed_h_ms": g["elapsed_h"].mean() * 1000,
                        "rounds": g["num_comm_rounds"].mean(),
                        "comm_bytes": int(total_comm),
                    })
    return pd.DataFrame(rows)


def _plot(df, dataset, delay, ycol, yerr_col, ylabel, out_path):
    sub = df[(df.dataset == dataset) & (np.isclose(df.delay, delay))]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for proto, g in sub.groupby("protocol"):
        g = g.sort_values("n_clients")
        yerr = g[yerr_col] if yerr_col else None
        ax.errorbar(g["n_clients"], g[ycol], yerr=yerr, marker="o", capsize=3,
                    label=proto, color=PROTO_COLOR.get(proto, "gray"), linewidth=1.8)
    ax.set_xlabel("number of clients")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset}  (delay={delay}s)")
    ax.set_xticks(sorted(sub["n_clients"].unique()))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = ArgumentParser(description="Compare LSH vs FastLloyd timing/scalability")
    ap.add_argument("folders", nargs="+", help="results folders to scan")
    ap.add_argument("--out", default="timing_compare", help="output folder")
    args = ap.parse_args()

    df = collect(args.folders)
    if df.empty:
        print("No timing data found in:", args.folders)
        return 1

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "timing_compare.csv")
    df.sort_values(["dataset", "delay", "protocol", "n_clients"]).to_csv(
        csv_path, index=False)

    for dataset in df.dataset.unique():
        for delay in sorted(df.delay.unique()):
            tag = f"{dataset}_d{delay}"
            _plot(df, dataset, delay, "elapsed_ms", "elapsed_h_ms",
                  "wall-time (ms)", os.path.join(args.out, f"time_{tag}.pdf"))
            _plot(df, dataset, delay, "comm_bytes", None,
                  "total communication (bytes)",
                  os.path.join(args.out, f"comm_{tag}.pdf"))

    print(f"Wrote {csv_path} and PDFs to {args.out}/\n")
    print(df.sort_values(["dataset", "delay", "protocol", "n_clients"])
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
