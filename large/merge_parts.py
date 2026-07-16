"""Merge the per-d' LSH result "parts" into the canonical results layout.

LSH jobs are split per d' (see jobs.sh) and each writes to its own
RESULT_FOLDER/parts/<id> (accuracy) or RESULT_FOLDER/timing/parts/<id> (timing)
to avoid concurrent writes to the same CSV. This script concatenates the rows of
those per-d' CSVs into the canonical files the plots expect:

    parts/*/accuracy/<ds>/variances_lsh.csv
        -> <root>/accuracy/<ds>/variances_lsh.csv
    timing/parts/*/timing_<n>/<ds>/variances_<rank>.csv
        -> <root>/timing/lsh/timing_<n>/<ds>/variances_<rank>.csv

Idempotent: re-running rebuilds the canonical files from whatever parts exist.

Usage:  python large/merge_parts.py [results_root]   (default: large)
"""

import collections
import glob
import os
import sys

import pandas as pd


def _concat_into(dest, srcs):
    dfs = []
    for s in sorted(srcs):
        df = pd.read_csv(s)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # drop old index col
        dfs.append(df)
    if not dfs:
        return
    merged = pd.concat(dfs, ignore_index=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    merged.to_csv(dest)   # write with a fresh index, like experiments.py
    print(f"  {dest}  <- {len(srcs)} part(s), {len(merged)} rows")


def main(root="large"):
    # --- accuracy: parts/*/accuracy/<ds>/variances_lsh.csv ---
    acc = collections.defaultdict(list)
    for f in glob.glob(os.path.join(root, "parts", "*", "accuracy", "*", "variances_lsh.csv")):
        ds = f.split(os.sep + "accuracy" + os.sep)[1].split(os.sep)[0]
        acc[ds].append(f)
    for ds, srcs in acc.items():
        _concat_into(os.path.join(root, "accuracy", ds, "variances_lsh.csv"), srcs)

    # --- timing: timing/parts/*/timing_<n>/<ds>/variances_<rank>.csv ---
    tim = collections.defaultdict(list)
    base = os.path.join(root, "timing", "parts") + os.sep
    for f in glob.glob(os.path.join(root, "timing", "parts", "*", "timing_*", "*", "variances_*.csv")):
        rel = f.split(base, 1)[1].split(os.sep, 1)[1]  # timing_<n>/<ds>/variances_<rank>.csv
        tim[rel].append(f)
    for rel, srcs in tim.items():
        _concat_into(os.path.join(root, "timing", "lsh", rel), srcs)

    if not acc and not tim:
        print("  (no LSH parts found to merge)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "large")
