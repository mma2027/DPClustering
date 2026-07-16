# Large-dataset experiments: LSH vs FastLloyd (headline)

Accuracy **and** scalability of the DP LSH prefix-tree clustering against the
FastLloyd baselines, on **large, dense, high-dimensional** datasets. This is the
**headline** group — the regime the method targets — where LSH is expected to
**win** on both axes:

- **Accuracy under DP:** LSH's leaf-sum / count queries have L2 sensitivity 1 on
  unit-norm data, so centroid noise is **independent of `d`**, whereas
  FastLloyd's scales with `√d`. At `d = 784` / `d = 100` this should let LSH
  match or beat FastLloyd at a given ε.
- **Scalability:** fixed, small number of communication rounds (~6 random / ~7
  PCA) vs FastLloyd's `≈ 2 × iters`, with `d`-light count rounds.

Reproduce everything with:

```bash
bash large/run_local.sh
```

## Prerequisite — download the data (opt-in, large)

These datasets are **not** created by the default `download_data.py` run:

```bash
python scripts/download_data.py --only mnist784 glove100
# glove100 downloads glove.6B.zip (~822 MB); to use a local copy instead:
GLOVE_TXT=/path/glove.6B.100d.txt python scripts/download_data.py --only glove100
# GLOVE_MAX_ROWS=100000 caps GloVe vectors for a lighter run
```

`large/run_local.sh` aborts with a hint if `data/<dataset>.txt` is missing.

## Threat model, baseline fairness & accounting (read before interpreting results)

These are the deliberate modelling choices behind the headline. State them in the
write-up so the comparison is read correctly.

- **Data precondition (unit norm).** Every point is projected onto the unit L2 ball
  during **data preparation** (`scripts/download_data.py`: per-feature min-max →
  `ensure_unit_norm`), so each contributed point has L2 sensitivity ≤ 1. This is the
  precondition `compute_dp_sigmas_zcdp` assumes; it runs **outside** the timed region.

- **FastLloyd baseline uses √d sum-sensitivity — by design, as published.** FastLloyd
  assumes data in the box `[-1,1]^d`, so its per-iteration sum query has L2 sensitivity
  `√d` (`parties/server.py`: `sum_sensitivity = √d`, or `2√d` constrained). On unit-norm
  data the *true* contribution norm is ≤ 1, so FastLloyd over-noises by ≈ `√d`. **This is
  exactly the gap the method exploits** (cosine/SimHash gives `d`-independent sensitivity)
  and the comparison is "FastLloyd **as published**." Caveat to disclose: a unit-norm-aware
  FastLloyd (clip to the unit ball, sensitivity ≈ 1) would narrow this gap; we do **not**
  claim FastLloyd is fundamentally `√d`-bound, only that the standard coordinate-wise DP
  k-means is. (FastLloyd's centroid init is also over the `[-1,1]^d` box, mismatched to
  the unit sphere in high `d` — this disfavours FastLloyd, i.e. is conservative for us.)

- **LSH-SVD is a NON-PRIVATE oracle.** `svd_pca` builds the basis from the exact data
  PCA with **zero** basis budget — it has *no* DP guarantee for the basis and is an
  upper-bound reference only. The plots label it "(non-private oracle)"; never read it as
  a DP method at the plotted ε. Only **LSH-Rand** (data-independent) and **LSH-DP-SGD**
  (DP basis) are end-to-end private.

- **DP-SGD-PCA timing is reported as three separate phases.** The protocol times, on rank 0,
  `basis_calib_ms` (the autodp σ search), `basis_build_ms` (the SGD loop / eigh / random
  draw), and `cluster_ms` (the count + sum rounds). The σ search is a pure function of public
  parameters — data-independent and precomputable from a table — but **empirically it is the
  dominant cost by far** (tens to thousands of seconds, and it *grows as ε shrinks*, since a
  smaller basis budget needs a more expensive moments-accountant search), whereas the SGD
  loop and the clustering rounds are only seconds. So it is **not** a negligible fixed offset:
  the timing plots show it as its own component (`Clustering` vs `LSH-DP-SGD = clustering +
  basis`), and `timing_compare.csv` keeps all three columns so the write-up can present the
  scalability comparison either including or excluding this one-time setup. The σ duration is
  memoized alongside its value, so it is measured **consistently across the network-delay
  sweep** (a previous artifact attributed the whole calibration to whichever delay ran first).
  It affects only the additive DP-SGD offset — rounds, bytes, and the clustering slope are
  unaffected.

- **Accuracy and timing measure one mechanism.** The accuracy path (`lsh_proto`) and the
  timing path (`mpi_lsh_proto`) build the DP-SGD-PCA basis with the **identical** mechanism
  — the same capped Poisson sub-sample and the same full-dataset amplified accounting
  (`full_n`, Poisson rate γ). So the accuracy and scalability claims refer to the same
  algorithm.

## Datasets, defaults, and chosen `d'`

`k` is the repo default (`configs/defaults.py: num_clusters`). It is set **large** —
comparable to the number of leaves the LSH tree yields across its `d'` sweep — so the
FastLloyd baseline is exercised in the large-`k` regime (FastLloyd's compute/comm grow
linearly in `k`; LSH's are `k`-independent). `k` only affects the FastLloyd lines; LSH's
cluster count is the (data-dependent) number of surviving leaves, ≤ `2^d'`.

| Dataset | n | d | k | `--d_primes` | LSH leaves (min_count=50) | Notes |
|---------|---------|-----|------|----------------|---------------------------|-------|
| mnist784 | 70,000  | 784 | 500  | `5 10 15 20`   | ~25–662 | full MNIST images; dense, very high-d |
| glove100 | ~400,000| 100 | 4000 | `8 12 16 20`   | ~256–4616 | GloVe embeddings; cosine-natural (no label ground truth) |

**Interpreting NICV across `d'` (important).** LSH's leaf count grows with `d'` and reaches
`k` only at the upper end of the sweep (mnist784 ≈ `d'` 9–15, glove100 ≈ `d'` 12–16). At
small `d'` LSH produces far fewer clusters than FastLloyd's `k`, so its NICV is
mechanically higher there — **not** a privacy/noise effect. Read the LSH-vs-FastLloyd NICV
comparison at the `d'` where the LSH leaf count ≈ `k`; the smaller-`d'` rows show the
basis/dimension axis, not a matched-cluster comparison.

## Fixed parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| ε (privacy budget) | `0.5 1 2 4` | accuracy sweep; passed via `--eps_budgets` for timing |
| LSH bases | `random svd_pca dpsgd_pca` | one LSH line each (random = no basis round) |
| `--tree_min_count` | `50` | noisy-count pruning threshold |
| `--basis_epsilon` | `0.1` | fraction of the total budget spent on the DP-SGD-PCA basis (dpsgd only); the other 90% goes to clustering |
| `--num_runs` | `5` | seeds per config (reduced from 10 — these runs are heavy) |
| baselines | Lloyd, GLloyd, FastLloyd | from the `local` protocol; SuLloyd dropped in plots |

## Part A — Accuracy

For each dataset: `local` baselines, then `lsh` with that dataset's `d_primes`,
sweeping ε and the three bases. Metrics: NICV (primary), Silhouette,
Davies-Bouldin, … (label-free, so glove100 qualifies). Plots
(`results_folder = large`):
- `plots.compare_methods large` — per dataset, one PDF per metric; lines for
  FastLloyd + LSH-{Rand,SVD,DP-SGD}, x = ε, one subplot row per `d'`.
- `plots.compare_basis large --eps 1.0` — bases across `d'` at ε = 1.

**Expected:** LSH (especially SVD/DP-SGD bases) matches or beats FastLloyd on
NICV under DP, with the margin widening at smaller ε and larger `d'`.

## Part B — Scalability

Federated timing (`mpi_lsh_proto` vs `mpi_proto`) over `mpirun`, sweeping the
number of clients. Measures communication rounds, bytes, and wall-time
(LAN + WAN). Plot:
- `plots.compare_timing large/timing/baselines large/timing/lsh` — per dataset,
  a grid (rows = `d'`, columns = ε), each cell x = #clients (log₂), y = metric.
  **Wall-time** is decomposed into three lines (plus the FastLloyd baseline):
  `Clustering` (basis-free count+sum rounds), `LSH-Rand` (= clustering + its
  near-free random basis), and `LSH-DP-SGD` (= clustering + SGD basis + σ
  calibration). The wall-time y-axis is **log-scaled** because the DP-SGD
  calibration offset is orders of magnitude above the clustering. **SVD is excluded
  from the timing plots** (it is the non-private oracle — its basis is "free" but
  undefined as a private method). `comm_bytes` / `rounds` keep FastLloyd + LSH-Rand
  + LSH-DP-SGD.

**Expected:** clustering rounds flat in #clients (≈6 for LSH vs 4 for FastLloyd —
comparable, *not* strictly below FastLloyd); **`Clustering` and `LSH-Rand`
wall-time 5–50× below FastLloyd**, with communication far lower because the count
rounds are `d`-independent while FastLloyd sends `k·d` per iteration (`k·d` =
500·784 = 392,000 for mnist784, 4000·100 = 400,000 for glove100). `LSH-DP-SGD` sits
far above the others, dominated by the one-time σ calibration (report it as a
precomputable setup cost, not a per-round cost).

## Cost / scoping (important)

These are the heavy runs:
- Each MPI rank loads the full dataset; glove100 (~400k × 100 ≈ 160 MB/rank) at
  many clients is memory-heavy. Start with **`NCLIENTS="2 4 8"`** (the default).
- `--num_runs 5` by default; lower further for a first pass.
- For a quick smoke, fetch a capped GloVe (`GLOVE_MAX_ROWS=100000`) and/or set
  `ACC_DATASETS=mnist784 SCALE_DATASETS=mnist784`.

## Output layout

```
large/
├── EXPERIMENT_PLAN.md
├── run_local.sh
├── accuracy/<dataset>/{variances.csv, variances_lsh.csv, <Metric>.pdf, basis_compare_*}
├── accuracy/{comparison_summary.csv, basis_comparison_summary.csv}
└── timing/
    ├── baselines/timing_<n>/<dataset>/variances_<rank>.csv
    ├── lsh/timing_<n>/<dataset>/variances_<rank>.csv
    └── timing_compare/<dataset>/{elapsed_ms_delay*,comm_bytes,rounds}.pdf
```

## Distributed execution (multiple machines)

For running across a cluster of passwordless-SSH machines that **share the
network filesystem**, use the distributed runner instead of `run_local.sh`:

- `large/jobs.sh` — shared job list (**192 jobs** with the defaults). LSH is cut
  into small, roughly equal-cost units so it doesn't dwarf FastLloyd: accuracy
  splits per `(dataset, d', eps)` into a cheap `random+svd` job and a heavy
  `dpsgd` job (the DP-SGD basis is rebuilt per eps); timing splits per
  `(dataset, clients, d')` with `dpsgd` further per-eps. Each carries a
  `JOB_WEIGHT` (approx runtime). Sourced by both scripts below so they always agree.
- `large/run_distributed.sh` — **edit the `HOSTS=( … )` list in `large/hosts.sh`**
  (and `CONDA_ACTIVATE` if conda isn't on the login PATH), then run it. It assigns
  the *pending* jobs **cost-aware** (longest-processing-time-first by `JOB_WEIGHT`:
  heaviest job to the least-loaded host), so hosts finish together instead of one
  grinding on glove100/d'=20/dpsgd while others idle. It writes a per-host script
  to the shared FS (`large/.dispatch/<host>.sh`) that `cd`s in, activates conda,
  runs its jobs, and on success touches `large/.status/<job>.done`; it also logs a
  per-host planned-load summary before launch. All hosts run in parallel; logs land
  in `large/logs/`. Idempotent: re-running only executes jobs without a marker.
  Plots are generated at the end over whatever completed.
- **Re-running with changed parameters: run `bash large/clean.sh -y` first.** The
  job ids encode the split, not the hyperparameters, so changing e.g.
  `basis_epsilon`/`basis_lr` does **not** invalidate old `.status` markers or
  `parts/` — a stale run would merge old and new rows. `clean.sh` wipes
  results/markers/parts for a clean slate.
  - **To re-run only the scalability part** (e.g. after a timing-instrumentation
    change), use `bash large/clean.sh -y --timing-only`: it wipes just
    `large/timing/` and the `timing_*` markers, keeps the (expensive, already-done)
    accuracy results, and lets `run_distributed.sh` re-run exactly the 126 timing
    jobs. This also prevents old timing CSVs (missing the new phase columns) from
    contaminating `merge_parts.py`.
- `large/check_status.sh` — lists DONE vs PENDING jobs (from the markers) and
  prints the exact commands to re-run the pending ones (or just re-run
  `run_distributed.sh`, which skips completed jobs).

```bash
python scripts/download_data.py --only mnist784 glove100   # once, on the shared FS
# edit HOSTS in large/run_distributed.sh
bash large/run_distributed.sh
bash large/check_status.sh        # later: see what's left, re-run if needed
```

Because the filesystem is shared, jobs are "delivered" just by writing the
per-host scripts to disk — nothing is copied — and every machine reads `data/`
and writes results under the same `large/` path.

## Knobs (env overrides for `run_local.sh`)

`ACC_DATASETS`, `SCALE_DATASETS`, `EPS_BUDGETS`, `BASIS`, `NUM_RUNS`,
`MIN_COUNT`, `NCLIENTS`, `RESULT_FOLDER`, `MPIRUN_FLAGS`. Per-dataset `d'` lives
in the `DPRIMES` map at the top of `run_local.sh`.
