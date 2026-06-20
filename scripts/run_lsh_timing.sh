#!/bin/bash
# Scalability / performance comparison: federated LSH (mpi_lsh_proto) vs
# FastLloyd (mpi_proto), measuring communication rounds, bytes, and wall-time
# (with simulated per-round delay) via the MPI timing harness.
#
# Sweeps the number of clients; n-scaling is done by passing timesynth datasets
# of different sizes via DATASETS. (d-scaling is deferred.)
#
# FastLloyd and LSH write to SEPARATE results folders because the timing harness
# names every protocol's per-rank output `variances_<rank>.csv` (so they would
# otherwise overwrite each other in the same timing_<n>/ folder).
#
# Usage:
#   bash scripts/run_lsh_timing.sh
#   NCLIENTS="2 4 8" DATASETS="timesynth_2_2_10000 timesynth_2_2_100000" \
#       bash scripts/run_lsh_timing.sh
#
# Background:
#   nohup bash scripts/run_lsh_timing.sh > logs/lsh_timing.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# ── Configuration (override via environment) ─────────────────────────────────
NCLIENTS=${NCLIENTS:-"2 4 8 16"}          # client counts to sweep (x-axis of the grid)
# n-scaling: same k,d (=2,2), increasing number of points
DATASETS=${DATASETS:-"mnist timesynth_2_2_10000 timesynth_2_2_100000"}
DPRIMES=${DPRIMES:-"5 10 15"}             # basis width / max tree depth (grid rows)
EPS_BUDGETS=${EPS_BUDGETS:-"0.5 1 2"}      # privacy budgets (grid columns)
MIN_COUNT=${MIN_COUNT:-20}                   # LSH pruning threshold
BASIS=${BASIS:-"random svd_pca dpsgd_pca"}   # SimHash bases -> one LSH line each
NUM_RUNS=${NUM_RUNS:-10}                      # seeds per config (averaged)
RESULT_FOLDER=${RESULT_FOLDER:-"submission_timing"}              # parent results folder
LOCAL_FOLDER=${LOCAL_FOLDER:-"$RESULT_FOLDER/baselines"} # FastLloyd results
LSH_FOLDER=${LSH_FOLDER:-"$RESULT_FOLDER/lsh"}           # LSH results (separate!)
OUT_FOLDER=${OUT_FOLDER:-"$RESULT_FOLDER/timing_compare"} # comparison plots/CSV
MPIRUN_FLAGS=${MPIRUN_FLAGS:-"--oversubscribe"}   # set to "" if not supported

echo "========================================"
echo " LSH vs FastLloyd — scalability (timing)"
echo " Clients   : $NCLIENTS"
echo " Datasets  : $DATASETS"
echo " d'        : $DPRIMES   eps: $EPS_BUDGETS   prune< $MIN_COUNT"
echo " Bases     : $BASIS"
echo " FastLloyd -> $LOCAL_FOLDER/   LSH -> $LSH_FOLDER/"
echo " Started   : $(date)"
echo "========================================"

for n in $NCLIENTS; do
    np=$((n + 1))   # +1 for the server (rank 0)
    echo ""
    echo "=== clients=$n (np=$np) ==="

    echo "--- FastLloyd ---"
    mpirun $MPIRUN_FLAGS -np "$np" python3 experiments.py \
        --exp_type timing --protocol local \
        --eps_budgets $EPS_BUDGETS \
        --datasets $DATASETS --num_runs "$NUM_RUNS" \
        --results_folder "$LOCAL_FOLDER"

    echo "--- Federated LSH ---"
    mpirun $MPIRUN_FLAGS -np "$np" python3 experiments.py \
        --exp_type timing --protocol lsh \
        --basis_method $BASIS --d_primes $DPRIMES \
        --eps_budgets $EPS_BUDGETS --tree_min_count "$MIN_COUNT" \
        --datasets $DATASETS --num_runs "$NUM_RUNS" \
        --results_folder "$LSH_FOLDER"
done

echo ""
echo "=== Generating comparison plots ==="
python3 -m plots.compare_timing "$LOCAL_FOLDER" "$LSH_FOLDER" --out "$OUT_FOLDER"

echo ""
echo "========================================"
echo " Done: $(date)"
echo " FastLloyd : $LOCAL_FOLDER/timing_<n>/<dataset>/variances_<rank>.csv"
echo " LSH       : $LSH_FOLDER/timing_<n>/<dataset>/variances_<rank>.csv"
echo " Plots/CSV : $OUT_FOLDER/  (time_*.pdf, comm_*.pdf, timing_compare.csv)"
echo "========================================"
