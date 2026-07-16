#!/bin/bash
# Large-dataset experiments (accuracy + scalability): LSH vs FastLloyd.
# Headline group: large, dense, high-dimensional. See large/EXPERIMENT_PLAN.md.
#
# Prerequisite (datasets are NOT in the default download):
#   python scripts/download_data.py --only mnist784 glove100
#
# Reproduce with:  bash large/run_local.sh
# Background:      nohup bash large/run_local.sh > large/run.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

# ── Configuration (override via environment) ─────────────────────────────────
RESULT_FOLDER=${RESULT_FOLDER:-"large"}            # results root (this folder)
EPS_BUDGETS=${EPS_BUDGETS:-"0.5 1 2 4"}            # privacy sweep
BASIS=${BASIS:-"random svd_pca dpsgd_pca"}         # one LSH line per basis
NUM_RUNS=${NUM_RUNS:-5}                             # seeds per config (heavy runs -> 5)
MIN_COUNT=${MIN_COUNT:-50}                          # LSH pruning threshold
NCLIENTS=${NCLIENTS:-"2 4"}                         # client counts for timing. n=8 OOMs on
                                                    # glove300 (400k x 300) even with the lean
                                                    # load_txt: 9 ranks x (to_fixed transient +
                                                    # eval copies) > 32 GB. Needs per-rank shard
                                                    # loading (Fix A) to re-enable n=8.
MPIRUN_FLAGS=${MPIRUN_FLAGS:-"--oversubscribe"}     # set "" if unsupported

ACC_DATASETS=${ACC_DATASETS:-"mnist784 glove100"}
SCALE_DATASETS=${SCALE_DATASETS:-"mnist784 glove100"}

# Per-dataset d' (basis width / tree depth): ceil(log2 k) <= d' <= data dim.
declare -A DPRIMES=(
  [mnist784]="5 10 15 20"
  [glove100]="8 12 16 20"
)

# ── Prerequisite check: datasets must be downloaded first ────────────────────
for ds in $ACC_DATASETS $SCALE_DATASETS; do
    if [ ! -f "data/$ds.txt" ]; then
        echo "ERROR: data/$ds.txt not found. Download it first:"
        echo "    python scripts/download_data.py --only $ds"
        exit 1
    fi
done

echo "========================================"
echo " Large-dataset experiments: LSH vs FastLloyd"
echo " Accuracy : $ACC_DATASETS"
echo " Scale    : $SCALE_DATASETS   clients=$NCLIENTS"
echo " eps=$EPS_BUDGETS  bases=$BASIS  prune<$MIN_COUNT  runs=$NUM_RUNS"
echo " Results  : $RESULT_FOLDER/"
echo " Started  : $(date)"
echo "========================================"

# ── Part A: Accuracy ─────────────────────────────────────────────────────────
echo ""
echo "=== Part A: accuracy ==="
for ds in $ACC_DATASETS; do
    echo "--- $ds (baselines) ---"
    python3 experiments.py --exp_type accuracy --protocol local \
        --datasets "$ds" --num_runs "$NUM_RUNS" --results_folder "$RESULT_FOLDER"

    echo "--- $ds (LSH, d'=${DPRIMES[$ds]}) ---"
    python3 experiments.py --exp_type accuracy --protocol lsh \
        --datasets "$ds" --basis_method $BASIS --d_primes ${DPRIMES[$ds]} \
        --tree_min_count "$MIN_COUNT" --basis_epsilon 0.1 \
        --basis_lr 0.1 --basis_epochs 10 \
        --num_runs "$NUM_RUNS" --results_folder "$RESULT_FOLDER"
done

echo "--- accuracy plots ---"
python3 -m plots.compare_methods "$RESULT_FOLDER" --ignore SuLloyd
python3 -m plots.compare_basis  "$RESULT_FOLDER" --eps 1.0

# ── Part B: Scalability (MPI timing) ─────────────────────────────────────────
TIMING="$RESULT_FOLDER/timing"
echo ""
echo "=== Part B: scalability ==="
for ds in $SCALE_DATASETS; do
    for n in $NCLIENTS; do
        np=$((n + 1))   # +1 for the server (rank 0)
        echo "--- $ds, clients=$n (FastLloyd) ---"
        mpirun $MPIRUN_FLAGS -np "$np" python3 experiments.py \
            --exp_type timing --protocol local \
            --eps_budgets $EPS_BUDGETS \
            --datasets "$ds" --num_runs "$NUM_RUNS" \
            --results_folder "$TIMING/baselines"

        echo "--- $ds, clients=$n (LSH, d'=${DPRIMES[$ds]}) ---"
        mpirun $MPIRUN_FLAGS -np "$np" python3 experiments.py \
            --exp_type timing --protocol lsh \
            --basis_method $BASIS --d_primes ${DPRIMES[$ds]} \
            --eps_budgets $EPS_BUDGETS --tree_min_count "$MIN_COUNT" \
            --basis_epsilon 0.1 --basis_lr 0.1 --basis_epochs 10 \
            --datasets "$ds" --num_runs "$NUM_RUNS" \
            --results_folder "$TIMING/lsh"
    done
done

echo "--- scalability plots ---"
python3 -m plots.compare_timing "$TIMING/baselines" "$TIMING/lsh" --out "$TIMING/timing_compare"

echo ""
echo "========================================"
echo " Done: $(date)"
echo " Accuracy : $RESULT_FOLDER/accuracy/   (compare_methods, compare_basis)"
echo " Scaling  : $TIMING/timing_compare/"
echo "========================================"
