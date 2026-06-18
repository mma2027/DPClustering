#!/bin/bash
# Run Lloyd/FastLloyd (local) and the LSH prefix-tree protocol across the
# accuracy datasets, then generate comparison plots.
#
# Usage:
#   bash scripts/run_lsh.sh                    # default results folder (submission)
#   bash scripts/run_lsh.sh my_results          # custom results folder
#
# Background:
#   nohup bash scripts/run_lsh.sh > logs/lsh.log 2>&1 &

set -e

RESULTS_FOLDER="${1:-submission}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

# ── Experiment configuration ─────────────────────────────────────────────────
# DATASETS="iris s1 house lsun wine yeast breast mnist"
DATASETS="mnist"
BASIS_METHODS="random svd_pca dpsgd_pca"   # LSH SimHash basis methods to sweep
D_PRIMES="5 10 15 20"                       # basis width / max tree depth to sweep
MIN_COUNT=30                               # prune branches below this noisy count
MAX_DEPTH=0                                 # 0 -> use d_prime as the max tree depth

echo "========================================"
echo " FastLloyd + LSH prefix tree"
echo " Results folder : $RESULTS_FOLDER"
echo " Datasets       : $DATASETS"
echo " Basis methods  : $BASIS_METHODS"
echo " d' (depth)     : $D_PRIMES"
echo " Prune <        : $MIN_COUNT noisy points"
echo " Started        : $(date)"
echo "========================================"

# ── 1. Run Lloyd/FastLloyd (local protocol) — baselines ──────────────────────
echo ""
echo "=== Running Lloyd / FastLloyd (local protocol) ==="
python3 experiments.py \
    --exp_type accuracy \
    --protocol local \
    --datasets $DATASETS \
    --results_folder "$RESULTS_FOLDER" &
LOCAL_PID=$!

# ── 2. Run LSH (random, SVD PCA, DP-SGD PCA) ─────────────────────────────────
echo ""
echo "=== Running LSH (random + svd_pca + dpsgd_pca) ==="
python3 experiments.py \
    --exp_type accuracy \
    --protocol lsh \
    --basis_method $BASIS_METHODS \
    --d_primes $D_PRIMES \
    --tree_min_count $MIN_COUNT \
    --tree_max_depth $MAX_DEPTH \
    --datasets $DATASETS \
    --results_folder "$RESULTS_FOLDER" &
LSH_PID=$!

# ── 3. Wait for both to finish ────────────────────────────────────────────────
echo ""
echo "=== Waiting for experiments (local PID=$LOCAL_PID, lsh PID=$LSH_PID) ==="
wait $LOCAL_PID
echo "Local protocol done: $(date)"
wait $LSH_PID
echo "LSH protocol done: $(date)"

# ── 4. Generate plots ─────────────────────────────────────────────────────────
# Positional results-folder FIRST so the greedy --ignore list doesn't swallow it.
echo ""
echo "=== Generating comparison plots ==="
python3 -m plots.compare_methods "$RESULTS_FOLDER" --ignore SuLloyd LSH-SVD
python3 -m plots.compare_basis "$RESULTS_FOLDER"

echo ""
echo "========================================"
echo " Done: $(date)"
echo " Results : $RESULTS_FOLDER/accuracy/"
echo "========================================"
