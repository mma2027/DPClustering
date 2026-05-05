#!/bin/bash
# Run Lloyd/FastLloyd (local) and ortho DP-SGD PCA experiments across all accuracy
# datasets, then generate comparison plots.
#
# Usage:
#   bash scripts/run_ortho_dpsgd.sh                   # default results folder
#   bash scripts/run_ortho_dpsgd.sh my_results         # custom results folder
#
# Background:
#   nohup bash scripts/run_ortho_dpsgd.sh > logs/ortho_dpsgd.log 2>&1 &

set -e

RESULTS_FOLDER="${1:-submission}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

echo "========================================"
echo " FastLloyd + Ortho DP-SGD PCA"
echo " Results folder : $RESULTS_FOLDER"
echo " Started        : $(date)"
echo "========================================"

DATASETS="iris s1 house lsun wine yeast breast mnist"

# ── 1. Run Lloyd/FastLloyd (local protocol) ───────────────────────────────────
echo ""
echo "=== Running Lloyd / FastLloyd (local protocol) ==="
python3 experiments.py \
    --exp_type accuracy \
    --protocol local \
    --datasets $DATASETS \
    --results_folder "$RESULTS_FOLDER" &
LOCAL_PID=$!

# ── 2. Run ortho (random, SVD PCA, DP-SGD PCA) ───────────────────────────────
echo ""
echo "=== Running ortho (random + svd_pca + dpsgd_pca) ==="
python3 experiments.py \
    --exp_type accuracy \
    --protocol ortho \
    --basis_method random svd_pca dpsgd_pca \
    --datasets $DATASETS \
    --results_folder "$RESULTS_FOLDER" &
ORTHO_PID=$!

# ── 3. Wait for both to finish ────────────────────────────────────────────────
echo ""
echo "=== Waiting for experiments (local PID=$LOCAL_PID, ortho PID=$ORTHO_PID) ==="
wait $LOCAL_PID
echo "Local protocol done: $(date)"
wait $ORTHO_PID
echo "Ortho protocol done: $(date)"

# ── 4. Generate plots ─────────────────────────────────────────────────────────
echo ""
echo "=== Generating comparison plots ==="
python3 -m plots.compare_protocols "$RESULTS_FOLDER"
python3 -m plots.compare_basis "$RESULTS_FOLDER"

echo ""
echo "========================================"
echo " Done: $(date)"
echo " Results : $RESULTS_FOLDER/accuracy/"
echo "========================================"
