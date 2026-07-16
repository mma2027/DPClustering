#!/bin/bash
# Clean slate for the distributed large-dataset run:
#   1. terminate running/pending jobs on every host (and the local dispatcher),
#   2. delete results, logs, status markers, and per-host dispatch scripts.
# Keeps the scripts and EXPERIMENT_PLAN.md.
#
# With --timing-only, wipe ONLY the timing outputs + timing job markers, keeping
# the (expensive, already-complete) accuracy results. Use this to re-run just the
# scalability part -- e.g. after changing the timing instrumentation -- so the old
# timing CSVs (missing new columns) can't contaminate merge_parts, and the timing
# jobs aren't skipped as already-done.
#
# DESTRUCTIVE — pass -y / --yes to skip the confirmation prompt.
# Usage:  bash large/clean.sh [-y] [--timing-only]

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/hosts.sh"
RESULT_FOLDER=${RESULT_FOLDER:-large}

YES=0
TIMING_ONLY=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes) YES=1 ;;
        -t|--timing-only) TIMING_ONLY=1 ;;
    esac
done

echo "This will, for RESULT_FOLDER='$RESULT_FOLDER':"
echo "  - terminate experiment jobs on ${#HOSTS[@]} hosts + the local dispatcher"
if [ "$TIMING_ONLY" -eq 1 ]; then
    echo "  - delete $RESULT_FOLDER/timing and $RESULT_FOLDER/.status/timing_*.done"
    echo "    (KEEPS accuracy results + accuracy markers)"
else
    echo "  - delete $RESULT_FOLDER/{accuracy,timing,logs,.status,.dispatch} and *.log"
fi
if [ "$YES" -ne 1 ]; then
    read -r -p "Proceed? [y/N] " ans
    case "$ans" in y|Y|yes|YES) ;; *) echo "Aborted."; exit 1 ;; esac
fi

# ── 1. Stop the local dispatcher (so it stops launching / waiting) ───────────
pkill -f 'large/run_distributed[.]sh' 2>/dev/null || true

# ── 2. Terminate running + pending jobs on every host (in parallel) ──────────
# Kill the per-host loop FIRST (so it can't launch the next pending job), then
# the running worker + its mpirun parent. The bracket trick ('[.]', '[p]') makes
# each regex NOT match the pkill command line itself:
#   '[.]dispatch/'             -> the per-host loop script  (bash .../.dispatch/<h>.sh)
#   '[p]ython3 experiments.py' -> the worker and its mpirun parent
echo "Terminating jobs on ${#HOSTS[@]} hosts..."
for h in "${HOSTS[@]}"; do
    ssh $SSH_OPTS "$h" \
        "pkill -f '[.]dispatch/' 2>/dev/null; pkill -f '[p]ython3 experiments.py' 2>/dev/null; true" \
        >/dev/null 2>&1 &
done
wait
echo "  ...kill signals sent."

# ── 3. Delete generated outputs (keep scripts + EXPERIMENT_PLAN.md) ──────────
if [ "$TIMING_ONLY" -eq 1 ]; then
    # Only the timing outputs (merged + parts) and the timing job markers. Accuracy
    # results and acc_*.done markers are preserved, so run_distributed.sh re-runs
    # exactly the timing jobs.
    rm -rf "$RESULT_FOLDER/timing"
    rm -f "$RESULT_FOLDER"/.status/timing_*.done
    echo "Cleaned TIMING only: $RESULT_FOLDER/timing + timing markers removed; accuracy kept."
else
    rm -rf "$RESULT_FOLDER/accuracy" "$RESULT_FOLDER/timing" "$RESULT_FOLDER/parts" \
           "$RESULT_FOLDER/logs" "$RESULT_FOLDER/.status" "$RESULT_FOLDER/.dispatch"
    rm -f "$RESULT_FOLDER"/*.log
    echo "Cleaned: results, logs, markers, and dispatch scripts removed; jobs terminated."
fi
echo "Re-run from scratch with: bash large/run_distributed.sh"
