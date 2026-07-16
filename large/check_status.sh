#!/bin/bash
# Report which distributed large-dataset jobs are done vs pending, by checking
# the .done markers in large/.status/ (shared FS). Pending jobs can be re-run by
# simply re-running large/run_distributed.sh (it skips completed jobs), or
# individually with the commands printed below.
#
# Usage:  bash large/check_status.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
source "$SCRIPT_DIR/jobs.sh"

done_n=0
pending=()
echo "Status of ${#JOB_IDS[@]} jobs (markers in $STATUS_DIR):"
echo "------------------------------------------------------------"
for id in "${JOB_IDS[@]}"; do
    if [ -f "$STATUS_DIR/$id.done" ]; then
        echo "  [DONE]     $id"
        done_n=$((done_n + 1))
    else
        echo "  [PENDING]  $id"
        pending+=("$id")
    fi
done
echo "------------------------------------------------------------"
echo "Done ${done_n}/${#JOB_IDS[@]}  |  pending ${#pending[@]}"

if [ ${#pending[@]} -gt 0 ]; then
    echo ""
    echo "Re-run all pending (parallel, skips completed):"
    echo "    bash large/run_distributed.sh"
    echo ""
    echo "Or run a pending job manually (after: conda activate fastlloyd):"
    for id in "${pending[@]}"; do
        echo "  # $id"
        echo "  ${JOB_CMD[$id]}"
    done
fi
