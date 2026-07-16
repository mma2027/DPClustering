#!/bin/bash
# Distributed large-dataset experiments: fan jobs out across machines.
#
# Assumes: passwordless SSH to every host, and a SHARED network filesystem (so
# this checkout, the data/, and the results all live at the same path on every
# machine — jobs are "delivered" simply by writing per-host scripts to the
# shared FS; nothing is copied).
#
# Each job writes a marker  large/.status/<job_id>.done  on success, so the run
# is idempotent: re-running this script only executes jobs that are not yet done
# (use large/check_status.sh to see what is pending).
#
# Usage:
#   1. Edit HOSTS below (and CONDA_* if conda isn't on a login PATH).
#   2. python scripts/download_data.py --only mnist784 glove100   # once, shared FS
#   3. bash large/run_distributed.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
source "$SCRIPT_DIR/jobs.sh"

# ── Machines + conda + ssh opts (shared with check_hosts.sh) ─────────────────
# Edit the HOSTS list in large/hosts.sh. Health-check them first:
#   bash large/check_hosts.sh
source "$SCRIPT_DIR/hosts.sh"

LOG_DIR="$RESULT_FOLDER/logs"
DISPATCH_DIR="$RESULT_FOLDER/.dispatch"
mkdir -p "$STATUS_DIR" "$LOG_DIR" "$DISPATCH_DIR"
# Resolve to absolute paths: a remote `bash -l` login shell starts in $HOME, so
# the per-host script path passed over ssh (and the .done markers it touches)
# must be absolute, not relative to the project root.
STATUS_DIR="$(cd "$STATUS_DIR" && pwd)"
LOG_DIR="$(cd "$LOG_DIR" && pwd)"
DISPATCH_DIR="$(cd "$DISPATCH_DIR" && pwd)"
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/dispatch_$STAMP.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

# ── Prerequisites ─────────────────────────────────────────────────────────────
for ds in $ACC_DATASETS $SCALE_DATASETS; do
    if [ ! -f "data/$ds.txt" ]; then
        echo "ERROR: data/$ds.txt missing. Download (once, on the shared FS):"
        echo "    python scripts/download_data.py --only $ds"
        exit 1
    fi
done
if [ ${#HOSTS[@]} -eq 0 ]; then
    echo "ERROR: no HOSTS configured — edit $SCRIPT_DIR/hosts.sh"
    exit 1
fi

# ── Pending jobs = those without a .done marker (idempotent resume) ──────────
PENDING=()
for id in "${JOB_IDS[@]}"; do
    [ -f "$STATUS_DIR/$id.done" ] || PENDING+=("$id")
done
log "Jobs total=${#JOB_IDS[@]} done=$(( ${#JOB_IDS[@]} - ${#PENDING[@]} )) pending=${#PENDING[@]}"
if [ ${#PENDING[@]} -eq 0 ]; then log "Nothing to run."; exit 0; fi

# ── Build one script per host (round-robin assignment) on the shared FS ──────
declare -A HOST_SCRIPT=()
declare -A HOST_NJOBS=()
for h in "${HOSTS[@]}"; do
    s="$DISPATCH_DIR/$h.sh"
    HOST_SCRIPT["$h"]="$s"
    HOST_NJOBS["$h"]=0
    {
        echo "#!/bin/bash"
        echo "cd '$PROJECT_ROOT' || exit 1"
        echo "export PYTHONPATH='$PROJECT_ROOT'"
        echo "set +u            # conda's activate.d scripts reference unbound vars"
        echo "$CONDA_ACTIVATE"
        echo "set -u"
    } > "$s"
done

# Cost-aware assignment (longest-processing-time-first): sort pending jobs by
# weight descending, then greedily place each on the currently least-loaded host.
# This balances the makespan far better than round-robin when job costs vary by
# ~100x (FastLloyd vs glove100/d'=20/dpsgd), so hosts finish together.
declare -A HOST_LOAD=()
for h in "${HOSTS[@]}"; do HOST_LOAD["$h"]=0; done
mapfile -t PENDING_SORTED < <(
    for id in "${PENDING[@]}"; do echo "${JOB_WEIGHT[$id]:-1} $id"; done | sort -rn | awk '{print $2}'
)
for id in "${PENDING_SORTED[@]}"; do
    w=${JOB_WEIGHT[$id]:-1}
    # pick the least-loaded host
    best=""; bestload=0
    for h in "${HOSTS[@]}"; do
        l=${HOST_LOAD[$h]}
        if [ -z "$best" ] || [ "$l" -lt "$bestload" ]; then best="$h"; bestload="$l"; fi
    done
    HOST_LOAD["$best"]=$(( bestload + w ))
    HOST_NJOBS["$best"]=$(( HOST_NJOBS["$best"] + 1 ))
    {
        echo "echo \"[\$(date +%H:%M:%S)] START $id on \$(hostname)\""
        echo "if ${JOB_CMD[$id]}; then"
        echo "    touch '$STATUS_DIR/$id.done'; echo \"[\$(date +%H:%M:%S)] DONE  $id\""
        echo "else echo \"[\$(date +%H:%M:%S)] FAIL  $id (exit \$?)\"; fi"
    } >> "${HOST_SCRIPT[$best]}"
    log "assign $id (w=$w) -> $best  [load=${HOST_LOAD[$best]}]"
done

# Per-host planned load summary (helps spot a straggler before launch).
for h in "${HOSTS[@]}"; do
    [ "${HOST_NJOBS[$h]}" -gt 0 ] && log "  plan: $h  jobs=${HOST_NJOBS[$h]}  load=${HOST_LOAD[$h]}"
done

# ── Dispatch: one ssh per host, all hosts in parallel ────────────────────────
pids=()
for h in "${HOSTS[@]}"; do
    [ "${HOST_NJOBS[$h]}" -eq 0 ] && continue
    log "ssh -> $h  (${HOST_NJOBS[$h]} jobs, log: $LOG_DIR/$h.log)"
    # bash -l so login profiles (and conda) are on PATH on the remote.
    ssh $SSH_OPTS "$h" "bash -l '${HOST_SCRIPT[$h]}'" > "$LOG_DIR/$h.log" 2>&1 &
    pids+=($!)
done

log "Dispatched to ${#pids[@]} host(s); waiting for completion..."
ssh_fail=0
for p in "${pids[@]}"; do wait "$p" || ssh_fail=$((ssh_fail + 1)); done

done_now=0
for id in "${JOB_IDS[@]}"; do [ -f "$STATUS_DIR/$id.done" ] && done_now=$((done_now + 1)); done
log "Hosts returned (ssh-level failures: $ssh_fail). Completed ${done_now}/${#JOB_IDS[@]} jobs."

# ── Merge per-d' LSH parts into the canonical layout, then plot ──────────────
log "Merging per-d' LSH parts..."
python3 "$SCRIPT_DIR/merge_parts.py" "$RESULT_FOLDER"                        >> "$MASTER_LOG" 2>&1 || true

log "Generating plots over completed results..."
python3 -m plots.compare_methods "$RESULT_FOLDER" --ignore SuLloyd          >> "$MASTER_LOG" 2>&1 || true
python3 -m plots.compare_basis   "$RESULT_FOLDER" --eps 1.0                  >> "$MASTER_LOG" 2>&1 || true
python3 -m plots.compare_timing  "$RESULT_FOLDER/timing/baselines" "$RESULT_FOLDER/timing/lsh" \
        --out "$RESULT_FOLDER/timing/timing_compare"                        >> "$MASTER_LOG" 2>&1 || true

log "Done. Master log: $MASTER_LOG  |  pending check: bash large/check_status.sh"
