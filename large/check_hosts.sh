#!/bin/bash
# Health-check the cluster before a distributed run: for every host in
# large/hosts.sh, test passwordless SSH and (unless --ssh-only) that the shared
# filesystem is visible, the conda env activates, and python3 / mpirun are
# available. All hosts are probed in parallel; dead ones fail fast (ConnectTimeout).
#
# Usage:
#   bash large/check_hosts.sh             # full check (ssh + fs + conda + py + mpi)
#   bash large/check_hosts.sh --ssh-only  # just reachability / passwordless ssh
#
# Exit status: 0 if every host passed, 1 otherwise (handy in CI / before dispatch).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/hosts.sh"

SSH_ONLY=0
[ "${1:-}" = "--ssh-only" ] && SSH_ONLY=1

if [ ${#HOSTS[@]} -eq 0 ]; then
    echo "No HOSTS configured in $SCRIPT_DIR/hosts.sh"
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Remote probe script lives on the shared FS, so every host can just run it.
CHECK="$SCRIPT_DIR/.dispatch/_hostcheck.sh"
mkdir -p "$(dirname "$CHECK")"
cat > "$CHECK" <<EOF
#!/bin/bash
echo "NAME=\$(hostname)"
echo "LOAD=\$(cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo NA)"
[ -d '$PROJECT_ROOT' ] && echo "FS=ok" || echo "FS=missing"
if $CONDA_ACTIVATE >/dev/null 2>&1; then
    echo "CONDA=ok"
    python3 -c 'import sys; print("PY=" + sys.version.split()[0])' 2>/dev/null || echo "PY=missing"
    command -v mpirun >/dev/null 2>&1 && echo "MPI=ok" || echo "MPI=missing"
else
    echo "CONDA=fail"
fi
EOF

_field() { printf '%s\n' "$1" | sed -n "s/^$2=//p" | head -1; }

check_one() {
    local h="$1" out rc
    if [ "$SSH_ONLY" -eq 1 ]; then
        out="$(ssh $SSH_OPTS "$h" 'echo NAME=$(hostname)' 2>/dev/null)"; rc=$?
    else
        out="$(ssh $SSH_OPTS "$h" "bash -l '$CHECK'" 2>/dev/null)"; rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        printf '%s\tFAIL\t-\t-\t-\t-\t-\n' "$h" > "$TMP/$h"
        return
    fi
    printf '%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$h" \
        "$(_field "$out" FS)" "$(_field "$out" CONDA)" \
        "$(_field "$out" PY)" "$(_field "$out" MPI)" "$(_field "$out" LOAD)" > "$TMP/$h"
}

echo "Probing ${#HOSTS[@]} hosts in parallel$( [ "$SSH_ONLY" -eq 1 ] && echo ' (ssh-only)')..."
for h in "${HOSTS[@]}"; do check_one "$h" & done
wait

printf '\n%-10s %-5s %-8s %-6s %-8s %-5s %-6s\n' HOST SSH FS CONDA PYTHON MPI LOAD
printf -- '-------------------------------------------------------------\n'
ok=0
bad=()
for h in "${HOSTS[@]}"; do
    IFS=$'\t' read -r H s fs conda py mpi load < "$TMP/$h"
    printf '%-10s %-5s %-8s %-6s %-8s %-5s %-6s\n' \
        "$H" "$s" "${fs:--}" "${conda:--}" "${py:--}" "${mpi:--}" "${load:--}"
    if [ "$s" = ok ] && { [ "$SSH_ONLY" -eq 1 ] || \
         { [ "$fs" = ok ] && [ "$conda" = ok ] && [ -n "$py" ] && \
           [ "$py" != missing ] && [ "$mpi" = ok ]; }; }; then
        ok=$((ok + 1))
    else
        bad+=("$h")
    fi
done
printf -- '-------------------------------------------------------------\n'
echo "Healthy: ${ok}/${#HOSTS[@]}"
[ ${#bad[@]} -gt 0 ] && echo "Problem hosts: ${bad[*]}"
[ ${#bad[@]} -eq 0 ]
