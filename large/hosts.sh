#!/bin/bash
# Cluster machines for distributed runs: passwordless SSH + shared filesystem.
# Single source of truth for the host list — sourced by both
# large/run_distributed.sh and large/check_hosts.sh.
# Hostnames may be short (resolved via ~/.ssh/config or domain search) or
# fully-qualified (name.main_domain).
# Override the host list for a single run with, e.g.
#   HOSTS_LIST="gray west johnson" bash large/run_distributed.sh
# (useful to restrict to healthy hosts). Otherwise the full cluster below is used.
if [ -n "${HOSTS_LIST:-}" ]; then
    read -ra HOSTS <<< "$HOSTS_LIST"
else
HOSTS=(
    karp
    hawes
    clay
    gray
    rao
    dahl
    west
    brooks
    naur
    goto
    joshi
    johnson
    sammet
    hall
    muthoni
    micali
    asakawa
    goldberg
)
fi

# How each machine enters the conda env before running. Override if `conda` is
# not on the non-interactive PATH (e.g. point at the install's conda.sh):
#   CONDA_ACTIVATE='source /opt/miniconda3/etc/profile.d/conda.sh && conda activate fastlloyd'
CONDA_ACTIVATE=${CONDA_ACTIVATE:-'eval "$(conda shell.bash hook)" && conda activate fastlloyd'}

# Non-interactive SSH: no password prompt, fail fast on dead hosts.
SSH_OPTS=${SSH_OPTS:-"-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"}
