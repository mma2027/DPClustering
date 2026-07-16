#!/bin/bash
# Shared job definitions for the distributed large-dataset experiments.
# Sourced by run_distributed.sh (dispatch) and check_status.sh (status), so both
# agree on the exact set of jobs and commands. Each job is one self-contained
# unit that runs on a single machine.
#
# Exposes: RESULT_FOLDER, STATUS_DIR, JOB_IDS (ordered), JOB_CMD[id] -> command,
#          JOB_WEIGHT[id] -> approx relative runtime (for load balancing),
#          ACC_DATASETS, SCALE_DATASETS (for the data-presence check).
#
# All config is env-overridable.

RESULT_FOLDER=${RESULT_FOLDER:-"large"}
STATUS_DIR=${STATUS_DIR:-"$RESULT_FOLDER/.status"}    # per-job .done markers (shared FS)
EPS_BUDGETS=${EPS_BUDGETS:-"0.5 1 2 4"}
NUM_RUNS=${NUM_RUNS:-5}
MIN_COUNT=${MIN_COUNT:-50}
NCLIENTS=${NCLIENTS:-"2 4 8"}
MPIRUN_FLAGS=${MPIRUN_FLAGS:-"--oversubscribe"}
# Large tier by default. The HUGE tier (glove840b, ~2.2M x 300, near FastLloyd's limit)
# is opt-in -- run it separately, e.g.  ACC_DATASETS="glove840b" SCALE_DATASETS="glove840b" \
# bash large/run_distributed.sh
ACC_DATASETS=${ACC_DATASETS:-"mnist784 glove100 glove300"}
SCALE_DATASETS=${SCALE_DATASETS:-"mnist784 glove100 glove300"}

# DP-SGD-PCA basis hyperparameters. lr=0.1 (the old lr=0.01 left the basis stuck at its
# random init). epochs=10: the noise-vs-optimization optimum is interior and small because
# sigma is calibrated to the step count T = epochs*(m/batch) -- more epochs -> larger sigma.
# At tight eps / high d (mnist784) EVR peaks near ~10 and 40 roughly halves it; glove100 is
# insensitive. basis_epsilon=0.1 spends 10% of the (eps, delta) budget on the basis.
BASIS_EPS=${BASIS_EPS:-0.1}
BASIS_LR=${BASIS_LR:-0.1}
BASIS_EPOCHS=${BASIS_EPOCHS:-10}

declare -A DPRIMES=(
  [mnist784]="5 10 15 20"
  [glove100]="8 12 16 20"
  [glove300]="8 12 16 20"
  [glove840b]="8 12 16 20"
)
# Relative cost weight per dataset (drives the load-balanced assignment in
# run_distributed.sh; absolute scale is irrelevant). Roughly the per-dataset work:
# glove100 ~5.7x mnist784 points; glove300 = glove100 x 3 dims; glove840b (huge) ~5.5x
# glove100 points x 3 dims, by far the heaviest -- esp. the FastLloyd baseline O(n*k*d).
declare -A DSW=( [mnist784]=2 [glove100]=11 [glove300]=33 [glove840b]=60 )

# ── Splitting strategy ───────────────────────────────────────────────────────
# LSH sweeps many d' and the DP-SGD basis (10 epochs) is far heavier than the
# rest, so LSH dwarfs FastLloyd and a coarse per-d' split leaves stragglers. We
# cut LSH into small units of roughly uniform cost:
#   ACCURACY  per (dataset, d', eps):
#     - randsvd : the random + svd_pca bases (cheap; no basis budget)
#     - dpsgd   : the dpsgd_pca basis (heavy; rebuilt per eps since the basis
#                 budget is eps_basis = basis_epsilon * eps, so eps is the
#                 natural finest cut -- this isolates the single heaviest units)
#   TIMING    per (dataset, clients, d'): randsvd vs dpsgd (eps stays bundled in
#             --eps_budgets to avoid launching one mpirun per eps).
# Each split job writes to its own parts/<id> dir; merge_parts.py stitches them.
# FastLloyd ("local") has no d' and stays one-per-(dataset[,clients]).
# JOB_WEIGHT[id] feeds longest-processing-time-first scheduling so hosts finish
# together rather than some idling while one grinds on glove100/d'=20/dpsgd.

JOB_IDS=()
declare -A JOB_CMD=()
declare -A JOB_WEIGHT=()
_add_job() { JOB_IDS+=("$1"); JOB_WEIGHT["$1"]="$2"; JOB_CMD["$1"]="$3"; }

# --- Accuracy ----------------------------------------------------------------
for ds in $ACC_DATASETS; do
    dsw=${DSW[$ds]:-2}
    _add_job "acc_${ds}_local" $(( dsw * 3 )) \
        "python3 experiments.py --exp_type accuracy --protocol local --datasets $ds --num_runs $NUM_RUNS --results_folder $RESULT_FOLDER"
    for dp in ${DPRIMES[$ds]}; do
        for eps in $EPS_BUDGETS; do
            _add_job "acc_${ds}_randsvd_d${dp}_e${eps}" $(( dsw * dp )) \
                "python3 experiments.py --exp_type accuracy --protocol lsh --datasets $ds --basis_method random svd_pca --d_primes $dp --eps_budgets $eps --tree_min_count $MIN_COUNT --num_runs $NUM_RUNS --results_folder $RESULT_FOLDER/parts/acc_${ds}_randsvd_d${dp}_e${eps}"
            _add_job "acc_${ds}_dpsgd_d${dp}_e${eps}" $(( dsw * dp * 2 )) \
                "python3 experiments.py --exp_type accuracy --protocol lsh --datasets $ds --basis_method dpsgd_pca --d_primes $dp --eps_budgets $eps --tree_min_count $MIN_COUNT --basis_epsilon $BASIS_EPS --basis_lr $BASIS_LR --basis_epochs $BASIS_EPOCHS --num_runs $NUM_RUNS --results_folder $RESULT_FOLDER/parts/acc_${ds}_dpsgd_d${dp}_e${eps}"
        done
    done
done

# --- Scalability (MPI) -------------------------------------------------------
for ds in $SCALE_DATASETS; do
    dsw=${DSW[$ds]:-2}
    for n in $NCLIENTS; do
        np=$((n + 1))   # +1 for the server (rank 0)
        _add_job "timing_${ds}_local_n${n}" $(( dsw * (n + 1) * 3 )) \
            "mpirun $MPIRUN_FLAGS -np $np python3 experiments.py --exp_type timing --protocol local --eps_budgets $EPS_BUDGETS --datasets $ds --num_runs $NUM_RUNS --results_folder $RESULT_FOLDER/timing/baselines"
        for dp in ${DPRIMES[$ds]}; do
            # random+svd: cheap and basis is eps-independent -> keep all eps bundled.
            _add_job "timing_${ds}_randsvd_n${n}_d${dp}" $(( dsw * dp * (n + 1) )) \
                "mpirun $MPIRUN_FLAGS -np $np python3 experiments.py --exp_type timing --protocol lsh --datasets $ds --basis_method random svd_pca --d_primes $dp --eps_budgets $EPS_BUDGETS --tree_min_count $MIN_COUNT --num_runs $NUM_RUNS --results_folder $RESULT_FOLDER/timing/parts/lsh_${ds}_randsvd_n${n}_d${dp}"
            # dpsgd: heavy basis (rebuilt per eps) -> split per eps so no single
            # mpirun unit (e.g. glove100/n=8/d'=20) dominates the makespan.
            for eps in $EPS_BUDGETS; do
                _add_job "timing_${ds}_dpsgd_n${n}_d${dp}_e${eps}" $(( dsw * dp * (n + 1) / 2 + 1 )) \
                    "mpirun $MPIRUN_FLAGS -np $np python3 experiments.py --exp_type timing --protocol lsh --datasets $ds --basis_method dpsgd_pca --d_primes $dp --eps_budgets $eps --tree_min_count $MIN_COUNT --basis_epsilon $BASIS_EPS --basis_lr $BASIS_LR --basis_epochs $BASIS_EPOCHS --num_runs $NUM_RUNS --results_folder $RESULT_FOLDER/timing/parts/lsh_${ds}_dpsgd_n${n}_d${dp}_e${eps}"
            done
        done
    done
done
