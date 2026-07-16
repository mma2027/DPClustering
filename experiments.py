"""
Experiments runner for multiparty DP clustering.
Handles experiment configuration, execution, and result collection.
"""

import hashlib
import itertools
import json
import os
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Dict, Any, Generator

# When launched under MPI (one rank per process, oversubscribed onto a single
# node), each rank's BLAS must stay single-threaded -- otherwise every rank
# spawns all-core OpenBLAS worker threads and they thrash. This is catastrophic
# for the DP-SGD basis (a loop of many tiny linear-algebra ops + QRs): on
# mnist784 (d=784) the W=784xd' matrices cross OpenBLAS's multithread threshold
# at d'>=15, turning a ~0.4s basis build into 30-60s. Must run BEFORE numpy is
# imported. Non-MPI runs (accuracy) are untouched and keep multithreaded BLAS.
if any(v in os.environ for v in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMI_RANK")):
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_var, "1")

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from configs import Params, exp_parameter_dict, num_clusters
from configs.defaults import accuracy_datasets, large_datasets
from data_io import shuffle_and_split, unscale, load_txt, txt_shape, ensure_unit_norm, to_fixed
from utils import evaluate, mean_confidence_interval, plot_clusters

# Above this many points, the O(n^2) metrics (silhouette, Dunn) are skipped —
# they would hang / OOM on large datasets (e.g. mnist784 70k, glove100 400k).
LARGE_N_EVAL = 20000


class ExperimentRunner:
    """Handles the execution and management of clustering experiments."""

    def __init__(
            self,
            protocol: callable,
            k: int,
            dataset: str,
            values: np.ndarray,
            params_list: Dict[str, Any],
            exp_type: str,
            results_folder: str,
            plot: bool = False,
            with_comm: bool = False,
            export_centroids: bool = False
    ):
        """
        Initialize the experiment runner.

        Args:
            protocol: The clustering protocol to run
            k: Number of clusters
            dataset: Name of the dataset
            values: Input data values
            params_list: Dictionary of experiment parameters
            exp_type: Type of experiment
            results_folder: Output directory for results
            plot: Whether to generate plots
            with_comm: Whether to use communication metrics
            export_centroids: Whether to dump the final centroids of every run to
                disk (see _export_centroids). Lets new point-to-centroid metrics
                (e.g. cosine similarity) be recomputed later without re-running.
        """
        self.results_folder = results_folder
        self.protocol = protocol
        self.k = k
        self.values = values
        self.dataset = dataset
        self.plot = plot
        self.with_comm = with_comm
        self.export_centroids = export_centroids
        self.comm = None
        if with_comm:
            from data_io.comm import comm
            self.comm = comm
        self.params_list = params_list
        self.exp_type = exp_type

        self.results_df = None
        self.failed_experiments = []
        # TODO: make eval metrics = nicv to save time
        if dataset in large_datasets:
            # Large datasets: NICV plus mean cosine similarity — the rest of the
            # suite (esp. O(n^2) silhouette / Dunn) is infeasible at this scale,
            # but cosine similarity is O(n*d) and memory-lean (chunked), so it is
            # cheap enough to compute alongside NICV.
            self.eval_metrics = ["nicv", "cosine_similarity"]
        elif exp_type == "accuracy" or dataset in accuracy_datasets:
            self.eval_metrics = "all"
        else:
            self.eval_metrics = "nicv"

        # Ground-truth KMeans centroids are only needed for the MSE metric.
        # Computed lazily so timing / NICV-only / large-dataset runs (which never
        # use MSE) don't pay an expensive full-data KMeans on every MPI rank.
        self._centroids_gt = None

    def gt_centroids(self):
        """Lazily compute (and cache) the KMeans ground-truth centroids."""
        if self._centroids_gt is None:
            values_unscaled = unscale(self.values.copy())
            self._centroids_gt = KMeans(n_clusters=self.k).fit(values_unscaled).cluster_centers_
        return self._centroids_gt

    def run_single_protocol(self, params: Params) -> Dict[str, float]:
        """
        Run a single instance of the clustering protocol.

        Args:
            params: Parameters for this protocol run

        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare data. The MPI server rank holds no data (see process_dataset), so it
        # skips the shard split and the (data-dependent) evaluation below; it still runs
        # the protocol and reports the timing (elapsed + phase stats read by the plots).
        server_rank = self.comm is not None and self.comm.rank == self.comm.root
        proportions = np.ones(params.num_clients) / params.num_clients
        value_lists = ([None] * params.num_clients if server_rank
                       else shuffle_and_split(self.values, params.num_clients, proportions))

        # Run protocol and time it
        start = timer()
        centroids, unassigned = self.protocol(value_lists, params)
        elapsed_time = timer() - start

        if server_rank:
            # No data on the server -> no NICV; keep wall-time + protocol phase stats.
            metrics = {"elapsed": elapsed_time}
            if isinstance(unassigned, dict):
                metrics.update(unassigned)
            else:
                metrics["unassigned"] = unassigned
            return metrics

        # Handle scaling
        values_unscaled = unscale(self.values.copy()) if params.fixed else self.values
        centroids_final = unscale(centroids) if params.fixed else centroids

        # Optionally persist the final centroids so future point-to-centroid
        # metrics can be recomputed offline (see _export_centroids).
        if self.export_centroids:
            self._export_centroids(centroids_final, params)

        # Evaluate results. Expand "all" to an explicit list so we can drop:
        #   - MSE          when centroid counts differ from the ground truth, and
        #   - silhouette / dunn_index  for large n (both are O(n^2) and would
        #     hang / OOM on datasets like mnist784 (70k) or glove100 (400k)).
        eval_metrics = self.eval_metrics
        if eval_metrics == "all":
            eval_metrics = ["nicv", "bcss", "empty_clusters", "silhouette",
                            "davies_bouldin", "calinski_harabasz", "dunn_index", "mse",
                            "cosine_similarity"]
            if centroids_final.shape[0] != self.k:   # MSE needs matching centroid count
                eval_metrics.remove("mse")
            if self.values.shape[0] > LARGE_N_EVAL:
                eval_metrics = [m for m in eval_metrics
                                if m not in ("silhouette", "dunn_index")]
        # only compute the (expensive) ground truth if MSE is actually evaluated
        gt = self.gt_centroids() if "mse" in eval_metrics else None
        metrics = evaluate(centroids_final, values_unscaled, gt, eval_metrics)
        metrics["elapsed"] = elapsed_time
        if isinstance(unassigned, dict):
            metrics.update(unassigned)
        else:
            metrics["unassigned"] = unassigned

        # Generate plots if requested
        if self.plot:
            self._generate_plot(centroids_final, values_unscaled, params)

        return metrics

    def _generate_plot(self, centroids: np.ndarray, values: np.ndarray, params: Params) -> None:
        """Generate and save clustering visualization."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (f"{timestamp}_{params.method}_{params.dp}_{params.eps}_"
                    f"[{params.post}-{params.alpha}]_{params.seed}")

        plot_clusters(centroids, values)
        plt.title(filename)

        folder = Path("results") / self.dataset / self.protocol.__name__
        folder.mkdir(parents=True, exist_ok=True)

        plt.savefig(folder / f"{filename}.png")
        plt.close()

    def _export_centroids(self, centroids: np.ndarray, params: Params) -> None:
        """Persist the final centroids of a single run to disk.

        Enabled by --export_centroids. Writes, under

            <results_folder>/<exp_type>/<dataset>/centroids/

        one ``.npy`` per run holding the (n_clusters, dim) final centroid array,
        alongside a same-stem ``.json`` sidecar recording the full parameter set
        (protocol, basis, d', eps, seed, ...). Both share a stem of the form

            <protocol>_<basis>_d<d'>_eps<eps>_seed<seed>_<hash8>

        where ``hash8`` is a short digest of the complete parameter set, so the
        stem is unique even when two configs share the human-readable fields.

        Why: the centroids are otherwise computed in memory and discarded after
        evaluation, so adding a new point-to-centroid metric (e.g. cosine
        similarity) requires re-running the whole sweep. With the centroids
        saved, any such metric can be recomputed offline by loading the dataset
        (data/<dataset>.txt, unit-normed the same way) and the ``.npy``, then
        calling utils.evaluate — no clustering re-run needed.

        In an MPI run the global centroids are identical across client ranks, so
        only the first client rank (root + 1) writes; the server rank never
        reaches this method (it returns early, holding no data).
        """
        if self.with_comm and self.comm.rank != self.comm.root + 1:
            return

        folder = Path(self.results_folder) / self.exp_type / self.dataset / "centroids"
        folder.mkdir(parents=True, exist_ok=True)

        meta = {attr: getattr(params, attr) for attr in vars(params)}
        meta.pop("attributes", None)
        meta["protocol"] = self.protocol.__name__
        meta["dataset"] = self.dataset

        digest = hashlib.md5(
            json.dumps(meta, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]
        stem = (f"{self.protocol.__name__}_{params.basis_method}_"
                f"d{params.d_prime}_eps{params.eps}_seed{params.seed}_{digest}")

        np.save(folder / f"{stem}.npy", np.asarray(centroids))
        with open(folder / f"{stem}.json", "w") as f:
            json.dump(meta, f, default=str, indent=2)

    def _get_parameter_combinations(self) -> Generator[Params, None, None]:
        """Generate all parameter combinations for experiments."""
        params_order = ["methods", "posts", "delays", "dps"]
        dimension, data_size = self.values.shape[1], self.values.shape[0]

        d_primes = self.params_list.get("d_primes", [None])
        sigma_fractions = self.params_list.get("sigma_fraction", [10.0])
        tree_max_depth = self.params_list.get("tree_max_depth", 0)
        tree_min_count = self.params_list.get("tree_min_count", 0.0)
        basis_methods = self.params_list.get("basis_methods", ["random"])
        basis_epsilons = self.params_list.get("basis_epsilons", [0.0])
        basis_clip_norms = self.params_list.get("basis_clip_norms", [1.0])
        basis_data_fractions = self.params_list.get("basis_data_fractions", [0.1])
        basis_lr = self.params_list.get("basis_lr", 0.1)              # scalar (not swept)
        basis_epochs = self.params_list.get("basis_epochs", 10)       # scalar (not swept)

        for method, post, delay, dp in itertools.product(
                *[self.params_list[key] for key in params_order]
        ):
            for eps_budget in self._get_eps_budgets(dp):
                for d_prime in d_primes:
                    for sigma_fraction in sigma_fractions:
                        for basis_method, basis_epsilon, basis_clip_norm, basis_data_fraction in itertools.product(
                                basis_methods, basis_epsilons, basis_clip_norms, basis_data_fractions
                        ):
                            params = Params(
                                num_clients=self.params_list["num_clients"],
                                k=self.k,
                                dim=dimension,
                                data_size=data_size,
                                dp=dp,
                                eps=eps_budget,
                                method=method,
                                post=post,
                                delay=delay,
                            )
                            if d_prime is not None:
                                params.d_prime = d_prime
                            params.sigma_fraction = sigma_fraction
                            params.tree_max_depth = tree_max_depth
                            params.min_count_in_node = tree_min_count
                            params.min_count_to_branch = tree_min_count
                            params.basis_method = basis_method
                            params.basis_epsilon = basis_epsilon
                            params.basis_clip_norm = basis_clip_norm
                            params.basis_data_fraction = basis_data_fraction
                            params.basis_lr = basis_lr
                            params.basis_epochs = basis_epochs

                            if method == "none":
                                params.alpha = 0
                                yield params
                            else:
                                for alpha in self.params_list["alphas"]:
                                    params.alpha = alpha
                                    yield params

    def _get_eps_budgets(self, dp: str) -> List[float]:
        """Get epsilon budgets based on privacy setting."""
        # The LSH protocol always sweeps the eps budgets, since its privacy comes
        # from eps directly rather than from `dp`.
        is_lsh = self.protocol.__name__ in ("lsh_proto", "mpi_lsh_proto")
        return [0] if dp == "none" and not is_lsh else self.params_list["eps_budgets"]

    def run_experiment(self, params: Params) -> None:
        """Run experiment with given parameters multiple times."""
        params.calculate_iters()
        total_metrics = defaultdict(list)
        successful_experiments = experiment_count = 0

        # Show current config
        if self.protocol.__name__ in ("lsh_proto", "mpi_lsh_proto"):
            max_depth = params.tree_max_depth or params.d_prime
            config_desc = (f"d_prime={params.d_prime}, max_depth={max_depth}, "
                           f"eps={params.eps}, min_count={params.min_count_in_node}")
        else:
            config_desc = f"dp={params.dp}, method={params.method}, eps={params.eps}, iters={params.iters}"
        print(f"\n[{self.dataset}] {self.protocol.__name__} | {config_desc} | k={params.k}")

        # Run multiple times with different seeds
        for seed in tqdm(self.params_list["seeds"], desc="  seeds", leave=False):
            params.seed = seed
            try:
                metrics = self.run_single_protocol(params)

                for metric, value in metrics.items():
                    total_metrics[metric].append(value)

                failed = any(
                    isinstance(value, float) and np.isnan(value)
                    for value in metrics.values()
                )
                successful_experiments += 1 if not failed else 0
                experiment_count += 1

            except Exception as e:
                print(f"\nExperiment failed (seed {seed}): {str(e)}")
                self.failed_experiments.append(vars(params))
                continue

        # Process and save results
        self._process_and_save_results(
            params, total_metrics, successful_experiments, experiment_count
        )

    def _process_and_save_results(
            self,
            params: Params,
            total_metrics: Dict[str, List[float]],
            successful_experiments: int,
            experiment_count: int
    ) -> None:
        """Process experiment results and save to DataFrame."""
        # Calculate statistics
        metric_stats = {
            metric: mean_confidence_interval(values)
            for metric, values in total_metrics.items()
        }

        # Prepare results dictionary
        result = {
            "protocol": self.protocol.__name__,
            **{attr: getattr(params, attr) for attr in vars(params)},
            "successes": successful_experiments,
            "experiments": experiment_count,
            "post_method": params.post,
            **{metric: stats[0] for metric, stats in metric_stats.items()},
            **{f"{metric}_h": stats[1] for metric, stats in metric_stats.items()}
        }

        # Remove unnecessary attributes
        result.pop("attributes", None)

        # Add communication stats if needed
        if self.with_comm:
            result.update(self.comm.get_comm_stats())

        # Update DataFrame
        new_results = pd.DataFrame([result])
        self.results_df = (
            pd.concat([self.results_df, new_results], ignore_index=True)
            if self.results_df is not None
            else new_results
        )

        self._save_results()

    def _save_results(self) -> None:
        """Save results to files."""
        folder = Path(self.results_folder) / self.exp_type / self.dataset
        folder.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to {folder}")
        # Sort and save results. The MPI server rank has no NICV column (it holds no data
        # and skips evaluation), so only sort when the column is present.
        if self.results_df is not None:
            nicv_col = "Normalized Intra-cluster Variance (NICV)"
            if nicv_col in self.results_df.columns:
                self.results_df = self.results_df.sort_values(nicv_col)

            # Determine filename based on protocol
            proto_name = self.protocol.__name__
            if proto_name != "local_proto":
                filename = f"variances_{proto_name.replace('_proto', '')}.csv"
            else:
                filename = "variances.csv"
            if self.with_comm:
                rank_str = f"_{self.comm.rank}" if self.comm.world_size > 1 else ""
                filename = f"variances{rank_str}.csv"

            self.results_df.to_csv(folder / filename)

        # Save failed experiments
        with open(folder / "failed.json", "w") as f:
            json.dump(self.failed_experiments, f, default=str)

    def run(self) -> None:
        """Run all experiments with different parameter combinations."""
        for params in self._get_parameter_combinations():
            self.run_experiment(params)


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(description="Run experiments for multiparty DP clustering")
    parser.add_argument("--exp_type", default="test", help="type of experiment")
    parser.add_argument("--datasets", nargs="+", default=accuracy_datasets, help="datasets to run")  # changed default from ["mnist"] to accuracy_datasets
    parser.add_argument("--plot", action="store_true", help="plot clusters")
    parser.add_argument(
        "--export_centroids",
        action="store_true",
        help="dump the final centroids of every run to "
             "<results_folder>/<exp_type>/<dataset>/centroids/ (one .npy + .json "
             "sidecar per run). Lets new point-to-centroid metrics (e.g. cosine "
             "similarity) be recomputed offline without re-running the sweep."
    )
    parser.add_argument("--num_runs", default=10, type=int, help="number of runs")
    parser.add_argument(
        "--eps_budgets",
        nargs="+",
        type=float,
        default=None,
        help="override the experiment's epsilon budgets (e.g. 0.5 1 2 4)"
    )
    parser.add_argument(
        "--method",
        default="diagonal_then_frac",
        choices=["none", "diagonal_then_frac", "stay_frac"],
        help="maxdist method"
    )
    parser.add_argument("--alpha", default=0.8, type=float, help="max distance alpha")
    parser.add_argument(
        "--post",
        default="fold",
        choices=["none", "truncate", "fold"],
        help="centroid post-processing method"
    )
    parser.add_argument(
        "--results_folder",
        default="submission",
        help="folder for results"
    )
    parser.add_argument(
        "--protocol",
        default="local",
        choices=["local", "lsh"],
        help="clustering protocol to use"
    )
    parser.add_argument(
        "--d_primes",
        nargs="+",
        type=int,
        default=None,
        help="d_prime values to sweep (lsh protocol only)"
    )
    parser.add_argument(
        "--basis_method",
        default=["dpsgd_pca"],
        nargs="+",
        choices=["random", "dpsgd_pca", "svd_pca"],
        help="basis generation method(s) for lsh protocol (space-separated, e.g. random svd_pca dpsgd_pca)"
    )
    parser.add_argument(
        "--d_prime",
        default=None,
        type=int,
        nargs="+",
        help="d_prime value(s) for dpsgd_pca mode (default: 1 2 3 4 5); ignored when --basis_method random"
    )
    parser.add_argument(
        "--basis_epsilon",
        default=0.1,
        type=float,
        help="FRACTION of the total (eps, delta) budget spent on the DP-SGD PCA basis "
             "(dpsgd_pca only); the remaining 1 - basis_epsilon goes to the aggregation"
    )
    parser.add_argument(
        "--basis_clip_norm",
        default=1.0,
        type=float,
        help="per-sample gradient clipping norm for DP-SGD PCA basis"
    )
    parser.add_argument(
        "--basis_lr",
        default=0.1,
        type=float,
        help="DP-SGD PCA learning rate (dpsgd_pca only; 0.01 was init-stuck, 0.1 converges)"
    )
    parser.add_argument(
        "--basis_epochs",
        default=10,
        type=int,
        help="DP-SGD PCA epochs (dpsgd_pca only). More epochs = more SGD steps = larger "
             "calibrated sigma, so the noise-vs-optimization optimum is INTERIOR and small: "
             "~10 at tight eps (mnist784, d=784) up to ~20 at looser eps; 40 over-noises and "
             "roughly halves basis EVR at eps=0.5. glove100 (small d, flat spectrum) is "
             "insensitive. Default 10."
    )
    parser.add_argument(
        "--basis_data_fraction",
        default=0.1,
        type=float,
        help="fraction of data to subsample before running DP-SGD PCA (default: 0.1)"
    )
    parser.add_argument(
        "--tree_max_depth",
        default=0,
        type=int,
        help="max LSH-tree depth for --protocol lsh (0 -> use d_prime)"
    )
    parser.add_argument(
        "--tree_min_count",
        default=0.0,
        type=float,
        help="noisy-count pruning threshold for --protocol lsh; a branch with "
             "fewer noisy points is pruned and not branched (default: 0 -> no pruning)"
    )
    return parser.parse_args()


def process_dataset(
        dataset: str,
        proto: callable,
        params_list: Dict[str, Any],
        fixed: bool,
        exp_type: str,
        results_folder: str,
        plot: bool,
        with_comm: bool,
        export_centroids: bool = False
) -> None:
    """Process a single dataset with given parameters."""
    # Determine number of clusters
    k = int(dataset.split("_")[1]) if "synth" in dataset.lower() else num_clusters[dataset]

    # Load and prepare dataset
    dataset_file = Path("data") / f"{dataset}.txt"
    if not dataset_file.is_file():
        return

    # Skip the data load on the MPI server (rank 0): in the federated timing path the
    # server only aggregates gathered messages and never uses its own shard, so it need
    # not hold the full dataset. With every rank oversubscribed onto one host, 9 full
    # copies at n=8 are what OOM large-d runs (glove300); dropping the server's copy is a
    # cheap stopgap. The server still needs the (n, d) SHAPE for params.data_size / dim
    # (=> delta and noise calibration), which txt_shape gets without allocating the array.
    server_rank = False
    if with_comm:
        from data_io.comm import comm
        server_rank = (comm.rank == comm.root)
    if server_rank:
        n, d = txt_shape(str(dataset_file))
        values = np.broadcast_to(np.float64(0.0), (n, d))   # O(1) memory, correct shape
    else:
        values = load_txt(str(dataset_file))
        # Files are written final (min-max + unit-norm) by scripts/download_data.py, so
        # load them as-is. ensure_unit_norm is the idempotent safeguard guaranteeing the
        # DP aggregation's L2 sensitivity-1 precondition: a no-op on prepared data, it
        # still rescales any legacy/un-prepared file. This runs OUTSIDE the timed run().
        values = ensure_unit_norm(values)
        if fixed:
            values = to_fixed(values)

    # Run experiments
    experiment = ExperimentRunner(
        proto, k, dataset, values, params_list,
        exp_type, results_folder, plot, with_comm, export_centroids
    )
    try:
        experiment.run()
    except Exception as e:
        import traceback
        print(f"\n[{dataset}] FATAL ERROR: {e}")
        traceback.print_exc()


def _configure_lsh(params_list: Dict[str, Any], args: Namespace) -> None:
    """Apply the LSH-protocol parameter sweep.

    Shared by the timing/MPI path (mpi_lsh_proto) and the centralized path
    (lsh_proto) so the two stay in sync. Does not touch num_clients or
    eps_budgets (set by the caller / experiment config).
    """
    params_list["tree_max_depth"] = args.tree_max_depth
    params_list["tree_min_count"] = args.tree_min_count
    # LSH doesn't use DP/method/post — collapse to single "none" values
    params_list.update({"dps": ["none"], "methods": ["none"], "posts": ["none"]})
    if args.d_primes is not None:
        params_list["d_primes"] = args.d_primes
    else:
        params_list.setdefault("d_primes", [1, 2, 3, 4, 5])
    params_list["basis_methods"] = args.basis_method
    params_list["basis_epsilons"] = [args.basis_epsilon]
    params_list["basis_clip_norms"] = [args.basis_clip_norm]
    params_list["basis_data_fractions"] = [args.basis_data_fraction]
    params_list["basis_lr"] = args.basis_lr            # scalar (not swept)
    params_list["basis_epochs"] = args.basis_epochs    # scalar (not swept)


def main() -> None:
    """Main entry point for running experiments."""
    args = parse_args()
    fixed = True  # Always use fixed-point
    exp_type = args.exp_type

    # Set up default parameters
    params_list = {
        "num_runs": args.num_runs,
        "seeds": range(args.num_runs),
        "posts": [args.post],
        "methods": [args.method],
        "alphas": [args.alpha],
        "datasets": args.datasets,
        "dps": ["none", "gaussiananalytic"],
        "delays": [0],
        "fixed": fixed,
    }

    # Override parameters if needed
    if exp_type in exp_parameter_dict:
        params_list.update(exp_parameter_dict[exp_type])
        params_list["datasets"] = args.datasets  # re-apply --datasets after update, since exp_parameter_dict entries override it
    if args.eps_budgets is not None:
        params_list["eps_budgets"] = args.eps_budgets  # CLI override of the epsilon sweep

    # Set up protocol and communication
    if "timing" in exp_type:
        from data_io.comm import comm
        with_comm = True
        num_clients = comm.world_size - 1
        params_list["num_clients"] = num_clients
        exp_type = f"timing_{num_clients}"
        if args.protocol == "lsh":
            from utils.protocols import mpi_lsh_proto
            proto = mpi_lsh_proto
            _configure_lsh(params_list, args)
        else:
            from utils.protocols import mpi_proto
            proto = mpi_proto
    elif args.protocol == "lsh":
        from utils.protocols import lsh_proto
        proto = lsh_proto
        with_comm = False
        params_list["num_clients"] = 2
        _configure_lsh(params_list, args)
    else:
        from utils.protocols import local_proto
        proto = local_proto
        with_comm = False
        params_list["num_clients"] = 2

    # Run experiments in parallel
    max_processes = min(max(os.cpu_count() - 16, 1), len(params_list["datasets"]))
    if "timing" in exp_type:
        max_processes = 1
    if max_processes > 1:
        print(f"Running {max_processes} processes in parallel")
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            partial_fn = partial(
                process_dataset,
                proto=proto,
                params_list=params_list,
                fixed=fixed,
                exp_type=exp_type,
                results_folder=args.results_folder,
                plot=args.plot,
                with_comm=with_comm,
                export_centroids=args.export_centroids
            )
            executor.map(partial_fn, params_list["datasets"])
    else:
        for dataset in params_list["datasets"]:
            process_dataset(
                dataset,
                proto,
                params_list,
                fixed,
                exp_type,
                args.results_folder,
                args.plot,
                with_comm,
                args.export_centroids
            )


if __name__ == "__main__":
    main()
