import numpy as np
from tqdm import tqdm

from parties import MaskedClient, UnmaskedClient, Server
from data_io import to_fixed, unscale
from configs import Params
from utils import set_seed


def mpi_proto(value_lists, params: Params, method="masked"):
    """Implements the MPI-based protocol for federated clustering.
    
    This protocol uses Message Passing Interface (MPI) for distributed computation,
    enabling communication between multiple processes representing different clients
    and a central server. It supports both masked (privacy-preserving) and unmasked
    operations.
    
    The protocol follows these steps in each iteration:
    1. Clients compute local statistics (totals and counts)
    2. These statistics are gathered at the server
    3. Server processes the aggregated statistics (possibly adding DP noise)
    4. Results are broadcast back to all clients
    5. Clients update their local centroids
    
    Args:
        value_lists (list): List of numpy arrays, where each array contains the data
                           points for one client
        params (Params): Configuration parameters for the clustering algorithm
        method (str, optional): Either "masked" for privacy-preserving computation
                              or "unmasked" for standard computation. Defaults to "masked"
    
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Final cluster centroids after all iterations
            - int: Always 0 for MPI protocol (kept for compatibility with local_proto)
            
    Note:
        - The protocol uses the `data_io.comm` module for MPI operations
        - Progress is shown only on the server process using tqdm
        - Communication statistics are tracked and printed at the end
    """
    from data_io.comm import comm, fail_together
    set_seed(params.seed)
    comm.reset_comm_stats()
    comm.set_delay(params.delay)
    server_process = (comm.rank == comm.root)

    def initialize_server():
        """Initialize the server process with given parameters."""
        return Server(params)

    def initialize_client():
        """Initialize a client (masked or unmasked) with appropriate data."""
        cls = MaskedClient if method == "masked" else UnmaskedClient
        return cls(comm.rank - 1, value_lists[comm.rank - 1], params)

    if server_process:
        server = fail_together(initialize_server, "Server Initialization Failure")
    else:
        client = fail_together(initialize_client, "Client Initialization Failure")
    pbar = tqdm(range(params.iters)) if server_process else range(params.iters)

    for i in pbar:
        params.update_maxdist(i)
        if not server_process:
            # Client-side computation
            total, count, _ = client.step(params)

            # Pack statistics into a single array for efficient communication
            total_count = np.concatenate((total.flatten(), count.flatten()))
            comm.gather_delay(total_count, root=comm.root)

            # Receive and unpack aggregated statistics from server
            total_count = comm.bcast_delay(None, root=comm.root)

            total, count = np.split(total_count, [params.k * params.dim])
            total = total.reshape((params.k, params.dim))
            count = count.reshape(params.k)
            client.update(total, count)

        if server_process:
            # Server-side computation
            total_counts = comm.gather_delay(None, root=comm.root)
            total_counts = [np.split(tc, [params.k * params.dim]) for tc in total_counts[1:]]
            totals, counts = zip(*[(total.reshape((params.k, params.dim)), count.reshape(params.k))
                                   for total, count in total_counts])
            total, count = server.step(totals, counts, params)

            # Pack and broadcast updated statistics
            total_count = np.concatenate((total.flatten(), count.flatten()))
            comm.bcast_delay(total_count, root=comm.root)

        # Synchronize centroids across all processes
        centroids = comm.bcast(client.centroids if not server_process else None, root=1)

    comm.print_comm_stats()
    return to_fixed(centroids), 0


def local_proto(value_lists, params: Params, method="masked"):
    """Implements the local protocol for federated clustering.
    
    This protocol simulates federated clustering in a single process, useful for
    testing and development. It maintains separate client and server instances
    in memory and simulates their interaction. Like the MPI protocol, it supports
    both masked and unmasked computation.
    
    The protocol follows these steps in each iteration:
    1. Each client computes local statistics
    2. The server aggregates these statistics
    3. Clients update their centroids using the aggregated statistics
    4. Progress is tracked through centroid movement
    
    The implementation also tracks the number of unassigned points (points too
    far from any centroid)
    
    Args:
        value_lists (list): List of numpy arrays, where each array contains the data
                           points for one client
        params (Params): Configuration parameters for the clustering algorithm
        method (str, optional): Either "masked" for privacy-preserving computation
                              or "unmasked" for standard computation. Defaults to "masked"
    
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Final cluster centroids after all iterations
            - int: Number of points not assigned to any cluster in the final iteration
            
    Note:
        - Progress bar shows the Euclidean norm of centroid movement between iterations
        - All clients maintain identical centroids due to synchronized updates
        - A history of centroids is maintained but not returned
    """
    set_seed(params.seed)
    cls = MaskedClient if method == "masked" else UnmaskedClient
    clients = [
        cls(client, value_lists[client], params)
        for client in range(params.num_clients)
    ]
    centroids = clients[0].centroids
    centroid_history = [centroids]
    server = Server(params)
    pbar = tqdm(range(params.iters))
    unassigned_last_iter = 0

    for i in pbar:
        params.update_maxdist(i)
        # Collect statistics from all clients
        totals = []
        counts = []
        unassigneds = []
        for client in clients:
            total, count, unassigned = client.step(params)
            totals.append(total)
            counts.append(count)
            unassigneds.append(unassigned)
        unassigned_last_iter = sum(unassigneds)

        # Server processes aggregated statistics
        total, count = server.step(totals, counts, params)

        # Update all clients
        for client in clients:
            client.update(total, count)

        # Track progress through centroid movement
        err = np.linalg.norm(clients[0].centroids - centroids)
        pbar.set_description(str(err))
        centroids = clients[0].centroids
        centroid_history.append(centroids)

    return to_fixed(centroids), unassigned_last_iter


def lsh_proto(value_lists, params: Params, method="masked"):
    """Protocol adapter for DP LSH prefix-tree clustering.

    Hashes points with a SimHash basis (random / SVD-PCA / DP-SGD-PCA, via
    orthogonal_basis) and grows an LSH prefix tree (see utils/LSHTree.py), pruning any
    branch whose *noisy* point count falls below a threshold. Each surviving leaf
    yields one (noisy) centroid, so the number of clusters is data-dependent.

    Privacy accounting:
        delta         = 1 / (n log n)
        basis (dpsgd) = (basis_epsilon * eps, basis_epsilon * delta), composed
                        with the aggregation by basic composition on eps.
        aggregation   = ((1 - basis_epsilon) * eps, (1 - basis_epsilon) * delta),
                        accounted RIGOROUSLY in zero-concentrated DP (zCDP) via
                        compute_dp_sigmas_zcdp. The aggregation is one leaf-sum
                        release plus one count histogram per tree level; the
                        L = max_depth + 1 count releases compose sequentially in
                        zCDP (each level is a single sensitivity-1 mechanism by
                        parallel composition). count_levels uses the
                        data-independent bound max_depth + 1, since the realized
                        tree depth is itself privacy-sensitive.

    Only basis_method == "dpsgd_pca" spends basis budget; "random"/"svd_pca"
    leave the whole (eps, delta) for aggregation, and eps == 0 means exact
    (non-private) counts/centroids.

    Args:
        value_lists (list): list of numpy arrays (one per client).
        params (Params): uses d_prime, seed, eps, basis_method, basis_epsilon,
            sigma_fraction, basis_clip_norm, basis_data_fraction, tree_max_depth,
            min_count_in_node, min_count_to_branch, data_size, fixed.
        method (str, optional): unused, kept for protocol interface compatibility.

    Returns:
        tuple: (leaf_centroids, stats).
    """
    from utils.ortho_clustering import orthogonal_basis, compute_dp_sigmas_zcdp
    from utils.LSHTree import build_lsh_tree

    values = np.vstack(value_lists)
    if params.fixed:
        values = unscale(values)

    n = params.data_size
    delta = 1.0 / (n * np.log(n))

    # --- split the total budget: only dpsgd_pca pays for the basis (as in ortho) ---
    if params.basis_method == "dpsgd_pca":
        assert params.eps > 0, "dpsgd_pca basis needs eps > 0 to take a budget fraction"
        assert 0.0 < params.basis_epsilon < 1.0, \
            "basis_epsilon must be a fraction in (0, 1) for dpsgd_pca"
        eps_basis,  delta_basis = params.basis_epsilon * params.eps, params.basis_epsilon * delta
        eps_agg,    delta_agg   = params.eps - eps_basis, delta - delta_basis
    else:
        eps_basis, delta_basis = 0.0, 0.0
        eps_agg,   delta_agg   = params.eps, delta

    basis = orthogonal_basis(
        values, params.d_prime,
        method=params.basis_method, seed=params.seed,
        epsilon=eps_basis, delta=delta_basis,
        clip_norm=params.basis_clip_norm,
        data_fraction=params.basis_data_fraction,
    )

    # Worst-case tree depth (the tree can't go deeper than the basis is wide).
    max_depth = min(params.tree_max_depth or params.d_prime, basis.shape[1])

    # --- noise levels for counts (pruning) and leaf centroids; exact when eps == 0 ---
    # Rigorous zCDP: 1 leaf-sum release + (max_depth + 1) sequential count releases.
    if params.eps > 0:
        sigma_centers, sigma_count = compute_dp_sigmas_zcdp(
            eps_agg, delta_agg, params.sigma_fraction, count_levels=max_depth + 1
        )
    else:
        sigma_centers, sigma_count = 0.0, 0.0

    tree = build_lsh_tree(
        values, basis,
        max_depth=max_depth,
        min_count_to_branch=params.min_count_to_branch,
        min_count_in_node=params.min_count_in_node,
        count_sigma=sigma_count,
        base_seed=params.seed,
    )

    centers = tree.private_centers(center_sigma=sigma_centers)  # (num_leaves, d)

    leaf_counts = np.array([len(leaf.points) for leaf in tree.leaves])  # true, diagnostics
    leaf_depths = np.array([leaf.depth for leaf in tree.leaves])

    print(
        f" LSH tree (d'={params.d_prime}, max_depth={max_depth}, basis={params.basis_method}, "
        f"eps={params.eps}, delta={delta:.3g}, "
        f"eps_basis={eps_basis:.3f}, eps_agg={eps_agg:.3f}, "
        f"min_count_in_node={params.min_count_in_node}, "
        f"min_count_to_branch={params.min_count_to_branch}, "
        f"sigma_centers={sigma_centers:.4f}, sigma_count={sigma_count:.4f}): "
        f"leaves={len(tree.leaves)}, depth={leaf_depths.min()}-{leaf_depths.max()}, "
        f"covered={int(leaf_counts.sum())}/{len(values)}"
    )

    if params.fixed:
        centers = to_fixed(centers)

    return centers, {
        "unassigned": 0,
        "eps_basis": eps_basis,
        "eps_agg": eps_agg,
        "sigma_centers": sigma_centers,
        "sigma_count": sigma_count,
        "num_leaves": len(tree.leaves),
        "min_leaf_depth": int(leaf_depths.min()),
        "max_leaf_depth": int(leaf_depths.max()),
        "count_min": int(leaf_counts.min()),
        "count_max": int(leaf_counts.max()),
        "count_mean": float(leaf_counts.mean()),
        "points_covered": int(leaf_counts.sum()),
    }
 


# ---------------------------------------------------------------------------
# Federated LSH (MPI): mirrors mpi_proto's server/client + gather/bcast
# structure so its communication (rounds, bytes, simulated delay) is comparable
# to FastLloyd. Rank 0 = LshServer (blind), ranks 1..N = LshClient shards.
# Two-shot aggregation: one count round (tree + pruning) and one sum round
# (centroids), plus a tiny leaf-set exchange and a basis step.
# ---------------------------------------------------------------------------

def _encode_leaves(leaves):
    """Surviving leaf prefixes -> (L, 2) int array [(depth, value), ...] for the wire."""
    if not leaves:
        return np.empty((0, 2), dtype=np.int64)
    return np.array([[len(p), int(p, 2) if p else 0] for p in leaves], dtype=np.int64)


def _decode_leaves(arr):
    """Inverse of _encode_leaves."""
    return [format(int(v), f"0{int(L)}b") if L else "" for L, v in np.asarray(arr)]


def mpi_lsh_proto(value_lists, params: Params, method="masked"):
    """MPI federated LSH prefix-tree clustering (scalability counterpart to mpi_proto).

    Rank 0 runs a blind LshServer (only sees masked aggregates, adds DP noise);
    ranks 1..N run LshClient shards. Communication uses comm.*_delay so the round
    count, byte count, and simulated delay accrue exactly as for FastLloyd. The
    result is identical (up to float summation order) to the centralized
    lsh_proto under the same seed.

    Returns (centroids, stats) on every rank.
    """
    from data_io.comm import comm, fail_together
    from parties import LshServer, LshClient, MaskedLshClient
    from utils.LSHTree import node_prefixes

    set_seed(params.seed)
    comm.reset_comm_stats()
    comm.set_delay(params.delay)
    root = comm.root
    server_process = (comm.rank == root)

    if server_process:
        server = fail_together(lambda: LshServer(params), "LSH Server init failure")
    else:
        ClientCls = MaskedLshClient if method == "masked" else LshClient
        client = fail_together(
            lambda: ClientCls(comm.rank - 1, value_lists[comm.rank - 1], params),
            "LSH Client init failure")

    # --- Step 1: basis (built at the server from a pooled subsample) ----------
    if params.basis_method == "random":
        subsample = None  # no data needed; basis comes from the shared seed
    else:
        gathered = comm.gather_delay(None if server_process else client.subsample(), root)
        subsample = (np.vstack([g for g in gathered[1:] if g is not None])
                     if server_process else None)
    basis = server.build_basis(subsample) if server_process else None
    basis = comm.bcast_delay(basis if server_process else None, root)
    max_depth = min(params.tree_max_depth or params.d_prime, basis.shape[1])
    if not server_process:
        client.set_basis(basis)

    # --- Step 2: count round (tree + pruning) --------------------------------
    gathered = comm.gather_delay(
        None if server_process else client.masked_leaf_hist(), root)
    order = node_prefixes(max_depth)
    if server_process:
        masked_hist = sum(gathered[1:])
        noisy_nodes = server.noisy_node_counts(masked_hist)
        node_arr = np.array([noisy_nodes[p] for p in order])
    node_arr = comm.bcast_delay(node_arr if server_process else None, root)
    if not server_process:
        noisy_nodes = {p: node_arr[i] for i, p in enumerate(order)}
        leaves = client.prune(noisy_nodes, params.min_count_to_branch,
                              params.min_count_in_node, max_depth)

    # --- Tiny leaf-set exchange: tell the blind server which leaves to noise --
    gathered = comm.gather_delay(
        None if server_process else _encode_leaves(leaves), root)
    if server_process:
        leaves = _decode_leaves(next(g for g in gathered[1:] if g is not None))

    # --- Step 3: sum round (centroids) ---------------------------------------
    gathered = comm.gather_delay(
        None if server_process else client.masked_leaf_sums(leaves), root)
    if server_process:
        masked_sums = sum(gathered[1:])
        noisy_sums = server.noisy_leaf_sums(masked_sums, leaves)
    noisy_sums = comm.bcast_delay(noisy_sums if server_process else None, root)
    if not server_process:
        centroids = client.finalize(noisy_nodes, noisy_sums, leaves)

    # Share centroids to every rank (incl. the server), like mpi_proto.
    centroids = comm.bcast(centroids if not server_process else None, root=1)

    comm.print_comm_stats()
    return centroids, {"unassigned": 0, "num_leaves": len(leaves)}
