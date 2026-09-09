import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from utils import _disjoint_pool_split_sample


GAS_PATH = os.path.join(os.path.dirname(__file__), "data", "Dataset")
GAS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "gas_cache.pkl")
GAS_BATCH_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "gas_cache_batches_v3.pkl")


def load_gas_dataset():
    if os.path.exists(GAS_CACHE_PATH):
        print("Loading Gas Sensor dataset from cache...")
        with open(GAS_CACHE_PATH, "rb") as f:
            X, y = pickle.load(f)
        print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        return X, y

    print("Reading Gas Sensor dataset from batch files...")
    all_X, all_y = [], []

    for batch_num in range(1, 11):
        batch_path = os.path.join(GAS_PATH, f"batch{batch_num}.dat")
        if not os.path.exists(batch_path):
            continue
        print(f"  Loading batch{batch_num}.dat...")
        with open(batch_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                label = int(float(parts[0])) - 1
                features = [float(p.split(":")[1]) for p in parts[1:] if ":" in p]
                all_X.append(features)
                all_y.append(label)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    print("Normalising features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    with open(GAS_CACHE_PATH, "wb") as f:
        pickle.dump((X, y), f)
    print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    return X, y


def load_gas_dataset_with_batches():
    if os.path.exists(GAS_BATCH_CACHE_PATH):
        print("Loading Gas Sensor dataset with batch info from cache (v3, reference-fit scaler)...")
        with open(GAS_BATCH_CACHE_PATH, "rb") as f:
            X, y, batch_ids = pickle.load(f)
        print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        return X, y, batch_ids

    print("Reading Gas Sensor dataset with batch info (v3)...")
    all_X, all_y, all_batch_ids = [], [], []

    for batch_num in range(1, 11):
        batch_path = os.path.join(GAS_PATH, f"batch{batch_num}.dat")
        if not os.path.exists(batch_path):
            continue
        print(f"  Loading batch{batch_num}.dat...")
        with open(batch_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                label = int(float(parts[0])) - 1
                features = [float(p.split(":")[1]) for p in parts[1:] if ":" in p]
                all_X.append(features)
                all_y.append(label)
                all_batch_ids.append(batch_num)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    batch_ids = np.array(all_batch_ids, dtype=np.int64)

    # V3 fix: fit the scaler on reference (early-batch, pre-drift) data only,
    # then transform every batch with that frozen scaler. The original
    # version fit on the entire dataset including late batches, exposing
    # preprocessing to future-batch statistics and partially removing the
    # sensor drift the temporal scenarios are supposed to be detecting.
    print("Normalising features (scaler fit on batches 1-3 only, applied to all batches)...")
    reference_mask = np.isin(batch_ids, (1, 2, 3))
    scaler = StandardScaler()
    scaler.fit(X[reference_mask])
    X = scaler.transform(X).astype(np.float32)

    with open(GAS_BATCH_CACHE_PATH, "wb") as f:
        pickle.dump((X, y, batch_ids), f)
    print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    return X, y, batch_ids


def partition_noniid_gas(X, y, num_clients=10, alpha=0.5):
    num_classes = len(np.unique(y))
    client_data = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        class_indices = np.where(y == c)[0]
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (proportions * len(class_indices)).astype(int)
        diff = len(class_indices) - proportions.sum()
        proportions[0] += diff
        start = 0
        for client_id, count in enumerate(proportions):
            end = start + count
            client_data[client_id].extend(class_indices[start:end].tolist())
            start = end

    return client_data


def get_gas_dataloader(X, y, indices, batch_size=32):
    X_client = torch.tensor(X[indices])
    y_client = torch.tensor(y[indices])
    dataset = TensorDataset(X_client, y_client)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_gas_dataloader_from_arrays(X_arr, y_arr, batch_size=32):
    """DataLoader directly from arrays , for sequential batch feeding."""
    X_client = torch.tensor(X_arr)
    y_client = torch.tensor(y_arr)
    dataset = TensorDataset(X_client, y_client)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def inject_gas_drift(X, y, indices, magnitude=1.0):
    """
    Scenario 1 , Label Shuffle: Artificial sudden concept drift.
    Simulates real drift P(Y|X) changes , decision boundary shifts.
    Standard FL research approach (Casado 2022).
    """
    X_drifted = X[indices].copy()
    y_drifted = y[indices].copy()
    np.random.shuffle(y_drifted)
    return X_drifted, y_drifted


def build_fixed_pool_split(pool, test_ratio=0.2, seed=0):
    """V3: one-time, seeded, disjoint train/test split of `pool`, sized
    proportionally to the pool itself. Call once at setup and reuse the
    returned indices for the whole run — a sample assigned to test can
    never later be drawn into train, since nothing re-rolls this split."""
    rng = np.random.RandomState(seed)
    pool = np.asarray(pool)
    if len(pool) == 0:
        return pool, pool
    shuffled = pool.copy()
    rng.shuffle(shuffled)
    if len(pool) < 2:
        return shuffled, shuffled
    n_test = min(max(1, int(len(pool) * test_ratio)), len(pool) - 1)
    test_idx = shuffled[:n_test]
    train_idx = shuffled[n_test:]
    return train_idx, test_idx


def client_size_weights(client_data, num_clients):
    """Each client's share of the overall (whole-dataset) non-IID
    partition. Used to replicate the same non-IID skew when re-partitioning
    within a single batch, without touching client_data itself (which
    Gas Label Shuffle also depends on and must stay untouched)."""
    sizes = np.array([len(client_data[cid]) for cid in range(num_clients)], dtype=float)
    total = sizes.sum()
    if total == 0:
        return np.full(num_clients, 1.0 / num_clients)
    return sizes / total


def client_class_weights(client_data, y, num_clients):
    """For each class, return the share assigned to every client by the
    original Dirichlet partition. These class-conditional weights preserve
    label-based Non-IID structure when temporal pools are rebuilt jointly
    inside each Gas batch."""
    classes = np.unique(y)
    weights = {}
    fallback = client_size_weights(client_data, num_clients)
    for cls in classes:
        counts = np.array([
            np.sum(y[np.asarray(client_data[cid], dtype=int)] == cls)
            for cid in range(num_clients)
        ], dtype=float)
        weights[int(cls)] = counts / counts.sum() if counts.sum() else fallback.copy()
    return weights


def partition_batch_with_floor(batch_indices, weights, num_clients, min_count, seed,
                               labels=None, class_weights=None):
    """V4 (client+batch-joint) design: split one batch's samples across
    clients proportionally to `weights` (each client's overall non-IID size
    share), guaranteeing every client at least `min_count` samples where the
    batch has enough total samples to support it.

    This replaces intersecting an independently-built non-IID partition with
    batch membership (which can produce empty or near-empty cells, verified
    to crash outright at seed 123) with a partition built jointly per batch,
    so temporal presence is guaranteed by construction rather than hoped for.
    When a batch is too small in absolute terms to give every client the
    target floor (some Gas batches have under 300 samples total across 10
    clients), this degrades gracefully to as-equal-a-split-as-possible
    instead of crashing or leaving some clients starved while others hoard
    the surplus — callers should check the returned counts against the
    target floor and report where it wasn't achievable, rather than assume
    it always was.
    """
    rng = np.random.RandomState(seed)
    pool = np.asarray(batch_indices).copy()
    rng.shuffle(pool)
    n = len(pool)

    if n == 0:
        return {cid: np.array([], dtype=pool.dtype) for cid in range(num_clients)}

    if labels is not None and class_weights is not None:
        assigned = {cid: [] for cid in range(num_clients)}
        for cls in np.unique(labels[pool]):
            cls_pool = pool[labels[pool] == cls].copy()
            rng.shuffle(cls_pool)
            cls_target = class_weights[int(cls)] * len(cls_pool)
            cls_counts = np.floor(cls_target).astype(int)
            remainder = len(cls_pool) - cls_counts.sum()
            if remainder:
                frac = cls_target - cls_counts
                cls_counts[np.argsort(-frac)[:remainder]] += 1
            start = 0
            for cid, count in enumerate(cls_counts):
                assigned[cid].extend(cls_pool[start:start + count].tolist())
                start += count

        desired_floor = min_count if n >= min_count * num_clients else n // num_clients
        while True:
            sizes = np.array([len(assigned[cid]) for cid in range(num_clients)])
            receivers = np.where(sizes < desired_floor)[0]
            donors = np.where(sizes > desired_floor)[0]
            if len(receivers) == 0 or len(donors) == 0:
                break
            receiver = int(receivers[np.argmin(sizes[receivers])])
            donor = int(donors[np.argmax(sizes[donors])])
            donor_items = assigned[donor]
            preferred = sorted(
                range(len(donor_items)),
                key=lambda i: class_weights[int(labels[donor_items[i]])][receiver],
                reverse=True,
            )
            assigned[receiver].append(donor_items.pop(preferred[0]))
        return {cid: np.asarray(assigned[cid], dtype=pool.dtype) for cid in range(num_clients)}

    if n < min_count * num_clients:
        # Batch too small to support the floor for everyone: split as
        # evenly as possible instead (still no client left at zero unless
        # n < num_clients itself).
        base = n // num_clients
        remainder = n - base * num_clients
        counts = np.full(num_clients, base, dtype=int)
        counts[:remainder] += 1
        rng.shuffle(counts)
    else:
        target = weights * n
        counts = np.floor(target).astype(int)
        shortfall = n - counts.sum()
        if shortfall > 0:
            frac = target - counts
            for idx in np.argsort(-frac)[:shortfall]:
                counts[idx] += 1

        deficit = np.clip(min_count - counts, 0, None)
        if deficit.sum() > 0:
            surplus_capacity = np.clip(counts - min_count, 0, None)
            surplus_total = surplus_capacity.sum()
            needed = deficit.sum()
            if surplus_total > 0:
                take = np.floor(surplus_capacity / surplus_total * needed).astype(int)
                take = np.minimum(take, surplus_capacity)
                counts = counts - take + deficit
                leftover = n - counts.sum()
                if leftover != 0:
                    counts[np.argmax(counts)] += leftover

    client_assignments = {}
    start = 0
    for cid in range(num_clients):
        cnt = max(0, int(counts[cid]))
        end = start + cnt
        client_assignments[cid] = pool[start:end]
        start = end
    return client_assignments


def print_pool_table(pools, num_clients, batches, title):
    """Transparency table of train/test sample counts per client per batch,
    printed before training so a too-sparse pool is visible up front rather
    than discovered later as a silent detection failure or a crash."""
    print(f"\n--- {title}: per-client / per-batch pool sizes (train/test) ---")
    header = "Client".ljust(8) + "".join(f"Batch {b}".rjust(12) for b in batches)
    print(header)
    for cid in range(num_clients):
        row = str(cid).ljust(8)
        for b in batches:
            p = pools[cid][b]
            row += f"{len(p['train'])}/{len(p['test'])}".rjust(12)
        print(row)


def build_sudden_batch_pools(client_data, batch_ids, num_clients, seed, y=None,
                              early_batches=(1, 2, 3), late_batches=(8, 9, 10),
                              test_ratio=0.2, min_count=20, verbose=True):
    """V4 (client+batch-joint) replacement for inject_batch_drift.
    Precomputes, once per client, fixed disjoint train/test pools for the
    pre-drift (early-batch) and post-drift (late-batch) periods of Gas
    Sudden Batch. Rather than intersecting each client's independently-built
    non-IID partition with batch membership (which produced cells as small
    as 2 samples, and crashed outright at seed 123 with an empty cell), the
    early-batch pool and late-batch pool are each re-partitioned jointly
    across all clients using their overall non-IID size weights, with a
    minimum sample floor per client (graceful, non-crashing degradation if
    a batch range is too small in absolute terms to support the floor for
    every client)."""
    weights = client_size_weights(client_data, num_clients)
    class_profile = client_class_weights(client_data, y, num_clients) if y is not None else None
    early_pool_all = np.where(np.isin(batch_ids, early_batches))[0]
    late_pool_all = np.where(np.isin(batch_ids, late_batches))[0]
    # min_count is meant as a floor on the TRAIN portion after the
    # train/test split, not the pre-split pool — inflate the target passed
    # to partition_batch_with_floor accordingly, so a client landing right
    # at the floor still has >= min_count training samples afterward.
    pre_split_floor = int(np.ceil(min_count / max(1 - test_ratio, 1e-6)))
    early_assign = partition_batch_with_floor(
        early_pool_all, weights, num_clients, pre_split_floor, seed, y, class_profile)
    late_assign = partition_batch_with_floor(
        late_pool_all, weights, num_clients, pre_split_floor, seed + 1, y, class_profile)

    pools = {}
    for cid in range(num_clients):
        early_train, early_test = build_fixed_pool_split(early_assign[cid], test_ratio, seed=seed * 1000 + cid)
        late_train, late_test = build_fixed_pool_split(late_assign[cid], test_ratio, seed=seed * 1000 + cid + 500)
        pools[cid] = {
            "pre_train": early_train, "pre_test": early_test,
            "post_train": late_train, "post_test": late_test,
        }
    if verbose:
        under = [cid for cid in range(num_clients)
                 if len(pools[cid]["pre_train"]) < min_count or len(pools[cid]["post_train"]) < min_count]
        print(f"Gas Sudden Batch pools: early_total={len(early_pool_all)}, late_total={len(late_pool_all)}, "
              f"target_floor(train)={min_count}, clients_below_floor={under if under else 'none'}")
    return pools


def build_sequential_batch_pools(client_data, batch_ids, num_clients, seed, y=None,
                                  num_batches=10, test_ratio=0.2, min_count=20, verbose=True):
    """V4 (client+batch-joint) replacement for get_sequential_batch_data.
    Precomputes, once per client per batch (1..num_batches), fixed disjoint
    train/test pools. Each batch is re-partitioned jointly across all
    clients using their overall non-IID size weights (same profile every
    batch, so heterogeneity is preserved), with a minimum sample floor per
    client-batch cell, rather than intersecting an independently-built
    partition with batch membership — which produced cells with as few as
    2 samples and, at seed 123, an empty cell that crashed training."""
    weights = client_size_weights(client_data, num_clients)
    class_profile = client_class_weights(client_data, y, num_clients) if y is not None else None
    # min_count is a floor on the TRAIN portion after the train/test split,
    # not the pre-split pool — inflate what's passed to
    # partition_batch_with_floor so a client at the floor still ends up
    # with >= min_count training samples once test_ratio is carved out.
    pre_split_floor = int(np.ceil(min_count / max(1 - test_ratio, 1e-6)))
    pools = {cid: {} for cid in range(num_clients)}
    below_floor = {}
    for b in range(1, num_batches + 1):
        batch_pool = np.where(batch_ids == b)[0]
        assign = partition_batch_with_floor(
            batch_pool, weights, num_clients, pre_split_floor, seed + b, y, class_profile)
        for cid in range(num_clients):
            train_idx, test_idx = build_fixed_pool_split(assign[cid], test_ratio, seed=seed * 1000 + cid * 100 + b)
            pools[cid][b] = {"train": train_idx, "test": test_idx}
        under = [cid for cid in range(num_clients) if len(pools[cid][b]["train"]) < min_count]
        if under:
            below_floor[b] = under
    if verbose:
        print(f"Gas Sequential Batch pools: target_floor(train)={min_count}, "
              f"batches_with_clients_below_floor={below_floor if below_floor else 'none'}")
        print_pool_table(pools, num_clients, list(range(1, num_batches + 1)), "Gas Sequential Batch")
    return pools
