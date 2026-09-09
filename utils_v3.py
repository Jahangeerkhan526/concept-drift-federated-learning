import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import random

# Path to UCI HAR dataset
HAR_PATH = os.path.join(os.path.dirname(__file__), "data", "UCI HAR Dataset")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "har_cache.pkl")
HAR_SUBJECT_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "har_cache_subjects.pkl")

# UCI HAR Activity labels
ACTIVITY_LABELS = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}


def load_har_dataset():
    """Load UCI HAR dataset from local folder."""
    if os.path.exists(CACHE_PATH):
        print("Loading UCI HAR from cache...")
        with open(CACHE_PATH, "rb") as f:
            X, y = pickle.load(f)
        print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        return X, y

    print("Reading UCI HAR dataset from folder...")
    X_train = np.loadtxt(os.path.join(HAR_PATH, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(HAR_PATH, "train", "y_train.txt"), dtype=int)
    X_test = np.loadtxt(os.path.join(HAR_PATH, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(HAR_PATH, "test", "y_test.txt"), dtype=int)

    X = np.vstack([X_train, X_test]).astype(np.float32)
    y = np.concatenate([y_train, y_test]).astype(np.int64)
    y = y - 1

    with open(CACHE_PATH, "wb") as f:
        pickle.dump((X, y), f)
    print(f"Cache saved to {CACHE_PATH}")
    print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    return X, y


def load_har_dataset_with_subjects():
    """Load UCI HAR dataset with subject IDs for subject-based drift."""
    if os.path.exists(HAR_SUBJECT_CACHE_PATH):
        print("Loading UCI HAR from cache (with subjects)...")
        with open(HAR_SUBJECT_CACHE_PATH, "rb") as f:
            X, y, subjects = pickle.load(f)
        print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(subjects))} subjects")
        return X, y, subjects

    print("Reading UCI HAR dataset with subject info...")
    X_train = np.loadtxt(os.path.join(HAR_PATH, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(HAR_PATH, "train", "y_train.txt"), dtype=int)
    subjects_train = np.loadtxt(os.path.join(HAR_PATH, "train", "subject_train.txt"), dtype=int)

    X_test = np.loadtxt(os.path.join(HAR_PATH, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(HAR_PATH, "test", "y_test.txt"), dtype=int)
    subjects_test = np.loadtxt(os.path.join(HAR_PATH, "test", "subject_test.txt"), dtype=int)

    X = np.vstack([X_train, X_test]).astype(np.float32)
    y = np.concatenate([y_train, y_test]).astype(np.int64)
    y = y - 1
    subjects = np.concatenate([subjects_train, subjects_test]).astype(np.int64)

    with open(HAR_SUBJECT_CACHE_PATH, "wb") as f:
        pickle.dump((X, y, subjects), f)
    print(f"Cache saved to {HAR_SUBJECT_CACHE_PATH}")
    print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(subjects))} subjects")
    return X, y, subjects


def partition_noniid(X, y, num_clients=10, alpha=0.5):
    """Split data across clients using Dirichlet distribution."""
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


def get_client_dataloader(X, y, indices, batch_size=32):
    """Create a PyTorch DataLoader for a specific client."""
    X_client = torch.tensor(X[indices])
    y_client = torch.tensor(y[indices])
    dataset = TensorDataset(X_client, y_client)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def inject_concept_drift(X, y, indices, drift_type="sudden", magnitude=1.0):
    """
    Scenario 1 , Label Shuffle: Artificial sudden concept drift.
    Simulates real drift P(Y|X) changes , decision boundary shifts.
    Standard FL research approach (Casado 2022).
    """
    X_drifted = X[indices].copy()
    y_drifted = y[indices].copy()

    if drift_type == "sudden":
        np.random.shuffle(y_drifted)

    return X_drifted, y_drifted


def _disjoint_pool_split_sample(pool, train_size, test_size):
    """Split `pool` into disjoint train/test sub-pools before sampling, so a
    with-replacement draw (needed when the pool is smaller than what's asked
    for) can never place the same underlying sample in both train and test —
    which a single combined draw sliced afterward could do. Shared by
    utils.py's inject_activity_drift and utils_gas.py's batch/sequential
    drift sampling."""
    pool = np.asarray(pool)
    if len(pool) == 0:
        return pool, pool
    shuffled = pool.copy()
    np.random.shuffle(shuffled)
    total = max(train_size + test_size, 1)
    split = int(len(shuffled) * train_size / total)
    split = min(max(split, 1), max(len(shuffled) - 1, 1)) if len(shuffled) > 1 else len(shuffled)
    train_pool, test_pool = shuffled[:split], shuffled[split:]
    if len(test_pool) == 0:
        test_pool = train_pool
    train_sampled = np.random.choice(train_pool, size=train_size, replace=len(train_pool) < train_size)
    test_sampled = np.random.choice(test_pool, size=test_size, replace=len(test_pool) < test_size)
    return train_sampled, test_sampled


def inject_activity_drift(X, y, train_indices, test_indices, seed=0,
                           target_activities=(1, 2), target_ratio=0.5):
    """
    Scenario 2 , Activity Distribution Drift for UCI HAR.
    Shift client data distribution toward specific activities.

    Real world: user behaviour changes over time ,
    e.g. more stair climbing, less sitting.

    Default: shift toward Walking Upstairs (1) and
    Walking Downstairs (2) , more dynamic activities.

    V4 fix: the V3 fix (restricting sampling to the client's own partition)
    exposed a second problem — a fixed 0.8 target_ratio applied to
    train_size/test_size derived from the client's FULL partition routinely
    asked for far more target-activity samples than a client's own partition
    actually has (e.g. one client needed ~500 but had 146 available),
    forcing 3-5x with-replacement duplication that diluted the actual
    distributional shift into mostly-repeated samples.

    V5 also preserves the client's original held-out boundary permanently:
    post-drift training samples are selected only from the original training
    split, and post-drift evaluation samples only from the original test
    split. This prevents a formerly held-out sample from later entering
    training after drift.

    Each split is sized from what the client actually has:
    every one of the client's own target-activity samples, plus a matching
    *unique, non-duplicated* downsample of non-target samples sized so the
    combined pool sits at `target_ratio` (default 0.5 — a client with 146
    target samples gets a clean 146/146 = 292-sample pool, a real 23%->50%
    shift with zero duplication). Train/test is a single fixed, seeded,
    non-overlapping split of that pool — computed once, never resampled.

    Returns (X_train, y_train, X_test, y_test).
    """
    def select_unique(split_indices, split_seed):
        split_indices = np.asarray(split_indices, dtype=int)
        if len(split_indices) == 0:
            return split_indices
        local_rng = np.random.RandomState(split_seed)
        target = split_indices[np.isin(y[split_indices], target_activities)].copy()
        remaining = split_indices[~np.isin(y[split_indices], target_activities)].copy()
        local_rng.shuffle(target)
        local_rng.shuffle(remaining)
        if len(target) == 0:
            return split_indices.copy()
        wanted_remaining = int(len(target) * (1 - target_ratio) / target_ratio) if target_ratio > 0 else 0
        selected = np.concatenate([target, remaining[:min(wanted_remaining, len(remaining))]])
        local_rng.shuffle(selected)
        return selected

    selected_train = select_unique(train_indices, seed)
    selected_test = select_unique(test_indices, seed + 1)
    return (X[selected_train].astype(np.float32), y[selected_train].copy(),
            X[selected_test].astype(np.float32), y[selected_test].copy())


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    return np.dot(v1, v2) / (norm1 * norm2)


def daaw_detect_drift(gradient_history, short_window=5, long_window=20, threshold=0.3):
    """
    DAAW , Double sliding window cosine similarity drift detection.
    Short window (5 rounds)  , detects sudden drift
    Long window  (20 rounds) , detects gradual drift
    """
    if len(gradient_history) < short_window + 1:
        return False, 1.0

    short = gradient_history[-short_window:]
    short_avg = np.mean(short, axis=0)

    long_end = max(0, len(gradient_history) - short_window)
    long_start = max(0, long_end - long_window)
    long = gradient_history[long_start:long_end]

    if len(long) == 0:
        return False, 1.0

    long_avg = np.mean(long, axis=0)
    similarity = cosine_similarity(short_avg, long_avg)
    drift_detected = similarity < threshold
    return drift_detected, similarity


def held_out_split(num_clients, client_data, seed, test_ratio=0.2):
    """Fixed train/test position split per client, seeded by seed+cid so it's
    stable across rounds within a run. Positions index into client_data[cid]
    and equally into any drift-injected array of the same length. Shared by
    server.py and server_gas.py so the split logic can't drift out of sync
    between the two pipelines.

    Clients with fewer than 2 samples can't support a real split (n_test
    would either be 0 or leave 0 training samples) — they fall back to using
    all their samples for both train and eval rather than crashing or
    silently training on nothing.
    """
    train_pos, test_pos = {}, {}
    for cid in range(num_clients):
        n = len(client_data[cid])
        rng = np.random.RandomState(seed * 1000 + cid)
        perm = rng.permutation(n)
        if n < 2:
            train_pos[cid] = perm
            test_pos[cid] = perm
            continue
        n_test = min(max(1, int(n * test_ratio)), n - 1)
        test_pos[cid] = perm[:n_test]
        train_pos[cid] = perm[n_test:]
    return train_pos, test_pos


def record_first_real_detection(detection_round_map, cid, round_num, is_real):
    """Record the first round a client's detector output aligned with real
    ground-truth drift. Called every round regardless of any ongoing
    alarm-episode gating state, so recall/detection-delay reflect when the
    real event was actually first observed — not suppressed just because an
    earlier, unrelated false alarm on the same client hadn't recovered yet."""
    if is_real and cid not in detection_round_map:
        detection_round_map[cid] = round_num


def annotate_zero_bars(ax, bars, missing=None, y=0.02):
    """Label zero-height (or undefined) bars with '0.0'/'N/A' text so they
    read as present-and-measured rather than invisible/left off the chart.
    Only labels bars that are actually zero-height or explicitly flagged
    missing — a bar with a real nonzero value is left alone, since its
    height already shows the value. Operates only on the passed-in ax/bars,
    so no matplotlib import needed here — shared by build_synthesis.py and
    significance_test.py."""
    if missing is None:
        missing = [False] * len(bars)
    for bar, is_missing in zip(bars, missing):
        if not is_missing and bar.get_height() != 0:
            continue
        text = "N/A" if is_missing else "0.0"
        ax.text(bar.get_x() + bar.get_width() / 2, y, text,
                 ha="center", fontsize=8, color="blue", rotation=90)
