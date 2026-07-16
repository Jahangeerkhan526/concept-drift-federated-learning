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


def inject_activity_drift(X, y, indices,
                           target_activities=(1, 2),
                           target_ratio=0.8):
    """
    Scenario 2 , Activity Distribution Drift for UCI HAR.
    Shift client data distribution toward specific activities.

    Real world: user behaviour changes over time ,
    e.g. more stair climbing, less sitting.

    Default: shift toward Walking Upstairs (1) and
    Walking Downstairs (2) , more dynamic activities.

    target_activities: activity class IDs to shift toward
    target_ratio: proportion of drifted data from target activities
    """
    X_drifted = X[indices].copy()
    y_drifted = y[indices].copy()

    n = len(indices)
    n_target = int(n * target_ratio)

    # Get target activity samples from full dataset
    target_mask = np.isin(y, target_activities)
    target_idx = np.where(target_mask)[0]

    if len(target_idx) == 0:
        # Fallback , label shuffle
        np.random.shuffle(y_drifted)
        return X_drifted, y_drifted

    if len(target_idx) >= n_target:
        sampled_target = np.random.choice(target_idx, size=n_target, replace=False)
    else:
        sampled_target = np.random.choice(target_idx, size=n_target, replace=True)

    # Remaining from original client data
    n_remaining = n - n_target
    remaining_idx = np.random.choice(len(indices), size=n_remaining, replace=False)

    X_drifted = np.concatenate([
        X[sampled_target],
        X_drifted[remaining_idx]
    ]).astype(np.float32)

    y_drifted = np.concatenate([
        y[sampled_target],
        y_drifted[remaining_idx]
    ])

    return X_drifted, y_drifted


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