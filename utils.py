import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset

# Path to UCI HAR dataset
HAR_PATH = os.path.join(os.path.dirname(__file__), "data", "UCI HAR Dataset")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "har_cache.pkl")


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


def partition_noniid(X, y, num_clients=10, alpha=0.5):
    """
    Split data across clients using Dirichlet distribution.
    alpha controls how Non-IID the split is:
    - Low alpha (0.1) = very Non-IID
    - High alpha (10) = nearly IID
    """
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


def inject_concept_drift(X, y, indices, drift_type="sudden"):
    """
    Simulate concept drift by shuffling labels for a client.
    sudden = all labels shuffled at once
    gradual = labels shuffled progressively
    """
    X_drifted = X[indices].copy()
    y_drifted = y[indices].copy()

    if drift_type == "sudden":
        np.random.shuffle(y_drifted)
    # gradual drift — to be implemented in next phase

    # elif drift_type == "gradual":
    #    n = len(y_drifted)
    #    shuffle_count = n // 3
     #   y_drifted[:shuffle_count] = np.random.permutation(y_drifted[:shuffle_count])

    return X_drifted, y_drifted


def cosine_similarity(v1, v2):
    """
    Compute cosine similarity between two vectors.
    Returns value between -1 and 1.
    1  = identical direction (no drift)
    0  = orthogonal (major change)
    -1 = opposite direction (complete drift)
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 1.0  # treat zero vector as no change

    return np.dot(v1, v2) / (norm1 * norm2)


def daaw_detect_drift(gradient_history, short_window=5, long_window=50, threshold=0.3):
    """
    DAAW — Double sliding window cosine similarity drift detection.

    Short window (5 rounds)  — detects sudden drift
    Long window  (50 rounds) — detects gradual drift

    Logic:
    - Compute average gradient in short window
    - Compute average gradient in long window
    - If cosine similarity between them drops below threshold → drift detected

    Returns: (drift_detected: bool, similarity_score: float)
    """
    if len(gradient_history) < short_window + 1:
        return False, 1.0  # not enough history yet

    # Short window — recent rounds
    short = gradient_history[-short_window:]
    short_avg = np.mean(short, axis=0)

    # Long window — older rounds (or all available if less than long_window)
    long_end = max(0, len(gradient_history) - short_window)
    long_start = max(0, long_end - long_window)
    long = gradient_history[long_start:long_end]

    if len(long) == 0:
        return False, 1.0

    long_avg = np.mean(long, axis=0)

    # Compute cosine similarity between short and long window averages
    similarity = cosine_similarity(short_avg, long_avg)

    drift_detected = similarity < threshold

    return drift_detected, similarity