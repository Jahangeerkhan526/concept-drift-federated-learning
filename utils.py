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

    # Load train
    X_train = np.loadtxt(os.path.join(HAR_PATH, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(HAR_PATH, "train", "y_train.txt"), dtype=int)

    # Load test
    X_test = np.loadtxt(os.path.join(HAR_PATH, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(HAR_PATH, "test", "y_test.txt"), dtype=int)

    # Combine train and test
    X = np.vstack([X_train, X_test]).astype(np.float32)
    y = np.concatenate([y_train, y_test]).astype(np.int64)

    # Convert labels 1-6 to 0-5
    y = y - 1

    # Save cache
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
    elif drift_type == "gradual":
        n = len(y_drifted)
        shuffle_count = n // 3
        y_drifted[:shuffle_count] = np.random.permutation(y_drifted[:shuffle_count])

    return X_drifted, y_drifted