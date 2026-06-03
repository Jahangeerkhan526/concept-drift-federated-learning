import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


GAS_PATH = os.path.join(os.path.dirname(__file__), "data", "Dataset")
GAS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "gas_cache.pkl")


def load_gas_dataset():
    """Load Gas Sensor Array Drift dataset from batch files."""

    if os.path.exists(GAS_CACHE_PATH):
        print("Loading Gas Sensor dataset from cache...")
        with open(GAS_CACHE_PATH, "rb") as f:
            X, y = pickle.load(f)
        print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        return X, y

    print("Reading Gas Sensor dataset from batch files...")

    all_X = []
    all_y = []

    for batch_num in range(1, 11):
        batch_path = os.path.join(GAS_PATH, f"batch{batch_num}.dat")
        if not os.path.exists(batch_path):
            print(f"  Skipping batch{batch_num}.dat — not found")
            continue

        print(f"  Loading batch{batch_num}.dat...")
        with open(batch_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                label = int(float(parts[0])) - 1
                features = []
                for part in parts[1:]:
                    if ":" in part:
                        val = float(part.split(":")[1])
                        features.append(val)
                all_X.append(features)
                all_y.append(label)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)

    # Normalise features before saving
    print("Normalising features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    with open(GAS_CACHE_PATH, "wb") as f:
        pickle.dump((X, y), f)
    print(f"Cache saved to {GAS_CACHE_PATH}")

    print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    return X, y


def partition_noniid_gas(X, y, num_clients=10, alpha=0.5):
    """Non-IID partition for gas dataset."""
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
    """DataLoader for gas sensor client."""
    X_client = torch.tensor(X[indices])
    y_client = torch.tensor(y[indices])
    dataset = TensorDataset(X_client, y_client)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def inject_gas_drift(X, y, indices):
    """Simulate sudden drift by shuffling labels."""
    X_drifted = X[indices].copy()
    y_drifted = y[indices].copy()
    np.random.shuffle(y_drifted)
    return X_drifted, y_drifted