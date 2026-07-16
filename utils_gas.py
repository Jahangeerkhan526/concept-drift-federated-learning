import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


GAS_PATH = os.path.join(os.path.dirname(__file__), "data", "Dataset")
GAS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "gas_cache.pkl")
GAS_BATCH_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "gas_cache_batches.pkl")


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
        print("Loading Gas Sensor dataset with batch info from cache...")
        with open(GAS_BATCH_CACHE_PATH, "rb") as f:
            X, y, batch_ids = pickle.load(f)
        print(f"Dataset ready: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        return X, y, batch_ids

    print("Reading Gas Sensor dataset with batch info...")
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

    print("Normalising features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

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


def inject_batch_drift(X, y, indices, batch_ids, early_batches=(1, 2, 3), late_batches=(8, 9, 10)):
    """
    Scenario 2 , Batch-based Temporal Drift (sudden switch):
    Switch client data from early batches to late batches.
    """
    late_mask = np.isin(batch_ids, late_batches)
    late_indices = np.where(late_mask)[0]
    n = len(indices)

    if len(late_indices) == 0:
        X_drifted = X[indices].copy()
        y_drifted = y[indices].copy()
        X_drifted = X_drifted + np.random.normal(1.0, 0.3, X_drifted.shape).astype(np.float32)
        return X_drifted, y_drifted

    if len(late_indices) >= n:
        sampled = np.random.choice(late_indices, size=n, replace=False)
    else:
        sampled = np.random.choice(late_indices, size=n, replace=True)

    return X[sampled].copy(), y[sampled].copy()


def get_sequential_batch_data(X, y, batch_ids, client_indices, round_num,
                               num_rounds=50, num_batches=10):
    """
    Professor Jing's approach , Sequential batch feeding.
    Model trains on Batch 1 first, then progressively sees
    Batch 2, 3... 10 over 50 rounds.

    Rounds 1-5   → Batch 1
    Rounds 6-10  → Batch 2
    Rounds 11-15 → Batch 3
    ...
    Rounds 46-50 → Batch 10
    """
    rounds_per_batch = num_rounds // num_batches  # = 5
    current_batch = min((round_num - 1) // rounds_per_batch + 1, num_batches)

    batch_mask = batch_ids == current_batch
    batch_indices = np.where(batch_mask)[0]

    n = len(client_indices)

    if len(batch_indices) == 0:
        return X[client_indices].copy(), y[client_indices].copy()

    if len(batch_indices) >= n:
        sampled = np.random.choice(batch_indices, size=n, replace=False)
    else:
        sampled = np.random.choice(batch_indices, size=n, replace=True)

    return X[sampled].copy(), y[sampled].copy()