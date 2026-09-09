import os
import sys
import json
import time
import numpy as np

if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_gas import get_model, get_parameters, set_parameters
from utils_gas_v3 import (
    load_gas_dataset,
    load_gas_dataset_with_batches,
    partition_noniid_gas,
    get_gas_dataloader,
    get_gas_dataloader_from_arrays,
    inject_gas_drift,
    build_sudden_batch_pools,
    build_sequential_batch_pools,
)
from utils import daaw_detect_drift, held_out_split, record_first_real_detection

import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def reseed_all():
    """Reset all RNGs to SEED so each scenario starts from an identical state,
    regardless of what ran before it in the same script execution."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _inject_drift_gas(drift_type, X, y, client_data, dclient, batch_ids, train_pos, test_pos):
    """Build (X_train, y_train, X_test, y_test) for a newly-drifted client.
    V3: only handles label-shuffle drift now — batch/sequential drift are
    handled by precomputed client-owned pools (build_sudden_batch_pools /
    build_sequential_batch_pools) instead of this function, since those
    need a one-time setup step rather than a per-drift-event call."""
    X_d, y_d = inject_gas_drift(X, y, client_data[dclient])
    return (X_d[train_pos[dclient]], y_d[train_pos[dclient]],
            X_d[test_pos[dclient]], y_d[test_pos[dclient]])


def _client_loaders(cid, X, y, client_data, drifted_data, drift_events, drift_type, batch_ids,
                     round_num, num_rounds, train_pos, test_pos,
                     sudden_pools=None, sequential_pools=None):
    """Build (train_loader, eval_loader, train_size, test_size) for one
    client/round. V3: batch/sequential branches now look up precomputed,
    client-owned, fixed-once pools instead of resampling from a global pool
    every call. train_size/test_size are the pools' actual sizes, not the
    client's unrelated full-partition size, so aggregation/accuracy
    weighting reflects what was really trained/evaluated on."""
    if drift_type == "sequential" and sequential_pools is not None:
        rounds_per_batch = num_rounds // 10
        current_batch = min((round_num - 1) // rounds_per_batch + 1, 10)
        pool = sequential_pools[cid][current_batch]
        train_idx, test_idx = pool["train"], pool["test"]
        train_loader = get_gas_dataloader(X, y, train_idx)
        eval_loader = get_gas_dataloader(X, y, test_idx)
        return train_loader, eval_loader, len(train_idx), len(test_idx)

    if drift_type == "batch" and sudden_pools is not None:
        dround = drift_events.get(cid, 9999)
        pool = sudden_pools[cid]
        if round_num >= dround:
            train_idx, test_idx = pool["post_train"], pool["post_test"]
        else:
            train_idx, test_idx = pool["pre_train"], pool["pre_test"]
        train_loader = get_gas_dataloader(X, y, train_idx)
        eval_loader = get_gas_dataloader(X, y, test_idx)
        return train_loader, eval_loader, len(train_idx), len(test_idx)

    if cid in drifted_data and round_num >= drift_events.get(cid, 9999):
        X_train_d, y_train_d, X_test_d, y_test_d = drifted_data[cid]
        train_loader = get_gas_dataloader_from_arrays(X_train_d, y_train_d)
        eval_loader = get_gas_dataloader_from_arrays(X_test_d, y_test_d)
        return train_loader, eval_loader, len(X_train_d), len(X_test_d)

    train_idx = [client_data[cid][p] for p in train_pos[cid]]
    test_idx = [client_data[cid][p] for p in test_pos[cid]]
    train_loader = get_gas_dataloader(X, y, train_idx)
    eval_loader = get_gas_dataloader(X, y, test_idx)
    return train_loader, eval_loader, len(train_idx), len(test_idx)


def run_fedavg_gas(num_rounds, num_clients, X, y, client_data,
                   drift_events=None, drift_type="none", batch_ids=None):
    print("\n" + "="*50)
    print("GAS SENSOR , BASELINE 1: FedAvg")
    print(f"Rounds: {num_rounds} | Clients: {num_clients} | Alpha: shared partition | Drift: {drift_type}")
    print("="*50)

    drift_events = drift_events or {}
    reseed_all()
    global_params = get_parameters(get_model())
    drifted_data = {}
    results_per_round = []
    losses_per_round = []
    train_pos, test_pos = held_out_split(num_clients, client_data, SEED)
    sudden_pools = build_sudden_batch_pools(client_data, batch_ids, num_clients, SEED, y=y) if (drift_type == "batch" and batch_ids is not None) else None
    sequential_pools = build_sequential_batch_pools(client_data, batch_ids, num_clients, SEED, y=y) if (drift_type == "sequential" and batch_ids is not None) else None
    start_time = time.time()

    for round_num in range(1, num_rounds + 1):
        for dclient, dround in drift_events.items():
            if round_num == dround and drift_type not in ("batch", "sequential"):
                print(f"  *** DRIFT INJECTED into Client {dclient} ***")
                drifted_data[dclient] = _inject_drift_gas(
                    drift_type, X, y, client_data, dclient, batch_ids, train_pos, test_pos)

        round_params, round_accuracies, round_sizes, round_test_sizes, round_losses = [], {}, {}, {}, {}

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            train_loader, eval_loader, train_size, test_size = _client_loaders(
                cid, X, y, client_data, drifted_data, drift_events, drift_type, batch_ids,
                round_num, num_rounds, train_pos, test_pos,
                sudden_pools=sudden_pools, sequential_pools=sequential_pools)

            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            total_loss = 0
            samples_seen = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_batch.size(0)
                samples_seen += X_batch.size(0)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in eval_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predicted = model(X_batch).argmax(dim=1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)

            round_accuracies[cid] = correct / total
            round_sizes[cid] = train_size
            round_test_sizes[cid] = test_size
            # Sample-weighted MEAN loss — pre/post-drift and per-batch pool
            # sizes now differ per client (V4), so a raw batch-count sum
            # would let a size change alone look like a loss change.
            round_losses[cid] = total_loss / max(samples_seen, 1)
            round_params.append(get_parameters(model))

        # Sample-weighted aggregation (standard FedAvg) — no drift-based reweighting,
        # since this baseline does nothing special when drift occurs.
        total_samples = sum(round_sizes.values())
        global_params = [
            sum((round_sizes[cid] / total_samples) * round_params[cid][i] for cid in range(num_clients))
            for i in range(len(global_params))
        ]
        total_test_samples = sum(round_test_sizes.values())
        # Mean of each client's own locally-adapted model evaluated on its
        # own held-out set, weighted by test-set size — not the newly-
        # aggregated global model evaluated on every client (each client's
        # evaluate() runs before this round's aggregation). Same definition
        # used identically across FedAvg/CDA-FedAvg/DAAW below.
        avg_accuracy = sum(round_accuracies[cid] * round_test_sizes[cid] for cid in range(num_clients)) / total_test_samples
        avg_loss = sum(round_losses[cid] * round_sizes[cid] for cid in range(num_clients)) / total_samples
        results_per_round.append(avg_accuracy)
        losses_per_round.append(avg_loss)
        print(f"  Round {round_num} accuracy: {avg_accuracy:.4f} | loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\nFedAvg Gas complete! Time: {total_time:.2f}s")
    return results_per_round, losses_per_round, total_time


def run_cda_fedavg_gas(num_rounds, num_clients, X, y, client_data, drift_events,
                        drift_threshold=0.15, drift_type="label", batch_ids=None):
    print("\n" + "="*50)
    print("GAS SENSOR , BASELINE 2: CDA-FedAvg")
    print(f"Rounds: {num_rounds} | Clients: {num_clients} | Threshold: {drift_threshold} | Drift: {drift_type}")
    print("="*50)
    print(f"\nDrift events: {drift_events}")

    reseed_all()
    client_loss_history = {i: [] for i in range(num_clients)}
    client_weights = {i: 1.0 for i in range(num_clients)}
    client_cda_flagged = {i: False for i in range(num_clients)}
    drifted_data = {}
    global_params = get_parameters(get_model())
    results_per_round = []
    losses_per_round = []
    # Track CDA detections per round for graph
    cda_detections_per_round = {}
    cda_detection_round = {}
    detection_time_total = 0.0
    train_pos, test_pos = held_out_split(num_clients, client_data, SEED)
    sudden_pools = build_sudden_batch_pools(client_data, batch_ids, num_clients, SEED, y=y) if (drift_type == "batch" and batch_ids is not None) else None
    sequential_pools = build_sequential_batch_pools(client_data, batch_ids, num_clients, SEED, y=y) if (drift_type == "sequential" and batch_ids is not None) else None
    start_time = time.time()

    # In sequential mode every client's data changes at every batch boundary,
    # not just the 4 clients named in drift_events , so ground truth for
    # "is this detection real" has to be computed differently in that mode.
    sequential_first_transition = (num_rounds // 10) + 1

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        for dclient, dround in drift_events.items():
            if round_num == dround and drift_type not in ("batch", "sequential"):
                print(f"  *** DRIFT INJECTED into Client {dclient} ***")
                drifted_data[dclient] = _inject_drift_gas(
                    drift_type, X, y, client_data, dclient, batch_ids, train_pos, test_pos)

        round_params, round_accuracies, round_sizes, round_test_sizes = [], {}, {}, {}
        round_fp_count = 0
        round_tp_count = 0

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            train_loader, eval_loader, train_size, test_size = _client_loaders(
                cid, X, y, client_data, drifted_data, drift_events, drift_type, batch_ids,
                round_num, num_rounds, train_pos, test_pos,
                sudden_pools=sudden_pools, sequential_pools=sequential_pools)

            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            total_loss = 0
            samples_seen = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_batch.size(0)
                samples_seen += X_batch.size(0)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in eval_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predicted = model(X_batch).argmax(dim=1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)

            round_accuracies[cid] = correct / total
            round_sizes[cid] = train_size
            round_test_sizes[cid] = test_size
            # Sample-weighted MEAN loss — see run_fedavg_gas's comment.
            mean_loss = total_loss / max(samples_seen, 1)
            round_params.append(get_parameters(model))

            detection_start = time.time()
            if len(client_loss_history[cid]) > 0:
                prev_loss = client_loss_history[cid][-1]
                loss_change = (mean_loss - prev_loss) / (prev_loss + 1e-8)
                if loss_change > drift_threshold:
                    client_weights[cid] = 0.3
                    if drift_type == "sequential":
                        is_real = round_num >= sequential_first_transition
                    else:
                        is_real = cid in drift_events and round_num >= drift_events[cid]
                    # Recorded every round regardless of the alarm-episode flag
                    # below, so an unrelated earlier false alarm that hasn't
                    # recovered yet can't suppress credit for a real event.
                    record_first_real_detection(cda_detection_round, cid, round_num, is_real)
                    # Count once per contiguous alarm, not once per round it stays
                    # above threshold — mirrors DAAW's client_drift_detected gate,
                    # so repeat triggers on the same ongoing event don't inflate TP/FP.
                    if not client_cda_flagged[cid]:
                        client_cda_flagged[cid] = True
                        if is_real:
                            round_tp_count += 1
                        else:
                            round_fp_count += 1
                    print(f"  Client {cid}: DRIFT DETECTED (loss change: {loss_change:.3f}) → weight reduced to 0.3")
                else:
                    client_weights[cid] = min(1.0, client_weights[cid] + 0.1)
                    if client_weights[cid] >= 1.0:
                        client_cda_flagged[cid] = False
            detection_time_total += time.time() - detection_start
            client_loss_history[cid].append(mean_loss)

        cda_detections_per_round[round_num] = {
            "tp": round_tp_count, "fp": round_fp_count}

        total_weight = sum(client_weights[cid] * round_sizes[cid] for cid in range(num_clients))
        global_params = [
            sum((client_weights[cid] * round_sizes[cid] / total_weight) * round_params[cid][i]
                for cid in range(num_clients))
            for i in range(len(global_params))
        ]
        total_test_samples = sum(round_test_sizes.values())
        avg_accuracy = sum(round_accuracies[cid] * round_test_sizes[cid]
                          for cid in range(num_clients)) / total_test_samples
        avg_loss = sum(client_loss_history[cid][-1] * round_sizes[cid]
                      for cid in range(num_clients)) / sum(round_sizes.values())
        results_per_round.append(avg_accuracy)
        losses_per_round.append(avg_loss)
        print(f"  Round {round_num} accuracy: {avg_accuracy:.4f}")

    print("\nCDA drift detection summary:")
    for cid, rnd in cda_detection_round.items():
        print(f"  Client {cid}: drift detected at round {rnd}")
    for dclient in drift_events:
        if dclient not in cda_detection_round:
            print(f"  Client {dclient}: drift NOT detected (missed!)")

    total_time = time.time() - start_time
    print(f"\nCDA-FedAvg Gas complete! Time: {total_time:.2f}s | Detection-only time: {detection_time_total*1000:.2f}ms")
    return (results_per_round, losses_per_round, cda_detection_round, cda_detections_per_round,
            total_time, detection_time_total)


def run_daaw_gas(num_rounds, num_clients, X, y, client_data, drift_events,
                 short_window=5, long_window=20, threshold=0.3,
                 drift_type="label", batch_ids=None):
    print("\n" + "="*50)
    print("GAS SENSOR , PROPOSED METHOD: DAAW")
    print(f"Rounds: {num_rounds} | Clients: {num_clients} | Drift: {drift_type}")
    print(f"Short window: {short_window} | Long window: {long_window} | Threshold: {threshold}")
    print("="*50)
    print(f"\nDrift events: {drift_events}")

    reseed_all()
    client_gradient_history = {i: [] for i in range(num_clients)}
    client_weights = {i: 1.0 for i in range(num_clients)}
    client_drift_detected = {i: False for i in range(num_clients)}
    drift_detection_round = {}
    drifted_data = {}
    global_params = get_parameters(get_model())
    results_per_round = []
    losses_per_round = []
    daaw_detections_per_round = {}
    similarity_history = {i: [] for i in range(num_clients)}
    detection_time_total = 0.0
    train_pos, test_pos = held_out_split(num_clients, client_data, SEED)
    sudden_pools = build_sudden_batch_pools(client_data, batch_ids, num_clients, SEED, y=y) if (drift_type == "batch" and batch_ids is not None) else None
    sequential_pools = build_sequential_batch_pools(client_data, batch_ids, num_clients, SEED, y=y) if (drift_type == "sequential" and batch_ids is not None) else None
    start_time = time.time()

    # See run_cda_fedavg_gas , same reasoning for sequential mode's ground truth.
    sequential_first_transition = (num_rounds // 10) + 1

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        for dclient, dround in drift_events.items():
            if round_num == dround and drift_type not in ("batch", "sequential"):
                print(f"  *** SUDDEN DRIFT INJECTED into Client {dclient} ***")
                drifted_data[dclient] = _inject_drift_gas(
                    drift_type, X, y, client_data, dclient, batch_ids, train_pos, test_pos)

        round_params, round_accuracies, round_sizes, round_test_sizes, round_losses = [], {}, {}, {}, {}
        round_tp_count = 0
        round_fp_count = 0

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            train_loader, eval_loader, train_size, test_size = _client_loaders(
                cid, X, y, client_data, drifted_data, drift_events, drift_type, batch_ids,
                round_num, num_rounds, train_pos, test_pos,
                sudden_pools=sudden_pools, sequential_pools=sequential_pools)

            params_before = [p.clone().detach() for p in model.parameters()]
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            total_loss = 0
            samples_seen = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_batch.size(0)
                samples_seen += X_batch.size(0)

            params_after = [p.clone().detach() for p in model.parameters()]
            gradient = np.concatenate([
                (after - before).cpu().numpy().flatten()
                for before, after in zip(params_before, params_after)
            ])
            client_gradient_history[cid].append(gradient)

            detection_start = time.time()
            drift_detected, similarity = daaw_detect_drift(
                client_gradient_history[cid],
                short_window=short_window,
                long_window=long_window,
                threshold=threshold
            )
            detection_time_total += time.time() - detection_start
            similarity_history[cid].append(similarity)

            if drift_detected:
                if drift_type == "sequential":
                    is_real = round_num >= sequential_first_transition
                else:
                    is_real = cid in drift_events and round_num >= drift_events[cid]
                # Recorded every round regardless of the alarm-episode flag
                # below, so an unrelated earlier false alarm that hasn't
                # recovered yet can't suppress credit for a real event.
                record_first_real_detection(drift_detection_round, cid, round_num, is_real)
                if not client_drift_detected[cid]:
                    client_drift_detected[cid] = True
                    client_weights[cid] = 0.3
                    if is_real:
                        round_tp_count += 1
                    else:
                        round_fp_count += 1
                    print(f"  Client {cid}: DRIFT DETECTED (similarity: {similarity:.4f}) → weight reduced to 0.3")
                else:
                    client_weights[cid] = max(0.3, client_weights[cid] - 0.05)
            else:
                if client_weights[cid] < 1.0:
                    client_weights[cid] = min(1.0, client_weights[cid] + 0.1)
                    if client_weights[cid] >= 1.0:
                        client_drift_detected[cid] = False
                        print(f"  Client {cid}: stabilised → weight restored to 1.0")

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in eval_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predicted = model(X_batch).argmax(dim=1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)

            round_accuracies[cid] = correct / total
            round_sizes[cid] = train_size
            round_test_sizes[cid] = test_size
            round_losses[cid] = total_loss / max(samples_seen, 1)
            round_params.append(get_parameters(model))

        daaw_detections_per_round[round_num] = {
            "tp": round_tp_count, "fp": round_fp_count}

        total_weight = sum(client_weights[cid] * round_sizes[cid] for cid in range(num_clients))
        global_params = [
            sum((client_weights[cid] * round_sizes[cid] / total_weight) * round_params[cid][i]
                for cid in range(num_clients))
            for i in range(len(global_params))
        ]
        total_samples = sum(round_sizes.values())
        total_test_samples = sum(round_test_sizes.values())
        avg_accuracy = sum(round_accuracies[cid] * round_test_sizes[cid]
                          for cid in range(num_clients)) / total_test_samples
        avg_loss = sum(round_losses[cid] * round_sizes[cid]
                      for cid in range(num_clients)) / total_samples
        results_per_round.append(avg_accuracy)
        losses_per_round.append(avg_loss)
        print(f"  Round {round_num} accuracy: {avg_accuracy:.4f} | loss: {avg_loss:.4f}")

    print("\nDrift detection summary:")
    for cid, rnd in drift_detection_round.items():
        print(f"  Client {cid}: drift detected at round {rnd}")
    for dclient in drift_events:
        if dclient not in drift_detection_round:
            print(f"  Client {dclient}: drift NOT detected (missed!)")

    total_time = time.time() - start_time
    print(f"\nDAAW Gas complete! Time: {total_time:.2f}s | Detection-only time: {detection_time_total*1000:.2f}ms")
    return (results_per_round, losses_per_round, drift_detection_round, daaw_detections_per_round,
            total_time, similarity_history, detection_time_total)


def compute_detection_metrics(detections_per_round, detection_round_map, drift_events):
    """Precision/Recall/F1/avg detection delay for one method on one scenario.

    Precision/recall/F1 use event-level TP (len(detected_events): each real
    drift event counted once, matching recall's existing definition) rather
    than total_tp, which counts alarm EPISODES and can both over-count
    (a client that recovers and re-drifts) and under-count (a real detection
    swallowed by an already-active false-alarm episode) relative to events.
    total_tp/total_fp (episode counts) are kept in the returned dict for the
    TP/FP diverging-bar charts, which visualise alarm episodes rather than
    the event-level detection rate."""
    total_tp = sum(d["tp"] for d in detections_per_round.values())
    total_fp = sum(d["fp"] for d in detections_per_round.values())

    detected_events = [cid for cid in drift_events if cid in detection_round_map]
    total_events = len(drift_events)
    event_tp = len(detected_events)

    precision = event_tp / (event_tp + total_fp) if (event_tp + total_fp) > 0 else 0.0
    recall = event_tp / total_events if total_events > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    delays = [detection_round_map[cid] - drift_events[cid] for cid in detected_events]
    avg_delay = sum(delays) / len(delays) if delays else None

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_detection_delay": round(avg_delay, 2) if avg_delay is not None else None,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "detected_events": len(detected_events),
        "total_events": total_events,
    }


def save_metrics_summary(cda_metrics, daaw_metrics, timings, save_path="results", title_suffix="", final_accuracy=None):
    """Save Precision/Recall/F1/Detection Delay/Computational Time as JSON next to the graphs."""
    os.makedirs(save_path, exist_ok=True)
    summary = {
        "title": title_suffix,
        "cda": cda_metrics,
        "daaw": daaw_metrics,
        "computational_time_seconds": timings,
        "final_accuracy": final_accuracy or {},
    }
    with open(os.path.join(save_path, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics summary saved: {save_path}/metrics_summary.json")


def plot_all_graphs(fedavg_acc, cda_acc, daaw_acc,
                    fedavg_loss, cda_loss, daaw_loss,
                    drift_events, daaw_detection_rounds,
                    cda_detections_per_round, daaw_detections_per_round,
                    fedavg_time=None, cda_time=None, daaw_time=None,
                    save_path="results", title_suffix="",
                    drift_type=None, num_rounds=None):
    """drift_type/num_rounds are only needed for "sequential" — every client
    shares one global batch schedule there (not the per-client DRIFT_EVENTS
    rounds), so the vertical markers need to reflect batch transitions
    instead of misleadingly labeling them as each client's own drift round."""
    os.makedirs(save_path, exist_ok=True)
    rounds = list(range(1, len(daaw_acc) + 1))
    colors = ["red", "purple", "orange", "cyan"]
    real_drift_clients = set(drift_events.keys())
    is_sequential = drift_type == "sequential" and num_rounds is not None

    # ── Graph 1: Accuracy Comparison ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(rounds, fedavg_acc, label="FedAvg (Baseline 1)", color="blue", linewidth=2)
    ax.plot(rounds, cda_acc, label="CDA-FedAvg (Baseline 2)", color="orange", linewidth=2)
    ax.plot(rounds, daaw_acc, label="DAAW (Proposed)", color="green", linewidth=2)

    for i, (dclient, dround) in enumerate(drift_events.items()):
        ax.axvline(x=dround, color=colors[i % len(colors)], linestyle="--", alpha=0.8, linewidth=1.5)
        ax.text(dround + 0.5, 0.32, f"Drift\nC{dclient}\nR{dround}",
                color=colors[i % len(colors)], fontsize=8)

    real_label_added = False
    fp_label_added = False
    if daaw_detection_rounds:
        for cid, rnd in daaw_detection_rounds.items():
            if cid in real_drift_clients:
                ax.scatter([rnd], [daaw_acc[rnd-1]], marker="^", color="green",
                           s=120, zorder=5,
                           label="DAAW Real Detection" if not real_label_added else "")
                real_label_added = True
            else:
                ax.scatter([rnd], [daaw_acc[rnd-1]], marker="X", color="red",
                           s=120, zorder=5,
                           label="DAAW False Positive" if not fp_label_added else "")
                fp_label_added = True

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Gas Sensor , Accuracy Comparison\n{title_suffix}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "1_accuracy_comparison.png"), dpi=150)
    plt.close()
    print(f"Graph 1 saved: 1_accuracy_comparison.png")

    # ── Graph 2: Training Loss ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    mevery = max(len(rounds) // 20, 1)
    ax.plot(rounds, fedavg_loss, marker="o", markevery=mevery, linestyle="--",
            label="FedAvg Loss", color="blue", linewidth=1.6, markersize=6)
    ax.plot(rounds, cda_loss, marker="s", markevery=mevery, linestyle="--",
            label="CDA-FedAvg Loss", color="orange", linewidth=1.6, markersize=6)
    ax.plot(rounds, daaw_loss, marker="^", markevery=mevery, linestyle="--",
            label="DAAW Loss", color="green", linewidth=1.6, markersize=6)

    if is_sequential:
        # All clients share one global batch schedule here — per-client
        # DRIFT_EVENTS rounds don't apply, so mark batch transitions instead.
        rounds_per_batch = num_rounds // 10
        for b in range(rounds_per_batch, num_rounds, rounds_per_batch):
            ax.axvline(x=b, color="gray", linestyle=":", alpha=0.4, linewidth=1)
    else:
        for i, (dclient, dround) in enumerate(drift_events.items()):
            ax.axvline(x=dround, color=colors[i % len(colors)], linestyle=":", alpha=0.5, linewidth=1.2)

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title(f"Gas Sensor , Training Loss\n{title_suffix}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "2_training_loss.png"), dpi=150)
    plt.close()
    print(f"Graph 2 saved: 2_training_loss.png")

    # ── Graph 3: True Positives vs False Positives (diverging bar) ──
    cda_total_tp = sum(d["tp"] for d in cda_detections_per_round.values())
    cda_total_fp = sum(d["fp"] for d in cda_detections_per_round.values())
    daaw_total_tp = sum(d["tp"] for d in daaw_detections_per_round.values())
    daaw_total_fp = sum(d["fp"] for d in daaw_detections_per_round.values())

    fig, ax = plt.subplots(figsize=(11, 5))
    for yy, label, tp, fp, color in [
        (0, "FedAvg", 0, 0, "blue"),
        (1, "CDA-FedAvg", cda_total_tp, cda_total_fp, "orange"),
        (2, "DAAW", daaw_total_tp, daaw_total_fp, "green"),
    ]:
        ax.barh(yy, tp, height=0.6, color=color, alpha=0.85)
        ax.barh(yy, -fp, height=0.6, color=color, alpha=0.4, hatch="//")
        if label == "FedAvg":
            ax.plot(0, yy, marker="D", color="blue", markersize=11,
                    zorder=5, markeredgecolor="black", markeredgewidth=1.2)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["FedAvg", "CDA-FedAvg", "DAAW"], fontsize=10)
    ax.set_xlabel("<- False Alarm Episodes          True Detection Episodes ->", fontsize=11)
    ax.set_title(f"Alarm Episode Counts: True vs False Detection Episodes\n{title_suffix}", fontsize=13)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "3_detection_comparison.png"), dpi=150)
    plt.close()
    print(f"Graph 3 saved: 3_detection_comparison.png")

    # ── Graph 4: Computational Time Comparison ───────────────────
    if fedavg_time is not None and cda_time is not None and daaw_time is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        methods = ["FedAvg", "CDA-FedAvg", "DAAW"]
        times = [fedavg_time, cda_time, daaw_time]
        bars = ax.bar(methods, times, color=["blue", "orange", "green"], alpha=0.8)
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{t:.1f}s", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Total Training Time (seconds)", fontsize=12)
        ax.set_title(f"Gas Sensor , Computational Time Comparison\n{title_suffix}", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "4_computational_time.png"), dpi=150)
        plt.close()
        print(f"Graph 4 saved: 4_computational_time.png")

    # Graph 5 (old standalone False Positive bar, CDA vs DAAW only, no
    # FedAvg) removed — Graph 3's diverging bar already shows false
    # positives (hatched, left side) alongside true positives and FedAvg,
    # so a separate FP-only chart was redundant and never got the FedAvg
    # fix the other graphs did.


def plot_cosine_similarity(similarity_history, drift_events, threshold, save_path, title_suffix="",
                            detection_round_map=None, drift_type=None, num_rounds=None):
    """Graph 6: DAAW's model-update (pseudo-gradient) cosine similarity per round for each client.
    For label/batch scenarios, each client genuinely drifts at its own
    DRIFT_EVENTS round, so per-client staggered labeling is accurate. For
    "sequential", every client shares one global batch schedule instead —
    labeling them with individual DRIFT_EVENTS rounds would be fabricated,
    so that case shows a single shared transition line and a few example
    clients instead."""
    os.makedirs(save_path, exist_ok=True)
    detection_round_map = detection_round_map or {}
    colors = ["red", "purple", "orange", "cyan"]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axhspan(-1.1, threshold, color="red", alpha=0.06, zorder=0)

    if drift_type == "sequential" and num_rounds is not None:
        first_transition = (num_rounds // 10) + 1
        ax.axvline(first_transition, color="gray", linestyle="--", linewidth=1.2, alpha=0.7,
                   label=f"Shared batch transition begins (round {first_transition})")
        show_clients = sorted(similarity_history.keys())[:4] if len(similarity_history) > 4 else sorted(similarity_history.keys())
        for i, cid in enumerate(show_clients):
            sims = similarity_history.get(cid, [])
            if not sims:
                continue
            rounds = list(range(1, len(sims) + 1))
            color = colors[i % len(colors)]
            ax.plot(rounds, sims, label=f"Client {cid}", color=color, linewidth=2)
            if cid in detection_round_map:
                r = detection_round_map[cid]
                if r - 1 < len(sims):
                    ax.scatter(r, sims[r - 1], marker="*", color=color, s=260,
                               edgecolor="black", linewidth=0.8, zorder=6)
    else:
        for i, (dclient, dround) in enumerate(drift_events.items()):
            sims = similarity_history.get(dclient, [])
            if not sims:
                continue
            rounds = list(range(1, len(sims) + 1))
            color = colors[i % len(colors)]
            ax.plot(rounds, sims, label=f"Client {dclient} (drift at round {dround})", color=color, linewidth=2)
            ax.axvline(x=dround, color=color, linestyle="--", alpha=0.6, linewidth=1.2)
            if dclient in detection_round_map:
                r = detection_round_map[dclient]
                if r - 1 < len(sims):
                    ax.scatter(r, sims[r - 1], marker="*", color=color, s=220,
                               edgecolor="black", linewidth=0.7, zorder=6)

    ax.axhline(y=threshold, color="black", linestyle=":", linewidth=1.5, label=f"Threshold ({threshold})")
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Model-Update Cosine Similarity", fontsize=12)
    ax.set_title(f"Gas Sensor — DAAW Model-Update Cosine Similarity per Round\n{title_suffix}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "6_cosine_similarity.png"), dpi=150)
    plt.close()
    print(f"Graph 6 saved: 6_cosine_similarity.png")


if __name__ == "__main__":
    ALPHA = 0.5
    NUM_ROUNDS = 50
    NUM_CLIENTS = 10
    DRIFT_EVENTS = {3: 5, 5: 20, 7: 35, 9: 45}

    # ══════════════════════════════════════════════════
    # SCENARIO 1 , Label Shuffle
    # ══════════════════════════════════════════════════
    print("\n" + "="*60)
    print("SCENARIO 1 , Label Shuffle (Artificial Concept Drift)")
    print("Simulates real drift P(Y|X) , decision boundary changes")
    print("="*60)

    X, y = load_gas_dataset()
    np.random.seed(SEED)
    shared_data_s1 = partition_noniid_gas(X, y, num_clients=NUM_CLIENTS, alpha=ALPHA)

    fedavg_acc1, fedavg_loss1, fedavg_t1 = run_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X, y, shared_data_s1, DRIFT_EVENTS, drift_type="label")
    cda_acc1, cda_loss1, cda_det1, cda_dpr1, cda_t1, cda_dettime1 = run_cda_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X, y, shared_data_s1, DRIFT_EVENTS, drift_type="label")
    daaw_acc1, daaw_loss1, det1, daaw_dpr1, daaw_t1, sim_hist1, daaw_dettime1 = run_daaw_gas(
        NUM_ROUNDS, NUM_CLIENTS, X, y, shared_data_s1, DRIFT_EVENTS, drift_type="label")

    plot_all_graphs(
        fedavg_acc1, cda_acc1, daaw_acc1,
        fedavg_loss1, cda_loss1, daaw_loss1,
        DRIFT_EVENTS, det1, cda_dpr1, daaw_dpr1,
        fedavg_time=fedavg_t1, cda_time=cda_t1, daaw_time=daaw_t1,
        save_path="myfinalresult_v3/spatial_drift/gas_label_shuffle",
        title_suffix="Scenario 1: Label Shuffle (Real Drift)")
    plot_cosine_similarity(sim_hist1, DRIFT_EVENTS, threshold=0.3,
                            save_path="myfinalresult_v3/spatial_drift/gas_label_shuffle",
                            title_suffix="Scenario 1: Label Shuffle",
                            detection_round_map=det1)

    cda_metrics1 = compute_detection_metrics(cda_dpr1, cda_det1, DRIFT_EVENTS)
    daaw_metrics1 = compute_detection_metrics(daaw_dpr1, det1, DRIFT_EVENTS)
    save_metrics_summary(cda_metrics1, daaw_metrics1,
                          {"fedavg": round(fedavg_t1, 2), "cda": round(cda_t1, 2), "daaw": round(daaw_t1, 2),
                           "cda_detection_ms": round(cda_dettime1 * 1000, 3),
                           "daaw_detection_ms": round(daaw_dettime1 * 1000, 3)},
                          save_path="myfinalresult_v3/spatial_drift/gas_label_shuffle",
                          title_suffix="Scenario 1: Label Shuffle",
                          final_accuracy={"fedavg": round(fedavg_acc1[-1], 4), "cda": round(cda_acc1[-1], 4), "daaw": round(daaw_acc1[-1], 4)})

    print(f"\n⏱️ Scenario 1: FedAvg={fedavg_t1:.1f}s | CDA={cda_t1:.1f}s | DAAW={daaw_t1:.1f}s")
    print(f"  CDA  → Precision: {cda_metrics1['precision']}, Recall: {cda_metrics1['recall']}, F1: {cda_metrics1['f1']}, Avg Delay: {cda_metrics1['avg_detection_delay']}")
    print(f"  DAAW → Precision: {daaw_metrics1['precision']}, Recall: {daaw_metrics1['recall']}, F1: {daaw_metrics1['f1']}, Avg Delay: {daaw_metrics1['avg_detection_delay']}")

    # Load batch dataset , used for both 2a and 2b
    X2, y2, batch_ids = load_gas_dataset_with_batches()

    # ══════════════════════════════════════════════════
    # SCENARIO 2a , Sudden Batch Switch
    # ══════════════════════════════════════════════════
    print("\n" + "="*60)
    print("SCENARIO 2a , Sudden Batch Drift (batches 1-3 → 8-10)")
    print("Simulates abrupt sensor degradation")
    print("="*60)

    np.random.seed(SEED)
    shared_data_s2a = partition_noniid_gas(X2, y2, num_clients=NUM_CLIENTS, alpha=ALPHA)

    fedavg_acc2a, fedavg_loss2a, fedavg_t2a = run_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, shared_data_s2a, DRIFT_EVENTS,
        drift_type="batch", batch_ids=batch_ids)
    cda_acc2a, cda_loss2a, cda_det2a, cda_dpr2a, cda_t2a, cda_dettime2a = run_cda_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, shared_data_s2a, DRIFT_EVENTS,
        drift_type="batch", batch_ids=batch_ids)
    daaw_acc2a, daaw_loss2a, det2a, daaw_dpr2a, daaw_t2a, sim_hist2a, daaw_dettime2a = run_daaw_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, shared_data_s2a, DRIFT_EVENTS,
        drift_type="batch", batch_ids=batch_ids)

    plot_all_graphs(
        fedavg_acc2a, cda_acc2a, daaw_acc2a,
        fedavg_loss2a, cda_loss2a, daaw_loss2a,
        DRIFT_EVENTS, det2a, cda_dpr2a, daaw_dpr2a,
        fedavg_time=fedavg_t2a, cda_time=cda_t2a, daaw_time=daaw_t2a,
        save_path="myfinalresult_v3/temporal_drift/gas_sudden_batch",
        title_suffix="Scenario 2a: Sudden Batch Drift (1-3 → 8-10)")
    plot_cosine_similarity(sim_hist2a, DRIFT_EVENTS, threshold=0.3,
                            save_path="myfinalresult_v3/temporal_drift/gas_sudden_batch",
                            title_suffix="Scenario 2a: Sudden Batch Drift",
                            detection_round_map=det2a)

    cda_metrics2a = compute_detection_metrics(cda_dpr2a, cda_det2a, DRIFT_EVENTS)
    daaw_metrics2a = compute_detection_metrics(daaw_dpr2a, det2a, DRIFT_EVENTS)
    save_metrics_summary(cda_metrics2a, daaw_metrics2a,
                          {"fedavg": round(fedavg_t2a, 2), "cda": round(cda_t2a, 2), "daaw": round(daaw_t2a, 2),
                           "cda_detection_ms": round(cda_dettime2a * 1000, 3),
                           "daaw_detection_ms": round(daaw_dettime2a * 1000, 3)},
                          save_path="myfinalresult_v3/temporal_drift/gas_sudden_batch",
                          title_suffix="Scenario 2a: Sudden Batch Drift",
                          final_accuracy={"fedavg": round(fedavg_acc2a[-1], 4), "cda": round(cda_acc2a[-1], 4), "daaw": round(daaw_acc2a[-1], 4)})

    print(f"\n⏱️ Scenario 2a: FedAvg={fedavg_t2a:.1f}s | CDA={cda_t2a:.1f}s | DAAW={daaw_t2a:.1f}s")
    print(f"  CDA  → Precision: {cda_metrics2a['precision']}, Recall: {cda_metrics2a['recall']}, F1: {cda_metrics2a['f1']}, Avg Delay: {cda_metrics2a['avg_detection_delay']}")
    print(f"  DAAW → Precision: {daaw_metrics2a['precision']}, Recall: {daaw_metrics2a['recall']}, F1: {daaw_metrics2a['f1']}, Avg Delay: {daaw_metrics2a['avg_detection_delay']}")

    # ══════════════════════════════════════════════════
    # SCENARIO 2b , Sequential Continuous Drift
    # ══════════════════════════════════════════════════
    print("\n" + "="*60)
    print("SCENARIO 2b , Sequential Batch Drift (batch 1 → 2 → ... → 10)")
    print("Simulates continuous gradual sensor degradation")
    print("Rounds 1-5=Batch1 | 6-10=Batch2 | ... | 46-50=Batch10")
    print("="*60)

    np.random.seed(SEED)
    shared_data_s2b = partition_noniid_gas(X2, y2, num_clients=NUM_CLIENTS, alpha=ALPHA)

    fedavg_acc2b, fedavg_loss2b, fedavg_t2b = run_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, shared_data_s2b, DRIFT_EVENTS,
        drift_type="sequential", batch_ids=batch_ids)
    cda_acc2b, cda_loss2b, cda_det2b, cda_dpr2b, cda_t2b, cda_dettime2b = run_cda_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, shared_data_s2b, DRIFT_EVENTS,
        drift_type="sequential", batch_ids=batch_ids)
    daaw_acc2b, daaw_loss2b, det2b, daaw_dpr2b, daaw_t2b, sim_hist2b, daaw_dettime2b = run_daaw_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, shared_data_s2b, DRIFT_EVENTS,
        drift_type="sequential", batch_ids=batch_ids,
        short_window=8,
        long_window=30,
        threshold=0.15)

    plot_all_graphs(
        fedavg_acc2b, cda_acc2b, daaw_acc2b,
        fedavg_loss2b, cda_loss2b, daaw_loss2b,
        DRIFT_EVENTS, det2b, cda_dpr2b, daaw_dpr2b,
        fedavg_time=fedavg_t2b, cda_time=cda_t2b, daaw_time=daaw_t2b,
        save_path="myfinalresult_v3/temporal_drift/gas_sequential_batch",
        title_suffix="Scenario 2b: Sequential Batch Drift (1→10)",
        drift_type="sequential", num_rounds=NUM_ROUNDS)
    plot_cosine_similarity(sim_hist2b, DRIFT_EVENTS, threshold=0.15,
                            save_path="myfinalresult_v3/temporal_drift/gas_sequential_batch",
                            title_suffix="Scenario 2b: Sequential Batch Drift",
                            detection_round_map=det2b, drift_type="sequential", num_rounds=NUM_ROUNDS)

    # In sequential mode every client genuinely drifts at every batch boundary,
    # not just the 4 clients in DRIFT_EVENTS , so ground truth for precision/
    # recall/delay here is "all 10 clients, real from the first transition
    # (round 6) onward," matching the is_real logic inside run_cda_fedavg_gas /
    # run_daaw_gas.
    SEQUENTIAL_GROUND_TRUTH = {i: (NUM_ROUNDS // 10) + 1 for i in range(NUM_CLIENTS)}
    cda_metrics2b = compute_detection_metrics(cda_dpr2b, cda_det2b, SEQUENTIAL_GROUND_TRUTH)
    daaw_metrics2b = compute_detection_metrics(daaw_dpr2b, det2b, SEQUENTIAL_GROUND_TRUTH)
    save_metrics_summary(cda_metrics2b, daaw_metrics2b,
                          {"fedavg": round(fedavg_t2b, 2), "cda": round(cda_t2b, 2), "daaw": round(daaw_t2b, 2),
                           "cda_detection_ms": round(cda_dettime2b * 1000, 3),
                           "daaw_detection_ms": round(daaw_dettime2b * 1000, 3)},
                          save_path="myfinalresult_v3/temporal_drift/gas_sequential_batch",
                          title_suffix="Scenario 2b: Sequential Batch Drift",
                          final_accuracy={"fedavg": round(fedavg_acc2b[-1], 4), "cda": round(cda_acc2b[-1], 4), "daaw": round(daaw_acc2b[-1], 4)})

    print(f"\n⏱️ Scenario 2b: FedAvg={fedavg_t2b:.1f}s | CDA={cda_t2b:.1f}s | DAAW={daaw_t2b:.1f}s")
    print(f"  CDA  → Precision: {cda_metrics2b['precision']}, Recall: {cda_metrics2b['recall']}, F1: {cda_metrics2b['f1']}, Avg Delay: {cda_metrics2b['avg_detection_delay']}")
    print(f"  DAAW → Precision: {daaw_metrics2b['precision']}, Recall: {daaw_metrics2b['recall']}, F1: {daaw_metrics2b['f1']}, Avg Delay: {daaw_metrics2b['avg_detection_delay']}")

    print("\n" + "="*60)
    print("ALL DONE!")
    print("Graphs + metrics saved:")
    print("  myfinalresult_v3/spatial_drift/gas_label_shuffle/    , Label Shuffle")
    print("  myfinalresult_v3/temporal_drift/gas_sudden_batch/    , Sudden Batch Drift")
    print("  myfinalresult_v3/temporal_drift/gas_sequential_batch/ , Sequential Batch Drift")
    print("="*60)
