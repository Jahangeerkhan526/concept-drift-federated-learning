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
from utils_gas import (
    load_gas_dataset,
    load_gas_dataset_with_batches,
    partition_noniid_gas,
    get_gas_dataloader,
    get_gas_dataloader_from_arrays,
    inject_gas_drift,
    inject_batch_drift,
    get_sequential_batch_data
)
from utils import daaw_detect_drift

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


def run_fedavg_gas(num_rounds, num_clients, X, y, client_data,
                   drift_type="none", batch_ids=None):
    print("\n" + "="*50)
    print("GAS SENSOR , BASELINE 1: FedAvg")
    print(f"Rounds: {num_rounds} | Clients: {num_clients} | Alpha: shared partition")
    print("="*50)

    reseed_all()
    global_params = get_parameters(get_model())
    results_per_round = []
    losses_per_round = []
    start_time = time.time()

    for round_num in range(1, num_rounds + 1):
        round_params, round_accuracies, round_sizes, round_losses = [], {}, {}, {}

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            if drift_type == "sequential" and batch_ids is not None:
                X_c, y_c = get_sequential_batch_data(
                    X, y, batch_ids, client_data[cid], round_num, num_rounds)
                loader = get_gas_dataloader_from_arrays(X_c, y_c)
            else:
                loader = get_gas_dataloader(X, y, client_data[cid])

            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predicted = model(X_batch).argmax(dim=1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)

            round_accuracies[cid] = correct / total
            round_sizes[cid] = len(client_data[cid])
            round_losses[cid] = total_loss / max(len(loader), 1)
            round_params.append(get_parameters(model))

        global_params = [
            np.mean([round_params[cid][i] for cid in range(num_clients)], axis=0)
            for i in range(len(global_params))
        ]
        total_samples = sum(round_sizes.values())
        avg_accuracy = sum(round_accuracies[cid] * round_sizes[cid] for cid in range(num_clients)) / total_samples
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
    drifted_data = {}
    global_params = get_parameters(get_model())
    results_per_round = []
    losses_per_round = []
    # Track CDA detections per round for graph
    cda_detections_per_round = {}
    cda_detection_round = {}
    detection_time_total = 0.0
    start_time = time.time()

    # In sequential mode every client's data changes at every batch boundary,
    # not just the 4 clients named in drift_events , so ground truth for
    # "is this detection real" has to be computed differently in that mode.
    sequential_first_transition = (num_rounds // 10) + 1

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        for dclient, dround in drift_events.items():
            if round_num == dround:
                print(f"  *** DRIFT INJECTED into Client {dclient} ***")
                if drift_type == "batch" and batch_ids is not None:
                    X_d, y_d = inject_batch_drift(X, y, client_data[dclient], batch_ids)
                else:
                    X_d, y_d = inject_gas_drift(X, y, client_data[dclient])
                drifted_data[dclient] = (X_d, y_d)

        round_params, round_accuracies, round_sizes = [], {}, {}
        round_fp_count = 0
        round_tp_count = 0

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            if drift_type == "sequential" and batch_ids is not None:
                X_c, y_c = get_sequential_batch_data(
                    X, y, batch_ids, client_data[cid], round_num, num_rounds)
                loader = get_gas_dataloader_from_arrays(X_c, y_c)
            elif cid in drifted_data and round_num >= drift_events.get(cid, 9999):
                X_d, y_d = drifted_data[cid]
                loader = get_gas_dataloader(X_d, y_d, list(range(len(client_data[cid]))))
            else:
                loader = get_gas_dataloader(X, y, client_data[cid])

            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predicted = model(X_batch).argmax(dim=1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)

            round_accuracies[cid] = correct / total
            round_sizes[cid] = len(client_data[cid])
            round_params.append(get_parameters(model))

            detection_start = time.time()
            if len(client_loss_history[cid]) > 0:
                prev_loss = client_loss_history[cid][-1]
                loss_change = (total_loss - prev_loss) / (prev_loss + 1e-8)
                if loss_change > drift_threshold:
                    client_weights[cid] = 0.3
                    if drift_type == "sequential":
                        is_real = round_num >= sequential_first_transition
                    else:
                        is_real = cid in drift_events and round_num >= drift_events[cid]
                    if is_real:
                        round_tp_count += 1
                        if cid not in cda_detection_round:
                            cda_detection_round[cid] = round_num
                    else:
                        round_fp_count += 1
                    print(f"  Client {cid}: DRIFT DETECTED (loss change: {loss_change:.3f}) → weight reduced to 0.3")
                else:
                    client_weights[cid] = min(1.0, client_weights[cid] + 0.1)
            detection_time_total += time.time() - detection_start
            client_loss_history[cid].append(total_loss)

        cda_detections_per_round[round_num] = {
            "tp": round_tp_count, "fp": round_fp_count}

        total_weight = sum(client_weights[cid] * round_sizes[cid] for cid in range(num_clients))
        global_params = [
            sum((client_weights[cid] * round_sizes[cid] / total_weight) * round_params[cid][i]
                for cid in range(num_clients))
            for i in range(len(global_params))
        ]
        avg_accuracy = sum(round_accuracies[cid] * round_sizes[cid]
                          for cid in range(num_clients)) / sum(round_sizes.values())
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
    start_time = time.time()

    # See run_cda_fedavg_gas , same reasoning for sequential mode's ground truth.
    sequential_first_transition = (num_rounds // 10) + 1

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        for dclient, dround in drift_events.items():
            if round_num == dround:
                print(f"  *** SUDDEN DRIFT INJECTED into Client {dclient} ***")
                if drift_type == "batch" and batch_ids is not None:
                    X_d, y_d = inject_batch_drift(X, y, client_data[dclient], batch_ids)
                else:
                    X_d, y_d = inject_gas_drift(X, y, client_data[dclient])
                drifted_data[dclient] = (X_d, y_d)

        round_params, round_accuracies, round_sizes, round_losses = [], {}, {}, {}
        round_tp_count = 0
        round_fp_count = 0

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            if drift_type == "sequential" and batch_ids is not None:
                X_c, y_c = get_sequential_batch_data(
                    X, y, batch_ids, client_data[cid], round_num, num_rounds)
                loader = get_gas_dataloader_from_arrays(X_c, y_c)
            elif cid in drifted_data and round_num >= drift_events.get(cid, 9999):
                X_d, y_d = drifted_data[cid]
                loader = get_gas_dataloader(X_d, y_d, list(range(len(client_data[cid]))))
            else:
                loader = get_gas_dataloader(X, y, client_data[cid])

            params_before = [p.clone().detach() for p in model.parameters()]
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            total_loss = 0
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

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

            if drift_detected and not client_drift_detected[cid]:
                client_drift_detected[cid] = True
                client_weights[cid] = 0.3
                if drift_type == "sequential":
                    is_real = round_num >= sequential_first_transition
                else:
                    is_real = cid in drift_events and round_num >= drift_events[cid]
                if is_real:
                    round_tp_count += 1
                    if cid not in drift_detection_round:
                        drift_detection_round[cid] = round_num
                else:
                    round_fp_count += 1
                print(f"  Client {cid}: DRIFT DETECTED (similarity: {similarity:.4f}) → weight reduced to 0.3")
            elif drift_detected:
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
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    predicted = model(X_batch).argmax(dim=1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)

            round_accuracies[cid] = correct / total
            round_sizes[cid] = len(client_data[cid])
            round_losses[cid] = total_loss / max(len(loader), 1)
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
        avg_accuracy = sum(round_accuracies[cid] * round_sizes[cid]
                          for cid in range(num_clients)) / total_samples
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
    """Precision/Recall/F1/avg detection delay for one method on one scenario."""
    total_tp = sum(d["tp"] for d in detections_per_round.values())
    total_fp = sum(d["fp"] for d in detections_per_round.values())

    detected_events = [cid for cid in drift_events if cid in detection_round_map]
    total_events = len(drift_events)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = len(detected_events) / total_events if total_events > 0 else 0.0
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


def save_metrics_summary(cda_metrics, daaw_metrics, timings, save_path="results", title_suffix=""):
    """Save Precision/Recall/F1/Detection Delay/Computational Time as JSON next to the graphs."""
    os.makedirs(save_path, exist_ok=True)
    summary = {
        "title": title_suffix,
        "cda": cda_metrics,
        "daaw": daaw_metrics,
        "computational_time_seconds": timings,
    }
    with open(os.path.join(save_path, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics summary saved: {save_path}/metrics_summary.json")


def plot_all_graphs(fedavg_acc, cda_acc, daaw_acc,
                    fedavg_loss, cda_loss, daaw_loss,
                    drift_events, daaw_detection_rounds,
                    cda_detections_per_round, daaw_detections_per_round,
                    fedavg_time=None, cda_time=None, daaw_time=None,
                    save_path="results", title_suffix=""):
    os.makedirs(save_path, exist_ok=True)
    rounds = list(range(1, len(daaw_acc) + 1))
    colors = ["red", "purple", "orange", "cyan"]
    real_drift_clients = set(drift_events.keys())

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
    ax.plot(rounds, fedavg_loss, label="FedAvg Loss", color="blue", linewidth=2)
    ax.plot(rounds, cda_loss, label="CDA-FedAvg Loss", color="orange", linewidth=2)
    ax.plot(rounds, daaw_loss, label="DAAW Loss", color="green", linewidth=2)

    for i, (dclient, dround) in enumerate(drift_events.items()):
        ax.axvline(x=dround, color=colors[i % len(colors)], linestyle="--", alpha=0.6, linewidth=1.5)

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title(f"Gas Sensor , Training Loss\n{title_suffix}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "2_training_loss.png"), dpi=150)
    plt.close()
    print(f"Graph 2 saved: 2_training_loss.png")

    # ── Graph 3: CDA vs DAAW Detection Comparison ────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    cda_tp = [cda_detections_per_round.get(r, {}).get("tp", 0) for r in rounds]
    cda_fp = [cda_detections_per_round.get(r, {}).get("fp", 0) for r in rounds]
    daaw_tp = [daaw_detections_per_round.get(r, {}).get("tp", 0) for r in rounds]
    daaw_fp = [daaw_detections_per_round.get(r, {}).get("fp", 0) for r in rounds]

    # CDA subplot
    axes[0].bar(rounds, cda_tp, color="green", alpha=0.7, label="True Positive (Real Drift)")
    axes[0].bar(rounds, cda_fp, bottom=cda_tp, color="red", alpha=0.7, label="False Positive")
    for i, (dclient, dround) in enumerate(drift_events.items()):
        axes[0].axvline(x=dround, color=colors[i % len(colors)], linestyle="--", alpha=0.8, linewidth=1.5)
    axes[0].set_ylabel("Detections", fontsize=11)
    axes[0].set_title("CDA-FedAvg , Drift Detections per Round", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # DAAW subplot
    axes[1].bar(rounds, daaw_tp, color="green", alpha=0.7, label="True Positive (Real Drift)")
    axes[1].bar(rounds, daaw_fp, bottom=daaw_tp, color="red", alpha=0.7, label="False Positive")
    for i, (dclient, dround) in enumerate(drift_events.items()):
        axes[1].axvline(x=dround, color=colors[i % len(colors)], linestyle="--", alpha=0.8, linewidth=1.5)
    axes[1].set_xlabel("Communication Round", fontsize=11)
    axes[1].set_ylabel("Detections", fontsize=11)
    axes[1].set_title("DAAW (Proposed) , Drift Detections per Round", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"CDA vs DAAW , Detection Comparison\n{title_suffix}", fontsize=13)
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

    # ── Graph 5: False Positive Comparison ───────────────────────
    cda_total_fp = sum(d["fp"] for d in cda_detections_per_round.values())
    daaw_total_fp = sum(d["fp"] for d in daaw_detections_per_round.values())
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = ["CDA-FedAvg", "DAAW"]
    fp_counts = [cda_total_fp, daaw_total_fp]
    bars = ax.bar(methods, fp_counts, color=["orange", "green"], alpha=0.8)
    for bar, c in zip(bars, fp_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(c), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Total False Positives", fontsize=12)
    ax.set_title(f"Gas Sensor , False Positive Comparison\n{title_suffix}", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "5_false_positive_comparison.png"), dpi=150)
    plt.close()
    print(f"Graph 5 saved: 5_false_positive_comparison.png")


def plot_cosine_similarity(similarity_history, drift_events, threshold, save_path, title_suffix=""):
    """Graph 6: DAAW's gradient cosine similarity per round for each drifted client,
    showing the similarity dropping below the threshold at/after injection."""
    os.makedirs(save_path, exist_ok=True)
    colors = ["red", "purple", "orange", "cyan"]
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (dclient, dround) in enumerate(drift_events.items()):
        sims = similarity_history.get(dclient, [])
        if not sims:
            continue
        rounds = list(range(1, len(sims) + 1))
        color = colors[i % len(colors)]
        ax.plot(rounds, sims, label=f"Client {dclient} (drift at round {dround})", color=color, linewidth=2)
        ax.axvline(x=dround, color=color, linestyle="--", alpha=0.6, linewidth=1.2)

    ax.axhline(y=threshold, color="black", linestyle=":", linewidth=1.5, label=f"Threshold ({threshold})")
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Gradient Cosine Similarity", fontsize=12)
    ax.set_title(f"Gas Sensor — DAAW Gradient Cosine Similarity per Round\n{title_suffix}", fontsize=13)
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
        NUM_ROUNDS, NUM_CLIENTS, X, y, shared_data_s1)
    cda_acc1, cda_loss1, cda_det1, cda_dpr1, cda_t1, cda_dettime1 = run_cda_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X, y, shared_data_s1, DRIFT_EVENTS, drift_type="label")
    daaw_acc1, daaw_loss1, det1, daaw_dpr1, daaw_t1, sim_hist1, daaw_dettime1 = run_daaw_gas(
        NUM_ROUNDS, NUM_CLIENTS, X, y, shared_data_s1, DRIFT_EVENTS, drift_type="label")

    plot_all_graphs(
        fedavg_acc1, cda_acc1, daaw_acc1,
        fedavg_loss1, cda_loss1, daaw_loss1,
        DRIFT_EVENTS, det1, cda_dpr1, daaw_dpr1,
        fedavg_time=fedavg_t1, cda_time=cda_t1, daaw_time=daaw_t1,
        save_path="results/spatial_drift/gas_label_shuffle",
        title_suffix="Scenario 1: Label Shuffle (Real Drift)")
    plot_cosine_similarity(sim_hist1, DRIFT_EVENTS, threshold=0.3,
                            save_path="results/spatial_drift/gas_label_shuffle",
                            title_suffix="Scenario 1: Label Shuffle")

    cda_metrics1 = compute_detection_metrics(cda_dpr1, cda_det1, DRIFT_EVENTS)
    daaw_metrics1 = compute_detection_metrics(daaw_dpr1, det1, DRIFT_EVENTS)
    save_metrics_summary(cda_metrics1, daaw_metrics1,
                          {"fedavg": round(fedavg_t1, 2), "cda": round(cda_t1, 2), "daaw": round(daaw_t1, 2),
                           "cda_detection_ms": round(cda_dettime1 * 1000, 3),
                           "daaw_detection_ms": round(daaw_dettime1 * 1000, 3)},
                          save_path="results/spatial_drift/gas_label_shuffle",
                          title_suffix="Scenario 1: Label Shuffle")

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
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, shared_data_s2a)
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
        save_path="results/temporal_drift/gas_sudden_batch",
        title_suffix="Scenario 2a: Sudden Batch Drift (1-3 → 8-10)")
    plot_cosine_similarity(sim_hist2a, DRIFT_EVENTS, threshold=0.3,
                            save_path="results/temporal_drift/gas_sudden_batch",
                            title_suffix="Scenario 2a: Sudden Batch Drift")

    cda_metrics2a = compute_detection_metrics(cda_dpr2a, cda_det2a, DRIFT_EVENTS)
    daaw_metrics2a = compute_detection_metrics(daaw_dpr2a, det2a, DRIFT_EVENTS)
    save_metrics_summary(cda_metrics2a, daaw_metrics2a,
                          {"fedavg": round(fedavg_t2a, 2), "cda": round(cda_t2a, 2), "daaw": round(daaw_t2a, 2),
                           "cda_detection_ms": round(cda_dettime2a * 1000, 3),
                           "daaw_detection_ms": round(daaw_dettime2a * 1000, 3)},
                          save_path="results/temporal_drift/gas_sudden_batch",
                          title_suffix="Scenario 2a: Sudden Batch Drift")

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
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, shared_data_s2b,
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
        save_path="results/temporal_drift/gas_sequential_batch",
        title_suffix="Scenario 2b: Sequential Batch Drift (1→10)")
    plot_cosine_similarity(sim_hist2b, DRIFT_EVENTS, threshold=0.15,
                            save_path="results/temporal_drift/gas_sequential_batch",
                            title_suffix="Scenario 2b: Sequential Batch Drift")

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
                          save_path="results/temporal_drift/gas_sequential_batch",
                          title_suffix="Scenario 2b: Sequential Batch Drift")

    print(f"\n⏱️ Scenario 2b: FedAvg={fedavg_t2b:.1f}s | CDA={cda_t2b:.1f}s | DAAW={daaw_t2b:.1f}s")
    print(f"  CDA  → Precision: {cda_metrics2b['precision']}, Recall: {cda_metrics2b['recall']}, F1: {cda_metrics2b['f1']}, Avg Delay: {cda_metrics2b['avg_detection_delay']}")
    print(f"  DAAW → Precision: {daaw_metrics2b['precision']}, Recall: {daaw_metrics2b['recall']}, F1: {daaw_metrics2b['f1']}, Avg Delay: {daaw_metrics2b['avg_detection_delay']}")

    print("\n" + "="*60)
    print("ALL DONE!")
    print("Graphs + metrics saved:")
    print("  results/spatial_drift/gas_label_shuffle/    , Label Shuffle")
    print("  results/temporal_drift/gas_sudden_batch/    , Sudden Batch Drift")
    print("  results/temporal_drift/gas_sequential_batch/ , Sequential Batch Drift")
    print("="*60)