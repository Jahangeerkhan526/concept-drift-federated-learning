import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_gas import get_model, get_parameters, set_parameters
from utils_gas import load_gas_dataset, partition_noniid_gas, get_gas_dataloader, inject_gas_drift
from utils import daaw_detect_drift


def run_fedavg_gas(num_rounds=50, num_clients=10, alpha=0.5):
    print("\n" + "="*50)
    print("GAS SENSOR — BASELINE 1: FedAvg")
    print(f"Rounds: {num_rounds} | Clients: {num_clients} | Alpha: {alpha}")
    print("="*50)

    X, y = load_gas_dataset()
    client_data = partition_noniid_gas(X, y, num_clients=num_clients, alpha=alpha)

    print("\nClient data distribution:")
    for cid, indices in client_data.items():
        unique, counts = np.unique(y[indices], return_counts=True)
        print(f"  Client {cid}: {len(indices)} samples | classes: {dict(zip(unique.tolist(), counts.tolist()))}")

    global_params = get_parameters(get_model())
    results_per_round = []

    for round_num in range(1, num_rounds + 1):
        round_params = []
        round_accuracies = {}
        round_sizes = {}

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            loader = get_gas_dataloader(X, y, client_data[cid])

            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    predicted = outputs.argmax(dim=1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)

            round_accuracies[cid] = correct / total
            round_sizes[cid] = len(client_data[cid])
            round_params.append(get_parameters(model))

        new_params = []
        for param_idx in range(len(global_params)):
            avg_param = np.mean([round_params[cid][param_idx] for cid in range(num_clients)], axis=0)
            new_params.append(avg_param)
        global_params = new_params

        avg_accuracy = sum(round_accuracies[cid] * round_sizes[cid] for cid in range(num_clients))
        avg_accuracy /= sum(round_sizes.values())
        results_per_round.append(avg_accuracy)
        print(f"  Round {round_num} accuracy: {avg_accuracy:.4f}")

    return results_per_round


def run_cda_fedavg_gas(num_rounds=50, num_clients=10, alpha=0.5, drift_threshold=0.15):
    print("\n" + "="*50)
    print("GAS SENSOR — BASELINE 2: CDA-FedAvg")
    print(f"Rounds: {num_rounds} | Clients: {num_clients} | Alpha: {alpha} | Threshold: {drift_threshold}")
    print("="*50)

    X, y = load_gas_dataset()
    client_data = partition_noniid_gas(X, y, num_clients=num_clients, alpha=alpha)

    client_loss_history = {i: [] for i in range(num_clients)}
    client_weights = {i: 1.0 for i in range(num_clients)}

    # Two drift events
    drift_events = {3: 5, 5: 20}
    drifted_data = {}

    print("\nClient data distribution:")
    for cid, indices in client_data.items():
        unique, counts = np.unique(y[indices], return_counts=True)
        print(f"  Client {cid}: {len(indices)} samples | classes: {dict(zip(unique.tolist(), counts.tolist()))}")

    print(f"\nDrift events: Client 3 at round 5, Client 5 at round 20")

    global_params = get_parameters(get_model())
    results_per_round = []

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        for dclient, dround in drift_events.items():
            if round_num == dround:
                print(f"  *** DRIFT INJECTED into Client {dclient} ***")
                X_d, y_d = inject_gas_drift(X, y, client_data[dclient])
                drifted_data[dclient] = (X_d, y_d)

        round_params = []
        round_accuracies = {}
        round_sizes = {}

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            if cid in drifted_data and round_num >= drift_events[cid]:
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
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    predicted = outputs.argmax(dim=1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)

            round_accuracies[cid] = correct / total
            round_sizes[cid] = len(client_data[cid])
            round_params.append(get_parameters(model))

            if len(client_loss_history[cid]) > 0:
                prev_loss = client_loss_history[cid][-1]
                loss_change = (total_loss - prev_loss) / (prev_loss + 1e-8)
                if loss_change > drift_threshold:
                    client_weights[cid] = 0.3
                    print(f"  Client {cid}: DRIFT DETECTED (loss change: {loss_change:.3f}) → weight reduced to 0.3")
                else:
                    client_weights[cid] = min(1.0, client_weights[cid] + 0.1)

            client_loss_history[cid].append(total_loss)

        total_weight = sum(client_weights[cid] * round_sizes[cid] for cid in range(num_clients))
        new_params = []
        for param_idx in range(len(global_params)):
            weighted_param = sum(
                (client_weights[cid] * round_sizes[cid] / total_weight) * round_params[cid][param_idx]
                for cid in range(num_clients)
            )
            new_params.append(weighted_param)
        global_params = new_params

        avg_accuracy = sum(round_accuracies[cid] * round_sizes[cid] for cid in range(num_clients))
        avg_accuracy /= sum(round_sizes.values())
        results_per_round.append(avg_accuracy)
        print(f"  Round {round_num} accuracy: {avg_accuracy:.4f}")

    print("\nCDA-FedAvg Gas Results per round:")
    for i, acc in enumerate(results_per_round, 1):
        print(f"  Round {i}: {acc:.4f}")

    return results_per_round


def run_daaw_gas(num_rounds=50, num_clients=10, alpha=0.5, short_window=5, long_window=50, threshold=0.3):
    print("\n" + "="*50)
    print("GAS SENSOR — PROPOSED METHOD: DAAW")
    print(f"Rounds: {num_rounds} | Clients: {num_clients} | Alpha: {alpha}")
    print(f"Short window: {short_window} | Long window: {long_window} | Threshold: {threshold}")
    print("="*50)

    X, y = load_gas_dataset()
    client_data = partition_noniid_gas(X, y, num_clients=num_clients, alpha=alpha)

    client_gradient_history = {i: [] for i in range(num_clients)}
    client_weights = {i: 1.0 for i in range(num_clients)}
    client_drift_detected = {i: False for i in range(num_clients)}
    drift_detection_round = {}

    drift_events = {3: 5, 5: 20}
    drifted_data = {}

    print("\nClient data distribution:")
    for cid, indices in client_data.items():
        unique, counts = np.unique(y[indices], return_counts=True)
        print(f"  Client {cid}: {len(indices)} samples | classes: {dict(zip(unique.tolist(), counts.tolist()))}")

    print(f"\nDrift events: Client 3 at round 5, Client 5 at round 20")

    global_params = get_parameters(get_model())
    results_per_round = []

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        for dclient, dround in drift_events.items():
            if round_num == dround:
                print(f"  *** SUDDEN DRIFT INJECTED into Client {dclient} ***")
                X_d, y_d = inject_gas_drift(X, y, client_data[dclient])
                drifted_data[dclient] = (X_d, y_d)

        round_params = []
        round_accuracies = {}
        round_sizes = {}

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            if cid in drifted_data and round_num >= drift_events[cid]:
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
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            params_after = [p.clone().detach() for p in model.parameters()]
            gradient = np.concatenate([
                (after - before).cpu().numpy().flatten()
                for before, after in zip(params_before, params_after)
            ])
            client_gradient_history[cid].append(gradient)

            drift_detected, similarity = daaw_detect_drift(
                client_gradient_history[cid],
                short_window=short_window,
                long_window=long_window,
                threshold=threshold
            )

            if drift_detected and not client_drift_detected[cid]:
                client_drift_detected[cid] = True
                drift_detection_round[cid] = round_num
                client_weights[cid] = 0.3
                print(f"  Client {cid}: DRIFT DETECTED (similarity: {similarity:.4f}) → weight reduced to 0.3")
            elif drift_detected:
                client_weights[cid] = max(0.3, client_weights[cid] - 0.05)
            else:
                if client_weights[cid] < 1.0:
                    client_weights[cid] = min(1.0, client_weights[cid] + 0.1)
                    if client_weights[cid] == 1.0:
                        client_drift_detected[cid] = False
                        print(f"  Client {cid}: stabilised → weight restored to 1.0")

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    predicted = outputs.argmax(dim=1)
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)

            round_accuracies[cid] = correct / total
            round_sizes[cid] = len(client_data[cid])
            round_params.append(get_parameters(model))

        total_weight = sum(client_weights[cid] * round_sizes[cid] for cid in range(num_clients))
        new_params = []
        for param_idx in range(len(global_params)):
            weighted_param = sum(
                (client_weights[cid] * round_sizes[cid] / total_weight) * round_params[cid][param_idx]
                for cid in range(num_clients)
            )
            new_params.append(weighted_param)
        global_params = new_params

        avg_accuracy = sum(round_accuracies[cid] * round_sizes[cid] for cid in range(num_clients))
        avg_accuracy /= sum(round_sizes.values())
        results_per_round.append(avg_accuracy)
        print(f"  Round {round_num} accuracy: {avg_accuracy:.4f}")

    print("\nDAAW Gas Sensor Results per round:")
    for i, acc in enumerate(results_per_round, 1):
        print(f"  Round {i}: {acc:.4f}")

    print("\nDrift detection summary:")
    for cid, round_detected in drift_detection_round.items():
        print(f"  Client {cid}: drift detected at round {round_detected}")

    for dclient in drift_events:
        if dclient not in drift_detection_round:
            print(f"  Client {dclient}: drift NOT detected (missed)")

    return results_per_round, drift_detection_round


def plot_gas_results(fedavg_results, cda_results, daaw_results, drift_events, save_path="results"):
    os.makedirs(save_path, exist_ok=True)
    rounds = range(1, len(daaw_results) + 1)

    plt.figure(figsize=(14, 7))
    plt.plot(rounds, fedavg_results, label="FedAvg (Baseline 1)", color="blue", linewidth=2)
    plt.plot(rounds, cda_results, label="CDA-FedAvg (Baseline 2)", color="orange", linewidth=2)
    plt.plot(rounds, daaw_results, label="DAAW (Proposed)", color="green", linewidth=2)

    colors = ["red", "purple"]
    for i, (dclient, dround) in enumerate(drift_events.items()):
        plt.axvline(x=dround, color=colors[i], linestyle="--", alpha=0.8, linewidth=1.5)
        plt.text(dround + 0.5, 0.35, f"Drift\nClient {dclient}\nRound {dround}",
                color=colors[i], fontsize=8)

    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Gas Sensor Array — Federated Learning Accuracy Comparison\nwith Concept Drift in Non-IID Environment", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "gas_accuracy_comparison.png"), dpi=150)
    plt.close()
    print(f"\nGraph saved to {save_path}/gas_accuracy_comparison.png")


if __name__ == "__main__":
    drift_events = {3: 5, 5: 20}

    fedavg_results = run_fedavg_gas(num_rounds=50, num_clients=10, alpha=0.5)
    print("\nFedAvg Gas complete!")

    cda_results = run_cda_fedavg_gas(num_rounds=50, num_clients=10, alpha=0.5)
    print("\nCDA-FedAvg Gas complete!")

    daaw_results, detected = run_daaw_gas(num_rounds=50, num_clients=10, alpha=0.5, short_window=5, long_window=20, threshold=0.5)
    print("\nDAAW Gas complete!")

    plot_gas_results(fedavg_results, cda_results, daaw_results, drift_events)
    print("\nAll done! Check results/gas_accuracy_comparison.png")