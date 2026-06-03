import os
import flwr as fl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from utils import load_har_dataset, partition_noniid, get_client_dataloader, inject_concept_drift
from client import HARClient

X_global, y_global, client_data_global = None, None, None


def get_client_fn(X, y, client_data):
    def client_fn(context: Context):
        client_id = int(context.node_id % 10)
        indices = client_data[client_id]
        return HARClient(client_id, X, y, indices).to_client()
    return client_fn


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def run_fedavg(num_rounds=50, num_clients=10, alpha=0.5):
    print("\n" + "="*50)
    print("BASELINE 1: FedAvg")
    print(f"Rounds: {num_rounds} | Clients: {num_clients} | Alpha: {alpha}")
    print("="*50)

    import torch
    import torch.nn as nn
    from model import get_model, get_parameters, set_parameters

    X, y = load_har_dataset()
    client_data = partition_noniid(X, y, num_clients=num_clients, alpha=alpha)

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

            loader = get_client_dataloader(X, y, client_data[cid])

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

        # Simple average — no drift detection
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


def run_cda_fedavg(num_rounds=50, num_clients=10, alpha=0.5, drift_threshold=0.15):
    """
    Baseline 2: CDA-FedAvg (Casado et al. 2022)
    Detects drift by comparing each client's loss to previous round.
    Problem: cannot distinguish Non-IID from actual drift.
    """
    print("\n" + "="*50)
    print("BASELINE 2: CDA-FedAvg")
    print(f"Rounds: {num_rounds} | Clients: {num_clients} | Alpha: {alpha} | Threshold: {drift_threshold}")
    print("="*50)

    import torch
    import torch.nn as nn
    from model import get_model, get_parameters, set_parameters

    X, y = load_har_dataset()
    client_data = partition_noniid(X, y, num_clients=num_clients, alpha=alpha)

    client_loss_history = {i: [] for i in range(num_clients)}
    client_weights = {i: 1.0 for i in range(num_clients)}

    print("\nClient data distribution:")
    for cid, indices in client_data.items():
        unique, counts = np.unique(y[indices], return_counts=True)
        print(f"  Client {cid}: {len(indices)} samples | classes: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Two drift events
    drift_events = {3: 5, 5: 20}
    print(f"\nDrift events: Client 3 at round 5, Client 5 at round 20")

    global_params = get_parameters(get_model())
    results_per_round = []

    # Storage for drifted data
    drifted_data = {}

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        # Inject drift at specified rounds
        for dclient, dround in drift_events.items():
            if round_num == dround:
                print(f"  *** DRIFT INJECTED into Client {dclient} ***")
                X_d, y_d = inject_concept_drift(X, y, client_data[dclient], drift_type="sudden")
                drifted_data[dclient] = (X_d, y_d)

        round_params = []
        round_losses = {}
        round_accuracies = {}
        round_sizes = {}

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            # Use drifted data if available for this client
            if cid in drifted_data and round_num >= drift_events[cid]:
                X_d, y_d = drifted_data[cid]
                loader = get_client_dataloader(X_d, y_d, list(range(len(client_data[cid]))))
            else:
                loader = get_client_dataloader(X, y, client_data[cid])

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

            round_losses[cid] = total_loss
            round_accuracies[cid] = correct / total
            round_sizes[cid] = len(client_data[cid])
            round_params.append(get_parameters(model))

            # CDA drift detection
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

    print("\nCDA-FedAvg Results per round:")
    for i, acc in enumerate(results_per_round, 1):
        print(f"  Round {i}: {acc:.4f}")

    return results_per_round


def run_daaw(num_rounds=50, num_clients=10, alpha=0.5, short_window=5, long_window=50, threshold=0.3):
    """
    Proposed Method: DAAW — Double Sliding Window Cosine Similarity
    Detects drift by comparing each client's OWN gradient history.
    Short window (5 rounds)  = sudden drift detection
    Long window  (50 rounds) = gradual drift detection
    Non-IID clients are NOT falsely flagged.
    """
    print("\n" + "="*50)
    print("PROPOSED METHOD: DAAW")
    print(f"Rounds: {num_rounds} | Clients: {num_clients} | Alpha: {alpha}")
    print(f"Short window: {short_window} | Long window: {long_window} | Threshold: {threshold}")
    print("="*50)

    import torch
    import torch.nn as nn
    from model import get_model, get_parameters, set_parameters
    from utils import daaw_detect_drift

    X, y = load_har_dataset()
    client_data = partition_noniid(X, y, num_clients=num_clients, alpha=alpha)

    client_gradient_history = {i: [] for i in range(num_clients)}
    client_weights = {i: 1.0 for i in range(num_clients)}
    client_drift_detected = {i: False for i in range(num_clients)}
    drift_detection_round = {}

    print("\nClient data distribution:")
    for cid, indices in client_data.items():
        unique, counts = np.unique(y[indices], return_counts=True)
        print(f"  Client {cid}: {len(indices)} samples | classes: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Two drift events
    drift_events = {3: 5, 5: 20}
    print(f"\nDrift events: Client 3 at round 5, Client 5 at round 20")

    global_params = get_parameters(get_model())
    results_per_round = []
    drifted_data = {}

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        # Inject drift at specified rounds
        for dclient, dround in drift_events.items():
            if round_num == dround:
                print(f"  *** SUDDEN DRIFT INJECTED into Client {dclient} ***")
                X_d, y_d = inject_concept_drift(X, y, client_data[dclient], drift_type="sudden")
                drifted_data[dclient] = (X_d, y_d)

        round_params = []
        round_accuracies = {}
        round_sizes = {}

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            # Use drifted data if available
            if cid in drifted_data and round_num >= drift_events[cid]:
                X_d, y_d = drifted_data[cid]
                loader = get_client_dataloader(X_d, y_d, list(range(len(client_data[cid]))))
            else:
                loader = get_client_dataloader(X, y, client_data[cid])

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

    print("\nDAAW Results per round:")
    for i, acc in enumerate(results_per_round, 1):
        print(f"  Round {i}: {acc:.4f}")

    print("\nDrift detection summary:")
    for cid, round_detected in drift_detection_round.items():
        print(f"  Client {cid}: drift detected at round {round_detected}")

    for dclient in drift_events:
        if dclient not in drift_detection_round:
            print(f"  Client {dclient}: drift NOT detected (missed)")

    return results_per_round, drift_detection_round


def plot_results(fedavg_results, cda_results, daaw_results, drift_events, save_path="results"):
    """Generate and save comparison graph."""
    os.makedirs(save_path, exist_ok=True)
    rounds = range(1, len(daaw_results) + 1)

    plt.figure(figsize=(14, 7))
    plt.plot(rounds, fedavg_results, label="FedAvg (Baseline 1)", color="blue", linewidth=2)
    plt.plot(rounds, cda_results, label="CDA-FedAvg (Baseline 2)", color="orange", linewidth=2)
    plt.plot(rounds, daaw_results, label="DAAW (Proposed)", color="green", linewidth=2)

    colors = ["red", "purple"]
    for i, (dclient, dround) in enumerate(drift_events.items()):
        plt.axvline(x=dround, color=colors[i], linestyle="--", alpha=0.8, linewidth=1.5)
        plt.text(dround + 0.5, 0.55, f"Drift\nClient {dclient}\nRound {dround}",
                color=colors[i], fontsize=8)

    plt.xlabel("Communication Round", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Federated Learning Accuracy Comparison\nwith Concept Drift in Non-IID Environment", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print(f"\nGraph saved to {save_path}/accuracy_comparison.png")


if __name__ == "__main__":
    drift_events = {3: 5, 5: 20}

    fedavg_results = run_fedavg(num_rounds=50, num_clients=10, alpha=0.5)
    print("\nFedAvg Baseline complete!")

    cda_results = run_cda_fedavg(num_rounds=50, num_clients=10, alpha=0.5)
    print("\nCDA-FedAvg Baseline complete!")

    daaw_results, detected = run_daaw(num_rounds=50, num_clients=10, alpha=0.5)
    print("\nDAAW complete!")

    plot_results(fedavg_results, cda_results, daaw_results, drift_events)
    print("\nAll done! Check results/accuracy_comparison.png")