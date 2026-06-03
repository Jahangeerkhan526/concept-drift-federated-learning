import flwr as fl
import numpy as np
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server import ServerAppComponents
from flwr.common import Context
from flwr.simulation import run_simulation
from utils import load_har_dataset, partition_noniid, get_client_dataloader, inject_concept_drift
from client import HARClient

# Load data once globally
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


def run_fedavg(num_rounds=10, num_clients=10, alpha=0.5):
    print("\n" + "="*50)
    print("BASELINE 1: FedAvg")
    print(f"Rounds: {num_rounds} | Clients: {num_clients} | Alpha: {alpha}")
    print("="*50)

    X, y = load_har_dataset()
    client_data = partition_noniid(X, y, num_clients=num_clients, alpha=alpha)

    print("\nClient data distribution:")
    for cid, indices in client_data.items():
        unique, counts = np.unique(y[indices], return_counts=True)
        print(f"  Client {cid}: {len(indices)} samples | classes: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # ServerApp using server_fn pattern (correct for flwr 1.13+)
    def server_fn(context: Context):
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=num_rounds),
        )

    client_app = ClientApp(client_fn=get_client_fn(X, y, client_data))
    server_app = ServerApp(server_fn=server_fn)

    history = run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
    )

    return history

def run_cda_fedavg(num_rounds=10, num_clients=10, alpha=0.5, drift_threshold=0.15):
    """
    Baseline 2: CDA-FedAvg (Casado et al. 2022)
    Detects drift by comparing each client's loss to previous round.
    If loss increases beyond threshold → client flagged as drifted → weight reduced.
    Problem: cannot distinguish Non-IID from actual drift.
    """
    print("\n" + "="*50)
    print("BASELINE 2: CDA-FedAvg")
    print(f"Rounds: {num_rounds} | Clients: {num_clients} | Alpha: {alpha} | Threshold: {drift_threshold}")
    print("="*50)

    X, y = load_har_dataset()
    client_data = partition_noniid(X, y, num_clients=num_clients, alpha=alpha)

    # Track each client's loss history
    client_loss_history = {i: [] for i in range(num_clients)}
    client_weights = {i: 1.0 for i in range(num_clients)}

    print("\nClient data distribution:")
    for cid, indices in client_data.items():
        unique, counts = np.unique(y[indices], return_counts=True)
        print(f"  Client {cid}: {len(indices)} samples | classes: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Inject drift into client 3 at round 5
    drift_round = 5
    drift_client = 3
    print(f"\nDrift will be injected into Client {drift_client} at round {drift_round}")

    # Manually simulate rounds
    import torch
    import torch.nn as nn
    from model import get_model, get_parameters, set_parameters

    global_model = get_model()
    global_params = get_parameters(global_model)

    results_per_round = []

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        # Inject drift at round 5 for client 3
        if round_num == drift_round:
            print(f"  *** DRIFT INJECTED into Client {drift_client} ***")
            X_drift, y_drift = inject_concept_drift(X, y, client_data[drift_client], drift_type="sudden")
        
        round_params = []
        round_losses = {}
        round_accuracies = {}
        round_sizes = {}

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            # Use drifted data for client 3 after drift round
            if cid == drift_client and round_num >= drift_round:
                loader = get_client_dataloader(X_drift, y_drift, 
                         list(range(len(client_data[cid]))))
            else:
                loader = get_client_dataloader(X, y, client_data[cid])

            # Train
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

            # Evaluate
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

            accuracy = correct / total
            round_losses[cid] = total_loss
            round_accuracies[cid] = accuracy
            round_sizes[cid] = len(client_data[cid])
            round_params.append(get_parameters(model))

            # CDA drift detection — compare loss to previous round
            drift_detected = False
            if len(client_loss_history[cid]) > 0:
                prev_loss = client_loss_history[cid][-1]
                loss_change = (total_loss - prev_loss) / (prev_loss + 1e-8)
                if loss_change > drift_threshold:
                    drift_detected = True
                    client_weights[cid] = 0.3  # penalise drifted client
                    print(f"  Client {cid}: DRIFT DETECTED (loss change: {loss_change:.3f}) → weight reduced to 0.3")
                else:
                    client_weights[cid] = min(1.0, client_weights[cid] + 0.1)  # gradually restore

            client_loss_history[cid].append(total_loss)

        # Weighted aggregation
        total_weight = sum(client_weights[cid] * round_sizes[cid] for cid in range(num_clients))
        new_params = []
        for param_idx in range(len(global_params)):
            weighted_param = sum(
                (client_weights[cid] * round_sizes[cid] / total_weight) * round_params[cid][param_idx]
                for cid in range(num_clients)
            )
            new_params.append(weighted_param)

        global_params = new_params

        # Round accuracy
        avg_accuracy = sum(round_accuracies[cid] * round_sizes[cid] for cid in range(num_clients))
        avg_accuracy /= sum(round_sizes.values())
        results_per_round.append(avg_accuracy)
        print(f"  Round {round_num} accuracy: {avg_accuracy:.4f}")

    print("\nCDA-FedAvg Results per round:")
    for i, acc in enumerate(results_per_round, 1):
        print(f"  Round {i}: {acc:.4f}")

    return results_per_round


def run_daaw(num_rounds=10, num_clients=10, alpha=0.5, short_window=5, long_window=50, threshold=0.3):
    """
    Proposed Method: DAAW — Double sliding Window Cosine Similarity
    Detects drift by comparing each client's OWN gradient history.
    Short window (5 rounds)  = sudden drift detection
    Long window  (50 rounds) = gradual drift detection
    Non-IID clients are NOT falsely flagged — only genuine drift detected.
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

    # Per client state
    client_gradient_history = {i: [] for i in range(num_clients)}
    client_weights = {i: 1.0 for i in range(num_clients)}
    client_drift_detected = {i: False for i in range(num_clients)}
    drift_detection_round = {}  # records which round drift was detected per client

    print("\nClient data distribution:")
    for cid, indices in client_data.items():
        unique, counts = np.unique(y[indices], return_counts=True)
        print(f"  Client {cid}: {len(indices)} samples | classes: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Inject sudden drift into client 3 at round 5
    drift_round = 5
    drift_client = 3
    print(f"\nSudden drift will be injected into Client {drift_client} at round {drift_round}")

    global_model = get_model()
    global_params = get_parameters(global_model)

    results_per_round = []
    similarity_scores = {i: [] for i in range(num_clients)}

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        # Inject drift at round 5 for client 3
        if round_num == drift_round:
            print(f"  *** SUDDEN DRIFT INJECTED into Client {drift_client} ***")
            X_drift, y_drift = inject_concept_drift(
                X, y, client_data[drift_client], drift_type="sudden"
            )

        round_params = []
        round_accuracies = {}
        round_sizes = {}

        for cid in range(num_clients):
            model = get_model()
            set_parameters(model, global_params)
            device = torch.device("cpu")
            model.to(device)

            # Use drifted data for client 3 after drift round
            if cid == drift_client and round_num >= drift_round:
                loader = get_client_dataloader(
                    X_drift, y_drift, list(range(len(client_data[cid])))
                )
            else:
                loader = get_client_dataloader(X, y, client_data[cid])

            # Store params before training
            params_before = [p.clone().detach() for p in model.parameters()]

            # Train
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

            # Compute gradient vector
            params_after = [p.clone().detach() for p in model.parameters()]
            gradient = np.concatenate([
                (after - before).cpu().numpy().flatten()
                for before, after in zip(params_before, params_after)
            ])

            # Store gradient in client history
            client_gradient_history[cid].append(gradient)

            # DAAW drift detection
            drift_detected, similarity = daaw_detect_drift(
                client_gradient_history[cid],
                short_window=short_window,
                long_window=long_window,
                threshold=threshold
            )

            similarity_scores[cid].append(similarity)

            if drift_detected and not client_drift_detected[cid]:
                client_drift_detected[cid] = True
                drift_detection_round[cid] = round_num
                client_weights[cid] = 0.3
                print(f"  Client {cid}: DRIFT DETECTED (similarity: {similarity:.4f}) → weight reduced to 0.3")
            elif drift_detected:
                client_weights[cid] = max(0.3, client_weights[cid] - 0.05)
            else:
                # Gradually restore weight if client stabilises
                if client_weights[cid] < 1.0:
                    client_weights[cid] = min(1.0, client_weights[cid] + 0.1)
                    if client_weights[cid] == 1.0:
                        client_drift_detected[cid] = False
                        print(f"  Client {cid}: stabilised → weight restored to 1.0")

            # Evaluate
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

        # Weighted aggregation using DAAW weights
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

    if drift_client not in drift_detection_round:
        print(f"  Client {drift_client}: drift NOT detected (missed)")

    return results_per_round, drift_detection_round


if __name__ == "__main__":
    run_fedavg(num_rounds=10, num_clients=10, alpha=0.5)
    print("\nFedAvg Baseline complete!")

    run_cda_fedavg(num_rounds=10, num_clients=10, alpha=0.5)
    print("\nCDA-FedAvg Baseline complete!")

    run_daaw(num_rounds=10, num_clients=10, alpha=0.5)
    print("\nDAAW complete!")