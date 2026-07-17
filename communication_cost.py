"""Communication cost — bytes transferred per experiment.

All three methods (FedAvg, CDA-FedAvg, DAAW) use the same communication
pattern: every client downloads the global model and uploads its updated
model, every round, regardless of drift status. DAAW/CDA only change how
much weight a client's update gets during aggregation — they don't skip
or compress communication. So this cost is identical across all three
methods; that's the expected, honestly-reported finding here.
"""
import json
import os
from model import get_model as get_har_model, get_parameters as get_har_params
from model_gas import get_model as get_gas_model, get_parameters as get_gas_params

NUM_CLIENTS = 10
NUM_ROUNDS = 50
BYTES_PER_PARAM = 4  # float32


def compute_cost(param_count, num_clients=NUM_CLIENTS, num_rounds=NUM_ROUNDS):
    bytes_per_direction = param_count * BYTES_PER_PARAM
    bytes_per_client_per_round = bytes_per_direction * 2  # download + upload
    bytes_per_round = bytes_per_client_per_round * num_clients
    total_bytes = bytes_per_round * num_rounds
    return {
        "param_count": param_count,
        "bytes_per_client_per_round": bytes_per_client_per_round,
        "bytes_per_round_all_clients": bytes_per_round,
        "total_bytes_full_experiment": total_bytes,
        "total_mb_full_experiment": round(total_bytes / (1024 * 1024), 2),
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "messages_total": num_clients * num_rounds * 2,
    }


if __name__ == "__main__":
    har_params = get_har_params(get_har_model())
    gas_params = get_gas_params(get_gas_model())
    har_count = sum(p.size for p in har_params)
    gas_count = sum(p.size for p in gas_params)

    har_cost = compute_cost(har_count)
    gas_cost = compute_cost(gas_count)

    summary = {
        "note": "Identical across FedAvg, CDA-FedAvg, and DAAW — none of the three "
                "methods change what gets communicated, only how updates are weighted "
                "during aggregation.",
        "har": har_cost,
        "gas_sensor": gas_cost,
    }

    os.makedirs("results/summary", exist_ok=True)
    with open("results/summary/communication_cost.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("HAR model:", har_count, "parameters")
    print(f"  Per client per round: {har_cost['bytes_per_client_per_round']:,} bytes")
    print(f"  Full experiment (10 clients, 50 rounds): {har_cost['total_mb_full_experiment']} MB, "
          f"{har_cost['messages_total']} messages")
    print()
    print("Gas Sensor model:", gas_count, "parameters")
    print(f"  Per client per round: {gas_cost['bytes_per_client_per_round']:,} bytes")
    print(f"  Full experiment (10 clients, 50 rounds): {gas_cost['total_mb_full_experiment']} MB, "
          f"{gas_cost['messages_total']} messages")
    print()
    print("Saved: results/summary/communication_cost.json")
