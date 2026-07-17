"""Alpha (0.1/0.5/1.0) and DAAW window/threshold sensitivity sweep."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import load_har_dataset, partition_noniid
from utils_gas import load_gas_dataset, partition_noniid_gas
import server
import server_gas

SEED = 42
NUM_ROUNDS = 50
NUM_CLIENTS = 10
DRIFT_EVENTS = {3: 5, 5: 20, 7: 35, 9: 45}


def run_har_at_alpha(alpha, save_path):
    X, y = load_har_dataset()
    np.random.seed(SEED)
    client_data = partition_noniid(X, y, num_clients=NUM_CLIENTS, alpha=alpha)

    fedavg_r, fedavg_loss, fedavg_t = server.run_fedavg(NUM_ROUNDS, NUM_CLIENTS, X, y, client_data)
    cda_r, cda_loss, cda_det, cda_dpr, cda_t, cda_dettime = server.run_cda_fedavg(
        NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="label")
    daaw_r, daaw_loss, daaw_det, daaw_dpr, daaw_t, sim_hist, daaw_dettime = server.run_daaw(
        NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="label")

    server.plot_all_graphs(fedavg_r, cda_r, daaw_r, fedavg_loss, cda_loss, daaw_loss,
                           DRIFT_EVENTS, daaw_det, cda_dpr, daaw_dpr,
                           fedavg_time=fedavg_t, cda_time=cda_t, daaw_time=daaw_t,
                           save_path=save_path, title_suffix=f"HAR Label Shuffle, alpha={alpha}")

    cda_metrics = server.compute_detection_metrics(cda_dpr, cda_det, DRIFT_EVENTS)
    daaw_metrics = server.compute_detection_metrics(daaw_dpr, daaw_det, DRIFT_EVENTS)
    server.save_metrics_summary(cda_metrics, daaw_metrics,
                                {"fedavg": round(fedavg_t, 2), "cda": round(cda_t, 2), "daaw": round(daaw_t, 2)},
                                save_path=save_path, title_suffix=f"alpha={alpha}")
    return {"accuracy": fedavg_r[-1], "cda": cda_metrics, "daaw": daaw_metrics,
            "cda_acc": cda_r[-1], "daaw_acc": daaw_r[-1]}


def run_gas_at_alpha(alpha, save_path):
    X, y = load_gas_dataset()
    np.random.seed(SEED)
    client_data = partition_noniid_gas(X, y, num_clients=NUM_CLIENTS, alpha=alpha)

    fedavg_r, fedavg_loss, fedavg_t = server_gas.run_fedavg_gas(NUM_ROUNDS, NUM_CLIENTS, X, y, client_data)
    cda_r, cda_loss, cda_det, cda_dpr, cda_t, cda_dettime = server_gas.run_cda_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="label")
    daaw_r, daaw_loss, daaw_det, daaw_dpr, daaw_t, sim_hist, daaw_dettime = server_gas.run_daaw_gas(
        NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="label")

    server_gas.plot_all_graphs(fedavg_r, cda_r, daaw_r, fedavg_loss, cda_loss, daaw_loss,
                               DRIFT_EVENTS, daaw_det, cda_dpr, daaw_dpr,
                               fedavg_time=fedavg_t, cda_time=cda_t, daaw_time=daaw_t,
                               save_path=save_path, title_suffix=f"Gas Label Shuffle, alpha={alpha}")

    cda_metrics = server_gas.compute_detection_metrics(cda_dpr, cda_det, DRIFT_EVENTS)
    daaw_metrics = server_gas.compute_detection_metrics(daaw_dpr, daaw_det, DRIFT_EVENTS)
    server_gas.save_metrics_summary(cda_metrics, daaw_metrics,
                                    {"fedavg": round(fedavg_t, 2), "cda": round(cda_t, 2), "daaw": round(daaw_t, 2)},
                                    save_path=save_path, title_suffix=f"alpha={alpha}")
    return {"accuracy": fedavg_r[-1], "cda": cda_metrics, "daaw": daaw_metrics,
            "cda_acc": cda_r[-1], "daaw_acc": daaw_r[-1]}


def run_gas_at_window_combo(short_window, long_window, threshold, label, save_path):
    X, y = load_gas_dataset()
    np.random.seed(SEED)
    client_data = partition_noniid_gas(X, y, num_clients=NUM_CLIENTS, alpha=0.5)

    fedavg_r, fedavg_loss, fedavg_t = server_gas.run_fedavg_gas(NUM_ROUNDS, NUM_CLIENTS, X, y, client_data)
    cda_r, cda_loss, cda_det, cda_dpr, cda_t, cda_dettime = server_gas.run_cda_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="label")
    daaw_r, daaw_loss, daaw_det, daaw_dpr, daaw_t, sim_hist, daaw_dettime = server_gas.run_daaw_gas(
        NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="label",
        short_window=short_window, long_window=long_window, threshold=threshold)

    server_gas.plot_all_graphs(fedavg_r, cda_r, daaw_r, fedavg_loss, cda_loss, daaw_loss,
                               DRIFT_EVENTS, daaw_det, cda_dpr, daaw_dpr,
                               fedavg_time=fedavg_t, cda_time=cda_t, daaw_time=daaw_t,
                               save_path=save_path, title_suffix=f"Gas Label Shuffle, {label}")

    cda_metrics = server_gas.compute_detection_metrics(cda_dpr, cda_det, DRIFT_EVENTS)
    daaw_metrics = server_gas.compute_detection_metrics(daaw_dpr, daaw_det, DRIFT_EVENTS)
    server_gas.save_metrics_summary(cda_metrics, daaw_metrics,
                                    {"fedavg": round(fedavg_t, 2), "cda": round(cda_t, 2), "daaw": round(daaw_t, 2)},
                                    save_path=save_path, title_suffix=label)
    return {"cda": cda_metrics, "daaw": daaw_metrics}


def plot_alpha_sensitivity(alpha_values, har_results, gas_results, save_path):
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, results, title in [(axes[0], har_results, "UCI HAR"), (axes[1], gas_results, "Gas Sensor")]:
        cda_f1 = [results[a]["cda"]["f1"] for a in alpha_values]
        daaw_f1 = [results[a]["daaw"]["f1"] for a in alpha_values]
        ax.plot(alpha_values, cda_f1, marker="o", label="CDA-FedAvg", color="orange", linewidth=2)
        ax.plot(alpha_values, daaw_f1, marker="o", label="DAAW", color="green", linewidth=2)
        ax.set_xlabel("Alpha (higher = closer to IID)", fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(f"{title} — F1 vs Alpha", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "f1_vs_alpha.png"), dpi=150)
    plt.close()
    print(f"Alpha sensitivity summary graph saved: {save_path}/f1_vs_alpha.png")


if __name__ == "__main__":
    alpha_values = [0.1, 0.5, 1.0]
    har_alpha_results = {}
    gas_alpha_results = {}

    print("=" * 60)
    print("ALPHA SENSITIVITY SWEEP")
    print("=" * 60)
    for alpha in alpha_values:
        print(f"\n--- HAR, alpha={alpha} ---")
        har_alpha_results[alpha] = run_har_at_alpha(alpha, f"results/alpha_sensitivity/har_alpha_{alpha}")

        print(f"\n--- Gas Sensor, alpha={alpha} ---")
        gas_alpha_results[alpha] = run_gas_at_alpha(alpha, f"results/alpha_sensitivity/gas_alpha_{alpha}")

    plot_alpha_sensitivity(alpha_values, har_alpha_results, gas_alpha_results, "results/alpha_sensitivity")

    print("\n" + "=" * 60)
    print("WINDOW/THRESHOLD SENSITIVITY SWEEP (Gas Sensor, Label Shuffle, alpha=0.5)")
    print("=" * 60)
    window_results = {}
    combos = [
        ("default_5_20_0.3", 5, 20, 0.3),
        ("faster_3_10_0.3", 3, 10, 0.3),
        ("conservative_8_30_0.3", 8, 30, 0.3),
    ]
    for label, sw, lw, thr in combos:
        print(f"\n--- {label} ---")
        window_results[label] = run_gas_at_window_combo(sw, lw, thr, label, f"results/window_sensitivity/{label}")

    summary = {
        "alpha_sensitivity": {
            "har": {str(a): v for a, v in har_alpha_results.items()},
            "gas": {str(a): v for a, v in gas_alpha_results.items()},
        },
        "window_sensitivity": window_results,
    }
    os.makedirs("results/alpha_sensitivity", exist_ok=True)
    with open("results/alpha_sensitivity/sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("results/alpha_sensitivity/  — alpha sweep graphs, metrics, f1_vs_alpha.png")
    print("results/window_sensitivity/ — window/threshold sweep graphs, metrics")
    print("=" * 60)
