"""5-seed significance testing across the 5 core scenarios.

Reuses run_fedavg/run_cda_fedavg/run_daaw (and _gas equivalents) from
server.py/server_gas.py, varying the seed instead of reimplementing
anything. No graphs are produced here — just metrics, since 25 runs x 6
graphs would be excessive; the point of this script is the numbers.

For each scenario: mean/std across 5 seeds, plus a paired t-test (DAAW vs
CDA on F1), justified by assumption rather than asserted, and reported
alongside the descriptive stats regardless of outcome.
"""
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from utils import load_har_dataset, partition_noniid
from utils_gas import load_gas_dataset, load_gas_dataset_with_batches, partition_noniid_gas
import server
import server_gas

SEEDS = [42, 123, 456, 789, 999]
NUM_ROUNDS = 50
NUM_CLIENTS = 10
ALPHA = 0.5
DRIFT_EVENTS = {3: 5, 5: 20, 7: 35, 9: 45}


def run_har_label_shuffle(seed):
    server.SEED = seed
    np.random.seed(seed)
    X, y = load_har_dataset()
    client_data = partition_noniid(X, y, num_clients=NUM_CLIENTS, alpha=ALPHA)
    fedavg_r, _, _ = server.run_fedavg(NUM_ROUNDS, NUM_CLIENTS, X, y, client_data)
    cda_r, _, cda_det, cda_dpr, _, _ = server.run_cda_fedavg(
        NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="label")
    daaw_r, _, daaw_det, daaw_dpr, _, _, _ = server.run_daaw(
        NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="label")
    cda_m = server.compute_detection_metrics(cda_dpr, cda_det, DRIFT_EVENTS)
    daaw_m = server.compute_detection_metrics(daaw_dpr, daaw_det, DRIFT_EVENTS)
    return {"seed": seed, "fedavg_acc": fedavg_r[-1], "cda_acc": cda_r[-1], "daaw_acc": daaw_r[-1],
            "cda": cda_m, "daaw": daaw_m}


def run_har_activity_drift(seed):
    server.SEED = seed
    np.random.seed(seed)
    X, y = load_har_dataset()
    client_data = partition_noniid(X, y, num_clients=NUM_CLIENTS, alpha=ALPHA)
    fedavg_r, _, _ = server.run_fedavg(NUM_ROUNDS, NUM_CLIENTS, X, y, client_data)
    cda_r, _, cda_det, cda_dpr, _, _ = server.run_cda_fedavg(
        NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="activity")
    daaw_r, _, daaw_det, daaw_dpr, _, _, _ = server.run_daaw(
        NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="activity")
    cda_m = server.compute_detection_metrics(cda_dpr, cda_det, DRIFT_EVENTS)
    daaw_m = server.compute_detection_metrics(daaw_dpr, daaw_det, DRIFT_EVENTS)
    return {"seed": seed, "fedavg_acc": fedavg_r[-1], "cda_acc": cda_r[-1], "daaw_acc": daaw_r[-1],
            "cda": cda_m, "daaw": daaw_m}


def run_gas_label_shuffle(seed):
    server_gas.SEED = seed
    np.random.seed(seed)
    X, y = load_gas_dataset()
    client_data = partition_noniid_gas(X, y, num_clients=NUM_CLIENTS, alpha=ALPHA)
    fedavg_r, _, _ = server_gas.run_fedavg_gas(NUM_ROUNDS, NUM_CLIENTS, X, y, client_data)
    cda_r, _, cda_det, cda_dpr, _, _ = server_gas.run_cda_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="label")
    daaw_r, _, daaw_det, daaw_dpr, _, _, _ = server_gas.run_daaw_gas(
        NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="label")
    cda_m = server_gas.compute_detection_metrics(cda_dpr, cda_det, DRIFT_EVENTS)
    daaw_m = server_gas.compute_detection_metrics(daaw_dpr, daaw_det, DRIFT_EVENTS)
    return {"seed": seed, "fedavg_acc": fedavg_r[-1], "cda_acc": cda_r[-1], "daaw_acc": daaw_r[-1],
            "cda": cda_m, "daaw": daaw_m}


def run_gas_sudden_batch(seed):
    server_gas.SEED = seed
    np.random.seed(seed)
    X2, y2, batch_ids = load_gas_dataset_with_batches()
    client_data = partition_noniid_gas(X2, y2, num_clients=NUM_CLIENTS, alpha=ALPHA)
    fedavg_r, _, _ = server_gas.run_fedavg_gas(NUM_ROUNDS, NUM_CLIENTS, X2, y2, client_data)
    cda_r, _, cda_det, cda_dpr, _, _ = server_gas.run_cda_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, client_data, DRIFT_EVENTS, drift_type="batch", batch_ids=batch_ids)
    daaw_r, _, daaw_det, daaw_dpr, _, _, _ = server_gas.run_daaw_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, client_data, DRIFT_EVENTS, drift_type="batch", batch_ids=batch_ids)
    cda_m = server_gas.compute_detection_metrics(cda_dpr, cda_det, DRIFT_EVENTS)
    daaw_m = server_gas.compute_detection_metrics(daaw_dpr, daaw_det, DRIFT_EVENTS)
    return {"seed": seed, "fedavg_acc": fedavg_r[-1], "cda_acc": cda_r[-1], "daaw_acc": daaw_r[-1],
            "cda": cda_m, "daaw": daaw_m}


def run_gas_sequential_batch(seed):
    server_gas.SEED = seed
    np.random.seed(seed)
    X2, y2, batch_ids = load_gas_dataset_with_batches()
    client_data = partition_noniid_gas(X2, y2, num_clients=NUM_CLIENTS, alpha=ALPHA)
    fedavg_r, _, _ = server_gas.run_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, client_data, drift_type="sequential", batch_ids=batch_ids)
    cda_r, _, cda_det, cda_dpr, _, _ = server_gas.run_cda_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, client_data, DRIFT_EVENTS, drift_type="sequential", batch_ids=batch_ids)
    daaw_r, _, daaw_det, daaw_dpr, _, _, _ = server_gas.run_daaw_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, client_data, DRIFT_EVENTS, drift_type="sequential", batch_ids=batch_ids,
        short_window=8, long_window=30, threshold=0.15)
    ground_truth = {i: (NUM_ROUNDS // 10) + 1 for i in range(NUM_CLIENTS)}
    cda_m = server_gas.compute_detection_metrics(cda_dpr, cda_det, ground_truth)
    daaw_m = server_gas.compute_detection_metrics(daaw_dpr, daaw_det, ground_truth)
    return {"seed": seed, "fedavg_acc": fedavg_r[-1], "cda_acc": cda_r[-1], "daaw_acc": daaw_r[-1],
            "cda": cda_m, "daaw": daaw_m}


SCENARIOS = {
    "har_label_shuffle": run_har_label_shuffle,
    "har_activity_drift": run_har_activity_drift,
    "gas_label_shuffle": run_gas_label_shuffle,
    "gas_sudden_batch": run_gas_sudden_batch,
    "gas_sequential_batch": run_gas_sequential_batch,
}


def aggregate_and_test(seed_results, save_path):
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "fedavg_acc", "cda_acc", "daaw_acc",
                          "cda_precision", "cda_recall", "cda_f1", "cda_delay", "cda_fp",
                          "daaw_precision", "daaw_recall", "daaw_f1", "daaw_delay", "daaw_fp"])
        for r in seed_results:
            writer.writerow([
                r["seed"], r["fedavg_acc"], r["cda_acc"], r["daaw_acc"],
                r["cda"]["precision"], r["cda"]["recall"], r["cda"]["f1"],
                r["cda"]["avg_detection_delay"], r["cda"]["total_fp"],
                r["daaw"]["precision"], r["daaw"]["recall"], r["daaw"]["f1"],
                r["daaw"]["avg_detection_delay"], r["daaw"]["total_fp"],
            ])

    def mean_std(values):
        arr = np.array([v for v in values if v is not None], dtype=float)
        return (round(float(arr.mean()), 4), round(float(arr.std(ddof=1)), 4)) if len(arr) > 1 else (
            round(float(arr[0]), 4) if len(arr) == 1 else None, 0.0)

    cda_f1s = [r["cda"]["f1"] for r in seed_results]
    daaw_f1s = [r["daaw"]["f1"] for r in seed_results]

    stats_summary = {
        "n_seeds": len(seed_results),
        "seeds": [r["seed"] for r in seed_results],
        "fedavg_acc": dict(zip(["mean", "std"], mean_std([r["fedavg_acc"] for r in seed_results]))),
        "cda_acc": dict(zip(["mean", "std"], mean_std([r["cda_acc"] for r in seed_results]))),
        "daaw_acc": dict(zip(["mean", "std"], mean_std([r["daaw_acc"] for r in seed_results]))),
        "cda": {
            "precision": dict(zip(["mean", "std"], mean_std([r["cda"]["precision"] for r in seed_results]))),
            "recall": dict(zip(["mean", "std"], mean_std([r["cda"]["recall"] for r in seed_results]))),
            "f1": dict(zip(["mean", "std"], mean_std(cda_f1s))),
            "avg_detection_delay": dict(zip(["mean", "std"], mean_std([r["cda"]["avg_detection_delay"] for r in seed_results]))),
            "total_fp": dict(zip(["mean", "std"], mean_std([r["cda"]["total_fp"] for r in seed_results]))),
        },
        "daaw": {
            "precision": dict(zip(["mean", "std"], mean_std([r["daaw"]["precision"] for r in seed_results]))),
            "recall": dict(zip(["mean", "std"], mean_std([r["daaw"]["recall"] for r in seed_results]))),
            "f1": dict(zip(["mean", "std"], mean_std(daaw_f1s))),
            "avg_detection_delay": dict(zip(["mean", "std"], mean_std([r["daaw"]["avg_detection_delay"] for r in seed_results]))),
            "total_fp": dict(zip(["mean", "std"], mean_std([r["daaw"]["total_fp"] for r in seed_results]))),
        },
    }

    # Paired t-test, DAAW vs CDA on F1 — justified by assumption (small n),
    # not asserted as simply "better" than Wilcoxon.
    if len(set(daaw_f1s)) > 1 or len(set(cda_f1s)) > 1:
        t_stat, p_value = stats.ttest_rel(daaw_f1s, cda_f1s)
        stats_summary["paired_ttest_daaw_vs_cda_f1"] = {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 4),
            "significant_at_0.05": bool(p_value < 0.05),
            "note": "Given n=5, a Wilcoxon signed-rank test cannot reach p<0.05 regardless of "
                    "consistency (minimum attainable two-sided p is 0.0625), so a paired t-test "
                    "is used instead, assuming approximately normal paired differences. "
                    "Descriptive stats (mean/std, per-seed results) are reported regardless of "
                    "significance so conclusions do not rest on this test alone.",
        }
    else:
        stats_summary["paired_ttest_daaw_vs_cda_f1"] = {"note": "F1 identical across all seeds — t-test undefined (zero variance)."}

    with open(os.path.join(save_path, "statistics.json"), "w") as f:
        json.dump(stats_summary, f, indent=2)

    print(f"  CDA  F1: {stats_summary['cda']['f1']['mean']} ± {stats_summary['cda']['f1']['std']}")
    print(f"  DAAW F1: {stats_summary['daaw']['f1']['mean']} ± {stats_summary['daaw']['f1']['std']}")
    if "p_value" in stats_summary["paired_ttest_daaw_vs_cda_f1"]:
        print(f"  Paired t-test p-value: {stats_summary['paired_ttest_daaw_vs_cda_f1']['p_value']}")
    print(f"  Saved: {save_path}/summary.csv, {save_path}/statistics.json")
    return stats_summary


def plot_significance_summary(all_scenario_stats, save_path):
    """One presentable figure: mean ± std for F1, Detection Delay, and False
    Positives, CDA vs DAAW, across all 5 scenarios — the "proof it's not
    luck" figure, as opposed to the raw per-seed CSV/JSON files."""
    os.makedirs(save_path, exist_ok=True)
    scenario_names = list(all_scenario_stats.keys())
    labels = [n.replace("_", " ").title() for n in scenario_names]
    x = np.arange(len(scenario_names))
    width = 0.35

    fig, axes = plt.subplots(3, 1, figsize=(14, 15))

    def safe(values):
        """None means 'never detected in any seed' — plot as 0 with no error
        bar, rather than crashing matplotlib or silently hiding the gap."""
        vals, errs, missing = [], [], []
        for v in values:
            if v["mean"] is None:
                vals.append(0.0)
                errs.append(0.0)
                missing.append(True)
            else:
                vals.append(v["mean"])
                errs.append(v["std"])
                missing.append(False)
        return vals, errs, missing

    metrics = [("f1", "F1 Score"), ("avg_detection_delay", "Detection Delay (rounds)"), ("total_fp", "False Positives")]
    for ax, (key, ylabel) in zip(axes, metrics):
        cda_means, cda_stds, cda_missing = safe([all_scenario_stats[s]["cda"][key] for s in scenario_names])
        daaw_means, daaw_stds, daaw_missing = safe([all_scenario_stats[s]["daaw"][key] for s in scenario_names])

        bars1 = ax.bar(x - width/2, cda_means, width, yerr=cda_stds, capsize=5, label="CDA-FedAvg", color="orange", alpha=0.85)
        bars2 = ax.bar(x + width/2, daaw_means, width, yerr=daaw_stds, capsize=5, label="DAAW", color="green", alpha=0.85)
        for bar, missing in zip(bars1, cda_missing):
            if missing:
                ax.text(bar.get_x() + bar.get_width()/2, 0.02, "N/A", ha="center", fontsize=8, rotation=90)
        for bar, missing in zip(bars2, daaw_missing):
            if missing:
                ax.text(bar.get_x() + bar.get_width()/2, 0.02, "N/A", ha="center", fontsize=8, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} — Mean ± Std across 5 seeds", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Significance Testing — CDA vs DAAW, Mean ± Std Across 5 Random Seeds", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "significance_summary.png"), dpi=150)
    plt.close()
    print(f"Saved: {save_path}/significance_summary.png")


if __name__ == "__main__":
    all_scenario_stats = {}

    for scenario_name, run_fn in SCENARIOS.items():
        print("\n" + "=" * 60)
        print(f"SIGNIFICANCE TESTING: {scenario_name}")
        print("=" * 60)

        seed_results = []
        for seed in SEEDS:
            print(f"\n--- {scenario_name}, seed={seed} ---")
            result = run_fn(seed)
            seed_results.append(result)
            seed_path = f"results/significance/{scenario_name}/seed_{seed}"
            os.makedirs(seed_path, exist_ok=True)
            with open(os.path.join(seed_path, "metrics_summary.json"), "w") as f:
                json.dump(result, f, indent=2)

        print(f"\n--- {scenario_name}: aggregating {len(seed_results)} seeds ---")
        all_scenario_stats[scenario_name] = aggregate_and_test(
            seed_results, f"results/significance/{scenario_name}")

    with open("results/significance/overall_summary.json", "w") as f:
        json.dump(all_scenario_stats, f, indent=2)

    plot_significance_summary(all_scenario_stats, "results/significance")

    print("\n" + "=" * 60)
    print("SIGNIFICANCE TESTING COMPLETE")
    print("results/significance/<scenario>/seed_<N>/metrics_summary.json — per-seed raw results")
    print("results/significance/<scenario>/summary.csv, statistics.json — aggregated")
    print("results/significance/overall_summary.json — everything combined")
    print("results/significance/significance_summary.png — presentable figure for supervisor")
    print("=" * 60)
