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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats

from utils_v3 import load_har_dataset, partition_noniid
from utils_gas_v3 import load_gas_dataset, load_gas_dataset_with_batches, partition_noniid_gas
import server_v3 as server
import server_gas_v3 as server_gas

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
    fedavg_r, _, _ = server.run_fedavg(NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="label")
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
    fedavg_r, _, _ = server.run_fedavg(NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="activity")
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
    fedavg_r, _, _ = server_gas.run_fedavg_gas(NUM_ROUNDS, NUM_CLIENTS, X, y, client_data, DRIFT_EVENTS, drift_type="label")
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
    fedavg_r, _, _ = server_gas.run_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, client_data, DRIFT_EVENTS, drift_type="batch", batch_ids=batch_ids)
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
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, client_data, DRIFT_EVENTS, drift_type="sequential", batch_ids=batch_ids)
    cda_r, _, cda_det, cda_dpr, _, _ = server_gas.run_cda_fedavg_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, client_data, DRIFT_EVENTS, drift_type="sequential", batch_ids=batch_ids)
    daaw_r, _, daaw_det, daaw_dpr, _, _, _ = server_gas.run_daaw_gas(
        NUM_ROUNDS, NUM_CLIENTS, X2, y2, client_data, DRIFT_EVENTS, drift_type="sequential", batch_ids=batch_ids,
        short_window=8, long_window=30, threshold=0.15)
    # Ground truth for Gas Sequential Batch: every client transitions off
    # its starting batch at the SAME round (round 6 with NUM_ROUNDS=50, i.e.
    # 50//10+1), since every client moves through batch 1 -> batch 2 -> ...
    # in lockstep. This is defined as one first-transition event per client;
    # any additional detections after that first one are alarm episodes
    # (operational behaviour, e.g. re-triggering on later batch boundaries),
    # not separate ground-truth events — event-level recall/precision below
    # are computed against this one-event-per-client definition only.
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
            "total_tp": dict(zip(["mean", "std"], mean_std([r["cda"]["total_tp"] for r in seed_results]))),
        },
        "daaw": {
            "precision": dict(zip(["mean", "std"], mean_std([r["daaw"]["precision"] for r in seed_results]))),
            "recall": dict(zip(["mean", "std"], mean_std([r["daaw"]["recall"] for r in seed_results]))),
            "f1": dict(zip(["mean", "std"], mean_std(daaw_f1s))),
            "avg_detection_delay": dict(zip(["mean", "std"], mean_std([r["daaw"]["avg_detection_delay"] for r in seed_results]))),
            "total_fp": dict(zip(["mean", "std"], mean_std([r["daaw"]["total_fp"] for r in seed_results]))),
            "total_tp": dict(zip(["mean", "std"], mean_std([r["daaw"]["total_tp"] for r in seed_results]))),
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
    """One presentable figure: mean ± std for F1 (with paired t-test
    significance annotations), Detection Delay (symlog scale), and False
    Positives, FedAvg vs CDA vs DAAW, across all 5 scenarios — the "proof
    it's not luck" figure, as opposed to the raw per-seed CSV/JSON files.
    FedAvg makes zero detection attempts, so its F1/FP are trivially 0 and
    its delay is undefined (N/A) — shown explicitly (bold outlined diamond
    marker) rather than omitted, since "no detection at all" is itself part
    of the comparison."""
    os.makedirs(save_path, exist_ok=True)
    scenario_names = list(all_scenario_stats.keys())
    labels = [n.replace("_", " ").title() for n in scenario_names]
    x = np.arange(len(scenario_names))
    width = 0.32

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

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

    fedavg_diamond = Line2D([0], [0], marker="D", color="none", markerfacecolor="blue",
                             markeredgecolor="black", markersize=12, markeredgewidth=1.3,
                             label="FedAvg: no detection mechanism")

    # ── Panel 1: F1, with paired t-test significance brackets ──
    ax = axes[0]
    cda_means, cda_stds, _ = safe([all_scenario_stats[s]["cda"]["f1"] for s in scenario_names])
    daaw_means, daaw_stds, _ = safe([all_scenario_stats[s]["daaw"]["f1"] for s in scenario_names])
    ax.bar(x - width / 2, cda_means, width, yerr=cda_stds, capsize=4, color="orange")
    ax.bar(x + width / 2, daaw_means, width, yerr=daaw_stds, capsize=4, color="green")
    for xx in x:
        ax.plot(xx, -0.06, marker="D", color="blue", markersize=11, clip_on=False,
                 zorder=5, markeredgecolor="black", markeredgewidth=1.2)
    for i, s in enumerate(scenario_names):
        tt = all_scenario_stats[s].get("paired_ttest_daaw_vs_cda_f1", {})
        p_value = tt.get("p_value")
        if p_value is None:
            continue
        top = max(cda_means[i] + cda_stds[i], daaw_means[i] + daaw_stds[i]) + 0.05
        if tt.get("significant_at_0.05"):
            stars = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else "*")
            label, color = f"{stars}\np={p_value:.4f}", "black"
        else:
            label, color = f"n.s.\np={p_value:.3f}", "gray"
        ax.plot([i - width / 2, i - width / 2, i + width / 2, i + width / 2],
                 [top - 0.02, top, top, top - 0.02], color=color, linewidth=1)
        ax.text(i, top + 0.015, label, ha="center", fontsize=8.5, color=color, fontweight="bold")
    ax.set_ylim(-0.15, 1.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, rotation=10)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("F1 Score — mean ± std across 5 seeds, with paired t-test significance (DAAW vs CDA)",
                 fontsize=12.5, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 2: Detection Delay, symlog scale ──
    ax = axes[1]
    cda_means, cda_stds, _ = safe([all_scenario_stats[s]["cda"]["avg_detection_delay"] for s in scenario_names])
    daaw_means, daaw_stds, _ = safe([all_scenario_stats[s]["daaw"]["avg_detection_delay"] for s in scenario_names])
    ax.bar(x - width / 2, cda_means, width, yerr=cda_stds, capsize=4, color="orange")
    ax.bar(x + width / 2, daaw_means, width, yerr=daaw_stds, capsize=4, color="green")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(-0.3, max(30, max(daaw_means, default=1) * 1.5))
    for xx in x:
        ax.plot(xx, -0.2, marker="D", color="blue", markersize=11, clip_on=False,
                 zorder=5, markeredgecolor="black", markeredgewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, rotation=10)
    ax.set_ylabel("Detection Delay (rounds, symlog scale)", fontsize=11)
    ax.set_title("Detection Delay — mean ± std across 5 seeds", fontsize=12.5, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y", which="both")

    # ── Panel 3: False Positives ──
    ax = axes[2]
    cda_means, cda_stds, _ = safe([all_scenario_stats[s]["cda"]["total_fp"] for s in scenario_names])
    daaw_means, daaw_stds, _ = safe([all_scenario_stats[s]["daaw"]["total_fp"] for s in scenario_names])
    ax.bar(x - width / 2, cda_means, width, yerr=cda_stds, capsize=4, color="orange")
    ax.bar(x + width / 2, daaw_means, width, yerr=daaw_stds, capsize=4, color="green")
    for xx in x:
        ax.plot(xx, -0.6, marker="D", color="blue", markersize=11, clip_on=False,
                 zorder=5, markeredgecolor="black", markeredgewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, rotation=10)
    ax.set_ylabel("False Positives", fontsize=11)
    ax.set_title("False Positives — mean ± std across 5 seeds", fontsize=12.5, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Significance Testing — FedAvg vs CDA-FedAvg vs DAAW (5 random seeds)",
                 fontsize=14.5, fontweight="bold", y=0.998)
    fig.legend(handles=[fedavg_diamond, Patch(facecolor="orange", label="CDA-FedAvg"),
                         Patch(facecolor="green", label="DAAW")],
               fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3, framealpha=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, "significance_summary.png"), dpi=150)
    plt.close()
    print(f"Saved: {save_path}/significance_summary.png")


def plot_tp_fp_summary(all_scenario_stats, save_path):
    """New chart: True Positives vs False Positives, diverging bar, across
    all 5 scenarios — replaces the old per-round Graph 3 concept at the
    summary level, using 5-seed mean totals."""
    os.makedirs(save_path, exist_ok=True)
    scenario_names = list(all_scenario_stats.keys())
    labels = [n.replace("_", " ").title() for n in scenario_names]
    y = np.arange(len(scenario_names))
    h = 0.25

    cda_tp = [all_scenario_stats[s]["cda"]["total_tp"]["mean"] for s in scenario_names]
    cda_fp = [all_scenario_stats[s]["cda"]["total_fp"]["mean"] for s in scenario_names]
    daaw_tp = [all_scenario_stats[s]["daaw"]["total_tp"]["mean"] for s in scenario_names]
    daaw_fp = [all_scenario_stats[s]["daaw"]["total_fp"]["mean"] for s in scenario_names]

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (label, tp, fp, color) in enumerate([
        ("FedAvg", [0] * len(scenario_names), [0] * len(scenario_names), "blue"),
        ("CDA-FedAvg", cda_tp, cda_fp, "orange"),
        ("DAAW", daaw_tp, daaw_fp, "green"),
    ]):
        offset = (i - 1) * h
        ax.barh(y + offset, tp, height=h, color=color, alpha=0.85)
        ax.barh(y + offset, [-f for f in fp], height=h, color=color, alpha=0.4, hatch="//")
        if label == "FedAvg":
            for yy in y:
                ax.plot(0, yy + offset, marker="D", color="blue", markersize=11,
                        zorder=5, markeredgecolor="black", markeredgewidth=1.2)

    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("<- False Alarm Episodes          True Detection Episodes ->", fontsize=11)
    ax.set_title("Alarm Episode Counts: True vs False Detection Episodes\n(mean over 5 seeds)", fontsize=13, fontweight="bold")
    legend_elems = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="blue", markeredgecolor="black",
               markersize=12, markeredgewidth=1.3, label="FedAvg (no detection mechanism)"),
        Patch(facecolor="orange", alpha=0.85, label="CDA-FedAvg"),
        Patch(facecolor="green", alpha=0.85, label="DAAW"),
        Patch(facecolor="gray", alpha=0.4, hatch="//", label="(hatched = False Positives)"),
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "tp_vs_fp_summary.png"), dpi=150)
    plt.close()
    print(f"Saved: {save_path}/tp_vs_fp_summary.png")


def plot_precision_f1_summary(all_scenario_stats, save_path):
    """New charts: Precision and F1 dumbbell plots, CDA-FedAvg vs DAAW,
    across all 5 scenarios (5-seed means)."""
    os.makedirs(save_path, exist_ok=True)
    scenario_names = list(all_scenario_stats.keys())
    labels = [n.replace("_", " ").title() for n in scenario_names]
    y = np.arange(len(scenario_names))

    for metric, ylabel, fname in [("precision", "Precision", "precision_summary.png"),
                                   ("f1", "F1 Score", "f1_summary.png")]:
        cda_vals = [all_scenario_stats[s]["cda"][metric]["mean"] for s in scenario_names]
        daaw_vals = [all_scenario_stats[s]["daaw"][metric]["mean"] for s in scenario_names]
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for i, yy in enumerate(y):
            ax.plot([cda_vals[i], daaw_vals[i]], [yy, yy], color="gray", linewidth=2, zorder=1)
            ax.scatter(cda_vals[i], yy, color="orange", s=160, zorder=3,
                       label="CDA-FedAvg" if i == 0 else None)
            ax.scatter(daaw_vals[i], yy, color="green", s=160, zorder=3,
                       label="DAAW" if i == 0 else None)
            ax.scatter(0, yy, marker="D", color="blue", s=170, zorder=3,
                       edgecolor="black", linewidth=1.2,
                       label="FedAvg (no detection)" if i == 0 else None)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel(f"{ylabel} (mean over 5 seeds)", fontsize=11)
        ax.set_title(f"{ylabel} — CDA-FedAvg vs DAAW", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, fname), dpi=150)
        plt.close()
        print(f"Saved: {save_path}/{fname}")


def plot_detection_delay_summary(all_scenario_stats, save_path):
    """Standalone Detection Delay chart (symlog scale, CDA-FedAvg vs DAAW),
    across all 5 scenarios — 5-seed means, same data as the middle panel of
    plot_significance_summary but as its own single-purpose figure."""
    os.makedirs(save_path, exist_ok=True)
    scenario_names = list(all_scenario_stats.keys())
    labels = [n.replace("_", " ").title() for n in scenario_names]
    y = np.arange(len(scenario_names))
    w = 0.35

    cda_delay = [all_scenario_stats[s]["cda"]["avg_detection_delay"]["mean"] or 0.0 for s in scenario_names]
    daaw_delay = [all_scenario_stats[s]["daaw"]["avg_detection_delay"]["mean"] or 0.0 for s in scenario_names]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(y - w / 2, cda_delay, w, color="orange", label="CDA-FedAvg")
    ax.bar(y + w / 2, daaw_delay, w, color="green", label="DAAW")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(-0.5, max(30, max(daaw_delay, default=1) * 1.5))
    for yy in y:
        ax.plot(yy, -0.35, marker="D", color="blue", markersize=11, clip_on=False,
                 markeredgecolor="black", markeredgewidth=1.2)
    for yy, v in zip(y, cda_delay):
        ax.text(yy - w / 2, v + v * 0.15 + 0.05, "0" if v == 0 else f"{v:.1f}",
                 fontsize=8.5, color="darkorange", ha="center", fontweight="bold")
    for yy, v in zip(y, daaw_delay):
        ax.text(yy + w / 2, v + v * 0.15 + 0.05, "0" if v == 0 else f"{v:.1f}",
                 fontsize=8.5, color="darkgreen", ha="center", fontweight="bold")
    ax.set_xticks(y)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Avg. Detection Delay (rounds late, symlog scale)", fontsize=10.5)
    ax.set_title("Detection Delay — CDA-FedAvg vs DAAW\n(mean over 5 seeds; lower is better)",
                 fontsize=13, fontweight="bold")
    legend_elems = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor="blue", markeredgecolor="black",
               markersize=12, markeredgewidth=1.3, label="FedAvg: no detection mechanism"),
        Patch(facecolor="orange", label="CDA-FedAvg"),
        Patch(facecolor="green", label="DAAW"),
    ]
    ax.legend(handles=legend_elems, fontsize=8.7, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "detection_delay_summary.png"), dpi=150)
    plt.close()
    print(f"Saved: {save_path}/detection_delay_summary.png")


def plot_single_seed_summary(result, save_path):
    """Turn one seed's own metrics_summary.json into a small 2-panel image,
    saved alongside the JSON in that same seed folder — so every per-seed
    result has a graph, not just the raw numbers."""
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    methods = ["FedAvg", "CDA-FedAvg", "DAAW"]
    accs = [result["fedavg_acc"], result["cda_acc"], result["daaw_acc"]]
    axes[0].bar(methods, accs, color=["blue", "orange", "green"], alpha=0.85)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Final Accuracy")
    axes[0].set_title(f"Seed {result['seed']} — Final Accuracy")
    axes[0].grid(True, alpha=0.3, axis="y")

    metric_names = ["precision", "recall", "f1"]
    x = np.arange(len(metric_names))
    width = 0.35
    cda_vals = [result["cda"][m] for m in metric_names]
    daaw_vals = [result["daaw"][m] for m in metric_names]
    axes[1].bar(x - width / 2, cda_vals, width, label="CDA-FedAvg", color="orange")
    axes[1].bar(x + width / 2, daaw_vals, width, label="DAAW", color="green")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.capitalize() for m in metric_names])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"Seed {result['seed']} — Detection Metrics")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "seed_summary.png"), dpi=150)
    plt.close()
    print(f"Saved: {save_path}/seed_summary.png")


def plot_seed_scatter(seed_results, save_path, scenario_name):
    """Per-scenario scatter of each seed's F1 (CDA vs DAAW), built straight
    from the same per-seed numbers already written to summary.csv/
    statistics.json — visual proof the mean±std bar in
    significance_summary.png isn't hiding wild seed-to-seed swings."""
    os.makedirs(save_path, exist_ok=True)
    seeds = [r["seed"] for r in seed_results]
    cda_f1 = [r["cda"]["f1"] for r in seed_results]
    daaw_f1 = [r["daaw"]["f1"] for r in seed_results]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([0.9] * len(cda_f1), cda_f1, color="orange", s=80, zorder=3, label="CDA-FedAvg (per seed)")
    ax.scatter([1.1] * len(daaw_f1), daaw_f1, color="green", s=80, zorder=3, label="DAAW (per seed)")
    ax.hlines(np.mean(cda_f1), 0.8, 1.0, color="orange", linewidth=2, label="CDA-FedAvg (mean)")
    ax.hlines(np.mean(daaw_f1), 1.0, 1.2, color="green", linewidth=2, label="DAAW (mean)")
    for x, y, s in zip([0.9] * len(cda_f1), cda_f1, seeds):
        ax.annotate(str(s), (x, y), fontsize=7, ha="right", xytext=(-4, 0), textcoords="offset points")
    for x, y, s in zip([1.1] * len(daaw_f1), daaw_f1, seeds):
        ax.annotate(str(s), (x, y), fontsize=7, ha="left", xytext=(4, 0), textcoords="offset points")
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([0.9, 1.1])
    ax.set_xticklabels(["CDA-FedAvg", "DAAW"])
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{scenario_name.replace('_', ' ').title()} — F1 per Seed (n=5)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "seed_scatter_f1.png"), dpi=150)
    plt.close()
    print(f"Saved: {save_path}/seed_scatter_f1.png")


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
            seed_path = f"myfinalresult_v3/significance/{scenario_name}/seed_{seed}"
            os.makedirs(seed_path, exist_ok=True)
            with open(os.path.join(seed_path, "metrics_summary.json"), "w") as f:
                json.dump(result, f, indent=2)
            plot_single_seed_summary(result, seed_path)

        print(f"\n--- {scenario_name}: aggregating {len(seed_results)} seeds ---")
        all_scenario_stats[scenario_name] = aggregate_and_test(
            seed_results, f"myfinalresult_v3/significance/{scenario_name}")
        plot_seed_scatter(seed_results, f"myfinalresult_v3/significance/{scenario_name}", scenario_name)

    with open("myfinalresult_v3/significance/overall_summary.json", "w") as f:
        json.dump(all_scenario_stats, f, indent=2)

    plot_significance_summary(all_scenario_stats, "myfinalresult_v3/significance")
    plot_tp_fp_summary(all_scenario_stats, "myfinalresult_v3/significance")
    plot_precision_f1_summary(all_scenario_stats, "myfinalresult_v3/significance")
    plot_detection_delay_summary(all_scenario_stats, "myfinalresult_v3/significance")

    print("\n" + "=" * 60)
    print("SIGNIFICANCE TESTING COMPLETE")
    print("myfinalresult_v3/significance/<scenario>/seed_<N>/metrics_summary.json — per-seed raw results")
    print("myfinalresult_v3/significance/<scenario>/summary.csv, statistics.json — aggregated")
    print("myfinalresult_v3/significance/<scenario>/seed_scatter_f1.png — per-seed F1 scatter (CDA vs DAAW)")
    print("myfinalresult_v3/significance/overall_summary.json — everything combined")
    print("myfinalresult_v3/significance/significance_summary.png — presentable figure for supervisor")
    print("=" * 60)
