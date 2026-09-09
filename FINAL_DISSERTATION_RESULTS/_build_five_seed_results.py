"""Aggregate the scattered V4 raw per-seed JSON files (label_shuffle_v4_all_seeds.json
+ gate_test_seed*_v4.json + gate_test_seed42_v3.json) into proper five_seed_results/
per scenario: individual seed JSONs, summary.csv, statistics.json (mean/SD/paired
t-test vs CDA-FedAvg), and a seed_scatter_f1.png.

This is pure aggregation of real, already-computed V4 numbers - no experiments are
re-run here. Source files are read-only; nothing outside FINAL_DISSERTATION_RESULTS/
is written.
"""
import os
import json
import csv
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # FL_reserach/
OUT = os.path.dirname(os.path.abspath(__file__))  # FINAL_DISSERTATION_RESULTS/
FIVE_SEED_DIR = os.path.join(OUT, "01_PRIMARY_V4_FIXED", "five_seed_results")

SEEDS = [42, 123, 456, 789, 999]

# Collect all raw per-seed records: scenario -> seed -> record
records = {}

with open(os.path.join(ROOT, "label_shuffle_v4_all_seeds.json")) as f:
    label_shuffle_data = json.load(f)
for scenario, seed_map in label_shuffle_data.items():
    records.setdefault(scenario, {})
    for seed_str, rec in seed_map.items():
        records[scenario][int(seed_str)] = rec

gate_files = {
    42: "gate_test_seed42_v3.json",
    123: "gate_test_seed123_v4.json",
    456: "gate_test_seed456_v4.json",
    789: "gate_test_seed789_v4.json",
    999: "gate_test_seed999_v4.json",
}
for seed, fname in gate_files.items():
    with open(os.path.join(ROOT, fname)) as f:
        data = json.load(f)
    for scenario, rec in data.items():
        records.setdefault(scenario, {})
        records[scenario][seed] = rec

print("Scenarios found:", list(records.keys()))
for scenario, seed_map in records.items():
    missing = [s for s in SEEDS if s not in seed_map]
    print(f"  {scenario}: seeds present = {sorted(seed_map.keys())}, missing = {missing}")

def mean_sd(values):
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean, sd

for scenario, seed_map in records.items():
    scenario_dir = os.path.join(FIVE_SEED_DIR, scenario)
    os.makedirs(scenario_dir, exist_ok=True)

    seed_results = []
    for seed in SEEDS:
        if seed not in seed_map:
            continue
        rec = seed_map[seed]
        with open(os.path.join(scenario_dir, f"seed_{seed}.json"), "w") as f:
            json.dump(rec, f, indent=2)
        seed_results.append(rec)

    # summary.csv: one row per seed, method-level metrics
    with open(os.path.join(scenario_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "method", "accuracy", "precision", "recall", "f1",
                          "avg_detection_delay", "total_tp", "total_fp",
                          "detected_events", "total_events"])
        for rec in seed_results:
            writer.writerow([rec["seed"], "fedavg", round(rec["fedavg_acc"], 4), "", "", "", "", "", "", "", ""])
            for method in ("cda", "daaw"):
                m = rec[method]
                acc_key = "cda_acc" if method == "cda" else "daaw_acc"
                writer.writerow([
                    rec["seed"], method, round(rec[acc_key], 4),
                    m["precision"], m["recall"], m["f1"], m["avg_detection_delay"],
                    m["total_tp"], m["total_fp"], m["detected_events"], m["total_events"],
                ])

    # statistics.json: mean +/- SD per method, paired t-test DAAW vs CDA on F1
    cda_f1 = [rec["cda"]["f1"] for rec in seed_results]
    daaw_f1 = [rec["daaw"]["f1"] for rec in seed_results]
    cda_acc = [rec["cda_acc"] for rec in seed_results]
    daaw_acc = [rec["daaw_acc"] for rec in seed_results]
    fedavg_acc = [rec["fedavg_acc"] for rec in seed_results]
    cda_precision = [rec["cda"]["precision"] for rec in seed_results]
    daaw_precision = [rec["daaw"]["precision"] for rec in seed_results]
    cda_recall = [rec["cda"]["recall"] for rec in seed_results]
    daaw_recall = [rec["daaw"]["recall"] for rec in seed_results]
    cda_fp = [rec["cda"]["total_fp"] for rec in seed_results]
    daaw_fp = [rec["daaw"]["total_fp"] for rec in seed_results]
    cda_delay = [rec["cda"]["avg_detection_delay"] for rec in seed_results if rec["cda"]["avg_detection_delay"] is not None]
    daaw_delay = [rec["daaw"]["avg_detection_delay"] for rec in seed_results if rec["daaw"]["avg_detection_delay"] is not None]

    ttest_note = None
    try:
        if len(set(np.round(np.array(daaw_f1) - np.array(cda_f1), 8))) == 1 and (np.array(daaw_f1) - np.array(cda_f1))[0] == 0:
            t_stat, p_value = float("nan"), float("nan")
            ttest_note = "F1 identical across all seeds - t-test undefined (zero variance)."
        else:
            t_stat, p_value = stats.ttest_rel(daaw_f1, cda_f1)
            t_stat, p_value = float(t_stat), float(p_value)
    except Exception as e:
        t_stat, p_value = None, None
        ttest_note = f"t-test failed: {e}"

    stats_out = {
        "scenario": scenario,
        "seeds": [rec["seed"] for rec in seed_results],
        "fedavg_accuracy": {"mean": mean_sd(fedavg_acc)[0], "sd": mean_sd(fedavg_acc)[1], "raw": fedavg_acc},
        "cda_accuracy": {"mean": mean_sd(cda_acc)[0], "sd": mean_sd(cda_acc)[1], "raw": cda_acc},
        "daaw_accuracy": {"mean": mean_sd(daaw_acc)[0], "sd": mean_sd(daaw_acc)[1], "raw": daaw_acc},
        "cda_precision": {"mean": mean_sd(cda_precision)[0], "sd": mean_sd(cda_precision)[1]},
        "daaw_precision": {"mean": mean_sd(daaw_precision)[0], "sd": mean_sd(daaw_precision)[1]},
        "cda_recall": {"mean": mean_sd(cda_recall)[0], "sd": mean_sd(cda_recall)[1]},
        "daaw_recall": {"mean": mean_sd(daaw_recall)[0], "sd": mean_sd(daaw_recall)[1]},
        "cda_f1": {"mean": mean_sd(cda_f1)[0], "sd": mean_sd(cda_f1)[1], "raw": cda_f1},
        "daaw_f1": {"mean": mean_sd(daaw_f1)[0], "sd": mean_sd(daaw_f1)[1], "raw": daaw_f1},
        "cda_total_fp": {"mean": mean_sd(cda_fp)[0], "sd": mean_sd(cda_fp)[1]},
        "daaw_total_fp": {"mean": mean_sd(daaw_fp)[0], "sd": mean_sd(daaw_fp)[1]},
        "cda_avg_detection_delay": {"mean": mean_sd(cda_delay)[0] if cda_delay else None, "sd": mean_sd(cda_delay)[1] if cda_delay else None, "n": len(cda_delay)},
        "daaw_avg_detection_delay": {"mean": mean_sd(daaw_delay)[0] if daaw_delay else None, "sd": mean_sd(daaw_delay)[1] if daaw_delay else None, "n": len(daaw_delay)},
        "paired_ttest_daaw_vs_cda_f1": {"t_stat": t_stat, "p_value": p_value, "note": ttest_note},
    }
    with open(os.path.join(scenario_dir, "statistics.json"), "w") as f:
        json.dump(stats_out, f, indent=2)

    # seed_scatter_f1.png
    fig, ax = plt.subplots(figsize=(7, 5))
    seeds_x = [rec["seed"] for rec in seed_results]
    ax.scatter(seeds_x, cda_f1, label="CDA-FedAvg", color="orange", s=80, zorder=3)
    ax.scatter(seeds_x, daaw_f1, label="DAAW", color="green", s=80, zorder=3)
    ax.axhline(mean_sd(cda_f1)[0], color="orange", linestyle="--", alpha=0.5)
    ax.axhline(mean_sd(daaw_f1)[0], color="green", linestyle="--", alpha=0.5)
    ax.set_xlabel("Seed")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{scenario.replace('_', ' ').title()} - F1 per Seed (n=5)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(scenario_dir, "seed_scatter_f1.png"), dpi=150)
    plt.close()

    print(f"{scenario}: wrote {len(seed_results)} seed JSONs, summary.csv, statistics.json, seed_scatter_f1.png")

print("\nDone.")
