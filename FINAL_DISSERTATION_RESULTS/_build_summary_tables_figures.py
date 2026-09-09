"""Build summary_tables/ (CSVs) and summary_figures/ (PNGs) for
01_PRIMARY_V4_FIXED from the aggregated five_seed_results/ produced by
_build_five_seed_results.py. Pure post-processing of already-aggregated,
already-verified numbers - no experiments are re-run here.
"""
import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
FIVE_SEED_DIR = os.path.join(OUT, "01_PRIMARY_V4_FIXED", "five_seed_results")
TABLES_DIR = os.path.join(OUT, "01_PRIMARY_V4_FIXED", "summary_tables")
FIGS_DIR = os.path.join(OUT, "01_PRIMARY_V4_FIXED", "summary_figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

SCENARIOS = ["har_label_shuffle", "har_activity_drift", "gas_label_shuffle",
             "gas_sudden_batch", "gas_sequential_batch"]
SCENARIO_LABELS = ["HAR Label\nShuffle", "HAR Activity\nDrift", "Gas Label\nShuffle",
                   "Gas Sudden\nBatch", "Gas Sequential\nBatch"]

stats_by_scenario = {}
for sc in SCENARIOS:
    with open(os.path.join(FIVE_SEED_DIR, sc, "statistics.json")) as f:
        stats_by_scenario[sc] = json.load(f)

# ---------- summary_tables/main_results.csv ----------
with open(os.path.join(TABLES_DIR, "main_results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "method", "precision_mean", "recall_mean", "f1_mean", "f1_sd",
                "false_alarms_mean", "detection_delay_mean", "accuracy_mean"])
    for sc in SCENARIOS:
        s = stats_by_scenario[sc]
        w.writerow([sc, "CDA-FedAvg", round(s["cda_precision"]["mean"], 4), round(s["cda_recall"]["mean"], 4),
                    round(s["cda_f1"]["mean"], 4), round(s["cda_f1"]["sd"], 4),
                    round(s["cda_total_fp"]["mean"], 2),
                    round(s["cda_avg_detection_delay"]["mean"], 2) if s["cda_avg_detection_delay"]["mean"] is not None else "",
                    round(s["cda_accuracy"]["mean"], 4)])
        w.writerow([sc, "DAAW", round(s["daaw_precision"]["mean"], 4), round(s["daaw_recall"]["mean"], 4),
                    round(s["daaw_f1"]["mean"], 4), round(s["daaw_f1"]["sd"], 4),
                    round(s["daaw_total_fp"]["mean"], 2),
                    round(s["daaw_avg_detection_delay"]["mean"], 2) if s["daaw_avg_detection_delay"]["mean"] is not None else "",
                    round(s["daaw_accuracy"]["mean"], 4)])
        w.writerow([sc, "FedAvg", "", "", "", "", "", "", round(s["fedavg_accuracy"]["mean"], 4)])

# ---------- summary_tables/per_seed_results.csv ----------
with open(os.path.join(TABLES_DIR, "per_seed_results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "seed", "method", "f1", "precision", "recall", "avg_detection_delay", "total_fp", "accuracy"])
    for sc in SCENARIOS:
        for seed in [42, 123, 456, 789, 999]:
            with open(os.path.join(FIVE_SEED_DIR, sc, f"seed_{seed}.json")) as sf:
                rec = json.load(sf)
            w.writerow([sc, seed, "CDA-FedAvg", rec["cda"]["f1"], rec["cda"]["precision"], rec["cda"]["recall"],
                        rec["cda"]["avg_detection_delay"], rec["cda"]["total_fp"], round(rec["cda_acc"], 4)])
            w.writerow([sc, seed, "DAAW", rec["daaw"]["f1"], rec["daaw"]["precision"], rec["daaw"]["recall"],
                        rec["daaw"]["avg_detection_delay"], rec["daaw"]["total_fp"], round(rec["daaw_acc"], 4)])
            w.writerow([sc, seed, "FedAvg", "", "", "", "", "", round(rec["fedavg_acc"], 4)])

# ---------- summary_tables/statistical_tests.csv ----------
with open(os.path.join(TABLES_DIR, "statistical_tests.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "cda_f1_mean", "cda_f1_sd", "daaw_f1_mean", "daaw_f1_sd", "t_statistic", "p_value", "significant_at_0.05"])
    for sc in SCENARIOS:
        s = stats_by_scenario[sc]
        tt = s["paired_ttest_daaw_vs_cda_f1"]
        p = tt["p_value"]
        sig = "Yes" if (p is not None and p == p and p < 0.05) else ("No" if p is not None and p == p else "N/A")
        w.writerow([sc, round(s["cda_f1"]["mean"], 4), round(s["cda_f1"]["sd"], 4),
                    round(s["daaw_f1"]["mean"], 4), round(s["daaw_f1"]["sd"], 4),
                    round(tt["t_stat"], 3) if tt["t_stat"] is not None and tt["t_stat"] == tt["t_stat"] else "",
                    f"{p:.4g}" if p is not None and p == p else "", sig])

# ---------- summary_tables/experimental_configuration.csv ----------
with open(os.path.join(TABLES_DIR, "experimental_configuration.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["component", "configuration"])
    for row in [
        ("Clients", "10"),
        ("Rounds", "50"),
        ("Datasets", "UCI HAR, Gas Sensor Array"),
        ("Scenarios", "5 (label shuffle x2, activity drift, sudden batch, sequential batch)"),
        ("Seeds", "5 (42, 123, 456, 789, 999)"),
        ("Methods", "FedAvg, CDA-FedAvg, DAAW"),
        ("Metrics", "Event-level Precision, Recall, F1, false alarm episodes, detection delay, accuracy"),
        ("Pipeline version", "V4 (corrected client-ownership partitioning, sample-weighted loss, permanent train/test boundary)"),
    ]:
        w.writerow(row)

print("Wrote 4 CSVs to summary_tables/")

# ============================================================
# Figures
# ============================================================
CDA_COLOR, DAAW_COLOR, FEDAVG_COLOR = "#F96167", "#02C39A", "#8A93B8"

def scenario_arrays(key_getter):
    return [key_getter(stats_by_scenario[sc]) for sc in SCENARIOS]

# 01_final_accuracy_comparison.png
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(SCENARIOS))
w_bar = 0.25
fedavg_acc = scenario_arrays(lambda s: s["fedavg_accuracy"]["mean"])
cda_acc = scenario_arrays(lambda s: s["cda_accuracy"]["mean"])
daaw_acc = scenario_arrays(lambda s: s["daaw_accuracy"]["mean"])
ax.bar(x - w_bar, fedavg_acc, w_bar, label="FedAvg", color=FEDAVG_COLOR)
ax.bar(x, cda_acc, w_bar, label="CDA-FedAvg", color=CDA_COLOR)
ax.bar(x + w_bar, daaw_acc, w_bar, label="DAAW", color=DAAW_COLOR)
ax.set_xticks(x); ax.set_xticklabels(SCENARIO_LABELS)
ax.set_ylabel("Final Accuracy (5-seed mean)")
ax.set_title("Final Accuracy Comparison - V4 (corrected)")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(FIGS_DIR, "01_final_accuracy_comparison.png"), dpi=150); plt.close()

# 02_f1_comparison.png
fig, ax = plt.subplots(figsize=(11, 6))
cda_f1 = scenario_arrays(lambda s: s["cda_f1"]["mean"])
daaw_f1 = scenario_arrays(lambda s: s["daaw_f1"]["mean"])
cda_f1_sd = scenario_arrays(lambda s: s["cda_f1"]["sd"])
daaw_f1_sd = scenario_arrays(lambda s: s["daaw_f1"]["sd"])
ax.bar(x - w_bar / 2, cda_f1, w_bar, yerr=cda_f1_sd, label="CDA-FedAvg", color=CDA_COLOR, capsize=4)
ax.bar(x + w_bar / 2, daaw_f1, w_bar, yerr=daaw_f1_sd, label="DAAW", color=DAAW_COLOR, capsize=4)
ax.set_xticks(x); ax.set_xticklabels(SCENARIO_LABELS)
ax.set_ylabel("F1 Score (5-seed mean +/- SD)")
ax.set_title("F1 Score Comparison - V4 (corrected)")
ax.set_ylim(0, 1.15)
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(FIGS_DIR, "02_f1_comparison.png"), dpi=150); plt.close()

# 03_precision_recall_comparison.png
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
cda_p = scenario_arrays(lambda s: s["cda_precision"]["mean"])
daaw_p = scenario_arrays(lambda s: s["daaw_precision"]["mean"])
cda_r = scenario_arrays(lambda s: s["cda_recall"]["mean"])
daaw_r = scenario_arrays(lambda s: s["daaw_recall"]["mean"])
axes[0].bar(x - w_bar / 2, cda_p, w_bar, label="CDA-FedAvg", color=CDA_COLOR)
axes[0].bar(x + w_bar / 2, daaw_p, w_bar, label="DAAW", color=DAAW_COLOR)
axes[0].set_xticks(x); axes[0].set_xticklabels(SCENARIO_LABELS, fontsize=8)
axes[0].set_title("Precision"); axes[0].set_ylim(0, 1.15); axes[0].legend(); axes[0].grid(True, alpha=0.3, axis="y")
axes[1].bar(x - w_bar / 2, cda_r, w_bar, label="CDA-FedAvg", color=CDA_COLOR)
axes[1].bar(x + w_bar / 2, daaw_r, w_bar, label="DAAW", color=DAAW_COLOR)
axes[1].set_xticks(x); axes[1].set_xticklabels(SCENARIO_LABELS, fontsize=8)
axes[1].set_title("Recall"); axes[1].set_ylim(0, 1.15); axes[1].legend(); axes[1].grid(True, alpha=0.3, axis="y")
fig.suptitle("Precision and Recall Comparison - V4 (corrected)")
plt.tight_layout(); plt.savefig(os.path.join(FIGS_DIR, "03_precision_recall_comparison.png"), dpi=150); plt.close()

# 04_detected_missed_false_alarms.png
fig, ax = plt.subplots(figsize=(11, 6))
cda_fp = scenario_arrays(lambda s: s["cda_total_fp"]["mean"])
daaw_fp = scenario_arrays(lambda s: s["daaw_total_fp"]["mean"])
ax.barh(x - w_bar / 2, [-v for v in cda_fp], w_bar, label="CDA-FedAvg false alarms", color=CDA_COLOR, hatch="//")
ax.barh(x + w_bar / 2, [-v for v in daaw_fp], w_bar, label="DAAW false alarms", color=DAAW_COLOR, hatch="//")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(x); ax.set_yticklabels(SCENARIO_LABELS)
ax.set_xlabel("<- False Alarm Episodes (mean over 5 seeds)")
ax.set_title("False Alarm Episodes - V4 (corrected)")
ax.legend(); ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout(); plt.savefig(os.path.join(FIGS_DIR, "04_detected_missed_false_alarms.png"), dpi=150); plt.close()

# 05_detection_delay.png
fig, ax = plt.subplots(figsize=(11, 6))
cda_delay = scenario_arrays(lambda s: s["cda_avg_detection_delay"]["mean"] or 0)
daaw_delay = scenario_arrays(lambda s: s["daaw_avg_detection_delay"]["mean"] or 0)
ax.bar(x - w_bar / 2, cda_delay, w_bar, label="CDA-FedAvg", color=CDA_COLOR)
ax.bar(x + w_bar / 2, daaw_delay, w_bar, label="DAAW", color=DAAW_COLOR)
ax.set_xticks(x); ax.set_xticklabels(SCENARIO_LABELS)
ax.set_ylabel("Avg Detection Delay (rounds, lower is better)")
ax.set_title("Detection Delay Comparison - V4 (corrected)")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(FIGS_DIR, "05_detection_delay.png"), dpi=150); plt.close()

# 06_per_seed_f1.png (combined scatter across all scenarios)
fig, ax = plt.subplots(figsize=(12, 6))
seeds = [42, 123, 456, 789, 999]
for i, sc in enumerate(SCENARIOS):
    with open(os.path.join(FIVE_SEED_DIR, sc, "statistics.json")) as f:
        s = json.load(f)
    cda_raw = s["cda_f1"]["raw"]
    daaw_raw = s["daaw_f1"]["raw"]
    ax.scatter([i - 0.1] * len(cda_raw), cda_raw, color=CDA_COLOR, alpha=0.7, s=50)
    ax.scatter([i + 0.1] * len(daaw_raw), daaw_raw, color=DAAW_COLOR, alpha=0.7, s=50)
ax.set_xticks(range(len(SCENARIOS))); ax.set_xticklabels(SCENARIO_LABELS)
ax.set_ylabel("F1 Score (individual seeds)")
ax.set_title("Per-Seed F1 Scatter (5 seeds each) - V4 (corrected)")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=CDA_COLOR, label="CDA-FedAvg", markersize=8),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor=DAAW_COLOR, label="DAAW", markersize=8)])
ax.set_ylim(-0.05, 1.1); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(FIGS_DIR, "06_per_seed_f1.png"), dpi=150); plt.close()

# 07_statistical_significance.png
fig, ax = plt.subplots(figsize=(11, 6))
pvals = []
for sc in SCENARIOS:
    p = stats_by_scenario[sc]["paired_ttest_daaw_vs_cda_f1"]["p_value"]
    pvals.append(p if (p is not None and p == p) else 1.0)
neg_log_p = [-np.log10(max(p, 1e-10)) for p in pvals]
colors = [DAAW_COLOR if stats_by_scenario[sc]["daaw_f1"]["mean"] > stats_by_scenario[sc]["cda_f1"]["mean"] else CDA_COLOR for sc in SCENARIOS]
bars = ax.bar(x, neg_log_p, color=colors)
ax.axhline(-np.log10(0.05), color="black", linestyle="--", label="p = 0.05 threshold")
ax.set_xticks(x); ax.set_xticklabels(SCENARIO_LABELS)
ax.set_ylabel("-log10(p-value)")
ax.set_title("Statistical Significance (paired t-test, DAAW vs CDA-FedAvg F1) - V4")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(FIGS_DIR, "07_statistical_significance.png"), dpi=150); plt.close()

# 08_computational_time.png - not available in the aggregated JSONs (no timing fields); placeholder skipped, noted in README
# 09_scenario_outcome_summary.png
fig, ax = plt.subplots(figsize=(11, 6))
outcomes = []
for sc in SCENARIOS:
    s = stats_by_scenario[sc]
    p = s["paired_ttest_daaw_vs_cda_f1"]["p_value"]
    sig = p is not None and p == p and p < 0.05
    daaw_higher = s["daaw_f1"]["mean"] > s["cda_f1"]["mean"]
    if sig and daaw_higher:
        outcomes.append(("DAAW wins\n(significant)", DAAW_COLOR))
    elif sig and not daaw_higher:
        outcomes.append(("CDA wins\n(significant)", CDA_COLOR))
    elif daaw_higher:
        outcomes.append(("DAAW higher\n(not significant)", "#F9D976"))
    else:
        outcomes.append(("CDA higher\n(not significant)", "#F9D976"))
bar_colors = [o[1] for o in outcomes]
ax.bar(x, [1] * len(SCENARIOS), color=bar_colors)
for i, (label, _) in enumerate(outcomes):
    ax.text(i, 0.5, label, ha="center", va="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(SCENARIO_LABELS)
ax.set_yticks([])
ax.set_title("Scenario Outcome Summary - V4 (corrected)")
plt.tight_layout(); plt.savefig(os.path.join(FIGS_DIR, "09_scenario_outcome_summary.png"), dpi=150); plt.close()

print("Wrote figures 01,02,03,04,05,06,07,09 to summary_figures/ (08_computational_time.png skipped - no timing data in aggregated JSONs, needs fresh run)")
