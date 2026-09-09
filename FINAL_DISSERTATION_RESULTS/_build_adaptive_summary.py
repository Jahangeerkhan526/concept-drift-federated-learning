"""Build the top-level adaptive_summary.csv, adaptive_statistics.json and
fixed_vs_adaptive_tradeoff.png for 03_EXPLORATORY_ADAPTIVE from the real
Stage 4 validation data already copied into validation_seeds/overall_summary.json.
Pure post-processing - no experiments re-run.
"""
import os
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(OUT, "03_EXPLORATORY_ADAPTIVE")

with open(os.path.join(DIR, "validation_seeds", "overall_summary.json")) as f:
    data = json.load(f)

SCENARIOS = ["har_label_shuffle", "har_activity_drift", "gas_label_shuffle", "gas_sudden_batch", "gas_sequential_batch"]
LABELS = ["HAR Label\nShuffle", "HAR Activity\nDrift", "Gas Label\nShuffle", "Gas Sudden\nBatch", "Gas Sequential\nBatch"]

with open(os.path.join(DIR, "adaptive_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["scenario", "daaw_fixed_f1_mean", "daaw_fixed_f1_sd", "daaw_adaptive_f1_mean", "daaw_adaptive_f1_sd",
                "daaw_fixed_total_fp_mean", "daaw_adaptive_total_fp_mean",
                "p_adaptive_vs_fixed", "p_adaptive_vs_cda"])
    for sc in SCENARIOS:
        d = data[sc]
        w.writerow([
            sc, d["daaw_fixed_f1"]["mean"], d["daaw_fixed_f1"]["std"],
            d["daaw_adaptive_f1"]["mean"], d["daaw_adaptive_f1"]["std"],
            d["daaw_fixed_total_fp"]["mean"], d["daaw_adaptive_total_fp"]["mean"],
            d["paired_ttest_adaptive_vs_fixed_f1"].get("p_value", d["paired_ttest_adaptive_vs_fixed_f1"].get("note", "")),
            d["paired_ttest_adaptive_vs_cda_f1"].get("p_value", d["paired_ttest_adaptive_vs_cda_f1"].get("note", "")),
        ])
print("Wrote adaptive_summary.csv")

adaptive_stats = {sc: data[sc] for sc in SCENARIOS}
with open(os.path.join(DIR, "adaptive_statistics.json"), "w") as f:
    json.dump(adaptive_stats, f, indent=2)
print("Wrote adaptive_statistics.json")

fixed_f1 = [data[sc]["daaw_fixed_f1"]["mean"] for sc in SCENARIOS]
adaptive_f1 = [data[sc]["daaw_adaptive_f1"]["mean"] for sc in SCENARIOS]
x = np.arange(len(SCENARIOS))
w_bar = 0.35
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - w_bar / 2, fixed_f1, w_bar, label="DAAW-Fixed", color="#02C39A")
ax.bar(x + w_bar / 2, adaptive_f1, w_bar, label="DAAW-Adaptive", color="#F9A825")
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_ylabel("F1 Score (mean, 4 validation seeds)")
ax.set_title("Fixed vs Adaptive Threshold - Trade-off (Stage 4 validation, unseen seeds)")
ax.set_ylim(0, 1.15)
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(DIR, "fixed_vs_adaptive_tradeoff.png"), dpi=150)
plt.close()
print("Wrote fixed_vs_adaptive_tradeoff.png")
