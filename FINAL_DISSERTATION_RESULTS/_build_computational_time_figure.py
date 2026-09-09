"""Build 08_computational_time.png from the fresh seed-42 timing data captured
in scenario_runs/ (the 5-seed aggregate JSONs never recorded timing, so this
uses the representative seed-42 run - labeled as such, per the reporting
convention in README.md).
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(OUT, "01_PRIMARY_V4_FIXED", "scenario_runs")
FIGS_DIR = os.path.join(OUT, "01_PRIMARY_V4_FIXED", "summary_figures")

SCENARIOS = ["har_label_shuffle", "har_activity_drift", "gas_label_shuffle", "gas_sudden_batch", "gas_sequential_batch"]
LABELS = ["HAR Label\nShuffle", "HAR Activity\nDrift", "Gas Label\nShuffle", "Gas Sudden\nBatch", "Gas Sequential\nBatch"]

fedavg_t, cda_t, daaw_t = [], [], []
for sc in SCENARIOS:
    with open(os.path.join(RUNS_DIR, sc, "seed_42_metrics.json")) as f:
        d = json.load(f)
    t = d["computational_time_seconds"]
    fedavg_t.append(t["fedavg"])
    cda_t.append(t["cda"])
    daaw_t.append(t["daaw"])

x = np.arange(len(SCENARIOS))
w = 0.25
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - w, fedavg_t, w, label="FedAvg", color="#8A93B8")
ax.bar(x, cda_t, w, label="CDA-FedAvg", color="#F96167")
ax.bar(x + w, daaw_t, w, label="DAAW", color="#02C39A")
ax.set_xticks(x); ax.set_xticklabels(LABELS)
ax.set_ylabel("Total Training Time (seconds, seed 42)")
ax.set_title("Computational Time Comparison - Representative Seed-42 Run (V4)")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "08_computational_time.png"), dpi=150)
plt.close()
print("Wrote 08_computational_time.png")
