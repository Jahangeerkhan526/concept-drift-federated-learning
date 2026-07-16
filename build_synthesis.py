"""Spatial vs temporal matrix + sudden vs gradual comparison graphs."""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCENARIOS = [
    {"name": "HAR Label Shuffle", "path": "results/spatial_drift/har_label_shuffle",
     "drift_type": "Spatial", "character": "Sudden", "provisional": False},
    {"name": "HAR Activity Drift", "path": "results/spatial_drift/har_activity_drift",
     "drift_type": "Spatial", "character": "Gradual", "provisional": False},
    {"name": "Gas Label Shuffle", "path": "results/spatial_drift/gas_label_shuffle",
     "drift_type": "Spatial", "character": "Sudden", "provisional": False},
    {"name": "Gas Sudden Batch", "path": "results/temporal_drift/gas_sudden_batch",
     "drift_type": "Temporal", "character": "Sudden", "provisional": False},
    {"name": "Gas Sequential Batch", "path": "results/temporal_drift/gas_sequential_batch",
     "drift_type": "Temporal", "character": "Gradual", "provisional": False},
]

def load_metrics():
    data = []
    for s in SCENARIOS:
        with open(os.path.join(s["path"], "metrics_summary.json")) as f:
            m = json.load(f)
        data.append({**s, "cda": m["cda"], "daaw": m["daaw"]})
    return data


def plot_spatial_temporal_matrix(data, save_path):
    os.makedirs(save_path, exist_ok=True)
    quadrants = [
        ("Spatial", "Sudden"), ("Spatial", "Gradual"),
        ("Temporal", "Sudden"), ("Temporal", "Gradual"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (dtype, char) in zip(axes, quadrants):
        subset = [d for d in data if d["drift_type"] == dtype and d["character"] == char]
        if not subset:
            ax.text(0.5, 0.5, "No scenario in this quadrant", ha="center", va="center", fontsize=11, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            names = [d["name"] for d in subset]
            cda_f1 = [d["cda"]["f1"] for d in subset]
            daaw_f1 = [d["daaw"]["f1"] for d in subset]
            x = range(len(names))
            width = 0.35
            ax.bar([i - width/2 for i in x], cda_f1, width, label="CDA-FedAvg", color="orange")
            ax.bar([i + width/2 for i in x], daaw_f1, width, label="DAAW", color="green")
            ax.set_xticks(list(x))
            labels = [n + ("*" if d["provisional"] else "") for n, d in zip(names, subset)]
            ax.set_xticklabels(labels, fontsize=9, rotation=10)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("F1 Score", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
        ax.set_title(f"{dtype} drift , {char}", fontsize=12, fontweight="bold")

    fig.suptitle("Spatial vs. Temporal Drift Matrix (F1 Score, CDA vs. DAAW)\n"
                  "* = provisional, pending Scenario 5 ground-truth fix", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "spatial_temporal_matrix.png"), dpi=150)
    plt.close()
    print(f"Saved: {save_path}/spatial_temporal_matrix.png")


def plot_sudden_vs_gradual(data, save_path):
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, char in zip(axes, ["Sudden", "Gradual"]):
        subset = [d for d in data if d["character"] == char]
        names = [f"{d['drift_type']}\n{d['name']}" for d in subset]
        cda_f1 = [d["cda"]["f1"] for d in subset]
        daaw_f1 = [d["daaw"]["f1"] for d in subset]
        x = range(len(names))
        width = 0.35
        ax.bar([i - width/2 for i in x], cda_f1, width, label="CDA-FedAvg", color="orange")
        ax.bar([i + width/2 for i in x], daaw_f1, width, label="DAAW", color="green")
        ax.set_xticks(list(x))
        labels = [n + ("*" if d["provisional"] else "") for n, d in zip(names, subset)]
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(f"{char} Drift Scenarios", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Sudden vs. Gradual Drift , F1 Comparison Across All Scenarios\n"
                  "* = provisional, pending Scenario 5 ground-truth fix", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "sudden_vs_gradual_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: {save_path}/sudden_vs_gradual_comparison.png")


def save_overall_table(data, save_path):
    table = []
    for d in data:
        table.append({
            "scenario": d["name"], "drift_type": d["drift_type"], "character": d["character"],
            "provisional": d["provisional"],
            "cda_precision": d["cda"]["precision"], "cda_recall": d["cda"]["recall"], "cda_f1": d["cda"]["f1"],
            "daaw_precision": d["daaw"]["precision"], "daaw_recall": d["daaw"]["recall"], "daaw_f1": d["daaw"]["f1"],
            "daaw_avg_delay": d["daaw"]["avg_detection_delay"],
        })
    with open(os.path.join(save_path, "overall_comparison.json"), "w") as f:
        json.dump(table, f, indent=2)
    print(f"Saved: {save_path}/overall_comparison.json")


if __name__ == "__main__":
    data = load_metrics()
    save_path = "results/summary"
    plot_spatial_temporal_matrix(data, save_path)
    plot_sudden_vs_gradual(data, save_path)
    save_overall_table(data, save_path)
    print("\nSynthesis complete: results/summary/")
