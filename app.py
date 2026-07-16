import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import random
import time
import json
import os
from datetime import datetime

from utils import (
    load_har_dataset,
    partition_noniid,
    get_client_dataloader,
    inject_concept_drift,
    daaw_detect_drift
)
from utils_gas import (
    load_gas_dataset,
    partition_noniid_gas,
    get_gas_dataloader,
    inject_gas_drift
)

st.set_page_config(page_title="FL Experiment Dashboard", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .metric-card { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 8px; padding: 16px; text-align: center; }
    .metric-val { font-size: 28px; font-weight: 700; }
    .metric-label { font-size: 12px; color: #718096; margin-top: 4px; }
    .fedavg { color: #4299e1; } .cda { color: #ed8936; } .daaw { color: #48bb78; }
    .winner { color: #ffd700; font-size: 13px; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

HISTORY_FILE = "experiment_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

if "history" not in st.session_state:
    st.session_state.history = load_history()
if "last_run" not in st.session_state:
    st.session_state.last_run = None

tab1, tab2 = st.tabs(["🧪 Run Experiment", "📋 Experiment History"])

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def get_model_for_dataset(dataset):
    if dataset == "UCI HAR":
        from model import get_model
    else:
        from model_gas import get_model
    return get_model()

def get_params_for_dataset(model, dataset):
    if dataset == "UCI HAR":
        from model import get_parameters
    else:
        from model_gas import get_parameters
    return get_parameters(model)

def set_params_for_dataset(model, params, dataset):
    if dataset == "UCI HAR":
        from model import set_parameters
    else:
        from model_gas import set_parameters
    set_parameters(model, params)

def get_loader(X, y, indices, dataset):
    if dataset == "UCI HAR":
        return get_client_dataloader(X, y, indices)
    else:
        return get_gas_dataloader(X, y, indices)

def inject_drift(X, y, indices, drift_magnitude, dataset):
    if dataset == "UCI HAR":
        return inject_concept_drift(X, y, indices, "sudden", drift_magnitude)
    else:
        return inject_gas_drift(X, y, indices, drift_magnitude)

def _train_client(model, loader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def _eval_client(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / total

def compute_detection_metrics(detections_per_round, detection_round_map, drift_events):
    """Precision/Recall/F1/avg detection delay for one method on one run."""
    total_tp = sum(d["tp"] for d in detections_per_round.values())
    total_fp = sum(d["fp"] for d in detections_per_round.values())

    detected_events = [cid for cid in drift_events if cid in detection_round_map]
    total_events = len(drift_events)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = len(detected_events) / total_events if total_events > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    delays = [detection_round_map[cid] - drift_events[cid] for cid in detected_events]
    avg_delay = sum(delays) / len(delays) if delays else None

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_detection_delay": round(avg_delay, 2) if avg_delay is not None else None,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "detected_events": len(detected_events),
        "total_events": total_events,
    }


def run_fedavg(num_rounds, num_clients, X, y, client_data, dataset, progress):
    start_time = time.time()
    model0 = get_model_for_dataset(dataset)
    global_params = get_params_for_dataset(model0, dataset)
    results = []
    losses = []
    device = torch.device("cpu")
    for r in range(1, num_rounds + 1):
        round_params, accs, sizes, round_losses = [], {}, {}, {}
        for cid in range(num_clients):
            model = get_model_for_dataset(dataset)
            set_params_for_dataset(model, global_params, dataset)
            model.to(device)
            loader = get_loader(X, y, client_data[cid], dataset)
            total_loss = _train_client(model, loader, device)
            accs[cid] = _eval_client(model, loader, device)
            sizes[cid] = len(client_data[cid])
            round_losses[cid] = total_loss / max(1, len(loader))
            round_params.append(get_params_for_dataset(model, dataset))
        global_params = [np.mean([round_params[c][i] for c in range(num_clients)], axis=0) for i in range(len(global_params))]
        total_samples = sum(sizes.values())
        avg = sum(accs[c] * sizes[c] for c in range(num_clients)) / total_samples
        avg_loss = sum(round_losses[c] * sizes[c] for c in range(num_clients)) / total_samples
        results.append(avg)
        losses.append(avg_loss)
        progress.progress(r / num_rounds, text=f"FedAvg round {r}/{num_rounds}")
    return results, losses, time.time() - start_time

def run_cda(num_rounds, num_clients, X, y, client_data, drift_events, threshold, drift_magnitude, dataset, progress):
    start_time = time.time()
    model0 = get_model_for_dataset(dataset)
    global_params = get_params_for_dataset(model0, dataset)
    loss_hist = {i: [] for i in range(num_clients)}
    weights = {i: 1.0 for i in range(num_clients)}
    drifted = {}; results = []; losses = []
    detections_per_round = {}
    cda_detection_round = {}
    device = torch.device("cpu")
    for r in range(1, num_rounds + 1):
        tp_this_round = 0
        fp_this_round = 0
        for dc, dr in drift_events.items():
            if r == dr:
                Xd, yd = inject_drift(X, y, client_data[dc], drift_magnitude, dataset)
                drifted[dc] = (Xd, yd)
        round_params, accs, sizes, round_losses = [], {}, {}, {}
        for cid in range(num_clients):
            model = get_model_for_dataset(dataset)
            set_params_for_dataset(model, global_params, dataset)
            model.to(device)
            if cid in drifted and r >= drift_events[cid]:
                Xd, yd = drifted[cid]
                loader = get_loader(Xd, yd, list(range(len(client_data[cid]))), dataset)
            else:
                loader = get_loader(X, y, client_data[cid], dataset)
            total_loss = _train_client(model, loader, device)
            accs[cid] = _eval_client(model, loader, device)
            sizes[cid] = len(client_data[cid])
            round_losses[cid] = total_loss / max(1, len(loader))
            round_params.append(get_params_for_dataset(model, dataset))
            if len(loss_hist[cid]) > 0:
                prev = loss_hist[cid][-1]
                change = (total_loss - prev) / (prev + 1e-8)
                if change > threshold:
                    weights[cid] = 0.3
                    is_real = cid in drift_events and r >= drift_events[cid]
                    if is_real:
                        tp_this_round += 1
                        if cid not in cda_detection_round:
                            cda_detection_round[cid] = r
                    else:
                        fp_this_round += 1
                else:
                    weights[cid] = min(1.0, weights[cid] + 0.1)
            loss_hist[cid].append(total_loss)
        detections_per_round[r] = {"tp": tp_this_round, "fp": fp_this_round}
        tw = sum(weights[c] * sizes[c] for c in range(num_clients))
        global_params = [sum((weights[c] * sizes[c] / tw) * round_params[c][i] for c in range(num_clients)) for i in range(len(global_params))]
        total_samples = sum(sizes.values())
        avg = sum(accs[c] * sizes[c] for c in range(num_clients)) / total_samples
        avg_loss = sum(round_losses[c] * sizes[c] for c in range(num_clients)) / total_samples
        results.append(avg); losses.append(avg_loss)
        progress.progress(r / num_rounds, text=f"CDA-FedAvg round {r}/{num_rounds}")
    return results, losses, cda_detection_round, detections_per_round, time.time() - start_time

def run_daaw(num_rounds, num_clients, X, y, client_data, drift_events, threshold, short_window, long_window, daaw_weight, drift_magnitude, dataset, progress):
    start_time = time.time()
    model0 = get_model_for_dataset(dataset)
    global_params = get_params_for_dataset(model0, dataset)
    grad_hist = {i: [] for i in range(num_clients)}
    weights = {i: 1.0 for i in range(num_clients)}
    drift_detected = {i: False for i in range(num_clients)}
    detection_rounds = {}; drifted = {}; results = []; losses = []
    detections_per_round = {}
    device = torch.device("cpu")
    for r in range(1, num_rounds + 1):
        tp_this_round = 0
        fp_this_round = 0
        for dc, dr in drift_events.items():
            if r == dr:
                Xd, yd = inject_drift(X, y, client_data[dc], drift_magnitude, dataset)
                drifted[dc] = (Xd, yd)
        round_params, accs, sizes, round_losses = [], {}, {}, {}
        for cid in range(num_clients):
            model = get_model_for_dataset(dataset)
            set_params_for_dataset(model, global_params, dataset)
            model.to(device)
            if cid in drifted and r >= drift_events[cid]:
                Xd, yd = drifted[cid]
                loader = get_loader(Xd, yd, list(range(len(client_data[cid]))), dataset)
            else:
                loader = get_loader(X, y, client_data[cid], dataset)
            pb = [p.clone().detach() for p in model.parameters()]
            total_loss = _train_client(model, loader, device)
            pa = [p.clone().detach() for p in model.parameters()]
            grad = np.concatenate([(a - b).cpu().numpy().flatten() for b, a in zip(pb, pa)])
            grad_hist[cid].append(grad)
            round_losses[cid] = total_loss / max(1, len(loader))
            dd, sim = daaw_detect_drift(grad_hist[cid], short_window, long_window, threshold)
            if dd and not drift_detected[cid]:
                drift_detected[cid] = True
                weights[cid] = daaw_weight
                is_real = cid in drift_events and r >= drift_events[cid]
                if is_real:
                    tp_this_round += 1
                    if cid not in detection_rounds:
                        detection_rounds[cid] = r
                else:
                    fp_this_round += 1
            elif dd:
                weights[cid] = max(daaw_weight, weights[cid] - 0.05)
            else:
                if weights[cid] < 1.0:
                    weights[cid] = min(1.0, weights[cid] + 0.1)
                    if weights[cid] >= 1.0:
                        drift_detected[cid] = False
            accs[cid] = _eval_client(model, loader, device)
            sizes[cid] = len(client_data[cid])
            round_params.append(get_params_for_dataset(model, dataset))
        detections_per_round[r] = {"tp": tp_this_round, "fp": fp_this_round}
        tw = sum(weights[c] * sizes[c] for c in range(num_clients))
        global_params = [sum((weights[c] * sizes[c] / tw) * round_params[c][i] for c in range(num_clients)) for i in range(len(global_params))]
        total_samples = sum(sizes.values())
        avg = sum(accs[c] * sizes[c] for c in range(num_clients)) / total_samples
        avg_loss = sum(round_losses[c] * sizes[c] for c in range(num_clients)) / total_samples
        results.append(avg); losses.append(avg_loss)
        progress.progress(r / num_rounds, text=f"DAAW round {r}/{num_rounds}")
    return results, losses, detection_rounds, detections_per_round, time.time() - start_time

def make_loss_graph(fedavg_l, cda_l, daaw_l, drift_events, num_rounds, figsize=(14,4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#0e1117')
    rounds = range(1, num_rounds + 1)
    ax.plot(rounds, fedavg_l, label="FedAvg Loss", color="#4299e1", linewidth=1.5)
    ax.plot(rounds, cda_l,    label="CDA Loss",    color="#ed8936", linewidth=1.5)
    ax.plot(rounds, daaw_l,   label="DAAW Loss",   color="#48bb78", linewidth=1.5)
    colors = ["#fc8181", "#b794f4", "#f6ad55", "#76e4f7"]
    for i, (dc, dr) in enumerate(drift_events.items()):
        if dr <= num_rounds:
            ax.axvline(x=dr, color=colors[i % len(colors)], linestyle="--", alpha=0.6, linewidth=1.2)
    ax.set_xlabel("Communication Round", color="#a0aec0", fontsize=11)
    ax.set_ylabel("Training Loss", color="#a0aec0", fontsize=11)
    ax.set_title("Training Loss", color="white", fontsize=12)
    ax.legend(fontsize=9, facecolor="#1a1f2e", labelcolor="white")
    ax.grid(True, alpha=0.2, color="#4a5568")
    ax.tick_params(colors="#a0aec0")
    for spine in ax.spines.values(): spine.set_edgecolor("#2d3748")
    plt.tight_layout()
    return fig


def make_graph(fedavg_r, cda_r, daaw_r, drift_events, num_rounds, figsize=(14,6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0e1117'); ax.set_facecolor('#0e1117')
    rounds = range(1, num_rounds + 1)
    ax.plot(rounds, fedavg_r, label="FedAvg (Baseline 1)", color="#4299e1", linewidth=2)
    ax.plot(rounds, cda_r,    label="CDA-FedAvg (Baseline 2)", color="#ed8936", linewidth=2)
    ax.plot(rounds, daaw_r,   label="DAAW (Proposed)", color="#48bb78", linewidth=2)
    colors = ["#fc8181", "#b794f4", "#f6ad55", "#76e4f7"]
    for i, (dc, dr) in enumerate(drift_events.items()):
        if dr <= num_rounds:
            ax.axvline(x=dr, color=colors[i % len(colors)], linestyle="--", alpha=0.8, linewidth=1.5)
            ax.text(dr + 0.3, min(fedavg_r + cda_r + daaw_r) + 0.005,
                    f"Drift\nClient {dc}\nRound {dr}", color=colors[i % len(colors)], fontsize=8)
    ax.set_xlabel("Communication Round", color="#a0aec0", fontsize=12)
    ax.set_ylabel("Accuracy", color="#a0aec0", fontsize=12)
    ax.set_title("Federated Learning Accuracy Comparison\nwith Concept Drift in Non-IID Environment", color="white", fontsize=13)
    ax.legend(fontsize=11, facecolor="#1a1f2e", labelcolor="white")
    ax.grid(True, alpha=0.2, color="#4a5568")
    ax.tick_params(colors="#a0aec0")
    for spine in ax.spines.values(): spine.set_edgecolor("#2d3748")
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Parameters")
    st.subheader("Experiment Label")
    test_label = st.text_input("Test name", value="Test 1")
    st.subheader("Dataset")
    dataset = st.selectbox("Select Dataset", ["UCI HAR", "Gas Sensor Array"])
    st.subheader("Data")
    alpha = st.slider("Alpha (Non-IID degree)", 0.1, 1.0, 0.3, 0.1)
    num_rounds = st.slider("Rounds", 10, 50, 50, 5)
    seed = st.number_input("Random Seed", value=42, step=1)
    st.subheader("Drift Events")
    drift_c3 = st.slider("Client 3 drift at round", 3, 20, 5, 1)
    drift_c5 = st.slider("Client 5 drift at round", 10, 45, 20, 1)
    drift_c7 = st.slider("Client 7 drift at round", 20, 48, 35, 1)
    drift_c9 = st.slider("Client 9 drift at round", 30, 49, 45, 1)
    st.subheader("Drift Magnitude")
    drift_magnitude = st.slider("Sensor Offset (X + ?)", 0.5, 3.0, 1.0, 0.5)
    st.subheader("CDA-FedAvg")
    cda_threshold = st.slider("CDA Loss Threshold", 0.05, 0.50, 0.15, 0.05)
    st.subheader("DAAW (Proposed)")
    daaw_threshold = st.slider("DAAW Cosine Threshold", 0.1, 0.9, 0.3, 0.05)
    short_window = st.slider("Short Window", 3, 10, 5, 1)
    long_window = st.slider("Long Window", 10, 50, 20, 5)
    daaw_weight = st.slider("Drifted Client Weight", 0.0, 0.3, 0.0, 0.05)
    st.markdown("---")
    run_btn = st.button("▶ Run Experiment", type="primary", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🧠 FL Experiment Dashboard")
    st.caption("DAAW vs CDA-FedAvg vs FedAvg — Concept Drift in Non-IID Federated Learning")

    if run_btn:
        set_seed(int(seed))
        drift_events = {3: drift_c3, 5: drift_c5, 7: drift_c7, 9: drift_c9}

        with st.spinner("Loading dataset..."):
            if dataset == "UCI HAR":
                X, y = load_har_dataset()
                client_data = partition_noniid(X, y, num_clients=10, alpha=alpha)
            else:
                X, y = load_gas_dataset()
                client_data = partition_noniid_gas(X, y, num_clients=10, alpha=alpha)

        st.info(f"✅ Shared partition | Dataset={dataset} | alpha={alpha} | seed={int(seed)} | {num_rounds} rounds | Drift X+{drift_magnitude}")

        p1 = st.progress(0, text="FedAvg starting...")
        fedavg_results, fedavg_losses, fedavg_time = run_fedavg(num_rounds, 10, X, y, client_data, dataset, p1)
        p1.empty()

        p2 = st.progress(0, text="CDA-FedAvg starting...")
        cda_results, cda_losses, cda_det, cda_dpr, cda_time = run_cda(
            num_rounds, 10, X, y, client_data, drift_events, cda_threshold, drift_magnitude, dataset, p2)
        p2.empty()

        p3 = st.progress(0, text="DAAW starting...")
        daaw_results, daaw_losses, daaw_det, daaw_dpr, daaw_time = run_daaw(
            num_rounds, 10, X, y, client_data, drift_events,
            daaw_threshold, short_window, long_window, daaw_weight, drift_magnitude, dataset, p3)
        p3.empty()

        st.success("✅ Experiment complete!")

        cda_metrics = compute_detection_metrics(cda_dpr, cda_det, drift_events)
        daaw_metrics = compute_detection_metrics(daaw_dpr, daaw_det, drift_events)

        st.session_state.last_run = {
            "label": test_label,
            "params": {
                "seed": int(seed), "alpha": alpha, "rounds": num_rounds,
                "dataset": dataset,
                "drift_c3": drift_c3, "drift_c5": drift_c5,
                "drift_c7": drift_c7, "drift_c9": drift_c9,
                "drift_magnitude": drift_magnitude,
                "cda_threshold": cda_threshold, "daaw_threshold": daaw_threshold,
                "short_window": short_window, "long_window": long_window,
                "daaw_weight": daaw_weight
            },
            "drift_events": drift_events,
            "fedavg_results": fedavg_results,
            "cda_results": cda_results,
            "daaw_results": daaw_results,
            "fedavg_losses": fedavg_losses,
            "cda_losses": cda_losses,
            "daaw_losses": daaw_losses,
            "cda_fp": [d["fp"] for d in cda_dpr.values()],
            "daaw_fp": [d["fp"] for d in daaw_dpr.values()],
            "daaw_det": daaw_det,
            "cda_metrics": cda_metrics,
            "daaw_metrics": daaw_metrics,
            "fedavg_time": fedavg_time,
            "cda_time": cda_time,
            "daaw_time": daaw_time,
        }

    if st.session_state.last_run is not None:
        lr = st.session_state.last_run
        fedavg_results = lr["fedavg_results"]
        cda_results    = lr["cda_results"]
        daaw_results   = lr["daaw_results"]
        cda_fp         = lr["cda_fp"]
        daaw_fp        = lr["daaw_fp"]
        daaw_det       = lr["daaw_det"]
        drift_events   = lr["drift_events"]
        num_rounds_disp = lr["params"]["rounds"]
        fedavg_time    = lr.get("fedavg_time", 0)
        cda_time       = lr.get("cda_time", 0)
        daaw_time      = lr.get("daaw_time", 0)
        fedavg_losses  = lr.get("fedavg_losses")
        cda_losses     = lr.get("cda_losses")
        daaw_losses    = lr.get("daaw_losses")
        cda_metrics    = lr.get("cda_metrics")
        daaw_metrics   = lr.get("daaw_metrics")

        st.subheader("📊 Final Results")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-val fedavg">{fedavg_results[-1]*100:.1f}%</div><div class="metric-label">FedAvg Final</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-val cda">{cda_results[-1]*100:.1f}%</div><div class="metric-label">CDA Final</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-val daaw">{daaw_results[-1]*100:.1f}%</div><div class="metric-label">DAAW Final</div><div class="winner">{"✅ Beats CDA" if daaw_results[-1] > cda_results[-1] else "❌ Below CDA"}</div></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card"><div class="metric-val cda">{sum(cda_fp)}</div><div class="metric-label">CDA False Positives</div></div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""<div class="metric-card"><div class="metric-val daaw">{sum(daaw_fp)}</div><div class="metric-label">DAAW False Positives</div><div class="winner">{"✅ Less FP" if sum(daaw_fp) < sum(cda_fp) else "❌ More FP"}</div></div>""", unsafe_allow_html=True)

        st.subheader("⏱️ Computational Overhead")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown(f"""<div class="metric-card"><div class="metric-val fedavg">{fedavg_time:.1f}s</div><div class="metric-label">FedAvg Total Time</div></div>""", unsafe_allow_html=True)
        with t2:
            st.markdown(f"""<div class="metric-card"><div class="metric-val cda">{cda_time:.1f}s</div><div class="metric-label">CDA Total Time</div></div>""", unsafe_allow_html=True)
        with t3:
            st.markdown(f"""<div class="metric-card"><div class="metric-val daaw">{daaw_time:.1f}s</div><div class="metric-label">DAAW Total Time</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(make_graph(fedavg_results, cda_results, daaw_results, drift_events, num_rounds_disp))

        if fedavg_losses and cda_losses and daaw_losses:
            st.pyplot(make_loss_graph(fedavg_losses, cda_losses, daaw_losses, drift_events, num_rounds_disp))

        if cda_metrics and daaw_metrics:
            st.subheader("🎯 Detection Performance (Precision / Recall / F1 / Delay)")
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f"""<div class="metric-card"><div class="metric-val cda">CDA-FedAvg</div>
                <div class="metric-label">Precision: {cda_metrics['precision']} | Recall: {cda_metrics['recall']} | F1: {cda_metrics['f1']}</div>
                <div class="metric-label">Avg Detection Delay: {cda_metrics['avg_detection_delay']} rounds</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card"><div class="metric-val daaw">DAAW</div>
                <div class="metric-label">Precision: {daaw_metrics['precision']} | Recall: {daaw_metrics['recall']} | F1: {daaw_metrics['f1']}</div>
                <div class="metric-label">Avg Detection Delay: {daaw_metrics['avg_detection_delay']} rounds</div></div>""", unsafe_allow_html=True)

        st.subheader("🔍 DAAW Detection Summary")
        real_drifts = set(drift_events.keys())
        for cid, rnd in sorted(daaw_det.items()):
            if cid in real_drifts:
                st.success(f"✅ Client {cid}: REAL drift detected at round {rnd} (injected at round {drift_events[cid]})")
            else:
                st.warning(f"⚠️ Client {cid}: FALSE POSITIVE at round {rnd}")
        for dc in real_drifts:
            if dc not in daaw_det:
                st.error(f"❌ Client {dc}: drift MISSED!")

        st.markdown("---")
        col_note, col_save = st.columns([3, 1])
        with col_note:
            note = st.text_input("Add a note (optional)", key="save_note",
                                 placeholder="e.g. Alpha 0.1 extreme NonIID — best result")
        with col_save:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔒 Save to History", use_container_width=True):
                entry = {
                    "label": lr["label"],
                    "timestamp": datetime.now().strftime("%d %b %Y %H:%M"),
                    "note": note,
                    "params": lr["params"],
                    "results": {
                        "fedavg": round(fedavg_results[-1] * 100, 1),
                        "cda": round(cda_results[-1] * 100, 1),
                        "daaw": round(daaw_results[-1] * 100, 1),
                        "cda_fp": sum(cda_fp),
                        "daaw_fp": sum(daaw_fp),
                        "daaw_beats_cda": daaw_results[-1] > cda_results[-1],
                        "less_fp": sum(daaw_fp) < sum(cda_fp),
                        "detection": {str(k): v for k, v in daaw_det.items()},
                        "fedavg_time": round(fedavg_time, 1),
                        "cda_time": round(cda_time, 1),
                        "daaw_time": round(daaw_time, 1),
                        "cda_metrics": cda_metrics,
                        "daaw_metrics": daaw_metrics,
                    },
                    "fedavg_curve": [round(x, 4) for x in fedavg_results],
                    "cda_curve":    [round(x, 4) for x in cda_results],
                    "daaw_curve":   [round(x, 4) for x in daaw_results],
                    "fedavg_loss_curve": [round(x, 4) for x in fedavg_losses] if fedavg_losses else None,
                    "cda_loss_curve":    [round(x, 4) for x in cda_losses] if cda_losses else None,
                    "daaw_loss_curve":   [round(x, 4) for x in daaw_losses] if daaw_losses else None,
                }
                st.session_state.history.append(entry)
                save_history(st.session_state.history)
                st.success(f"✅ '{lr['label']}' saved! Go to 📋 Experiment History tab.")

    elif not run_btn:
        st.info("👈 Set parameters in the sidebar and click **▶ Run Experiment**")
        st.markdown("""
        ### How to use
        - **Dataset** — UCI HAR or Gas Sensor Array
        - **Alpha** — controls Non-IID degree. Lower = more skewed data per client
        - **Drift Magnitude** — sensor offset value (X + ?)
        - **CDA Threshold** — loss change sensitivity. Original paper uses 0.15
        - **DAAW Threshold** — cosine similarity cutoff
        - **DAAW Weight** — drifted client contribution (0.0 = complete exclusion)
        - Give your test a name, run, then click 🔒 Save to History
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — History
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📋 Experiment History")
    st.caption("All saved experiment runs — locked and comparable")

    if len(st.session_state.history) == 0:
        st.info("No experiments saved yet. Run an experiment and click 🔒 Save to History.")
    else:
        st.subheader(f"📊 {len(st.session_state.history)} Experiments Saved")

        if len(st.session_state.history) > 1:
            fig2, ax2 = plt.subplots(figsize=(14, 5))
            fig2.patch.set_facecolor('#0e1117'); ax2.set_facecolor('#0e1117')
            cmap = plt.cm.get_cmap('tab10')
            for i, entry in enumerate(st.session_state.history):
                rds = range(1, len(entry["daaw_curve"]) + 1)
                ax2.plot(rds, entry["daaw_curve"], label=f"{entry['label']} — DAAW", color=cmap(i), linewidth=2)
                ax2.plot(rds, entry["cda_curve"],  label=f"{entry['label']} — CDA",  color=cmap(i), linewidth=1.5, linestyle="--", alpha=0.6)
            ax2.set_xlabel("Round", color="#a0aec0"); ax2.set_ylabel("Accuracy", color="#a0aec0")
            ax2.set_title("All Experiments — DAAW vs CDA", color="white")
            ax2.legend(fontsize=9, facecolor="#1a1f2e", labelcolor="white", loc="lower right")
            ax2.grid(True, alpha=0.2, color="#4a5568"); ax2.tick_params(colors="#a0aec0")
            for spine in ax2.spines.values(): spine.set_edgecolor("#2d3748")
            plt.tight_layout()
            st.pyplot(fig2)

        st.markdown("---")

        for i, entry in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - i
            r = entry["results"]; p = entry["params"]
            with st.expander(f"🔒 {entry['label']} — {entry['timestamp']}", expanded=(i == 0)):
                if entry["note"]:
                    st.markdown(f"📝 *{entry['note']}*")

                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                with mc1: st.metric("FedAvg", f"{r['fedavg']}%")
                with mc2: st.metric("CDA-FedAvg", f"{r['cda']}%")
                with mc3: st.metric("DAAW", f"{r['daaw']}%", delta=f"{round(r['daaw']-r['cda'],1)}% vs CDA")
                with mc4: st.metric("CDA False Positives", r['cda_fp'])
                with mc5: st.metric("DAAW False Positives", r['daaw_fp'], delta=f"{r['daaw_fp']-r['cda_fp']} vs CDA", delta_color="inverse")

                if "fedavg_time" in r:
                    t1, t2, t3 = st.columns(3)
                    with t1: st.metric("FedAvg Time", f"{r['fedavg_time']}s")
                    with t2: st.metric("CDA Time", f"{r['cda_time']}s")
                    with t3: st.metric("DAAW Time", f"{r['daaw_time']}s")

                if r.get("cda_metrics") and r.get("daaw_metrics"):
                    cm, dm = r["cda_metrics"], r["daaw_metrics"]
                    st.markdown(f"**CDA** — Precision: {cm['precision']} | Recall: {cm['recall']} | F1: {cm['f1']} | Avg Delay: {cm['avg_detection_delay']} rounds")
                    st.markdown(f"**DAAW** — Precision: {dm['precision']} | Recall: {dm['recall']} | F1: {dm['f1']} | Avg Delay: {dm['avg_detection_delay']} rounds")

                fig3, ax3 = plt.subplots(figsize=(10, 3))
                fig3.patch.set_facecolor('#0e1117'); ax3.set_facecolor('#0e1117')
                rds = range(1, len(entry["daaw_curve"]) + 1)
                ax3.plot(rds, entry["fedavg_curve"], color="#4299e1", linewidth=1.5, label="FedAvg")
                ax3.plot(rds, entry["cda_curve"],    color="#ed8936", linewidth=1.5, label="CDA")
                ax3.plot(rds, entry["daaw_curve"],   color="#48bb78", linewidth=1.5, label="DAAW")
                ax3.legend(fontsize=9, facecolor="#1a1f2e", labelcolor="white")
                ax3.grid(True, alpha=0.2, color="#4a5568"); ax3.tick_params(colors="#a0aec0")
                for spine in ax3.spines.values(): spine.set_edgecolor("#2d3748")
                plt.tight_layout(); st.pyplot(fig3)

                st.markdown(f"**Params:** Dataset={p.get('dataset','UCI HAR')} | Alpha={p['alpha']} | Rounds={p['rounds']} | Seed={p['seed']} | Drift X+{p.get('drift_magnitude', '?')} | CDA thr={p['cda_threshold']} | DAAW thr={p['daaw_threshold']} | SW={p['short_window']} | LW={p['long_window']} | DAAW weight={p.get('daaw_weight', '?')} | Drift C3=R{p['drift_c3']} | C5=R{p['drift_c5']} | C7=R{p.get('drift_c7','?')} | C9=R{p.get('drift_c9','?')}")

                det = r["detection"]
                det_str = " | ".join([f"Client {k}: R{v} {'✅' if k in ['3','5','7','9'] else '⚠️'}" for k, v in det.items()])
                st.markdown(f"**DAAW Detections:** {det_str if det_str else 'None'}")

                if st.button(f"🗑️ Delete", key=f"del_{idx}_{entry['timestamp']}"):
                    st.session_state.history = [e for e in st.session_state.history if e["timestamp"] != entry["timestamp"]]
                    save_history(st.session_state.history)
                    st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear All History", type="secondary"):
            st.session_state.history = []
            save_history([])
            st.rerun()