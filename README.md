# Detecting Concept Drift in Non-IID Federated Learning

This repository contains the corrected experimental pipeline and supporting
evidence for an MSc project evaluating **Double Adaptive Aggregation Weighting
(DAAW)** against **FedAvg** and **CDA-FedAvg**.

DAAW monitors the cosine similarity between each client's current model-update
vector and its own update history. When drift is detected, that client's
aggregation contribution is temporarily reduced. The study uses the UCI Human
Activity Recognition and Gas Sensor Array Drift datasets, ten simulated
clients, five drift scenarios and five independent seeds.

## Official implementation

The files ending in `_v3.py` are the **corrected V4 methodology** used for the
reported dissertation results. The filename and methodology-version numbers
differ because filenames were retained to avoid breaking imports during the
final correction.

| File | Purpose |
|---|---|
| `server_v3.py` | HAR experiments: FedAvg, CDA-FedAvg and DAAW |
| `server_gas_v3.py` | Gas Sensor experiments |
| `significance_test_v3.py` | Five scenarios × five seeds and paired tests |
| `utils_v3.py` | Corrected HAR partitioning and drift construction |
| `utils_gas_v3.py` | Corrected Gas class/batch partitioning |
| `utils.py` | Shared detector and held-out split helpers |
| `model.py`, `model_gas.py` | Dataset-specific neural networks |
| `test_v3_ownership.py` | 414 data-integrity and ownership checks |

Earlier V2 results were superseded after an audit identified client-ownership,
train/test-boundary and evaluation-unit problems. They are intentionally not
presented as the primary evidence.

## Installation

Python 3.12 was used for the final experiments.

```bash
python -m venv fl_env
fl_env\Scripts\activate
python -m pip install -r requirements.txt
```

The raw datasets are not committed. Download them from the official UCI pages:

- [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- [Gas Sensor Array Drift at Different Concentrations](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations)

Place the extracted files in the paths expected by `utils_v3.py` and
`utils_gas_v3.py`, or retain the same `data/` directory structure locally.

## Reproduce the corrected results

Run the integrity suite first:

```bash
python test_v3_ownership.py
```

The expected outcome is **414 passes and zero failures**. Then run:

```bash
python server_v3.py
python server_gas_v3.py
python significance_test_v3.py
```

The first two commands generate representative seed-42 trajectory figures.
The final command runs the five scenarios across seeds `42`, `123`, `456`,
`789` and `999`.

## Experimental configuration

- Clients: 10
- Communication rounds: 50
- Seeds: 42, 123, 456, 789, 999
- Four scenarios: short/long windows `5/20`, threshold `0.30`
- Gas Sequential Batch: windows `8/30`, threshold `0.15`
- Primary metrics: event-level precision, recall and F1
- Additional metrics: false-alarm episodes, detection delay and locally
  adapted client accuracy

## Results and evidence

`FINAL_DISSERTATION_RESULTS/` is the curated evidence package:

- `01_PRIMARY_V4_FIXED/`: official corrected five-seed results, figures,
  tables, code snapshot and verification evidence.
- `02_SENSITIVITY_ANALYSIS/`: historical pre-correction sensitivity studies,
  retained only with that limitation clearly stated.
- `03_EXPLORATORY_ADAPTIVE/`: an isolated adaptive-threshold extension. It
  improved recall in weak scenarios but introduced false alarms and was not
  adopted as the primary method.
- `04_SUPPORTING_EVIDENCE/`: environment, dataset and reproducibility records.

The main V4 finding is mixed rather than universal: DAAW performs strongly on
both label-shuffle scenarios, trends favourably on Gas Sudden Batch, and is
weaker than CDA-FedAvg on HAR Activity Drift and Gas Sequential Batch.

## Scope

The experiments are a controlled single-machine simulation. No raw personal
data, signed ethics forms, dissertation drafts or virtual environments are
included in this repository.
