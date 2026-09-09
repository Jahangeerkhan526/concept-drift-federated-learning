# Detecting Concept Drift in Non-IID Federated Learning Environments Using Cosine Similarity

This repository contains the corrected experimental pipeline and supporting evidence for an MSc Artificial Intelligence dissertation evaluating **Double Adaptive Aggregation Weighting (DAAW)** against **FedAvg** and **CDA-FedAvg**.

DAAW monitors the cosine similarity between each client's current model-update vector and its own update history. When the configured detection threshold is crossed, the client's contribution to the shared model is temporarily reduced and subsequently restored after stable behaviour returns.

The study uses the UCI Human Activity Recognition and Gas Sensor Array Drift datasets, ten simulated clients, five drift scenarios and five independent experimental seeds.

## Research question

Can model-update cosine similarity reliably detect concept drift in non-IID federated learning environments?

## Proposed method

DAAW compares each client's current model-update direction with its own short-term and long-term histories. This temporal self-comparison aims to distinguish genuine within-client change from the normal differences that exist between non-IID clients.

The method was evaluated against:

- **FedAvg:** the standard federated-averaging baseline without drift detection
- **CDA-FedAvg:** a loss-based concept-drift detection baseline
- **DAAW:** the proposed model-update cosine-similarity method

## Official implementation

The files ending in `_v3.py` implement the **corrected V4 methodology** used for the reported dissertation results. The filename and methodology-version numbers differ because the filenames were retained to avoid breaking imports during the final methodological correction.

| File | Purpose |
|---|---|
| `server_v3.py` | HAR experiments using FedAvg, CDA-FedAvg and DAAW |
| `server_gas_v3.py` | Gas Sensor experiments using all three methods |
| `significance_test_v3.py` | Five scenarios, five seeds and paired statistical tests |
| `utils_v3.py` | Corrected HAR partitioning and drift construction |
| `utils_gas_v3.py` | Corrected Gas class-and-batch partitioning |
| `utils.py` | Shared detector and held-out split helpers |
| `model.py` | Neural-network model for UCI HAR |
| `model_gas.py` | Neural-network model for the Gas Sensor dataset |
| `test_v3_ownership.py` | 414 data-integrity and client-ownership checks |

Earlier V2 results were superseded after a methodology audit identified client-ownership, train/test-boundary and evaluation-unit problems. The corrected V4 pipeline preserves client ownership, maintains permanent train/test separation and reports event-level detection metrics consistently.

## Experimental configuration

- Simulated clients: 10
- Communication rounds: 50
- Independent seeds: 42, 123, 456, 789 and 999
- Datasets: UCI HAR and Gas Sensor Array Drift
- Drift scenarios: 5
- Compared methods: FedAvg, CDA-FedAvg and DAAW
- Four scenarios: short/long windows `5/20`, threshold `0.30`
- Gas Sequential Batch: short/long windows `8/30`, threshold `0.15`
- Primary metrics: event-level precision, recall and F1-score
- Additional metrics: false-alarm episodes, detection delay and reported mean locally adapted client accuracy
- Statistical analysis: paired t-tests comparing DAAW and CDA-FedAvg F1-scores

## Drift scenarios

The evaluation covers five drift scenarios:

1. HAR Label Shuffle
2. HAR Activity Drift
3. Gas Label Shuffle
4. Gas Sudden Batch
5. Gas Sequential Batch

The scenarios include pronounced label-altering drift, activity-related drift and sudden or sequential changes between Gas Sensor batches.

## Main corrected V4 results

The corrected V4 evaluation produced mixed but clearly interpretable results:

| Scenario | CDA-FedAvg F1 | DAAW F1 | Interpretation |
|---|---:|---:|---|
| HAR Label Shuffle | 0.398 | 0.978 | DAAW performed significantly better and produced very few false alarms |
| HAR Activity Drift | 0.397 | 0.000 | DAAW missed the injected events; CDA-FedAvg performed significantly better |
| Gas Label Shuffle | 0.401 | 1.000 | DAAW achieved perfect event-level F1 with no false alarms |
| Gas Sudden Batch | 0.412 | 0.579 | DAAW had a higher mean F1, but the difference was not statistically significant |
| Gas Sequential Batch | 1.000 | 0.236 | CDA-FedAvg performed significantly better; DAAW recall was 0.140 |

### Label-shuffle scenarios

DAAW's clearest strength was pronounced label-altering drift. It achieved an F1-score of `0.978` for HAR Label Shuffle and `1.000` for Gas Label Shuffle. It also produced almost no false-alarm episodes.

DAAW performed significantly better than CDA-FedAvg in both label-shuffle scenarios.

### Gas Sudden Batch

DAAW achieved a higher mean F1-score than CDA-FedAvg for Gas Sudden Batch: `0.579` compared with `0.412`.

However, the result varied across the five seeds, and the difference was not statistically significant (`p = 0.1204`).

### Weaker scenarios

DAAW was less effective for HAR Activity Drift and Gas Sequential Batch.

It did not detect the injected HAR Activity Drift events across the five seeds, resulting in an F1-score of `0.000`.

For Gas Sequential Batch, DAAW achieved:

- Precision: `0.800`
- Recall: `0.140`
- F1-score: `0.236`
- Mean detection delay: `16.12` rounds

CDA-FedAvg performed significantly better in both HAR Activity Drift and Gas Sequential Batch.

### Overall interpretation

Reported mean locally adapted client accuracy remained broadly comparable between FedAvg, CDA-FedAvg and DAAW. The main contribution of DAAW was therefore selective drift detection with substantially fewer false alarms, rather than a universal improvement in predictive accuracy.

The results show that temporal model-update cosine similarity is effective for pronounced label-altering drift. However, it is not equally sensitive to every gradual or sequential drift pattern under the configured history windows and thresholds.

## Installation

Python 3.12 was used for the final experiments.

Create and activate a virtual environment on Windows:

```bash
python -m venv fl_env
fl_env\Scripts\activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Datasets

The raw datasets are not committed to this repository. Download them from the official UCI Machine Learning Repository pages:

- [Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- [Gas Sensor Array Drift Dataset at Different Concentrations](https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations)

Place the extracted files in the paths expected by `utils_v3.py` and `utils_gas_v3.py`, retaining the required local `data/` directory structure.

The datasets are public secondary datasets and are not redistributed in this repository.

## Reproducing the corrected results

Run the data-integrity suite first:

```bash
python test_v3_ownership.py
```

The expected outcome is **414 successful checks and zero failures**.

Run the representative seed-42 experiments:

```bash
python server_v3.py
python server_gas_v3.py
```

These commands generate the per-round accuracy, training-loss, cosine-similarity, detection-timeline and computational-time figures.

Run the complete five-scenario, five-seed evaluation:

```bash
python significance_test_v3.py
```

This command evaluates seeds `42`, `123`, `456`, `789` and `999` and produces the data required for the paired statistical comparisons.

## Results and evidence

`FINAL_DISSERTATION_RESULTS/` is the curated evidence package.

### `01_PRIMARY_V4_FIXED/`

Contains the official corrected V4 evidence:

- Five-seed scenario results
- Per-seed metric files
- Consolidated result tables
- Statistical-test results
- Summary figures
- Representative seed-42 trajectories
- Integrity and metric-verification evidence
- Snapshot of the corrected source code

These are the primary results reported in the dissertation.

### `02_SENSITIVITY_ANALYSIS/`

Contains the alpha, threshold and window sensitivity studies.

These studies predate the final V4 methodological correction. They are retained as historical exploratory evidence and must not be presented as corrected V4 validation unless rerun using the final pipeline.

### `03_EXPLORATORY_ADAPTIVE/`

Contains a separate adaptive-threshold extension using short/long windows of `3/10`, an adaptive threshold and two-of-three confirmation.

The extension improved recall in HAR Activity Drift and Gas Sequential Batch. However, it introduced additional false alarms and reduced performance in the HAR and Gas Label Shuffle scenarios.

It was therefore not adopted as a replacement for fixed-threshold DAAW. It is reported as an exploratory trade-off finding.

### `04_SUPPORTING_EVIDENCE/`

Contains additional reproducibility evidence:

- Complete per-seed results
- Software-environment information
- Dataset-source information
- Reproducibility instructions
- Saved experimental logs

## Reporting convention

The results in `01_PRIMARY_V4_FIXED/five_seed_results/` and `01_PRIMARY_V4_FIXED/summary_tables/` are the official five-seed findings.

Figures in `01_PRIMARY_V4_FIXED/scenario_runs/` represent one seed and should be captioned as:

> Representative seed-42 trajectory under the corrected V4 pipeline.

These trajectory figures must not be described as five-seed averages.

The contents of `02_SENSITIVITY_ANALYSIS/` must be identified as historical pre-correction evidence.

The contents of `03_EXPLORATORY_ADAPTIVE/` must be identified as an exploratory extension rather than the primary DAAW result.

## Scope and limitations

The experiments use a controlled, single-machine federated-learning simulation. They do not represent deployment across physical federated devices.

Five independent seeds improve repeatability but limit statistical inference. The fixed thresholds and history-window configurations do not generalise equally well across every drift scenario.

No raw personal data, signed ethics forms, dissertation drafts, virtual environments or raw UCI datasets are included in this repository.

## Completion status

The curated repository contains the corrected V4 pipeline, official five-seed results, representative trajectory figures, consolidated tables, statistical tests and saved data-integrity evidence required to support the dissertation.
