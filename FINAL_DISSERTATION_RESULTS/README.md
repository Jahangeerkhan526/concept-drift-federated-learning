# Final Dissertation Results

This folder contains the experimental evidence for the dissertation:

**Detecting Concept Drift in Non-IID Federated Learning Environments Using Cosine Similarity**

The corrected V4 results are kept separate from historical sensitivity studies
and the exploratory adaptive-threshold extension.

## Repository structure

### `01_PRIMARY_V4_FIXED/`

This folder contains the official results reported in the dissertation.

The corrected fixed-threshold DAAW method is compared with FedAvg and
CDA-FedAvg across five drift scenarios and five independent seeds:
`42`, `123`, `456`, `789` and `999`.

Four scenarios use short/long history windows of `5/20` and a detection
threshold of `0.30`. Gas Sequential Batch uses windows of `8/30` and a
threshold of `0.15`.

#### `five_seed_results/`

Contains the saved results for all five scenarios and all five seeds.

Each scenario folder includes:

- Per-seed metric files
- Per-seed summary figures
- Mean and standard-deviation summaries
- Paired statistical-test results
- Seed-level F1 visualisations

These files provide the evidence used to calculate the official five-seed
precision, recall, F1-score, false-alarm episodes, detection delay and reported
mean locally adapted client accuracy.

#### `scenario_runs/`

Contains representative seed-42 trajectory evidence generated using
`server_v3.py` and `server_gas_v3.py`.

The available figures include:

- Accuracy by communication round
- Training loss
- Drift-detection timeline
- Computational time
- Model-update cosine similarity

These are single-seed trajectory figures and must not be described as
five-seed averages.

#### `summary_tables/`

Contains the consolidated results used in the dissertation:

- `main_results.csv`
- `per_seed_results.csv`
- `statstat_tests.csv`
- `experimental_configuration.csv`

The files report the complete corrected V4 outcomes across all methods,
scenarios and seeds.

#### `summary_figures/`

Contains the figures produced from the consolidated experimental results,
including:

- Predictive-accuracy comparison
- F1-score comparison
- Precision and recall comparison
- Detected, missed and false-alarm episodes
- Detection delay
- Per-seed F1 variation
- Statistical significance
- Computational time
- Overall scenario outcomes

#### `verification/`

Contains the saved data-integrity and metric-validation evidence.

The evidence includes:

- Output from the 414-check invariant suite
- Client-ownership checks
- Train/test-overlap checks
- Duplication and allocation checks
- Event-metric validation
- Seed-42 gate-test evidence

The saved invariant-suite output reports 414 successful checks and zero
failures.

#### `code_snapshot/`

Contains a snapshot of the corrected V4 source files used to produce the
official results.

The filenames retain the `_v3.py` suffix because they were preserved during
the final methodological correction. The README at the repository root
explains the difference between the filename version and the corrected V4
methodology.

---

### `02_SENSITIVITY_ANALYSIS/`

Contains the alpha, threshold and window sensitivity studies.

The available analyses include:

- Dirichlet alpha sensitivity
- Detection-threshold sensitivity
- Short/long-window sensitivity

These studies were produced before the final V4 methodological correction.
They are retained as historical exploratory evidence and are clearly
separated from the official corrected results.

They must not be presented as final V4 validation unless they are rerun using
the corrected pipeline.

---

### `03_EXPLORATORY_ADAPTIVE/`

Contains the separate adaptive-threshold extension.

The experiment uses:

- Short/long windows of `3/10`
- An adaptive threshold
- Two-of-three confirmation
- A development run using seed 42
- Validation using four previously unseen seeds

The protocol is documented in `MANIFEST.md`.

The extension improved recall for HAR Activity Drift and Gas Sequential Batch.
However, it also introduced additional false alarms and reduced performance in
the HAR and Gas Label Shuffle scenarios.

For this reason, the adaptive extension was not adopted as a replacement for
the fixed-threshold DAAW method. It is reported as an exploratory trade-off
finding.

---

### `04_SUPPORTING_EVIDENCE/`

Contains additional material supporting reproducibility and auditability:

- Complete per-seed results
- Dataset-source information
- Software-environment information
- Reproducibility instructions
- Full saved experimental logs

The raw UCI datasets are not included in the repository. Their official
download sources are documented in the repository README and dataset-source
file.

## Official findings

The corrected V4 results provide a qualified rather than universally positive
outcome.

- DAAW performed significantly better than CDA-FedAvg for HAR Label Shuffle.
- DAAW achieved perfect five-seed event-level F1 for Gas Label Shuffle.
- DAAW produced substantially fewer false alarms than CDA-FedAvg in most
  scenarios.
- DAAW showed a favourable but statistically non-significant result for Gas
  Sudden Batch.
- CDA-FedAvg performed significantly better for HAR Activity Drift and Gas
  Sequential Batch.
- Reported mean locally adapted client accuracy remained broadly comparable
  between the methods.

The results therefore show that temporal model-update cosine similarity is
effective for pronounced label-altering drift but is not equally sensitive to
every gradual or sequential drift pattern under the configured windows and
thresholds.

## Reporting convention

The results in `five_seed_results/` and `summary_tables/` are the official
five-seed findings.

Figures from `scenario_runs/` must be captioned as:

> Representative seed-42 trajectory under the corrected V4 pipeline.

These trajectory figures illustrate per-round behaviour for one representative
seed. They must not be described as five-seed means.

The material in `02_SENSITIVITY_ANALYSIS/` must be identified as historical
pre-correction evidence.

The material in `03_EXPLORATORY_ADAPTIVE/` must be identified as an exploratory
extension rather than the primary DAAW method.

## Completion status

The curated evidence package is complete. It includes the corrected V4 code,
five-seed results, consolidated tables, dissertation figures, representative
seed-42 trajectories, statistical tests and data-integrity verification
evidence.
