# Final Dissertation Results

Corrected V4 results, kept clearly separate from exploratory work, for the
dissertation "Detecting Concept Drift in Non-IID Federated Learning
Environments."

## Structure

- **`01_PRIMARY_V4_FIXED/`** - the official reported results. Fixed-threshold
  DAAW (window 5/20, threshold 0.30; Gas Sequential Batch uses
  8/30 and 0.15) vs FedAvg vs CDA-FedAvg, corrected V4 methodology, 5 scenarios x 5 seeds.
  - `five_seed_results/` - per-scenario 5-seed aggregates (mean +/- SD, paired
    t-tests). **Complete, real data** from a fresh full run of
    `significance_test_v3.py` (all 5 scenarios x 5 seeds). Cross-checked
    field-for-field against the earlier reconstruction built from the raw
    per-seed JSONs at the repo root (`label_shuffle_v4_all_seeds.json`,
    `gate_test_seed*.json`) - identical to the last decimal - before replacing
    it.
  - `scenario_runs/` - representative seed-42 trajectory curves (accuracy/loss
    by round, cosine-similarity trace, detection timeline, computational time).
    **Complete** - fresh seed-42 run via `server_v3.py`/`server_gas_v3.py`,
    verified to match `five_seed_results/*/seed_42.json` exactly for all 5
    scenarios before being copied in here.
  - `summary_figures/`, `summary_tables/` - built from the five-seed aggregates.
  - `verification/` - the 414-check invariant suite output, seed-42 gate record and
    saved validation evidence.
  - `code_snapshot/` - the exact corrected V4 source files.

- **`02_SENSITIVITY_ANALYSIS/`** - alpha (0.1/0.5/1.0), threshold and window
  sensitivity sweeps. **All three predate the V3/V4 methodology correction**
  (confirmed by file timestamps, all before the Aug 31 fix). Kept as historical
  evidence only. Do not present these as verified under the corrected
  methodology unless re-run.

- **`03_EXPLORATORY_ADAPTIVE/`** - the adaptive-threshold extension experiment
  (window 3/10, adaptive lambda, 2-of-3 confirmation). Pre-registered protocol
  in `MANIFEST.md`. Validated on 4 unseen seeds. **Not adopted as the primary
  result** - it recovers recall on the two weak scenarios (HAR Activity Drift,
  Gas Sequential Batch) but introduces new false alarms on two previously
  perfect scenarios (HAR/Gas Label Shuffle). Reported as an honest trade-off
  finding, not a replacement for the fixed-threshold DAAW.

- **`04_SUPPORTING_EVIDENCE/`** - reproducibility instructions, software
  environment, dataset sources, full experiment logs, and a flattened
  per-seed results CSV.

## What's real vs. what's still needed

`01_PRIMARY_V4_FIXED/five_seed_results/` is a fresh full run of
`significance_test_v3.py` (all 5 scenarios x 5 seeds) - a real experiment run,
not a reconstruction. Its numbers were cross-checked field-for-field against
the earlier version (built from already-computed raw per-seed JSONs at the
repo root) and matched exactly before replacing it. `summary_tables/`,
`summary_figures/`, `verification/`, and all of `03_EXPLORATORY_ADAPTIVE/` are
built from **actual, already-computed V4 results** - no separate re-run was
needed to assemble those parts.

`scenario_runs/` and `summary_figures/08_computational_time.png` required one
additional step: a fresh single-seed (seed 42) run per scenario via
`server_v3.py`/`server_gas_v3.py` with full per-round history logging, since
this data was never saved during the original V4 runs. That run is complete -
its metrics were cross-checked field-for-field against
`five_seed_results/*/seed_42.json` (exact match on precision/recall/F1 for all
5 scenarios) before the plots were copied in and before the timing figure was
built from it.

**Nothing outstanding remains in this folder.**

## Reporting convention

Caption single-seed trajectory graphs (once produced) as:
> "Representative seed-42 trajectory under the corrected V4 pipeline."

Never describe these as five-seed means. The five-seed aggregate results in
`five_seed_results/` are the official reported findings.
