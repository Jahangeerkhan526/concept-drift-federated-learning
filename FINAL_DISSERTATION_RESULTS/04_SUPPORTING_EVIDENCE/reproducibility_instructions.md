# Reproducibility Instructions

## Environment
See `software_environment.txt` for exact package versions. Create the environment with:

```bash
python -m venv fl_env
fl_env/Scripts/activate   # Windows
pip install torch numpy scipy matplotlib
```

## Reproducing the primary V4 results
The corrected pipeline lives in the repository root as `*_v3.py` files (the "V4"
label refers to the corrected methodology; the filenames were not renamed to
avoid breaking existing imports mid-project):

- `utils_v3.py`, `utils_gas_v3.py` - corrected dataset loading and drift-injection logic
- `server_v3.py`, `server_gas_v3.py` - FedAvg / CDA-FedAvg / DAAW training loops
- `significance_test_v3.py` - runs all 5 scenarios across 5 seeds
- `test_v3_ownership.py` - the 414-check invariant suite (client ownership,
  train/test disjointness, no duplication) that must pass before trusting any
  run's output

Run the invariant suite first:
```bash
python test_v3_ownership.py
```
All checks must print PASS before proceeding.

Then run the 5-seed significance sweep:
```bash
python significance_test_v3.py
```

## Reproducing the adaptive-DAAW extension (03_EXPLORATORY_ADAPTIVE)
Lives in `testing_adaptive_daaw/`, a fully isolated copy of the V4 pipeline with
its own pre-registered `MANIFEST.md`. See that file for the exact staged
protocol (Stage 1: window selection, Stage 2: adaptive threshold, Stage 3:
confirmation rule, Stage 4: validation on 4 unseen seeds).

## How this folder was assembled
`01_PRIMARY_V4_FIXED/five_seed_results/` is a fresh full run of
`significance_test_v3.py` (all 5 scenarios x 5 seeds) - a real experiment run.
Its output was cross-checked field-for-field against an earlier reconstruction
built from already-computed raw per-seed JSON files (`label_shuffle_v4_all_seeds.json`,
`gate_test_seed*.json` at the repo root) and matched exactly before replacing
it. `summary_tables/` and most of `summary_figures/` were built from that
(now-confirmed) data. The build scripts used are `_build_five_seed_results.py`,
`_build_summary_tables_figures.py` and `_build_adaptive_summary.py` in this
folder's parent directory, kept for transparency/audit purposes.

`scenario_runs/` (per-round accuracy/loss/cosine-similarity trajectory graphs)
came from a fresh single-seed (seed 42) run per scenario with full per-round
logging enabled, since this data was never saved during the original V4 runs.
That run's metrics were verified against `five_seed_results/*/seed_42.json`
before its plots were copied in. See the top-level README.md for status.
