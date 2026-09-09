# Adaptive DAAW Extension — Pre-registration Manifest

Written BEFORE any adaptive/window-development code exists, so the success
criteria and configuration space below are genuinely pre-declared, not
written after seeing results.

## Status
This is a **separate, post-baseline extension experiment**. It does not
replace, overwrite, or get merged with the corrected V4 baseline
(`utils_v3.py`, `utils_gas_v3.py`, `server_v3.py`, `server_gas_v3.py`,
`significance_test_v3.py`), which remains the official, primary, reported
result set for the dissertation regardless of this experiment's outcome.

## Frozen V4 baseline files this experiment starts from (SHA-256)
```
155f01f9f69cce9ea72f19248467df3fcc8eae1324dd423d358d9780883bdb17  utils_v3.py
844d05c0b2c1323014c1e95c06b34a4678fbc0a75f358f43e0d79aa175c24f4b  utils_gas_v3.py
ea2f16ed7702322a9f6a3f4d35d0c118776397fcd3571ff2e9ad1d5ed1c01134  server_v3.py
90817b4fb66271607c1fec1246209b06ce63cf8503e2dd23e41c53c7d61590f3  server_gas_v3.py
9ccec770493fb8a0d90434d5a75914e5eb0fa225cb41e85b36827d5738e5d0e5  significance_test_v3.py
```

## Copy record
- Copied: 2026-08-31T16:33:52Z
- Copies verified byte-identical (SHA-256 match) to their V4 sources at copy
  time: utils_adaptive.py, utils_gas_adaptive.py, server_adaptive.py,
  server_gas_adaptive.py, significance_test_adaptive.py.
- Development seed: 42
- Validation seeds: 123, 456, 789, 999
- Output restriction: every result this experiment produces is written only
  under `testing_adaptive_daaw/outputs/`. Nothing outside this folder is
  ever written to.
- Import discipline: the copied files still `import utils_v3`/`utils_gas_v3`/
  `server_v3`/`server_gas_v3` internally (identical to their source, since
  they're byte-for-byte copies). Before any modification, these internal
  imports must be repointed to the adaptive copies (`utils_adaptive`,
  `utils_gas_adaptive`, `server_adaptive`, `server_gas_adaptive`) so nothing
  in this experiment silently runs against the frozen V4 modules instead of
  its own copies. Compile-checked after repointing, before the first smoke
  test.

## Current V4 baseline — all 5 scenarios, for later comparison
(5-seed mean ± sample SD, from the official, unmodified V4 pipeline)

| Scenario | CDA F1 | DAAW F1 | p-value | Result |
|---|---|---|---|---|
| HAR Label Shuffle | 0.398 ± 0.032 | 0.978 ± 0.050 | 5.04e-06 | DAAW wins, significant |
| HAR Activity Drift | 0.397 ± 0.025 | 0.000 ± 0.000 | 3.87e-06 | CDA wins, significant |
| Gas Label Shuffle | 0.401 ± 0.026 | 1.000 ± 0.000 | 8.40e-07 | DAAW wins, significant |
| Gas Sudden Batch | 0.412 ± 0.054 | 0.579 ± 0.193 | 0.120 | DAAW higher, not significant |
| Gas Sequential Batch | 1.000 ± 0.000 | 0.236 ± 0.148 | 0.000318 | CDA wins, significant |

DAAW recall detail (V4 baseline): HAR Activity Drift 0/4 every seed (0.0);
Gas Sudden Batch mean recall 0.450; Gas Sequential Batch mean recall 0.140.
These three numbers are exactly what this extension is trying to improve.

## Additional safeguards (agreed before implementation)

**1. Adaptive baseline must be online and ground-truth-blind.** The detector
never sees which rounds are the true drift rounds. The per-client
baseline/MAD is updated only from rounds currently classified as stable
(no confirmed or candidate alarm active that round) — a confirmed or
candidate drift round is excluded from updating the "normal" baseline, so
drifted behaviour can't contaminate the client's own reference distribution.

**2. Minimum calibration history.** Adaptive detection does not activate
until a client has accumulated at least `short_window + long_window`
valid similarity observations (same as the warm-up gate) — this is the
calibration period during which baseline/MAD are computed but no adaptive
decision is made. MAD epsilon: `1e-6` (if MAD < epsilon, fall back to the
absolute-only condition rather than dividing by a near-zero value).

**3. Confirmation behaviour, precisely.**
- A *candidate* alarm (absolute OR adaptive condition true this round)
  does NOT change client weight and does NOT get recorded as a detection.
- Only a *confirmed* alarm (candidate condition true in >= 2 of the latest
  3 rounds) changes the client's aggregation weight.
- Detection delay is measured from the true drift round to the
  CONFIRMATION round (not the first candidate round).
- One confirmed episode counts once (same alarm-episode gating as V4 —
  no re-counting every round the confirmed state persists).

**4. Complete diagnostic traces.** For every client and every round, save:
`similarity`, `absolute_threshold` (0.30), `adaptive_threshold`,
`candidate_alarm` (bool), `confirmed_alarm` (bool), `client_weight`, and
`ground_truth_status` (evaluation-only — never fed back into the detector
itself, per safeguard 1). Saved per scenario per seed under
`testing_adaptive_daaw/outputs/`.

**5. Internal controls stay genuinely separate.** `DAAW-Fixed (revised
warm-up)` uses ONLY: the selected shared window from Stage 1, the absolute
threshold 0.30, no adaptive threshold, no 2-of-3 confirmation. It is not
allowed to pick up any part of the adaptive/confirmation logic — otherwise
Stage 2's comparison can't isolate what the adaptive mechanism itself
contributed.

**6. Identical data for all methods.** For each seed/scenario, client
partitions and fixed train/test pools are built exactly once, then reused
unchanged across FedAvg, CDA-FedAvg, DAAW-Fixed-revised-warmup, and
DAAW-Adaptive — same sample IDs, same drift timing, every method.

**7. Gas Sequential Batch definition, explicit.**
```
Batch 1 warm-up: rounds 1-24
First transition / ground truth: round 25
Primary event: first transition per client (this is what recall/precision/F1 are computed against)
Later transitions (batch 3 onward): operational alarm episodes, not separate ground-truth events
```

**8. Selection terminology.** Reports say "selected configuration," never
"winning configuration." Every candidate's full result (Stage 1's three
window candidates, Stage 2's three lambda candidates) is recorded in
`outputs/seed_42_development/`, including the ones not selected — nothing
that was tried gets silently dropped from the record.

**9. Regression protection, checked before validation runs.** Before
proceeding from Stage 2 (frozen config) to Stage 4 (validation):
- Zero train/test overlap in every client/scenario pool (automated check).
- Zero duplicated or unassigned samples (automated check, same style as
  the V4 ownership tests).
- Deterministic rerun check: running the same seed twice produces
  identical results.
- No files written outside `testing_adaptive_daaw/outputs/`.
- V4's original 5 file hashes (recorded above) still match — confirming
  nothing in this experiment ever touched the frozen baseline files.

## Universal experimental configuration (identical across all 5 scenarios)
- Clients: 10
- Rounds: 50
- Non-IID alpha: 0.5
- Local epochs: 1
- Learning rate: 0.001
- DAAW absolute threshold: 0.30 (unchanged)
- Drifted-client weight: 0.30, recovery +0.10/stable round (unchanged)
- Seeds: 42 (development), 123/456/789/999 (validation)
- Gas Sequential's special 8/30/0.15 config is NOT used here — every
  scenario uses the one shared window/threshold selected below.

## Revised warm-up schedule (applies to ALL methods in this experiment:
## FedAvg, CDA-FedAvg, DAAW-Fixed-revised-warmup, DAAW-Adaptive)
- Event-based scenarios (HAR Label Shuffle, HAR Activity Drift, Gas Label
  Shuffle, Gas Sudden Batch): drift rounds moved to Client 3 -> round 25,
  Client 5 -> round 30, Client 7 -> round 35, Client 9 -> round 40 (>= 24
  clean rounds before any client's first possible drift).
- Gas Sequential Batch: rounds 1-24 = Batch 1 (clean warm-up), then batches
  2-9 get 3 rounds each (25-27, 28-30, ..., 46-48), batch 10 gets rounds
  49-50. Ground truth first-transition event = round 25 for every client.
- Detector warm-up gate: `if len(update_history) < short_window + long_window:
  return False, 1.0` — no decision until both windows are fully populated.

## Stage 1 — Window development (seed 42 ONLY, all 5 scenarios)
Candidates (short_window, long_window, threshold — threshold fixed at 0.30
for this stage):
| Candidate | Short | Long | Threshold |
|---|---|---|---|
| Fast | 3 | 10 | 0.30 |
| Balanced | 5 | 15 | 0.30 |
| Existing | 5 | 20 | 0.30 |

Selection rule (in order): (1) highest macro-average event-level F1 across
all 5 scenarios, (2) if approximately tied, fewer false-alarm episodes,
(3) if still tied, lower detection delay. Must NOT be selected because it
specifically favours HAR Activity Drift or Gas Sequential Batch alone.

## Stage 2 — Adaptive threshold development (seed 42 ONLY, winning window from Stage 1)
Candidate drift condition: `similarity < 0.30 OR similarity < adaptive_threshold`,
where `adaptive_threshold = baseline - lambda * MAD`, `baseline = median(normal
similarities during warm-up)`, `MAD = median absolute deviation of same`.
If MAD == 0, fall back to the absolute condition only (or a small numerical
epsilon) — never divide by zero.
Candidates: lambda in {2.0, 2.5, 3.0}. Same lambda for every client, dataset,
and scenario. Selected the same way as Stage 1 (macro-F1, then false alarms,
then delay).

## Stage 3 — Confirmation rule (fixed, not tuned)
Confirm drift when the candidate condition (absolute OR adaptive) is true in
at least 2 of the latest 3 rounds. Only CONFIRMED events affect aggregation
weight and primary metrics. Candidate (unconfirmed) alarms are recorded
separately for transparency.

## Stage 4 — Freeze and validate
Once Stage 1 and Stage 2 have each selected one shared configuration, ALL
code and parameters are frozen. Validation runs on seeds 123, 456, 789, 999
— nothing is changed after seeing these results, regardless of outcome.

## Final comparison (validation seeds)
| Method | Description |
|---|---|
| FedAvg | No drift detection |
| CDA-FedAvg | Loss-based baseline |
| DAAW-Fixed (revised warm-up) | V4's fixed-threshold logic, but re-run under this experiment's revised warm-up schedule for a fair internal control — NOT the same run as the official V4 baseline, which used the original round 5/20/35/45 schedule |
| DAAW-Adaptive | Universal enhanced method (winning window + winning lambda + 2-of-3 confirmation) |

## Pre-declared success criteria (stated before Stage 1 has been run)
Enhanced DAAW is only considered a successful extension if, on the
validation seeds, it:
- Improves recall on HAR Activity Drift (baseline: 0/4, i.e. 0.0) and on
  Gas Sequential Batch (baseline: mean recall 0.14).
- Retains materially fewer false alarms than CDA-FedAvg (baseline CDA: ~10-12
  alarm episodes on the affected scenarios).
- Does not substantially damage HAR Label Shuffle or Gas Label Shuffle
  performance (V4 DAAW baseline: F1 0.978 and 1.000 respectively).
- Performs consistently across all 4 validation seeds, not just one.
- Uses exactly the same window/lambda/confirmation rule everywhere — no
  scenario-specific tuning at any point.

Illustrative (non-binding) practical targets: HAR Activity recall > 0; Gas
Sudden Batch recall > 0.45; Gas Sequential Batch recall substantially above
0.14 (V4 baseline); mean false alarms still far below CDA's.

These are evaluation targets to interpret the outcome honestly — they are
explicitly NOT thresholds for hiding an unsuccessful result. If the
criteria are not met on the validation seeds, the correct action is to
stop, report the extension as unsuccessful, and treat V4 as final — not to
keep tuning until something passes.

## Reporting rule
V4 is never deleted, replaced, or presented as inferior to whatever this
experiment produces. The dissertation reports, in order: (1) the corrected
V4 fixed-threshold baseline, (2) its diagnosed limitation (poor recall on
gradual/semantic drift), (3) this universally-configured adaptive
extension and how it was developed/validated, (4) an honest verdict on
whether it improved the precision-recall trade-off — whichever way that
verdict comes out.
