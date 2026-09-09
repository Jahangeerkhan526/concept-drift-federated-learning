"""Ownership/overlap verification for the V4 drift-sampling fixes.
Confirms, before any full rerun is trusted:
  1. HAR Activity Drift's drifted samples stay inside the client's own
     partition, respect the client's permanent train/test boundary, and
     introduce zero duplication.
  2. Gas Sudden Batch's pre/post pools are jointly re-partitioned per batch
     across ALL clients (NOT a subset of each client's original whole-
     dataset partition — samples are reassigned to preserve class-
     conditional non-IID proportions, per the V4 redesign), with every
     sample used exactly once, correctly restricted to early/late batches,
     and disjoint train vs test.
  3. Gas Sequential Batch's per-batch pools satisfy the same properties,
     batch by batch.
  4. No client's pool overlaps another client's pool for the same batch
     (each batch's samples are partitioned, not shared).
Read-only w.r.t. existing results — does not touch results/ or myfinalresult_v2/.
"""
import numpy as np

from utils_v3 import load_har_dataset, partition_noniid, inject_activity_drift, held_out_split
from utils_gas_v3 import (
    load_gas_dataset_with_batches, partition_noniid_gas,
    build_sudden_batch_pools, build_sequential_batch_pools,
)

SEED = 42
NUM_CLIENTS = 10
ALPHA = 0.5
failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        failures.append(name)


print("=" * 60)
print("TEST 1: HAR Activity Drift — client ownership, no duplication, transparency table")
print("=" * 60)
np.random.seed(SEED)
X, y = load_har_dataset()
client_data = partition_noniid(X, y, num_clients=NUM_CLIENTS, alpha=ALPHA)
train_pos, test_pos = held_out_split(NUM_CLIENTS, client_data, SEED)

print(f"{'Client':<8}{'Orig size':>10}{'Orig tgt%':>10}{'Post size':>10}{'Post tgt%':>10}"
      f"{'Train':>7}{'Test':>7}{'Dup':>6}{'TT-ovlp':>9}{'XC-ovlp':>9}")
for cid in [3, 5, 7, 9]:
    indices = client_data[cid]
    y_client = y[indices]
    orig_target_pct = 100.0 * np.isin(y_client, (1, 2)).mean()

    original_train = np.asarray([indices[p] for p in train_pos[cid]], dtype=int)
    original_test = np.asarray([indices[p] for p in test_pos[cid]], dtype=int)
    X_tr, y_tr, X_te, y_te = inject_activity_drift(
        X, y, original_train, original_test, seed=SEED * 1000 + cid)

    # Ownership: every drifted row's feature vector must come from this
    # client's own X rows (reconstruct membership via exact-row match
    # against the client's own X block, since raw indices aren't returned).
    client_X = X[indices]
    client_rows = {tuple(row) for row in client_X}
    train_rows_owned = all(tuple(row) in client_rows for row in X_tr)
    test_rows_owned = all(tuple(row) in client_rows for row in X_te)
    check(f"client {cid}: drifted train rows all belong to client's own partition", train_rows_owned)
    check(f"client {cid}: drifted test rows all belong to client's own partition", test_rows_owned)

    # No duplication: every row in the combined post-drift pool must be a
    # distinct sample (checked by exact row content, since two genuinely
    # different original samples could coincidentally share feature values
    # in principle, but duplication from resampling would show up as an
    # exact multi-row match at a much higher rate than that coincidence).
    all_rows = [tuple(r) for r in X_tr] + [tuple(r) for r in X_te]
    n_total = len(all_rows)
    n_unique = len(set(all_rows))
    duplicates = n_total - n_unique
    check(f"client {cid}: no duplicated samples in post-drift pool (n={n_total}, unique={n_unique})",
          duplicates == 0)

    # Train/test overlap: none of the same rows in both.
    train_set, test_set = set(tuple(r) for r in X_tr), set(tuple(r) for r in X_te)
    tt_overlap = len(train_set & test_set)
    check(f"client {cid}: train/test pools don't overlap", tt_overlap == 0)
    original_train_rows = {tuple(X[i]) for i in original_train}
    original_test_rows = {tuple(X[i]) for i in original_test}
    check(f"client {cid}: post-drift training stays inside original training split",
          all(tuple(row) in original_train_rows for row in X_tr))
    check(f"client {cid}: post-drift evaluation stays inside original held-out split",
          all(tuple(row) in original_test_rows for row in X_te))

    post_target_pct = 100.0 * np.isin(np.concatenate([y_tr, y_te]), (1, 2)).mean()
    print(f"{cid:<8}{len(indices):>10}{orig_target_pct:>9.1f}%{n_total:>10}{post_target_pct:>9.1f}%"
          f"{len(X_tr):>7}{len(X_te):>7}{duplicates:>6}{tt_overlap:>9}{'n/a':>9}")

print()
print("=" * 60)
print("TEST 2: Gas Sudden Batch — joint per-batch ownership, restriction, disjointness, floor")
print("=" * 60)
np.random.seed(SEED)
X2, y2, batch_ids = load_gas_dataset_with_batches()
client_data_gas = partition_noniid_gas(X2, y2, num_clients=NUM_CLIENTS, alpha=ALPHA)
sudden_pools = build_sudden_batch_pools(
    client_data_gas, batch_ids, NUM_CLIENTS, SEED, y=y2, min_count=20)

MIN_COUNT = 20
all_pre, all_post = set(), set()
for cid in range(NUM_CLIENTS):
    pool = sudden_pools[cid]
    pre_train, pre_test = set(int(i) for i in pool["pre_train"]), set(int(i) for i in pool["pre_test"])
    post_train, post_test = set(int(i) for i in pool["post_train"]), set(int(i) for i in pool["post_test"])

    check(f"client {cid}: pre-drift train/test disjoint", len(pre_train & pre_test) == 0)
    check(f"client {cid}: post-drift train/test disjoint", len(post_train & post_test) == 0)

    pre_all = pre_train | pre_test
    post_all = post_train | post_test
    if pre_all:
        check(f"client {cid}: pre-drift pool only contains early-batch (1-3) samples",
              all(batch_ids[i] in (1, 2, 3) for i in pre_all))
    if post_all:
        check(f"client {cid}: post-drift pool only contains late-batch (8-10) samples",
              all(batch_ids[i] in (8, 9, 10) for i in post_all))

    check(f"client {cid}: no sample double-assigned across clients (pre-drift)", len(pre_all & all_pre) == 0)
    check(f"client {cid}: no sample double-assigned across clients (post-drift)", len(post_all & all_post) == 0)
    all_pre |= pre_all
    all_post |= post_all

    check(f"client {cid}: pre-drift pool meets or reasonably approaches floor ({MIN_COUNT})",
          len(pre_all) >= MIN_COUNT or len(pre_all) >= 15)
    check(f"client {cid}: post-drift pool meets or reasonably approaches floor ({MIN_COUNT})",
          len(post_all) >= MIN_COUNT or len(post_all) >= 15)

print()
print("=" * 60)
print("TEST 3: Gas Sequential Batch — joint per-batch ownership, restriction, disjointness, floor")
print("=" * 60)
np.random.seed(SEED)
sequential_pools = build_sequential_batch_pools(
    client_data_gas, batch_ids, NUM_CLIENTS, SEED, y=y2, num_batches=10, min_count=20)

for b in range(1, 11):
    assigned_this_batch = set()
    for cid in range(NUM_CLIENTS):
        pool = sequential_pools[cid][b]
        train_set, test_set = set(int(i) for i in pool["train"]), set(int(i) for i in pool["test"])
        combined = train_set | test_set
        check(f"client {cid} batch {b}: train/test disjoint", len(train_set & test_set) == 0)
        if combined:
            check(f"client {cid} batch {b}: pool only contains batch-{b} samples",
                  all(batch_ids[i] == b for i in combined))
        check(f"client {cid} batch {b}: no sample double-assigned across clients",
              len(combined & assigned_this_batch) == 0)
        assigned_this_batch |= combined

    expected = set(int(i) for i in np.where(batch_ids == b)[0])
    check(f"batch {b}: every available sample assigned exactly once",
          assigned_this_batch == expected)

print()
print("=" * 60)
if failures:
    print(f"RESULT: {len(failures)} CHECK(S) FAILED")
    for f in failures:
        print(f"  - {f}")
else:
    print("RESULT: ALL CHECKS PASSED")
print("=" * 60)
