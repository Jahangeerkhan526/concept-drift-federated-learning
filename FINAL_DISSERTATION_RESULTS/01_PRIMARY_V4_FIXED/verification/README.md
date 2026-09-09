# Verification

- `ownership_tests.txt` - the full 416-check invariant suite output. This single
  file covers client ownership, train/test overlap (disjointness), duplication/
  unassigned-sample checks, and batch restriction together (the underlying
  `test_v3_ownership.py` script tests all of these in one run, so they are not
  split into separate files).
- `metric_validation.txt` - independent recalculation log confirming the
  corrected event-level precision/F1 formula matches the pipeline's own output
  to 4 decimal places across all seed files.
- `complete_test_output.txt` - full gate-test log covering the remaining
  validation seeds (123, 456, 789, 999) run after the seed-42 gate test passed.
