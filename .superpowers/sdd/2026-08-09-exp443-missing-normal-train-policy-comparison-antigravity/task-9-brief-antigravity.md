# Exp443 Task 9 — freeze and inspect three label-free predictions

## Purpose

Turn one completed 57-unit Exp443 attempt into a fixed, replayable publication
before any TEST label is opened. This task does not change preprocessing,
MiniROCKET scores, thresholds, candidates, routing, or policy choice.

## Code-map impact check

- What calls the new module: Task 11 CLI publication/inspection commands; no
  current caller exists.
- What it affects: it reads the completed Exp443 checkpoint, Task 6 control,
  Task 8 parity evidence, frozen source snapshot, and pure routing output; it
  writes only a publication directory beneath the completed attempt.
- Tests: create `test_reporting_antigravity.py` and retain the existing 15
  Exp443 test modules as regression coverage.

The current grouped node is `exp443_environment_contracts`; it has no incoming
edge, imports `artifact_utils`, reads `sqlite_store`, and participates in the
source-freeze flow. Keep the map at no more than 20 primary nodes and regenerate
HTML, JSON, and lock together.

## Owned files

- Create `experiments/exp443_missing_normal_train_policy_comparison_antigravity/reporting_antigravity.py`
- Create `experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests/test_reporting_antigravity.py`
- Create `.superpowers/sdd/2026-08-09-exp443-missing-normal-train-policy-comparison-antigravity/task-9-report-antigravity.md`
- Narrow checkpoint/runtime integration is allowed only when a read-only
  completed-checkpoint inspection seam is required.
- Regenerate `docs/codemap/codemap.html`, `docs/codemap/codemap.json`, and
  `docs/codemap/codemap.lock` together.

## Required input gates

1. The attempt checkpoint is complete with exactly 57 successful records and
   no error record.
2. Progress is complete, PID is inactive, and the Task 6 control passed.
3. Results contain exactly 19 case-sensitive sorted datasets for each of the
   three frozen policies.
4. Frozen source/config/database identities still match and label access is 0.
5. Task 8 parity evidence says 19 units, workers=11, inner_threads=1,
   semantic_equal=true, equal sequential/parallel payload hashes, and label
   access 0.
6. Re-routing the checkpoint results recreates each runtime prediction byte for
   byte. Publication never imports the evaluator or reads TEST labels.

## Publication inventory

Create exactly these files under `publication_antigravity`:

1. `prediction_linear_edge_nearest_control_antigravity.jsonl`
2. `prediction_exclude_nonfinite_row_antigravity.jsonl`
3. `prediction_leave_one_out_peer_median_antigravity.jsonl`
4. `policy_score_audit_antigravity.jsonl`
5. `train_policy_audit_antigravity.jsonl`
6. `control_gate_antigravity.json`
7. `worker_parity_antigravity.json`
8. `source_audit_antigravity.json`
9. `backend_decision_antigravity.json`
10. `manifest_antigravity.json`
11. `status_antigravity.json`
12. `artifact_sha256_antigravity.json`

Each prediction contains exactly 1,117 case-sensitive sorted dataset rows. The
score audit contains 57 policy-major result summaries and binds the checkpoint
record hash. The TRAIN audit remains compact: one row per policy/dataset with
counts, action counts, and a canonical detailed-audit digest; the durable
checkpoint remains the detailed source of truth.

The manifest binds every semantic artifact hash and the three separate
prediction hashes. Status binds the manifest and reports
`complete_review_only`. The artifact map hashes every other publication file
and explicitly excludes itself. Manifest and status must state:

- `contains_test_labels=false`
- `contains_evaluation_metrics=false`
- `label_access_count=0`
- `review_only=true`
- `operational_alert_eligible=false`

## Write and replay behavior

- Build all bytes in a private same-parent antigravity staging directory.
- Use existing Exp443 create-only/fsync/exclusive-rename primitives.
- An existing final publication is inspection-only: replay and return it when
  every byte matches; never overwrite it.
- Inspection validates the exact 12-file inventory, hashes, row counts/order,
  policy identities, result/source/control/parity identities, and recomputes
  all three routing outputs from the completed checkpoint.
- Keep tests centered on real execution/science failures: incomplete attempt,
  failed control, missing/extra result, changed runtime prediction, label or
  evaluation field, hash mismatch, and successful replay. Do not add unrelated
  in-process object-tamper or publication-security research.

## Verification

1. Behavior TDD for reporting/replay.
2. Focused reporting tests.
3. Full Exp443 regression.
4. Fixture publication followed by independent inspection with exact byte
   identity and zero label access.
5. Regenerate/validate the code-map triplet and run `git diff --check`.
6. One focused Sol review limited to scientific correctness, reproducibility,
   label-free boundaries, and whether Task 10 can evaluate the frozen bytes.

