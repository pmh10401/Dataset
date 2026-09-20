# Exp443 Task 9 — label-free publication report

## Outcome

Task 9 now converts one completed Exp443 attempt into three immutable,
label-free prediction publications and independently reconstructs every byte
from the completed checkpoint. It does not change preprocessing, MiniROCKET,
alpha, seeds, candidates, thresholds, routing, or policy choice.

No official Exp443 prediction has been created by this implementation task.
The production publisher will run only after the official 57-unit Task 12
attempt completes.

## Read-only completed checkpoint

`Exp443CheckpointStoreAntigravity.inspect_complete_antigravity()` validates the
context, terminal state, exact 57-record inventory, record-SHA prefix, policy
and dataset identities, and decoded result contracts. Unlike resume, it does
not accept a replacement process, probe/rebind PID ownership, update a
timestamp, clear a temporary file, or write any byte. The fixture test compares
the complete checkpoint tree before and after inspection and requires byte
identity.

## Publication and replay

`reporting_antigravity.py` requires:

- complete 57/57 progress and inactive PID evidence;
- 57 successful checkpoint results, exactly 19 per frozen policy;
- matching config, environment lock, database, source snapshot, and reviewed
  commit identities;
- all 76 Policy A control comparisons finite, candidate-identical, and within
  `1e-6`;
- Policy A routing algebra `1,117 / 137 / 2,253 / 35 / 2,218` with exact
  Exp291/Exp404 coordinate equality;
- Task 8 parity `19 units / 11 workers / 1 inner thread`, equal sequential and
  parallel semantic hashes, and zero label access;
- byte-identical reconstruction of each runtime top-level prediction JSON.

Each rerouted prediction becomes exactly 1,117 canonical JSONL lines in
case-sensitive dataset order. The manifest connects each runtime JSON SHA256
to its published JSONL SHA256. Inspection repeats the checkpoint/source/control
gates, reroutes all three policies, reconstructs all 12 files, and requires
byte identity.

The create-only inventory is:

1. three policy prediction JSONL files;
2. one 57-row compact score audit;
3. one 57-row compact TRAIN-policy audit;
4. control, worker parity, source, and backend evidence;
5. manifest, status, and artifact map.

The hash graph is acyclic: nine leaves feed the manifest, the manifest feeds
status, and the artifact map hashes the other 11 files while excluding itself.
An existing publication is replay-only and is never overwritten.

## Sol review fix round

The first final review found that the publisher structurally checked the saved
Task 6 control JSON but did not independently reproduce it. Publication now
reruns `compare_policy_a_to_archive_antigravity()` from the inspected Policy A
checkpoint results and frozen archive snapshot. The recomputed pass evidence
must be canonical-byte identical to the runtime control artifact, including all
76 score/candidate comparisons and complete coordinate identities.

The completed-checkpoint inspection now requires the exact frozen topology:
57 units, three policies, 19 identically ordered datasets per policy. Progress
and PID evidence use exact schemas and must point to the inspected checkpoint
state and its PID/start identity, with counters and final dimensions equal to
the terminal checkpoint.

Publication failures before the first file, after one file, before rename, and
after final rename verification now leave a unique `_failed_*_antigravity`
audit sibling with canonical `failure_antigravity.json`. A verified final
publication is never removed or replaced when a later audit hook fails.

## Governance boundary

Prediction construction and inspection import no evaluator and execute no
TEST-label query. Manifest and status freeze:

- `contains_test_labels=false`;
- `contains_evaluation_metrics=false`;
- `label_access_count=0`;
- `review_only=true`;
- `operational_alert_eligible=false`.

Exact forbidden label/evaluation keys in inherited evidence are rejected.
TEST labels remain exclusively Task 10 post-freeze input.

## Verification

- Focused reporting suite: 14/14 passed.
- Full Exp443 suite: 141 tests passed with two expected opt-in skips.
- Python compileall: passed.
- `git diff --check`: passed before commit.
- Fixture publication: exact 57 records, 3 x 19 results, 3 x 1,117 published
  rows, Policy A routing 137/2,253/35/2,218, zero label access, exact replay.
- Negative coverage: incomplete or wrong 56/58/policy/dataset checkpoint,
  failed/fabricated control, changed runtime prediction, progress/PID drift,
  forbidden label payload, artifact-map change, and four publication-failure
  phases with preserved audit evidence.

## Next boundary

Task 10 may read only the three frozen JSONL predictions after successful
publication inspection. It may open TEST labels only after those hashes are
fixed and must not alter any policy, score, candidate, or routing input.
