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

- Focused reporting suite: 7/7 passed.
- Full Exp443 suite: 134 tests passed with two expected opt-in skips.
- Python compileall: passed.
- `git diff --check`: passed before commit.
- Fixture publication: exact 57 records, 3 x 19 results, 3 x 1,117 published
  rows, Policy A routing 137/2,253/35/2,218, zero label access, exact replay.
- Negative coverage: incomplete checkpoint, failed/mismatched control,
  changed runtime prediction, forbidden label payload, and artifact-map change.

## Next boundary

Task 10 may read only the three frozen JSONL predictions after successful
publication inspection. It may open TEST labels only after those hashes are
fixed and must not alter any policy, score, candidate, or routing input.

