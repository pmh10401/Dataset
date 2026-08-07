# Exp439 Scale-Aware Reconstruction Repair Design

## Purpose

Exp438 stopped after 89 of 1,117 datasets because its FFT reconstruction guard
used the fixed absolute tolerance `1e-5`. The failing dataset was
`Chinatown_normal_1`: its finite TRAIN values reach roughly 1,700, and normal
float32 rounding produced reconstruction errors from `1.52587890625e-05` to
`6.103515625e-05`. The guard incorrectly treated those errors as an invalid
decomposition.

Exp439 repairs only this numeric validation boundary. It does not change the
FFT split, depth selection, ASPE features, normal-reference distances,
thresholds, candidate rules, frozen lanes, or review-only policy.

## Evidence and predecessor status

- Exp438 reviewed code commit: `d1da286e9c74c64abe76e0c255c86c30141813bd`.
- Exp438 run root:
  `/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807`.
- Exp438 terminal state: failed, 89/1,117 committed, 74 successful datasets,
  15 abstained datasets, one runtime error, checkpoint
  `BeetleFly_normal_1`.
- Exact exception: `ValueError: decomposition reconstruction tolerance failed`.
- The failing dispatch window is dataset positions 89 through 99. A TRAIN-only
  diagnostic found failures only in `Chinatown_normal_1`, with 22 finite TRAIN
  rows rejected at depth 1.
- The same diagnostic found zero nonfinite TRAIN rows in that dispatch window.
- The failed Exp438 output, checkpoint, readiness, logs, and records remain
  immutable evidence.

## Chosen approach

Exp439 is a mechanically isolated fork of the reviewed Exp438 implementation.
It receives a new experiment identity, new run root, new readiness artifact,
new checkpoint, new publication, and new evaluation root. The package keeps the
same component boundaries so that the Exp438 regression suite can be replayed
against Exp439 with only identity and repair-specific expectations changed.

The only numeric behavior change is the reconstruction guard:

1. Compute `low64` and `high64` exactly as Exp438 does.
2. Cast both components to float32 exactly as Exp438 does.
3. Reconstruct from the stored float32 components.
4. Compute the maximum absolute reconstruction error.
5. Compute `peak = max(abs(original), abs(low), abs(high), 1.0)`.
6. Compute one float32 ULP at `peak` with `np.spacing(np.float32(peak))`.
7. Accept only when the error is at most `max(1e-5, one_ulp)`.

This preserves the original strict floor for small normalized signals and adds
only the scale-dependent rounding allowance required for large finite signals.
The guard continues to reject nonfinite outputs and errors larger than one ULP.

## Rejected approaches

### Increase the fixed tolerance to `1e-4`

This would pass Chinatown, but it would also weaken validation for small
normalized signals and could hide real decomposition errors. It is rejected.

### Exclude `Chinatown_normal_1`

The rows are finite and valid. Excluding the dataset would reduce coverage and
confuse ordinary numeric scale with missing data. It is rejected.

### Modify or resume the failed Exp438 root

Exp438 is a frozen failed experiment. Mutating its reviewed code identity or
failed checkpoint would break the audit trail. It is rejected.

## Missing-value policy

The Exp438 policy is retained unchanged:

- A TRAIN row containing any NaN, positive infinity, or negative infinity is
  excluded as a whole row before grouping or calibration.
- Finite sibling TRAIN rows remain eligible.
- No missing value is imputed or replaced.
- If an exact-length TRAIN group has fewer than the frozen minimum number of
  finite rows, that group is unavailable.
- A nonfinite TEST row abstains. It is never repaired from labels or sibling
  TEST values.

## Checkpoint reuse

Exp439 may import the 89 committed Exp438 records because each record contains
only canonical `dataset_metadata` and `result` payloads, not an Exp438 context
hash. Import is allowed only when all of these checks pass before any new work:

- the Exp438 readiness, context, checkpoint state, and every source record hash
  validate as a complete ordered prefix;
- the Exp438 reviewed commit, database hash, source hashes, dataset order, lane
  mappings, backend, frozen config, and zero-label-access counters match the
  preregistered predecessor identity;
- every imported dataset precedes the failed dataset window;
- the changed guard is proven not to modify `low`, `high`, feature, score,
  threshold, candidate, abstention, or audit bytes for inputs that passed the
  old `1e-5` guard;
- Exp439 writes the source record path and SHA256 into an import manifest and
  reports `import_count=89` separately from recomputation.

If any import condition fails, Exp439 imports zero records and recomputes from
dataset zero. Partial or best-effort import is forbidden.

## Prediction boundary

Exp439 prediction is label-free. Prediction code may read normal TRAIN values,
TEST values, exact row identity and length, frozen lanes, Exp438 frozen
predecessor records, and pinned source manifests. It may not read TEST labels,
actual anomaly locations, family-level historical TEST performance, or any
post-freeze evaluation result.

TEST labels remain behind the explicit `evaluate` command and may be read only
after the Exp439 prediction publication and hashes are frozen. Evaluation may
compare Exp439 against the frozen Hard comparator, Exp437, and Exp438 as
preregistered evidence; it may not change Exp439 prediction parameters.

## Runtime and monitoring

- Official CPU workers are `max logical CPUs - 1`, which is 11 on this Mac.
- Inner numeric libraries use one thread per worker.
- MLX, MPS, and Numba are benchmarked during readiness. They are selected only
  if candidate parity, finite output, numeric tolerance, and at least 1.1x
  end-to-end speedup all pass.
- Only one MLX/MPS-heavy process may run at a time.
- Progress is written atomically to JSON plus append-only event and text logs.
- Logs include PID, current dataset, TRAIN/TEST counts and lengths, stage,
  backend, completed/total, import/recompute/success/error counts, throughput,
  elapsed time, ETA, checkpoint, and latest error.
- Worker failures must include the dataset name in progress and terminal error
  evidence so the next failure cannot be anonymous.

## Test design

The repair must be developed test-first and prove all of the following:

1. The real frozen `Chinatown_normal_1` TRAIN row identity and SHA that failed
   Exp438 passes the scale-aware guard.
2. The returned float32 `low` and `high` bytes are identical to the components
   computed by Exp438 before its old guard raises.
3. Small normalized signals retain the `1e-5` floor.
4. A reconstruction error larger than one ULP is rejected.
5. NaN and infinity remain rejected or excluded under the existing contracts.
6. Candidate coordinates and all score bytes remain identical between Exp438
   and Exp439 for deterministic inputs that pass the old guard.
7. The Exp438 89-record prefix imports only under exact identity and hash
   parity; any mismatch forces a zero-import fresh run.
8. Runtime exceptions name the failing dataset.
9. New, interrupted resume, inspect, publication, and lazy evaluation preserve
   zero TEST-label access during prediction.
10. The full Exp439, shared Exp438/Exp436, and Exp425 regression suites pass.

## Success and stop conditions

Implementation readiness succeeds only when the code map is synchronized, the
working tree is clean, all focused tests pass, the database has one link before
snapshot creation, backend parity is recorded, the Exp438 failed root validates
read-only, and the real Chinatown reproduction passes.

The official Exp439 run may start after readiness. It stops safely on any hash,
identity, finite-value, candidate-budget, label-access, checkpoint-prefix, or
numeric-bound violation. It does not overwrite Exp438 or select parameters from
TEST results.

## Approved post-start scheduler and calibration-cache repair

The interrupted 101-record official root remains immutable. A distinct run root
with a new clean reviewed commit may reuse only the already preregistered exact
89-record Exp438 prefix.

The repair may keep up to `workers` futures running while earlier completed
results wait for frozen-order commit. Pending plus buffered results are bounded
by `workers ** 2`. Dataset commit order does not change.

TRAIN-only preferred depths may be computed once per row because a fold's depth
is exactly the lower median after excluding that row. Decomposition features may
be computed once per depth and reused by deleting the held-out feature row.
Robust transforms remain fold-local, and score, threshold, candidate, lane,
budget, missing-value, and label-access rules remain unchanged. Acceptance
requires byte identity against at least one predecessor record and a real
label-free bottleneck-dataset smoke.
