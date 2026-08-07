# Exp437 Canonical Metadata Roundtrip Repair Design

## Status

Approved direction: thin successor to Exp436. Exp437 changes execution identity,
external roots, and metadata persistence only. The anomaly-detection science is
unchanged.

## Background

The official Exp436 CPU run preserved 47 successful ordered records and failed
while committing `AllGestureWiimoteX_normal_1`. Its TRAIN and TEST rows have
many different lengths. Metadata was built in numeric length order, but
canonical JSON sorts object keys lexicographically. The checkpoint reader then
incorrectly required the decoded object to retain numeric insertion order.
For example, numeric order places `29` before `102`, while canonical JSON places
`102` before `29`.

This is a persistence bug, not a scoring failure. The failed Exp436 root remains
immutable evidence and is never deleted, repaired, imported, or resumed.

## Goal

Run the exact Exp436 label-free method as Exp437 after making length-histogram
validation independent of JSON object-key order. Complete, freeze, and evaluate
all 1,117 datasets under a new root.

## Non-goals

- No feature, score, alpha, threshold, lane, candidate budget, or routing change.
- No TEST-label, anomaly-location, or historical family-TEST input during prediction.
- No reuse of Exp436's 47 committed results.
- No new MLX/MPS algorithm and no attempt to force accelerator adoption.
- No deletion or mutation of the failed Exp436 root.

## Architecture

### Shared scientific engine

Exp437 reuses the existing Exp436 feature, scoring, backend, input, checkpoint,
progress, reporting, and evaluation implementation. A small immutable execution
profile supplies the experiment identity, run identity, and fixed external
roots. There are no user-selectable path, worker, backend, seed, threshold, or
budget flags.

The Exp436 wrapper remains bound to the preserved Exp436 evidence. A new
`run_exp437_metadata_roundtrip_antigravity.py` wrapper selects the frozen Exp437
profile. Shared scientific configuration values remain byte-for-byte equal
apart from execution identity fields.

### Exp437 fixed identity

- experiment: `Exp437`
- run identity: `exp437-metadata-roundtrip-antigravity-20260807`
- run parent:
  `/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807`
- children: `readiness_antigravity.json`, `output_antigravity`,
  `checkpoint_antigravity`, and `evaluation_antigravity`
- CPU policy: `max(1, logical_cpus - 1)`, currently 11 workers, one numerical
  thread per worker

Readiness is newly generated from the final reviewed clean commit. It runs the
same deterministic CPU/MLX/MPS parity and speed gate. CPU fallback is expected
and valid if neither accelerator reaches 1.1x.

### Metadata repair

Length histograms remain strict semantic maps:

- keys are positive canonical decimal length strings;
- values are nonnegative integers;
- counts sum exactly to the TRAIN or TEST row count;
- unknown or label-like keys remain forbidden.

Validators do not require caller insertion order. They normalize accepted maps
to canonical lexicographic key order, matching canonical JSON. This same rule
is used for checkpoint records, progress snapshots, and completion events.

## Data flow

1. Exp437 readiness binds the final clean commit, frozen sources, 1,117-dataset
   population, zero label access, backend evidence, and the Exp437 profile.
2. The new run recomputes every dataset from normal TRAIN values and label-free
   TEST values.
3. Parent workers return unchanged Exp436 scientific results plus non-label
   dataset metadata.
4. Checkpoint and progress writers validate metadata semantically, serialize it
   canonically, and replay it without relying on object-key insertion order.
5. Publication freezes the exact review-only candidate set.
6. Only after prediction hashes are frozen may retrospective evaluation open
   TEST labels.

## Error handling

Exp437 keeps the existing fail-closed lifecycle. A runtime or persistence error
marks the new root failed and immutable. It is never overwritten or silently
resumed. Readiness and publication remain create-only and no-clobber.

## Verification

- RED/GREEN checkpoint test with length keys `29` and `102`.
- RED/GREEN progress replay test with the same ordering conflict.
- The actual failed Exp436 record parses successfully with the repaired reader.
- Full Exp437/Exp436 shared suite and predecessor regressions pass.
- Code-map HTML, JSON, and lock are regenerated together.
- Official readiness verifies zero label access and the fixed 12/11/1 topology.
- The official run must complete 1,117/1,117 with zero runtime errors before
  publication or evaluation.

## Success criteria

Exp437 succeeds operationally when all 1,117 records are committed and replayed,
the global candidate budget remains at most 100, publication is review-only,
and TEST-label access stays zero during prediction. Scientific improvement is
judged only by the already frozen post-publication paired evaluation gates.
