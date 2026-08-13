# r4 Low-Cost Readiness Gate Design

## Goal

Catch known data and preprocessing failures before an expensive 1,117-dataset
paper proposal run. The gate must not train models, read TEST labels, create an
official r4 root, or change the frozen anomaly-detection mathematics.

## Operating model

- Grok CLI implements the bounded change with model `grok-4.6` and reasoning
  effort `high` in an isolated Git worktree.
- Codex reviews the diff, runs the focused verification, and integrates only a
  result-equivalent implementation.
- Grok may modify only the files named by the approved implementation plan. It
  must not write to the database, the immutable r3 root, or any proposed r4
  root.
- The active experiment remains Exp448. Before changing a module, the Exp448
  code-map triplet must identify its callers, effects, and covering tests; all
  three map files are regenerated together if the map is insufficient.

## Gate A: values-only population inventory

Read the frozen catalog and the label-free numeric TRAIN/TEST value blobs for
all 1,117 datasets. Do not construct or fit any detector. Validate:

- catalog dataset count is exactly 1,117;
- the known nonfinite inventory contains exactly 19 affected datasets;
- affected normal TRAIN rows total 164;
- nonfinite TRAIN scalar values total 983;
- all-missing TRAIN rows total zero;
- every affected row can be repaired by the frozen Exp249
  linear-interpolation/nearest-edge rule to a finite float32 row.

The inventory is a fail-closed readiness check, not a new scientific choice.
Any mismatch stops before model work and reports exact dataset and row
coordinates without opening TEST labels.

## Gate B: bounded MiniRocket parity

Run only the 19 affected datasets through the already restored paper
MiniRocket input path and the frozen comparison authority. Require:

- finite TRAIN OOF and TEST scores for both frozen kernel configurations;
- maximum score difference at most `1e-6` against the frozen authority;
- identical candidate coordinates at every frozen alpha;
- no change to thresholds, seeds, folds, routing, or downstream policy rules.

This gate reuses the capable interpreter and frozen aeon implementation. It is
not an official 1,117-dataset run and produces no evaluation metrics.

## Result and failure handling

The readiness command returns one canonical result containing inventory counts,
the ordered affected dataset identities, parity summaries, environment identity,
and SHA-256 bindings for the database, code closure, and relevant provenance.
It reports one of two terminal states:

- `ready`: both gates passed exactly;
- `not_ready`: the first mismatch plus the complete bounded failure list.

No partial result may be interpreted as readiness. A failed gate does not create
or modify an r4 run root. Temporary files, if required by a test, live only in a
temporary directory and are deleted after verification.

## Acceleration and worker decision

- Values-only inventory is I/O-bound and small; it remains sequential CPU to
  avoid database contention and unnecessary process-transfer cost.
- The MiniRocket gate uses the official frozen aeon CPU transformer. Rewriting
  it for MLX, MPS, or PyTorch would change the comparison implementation.
- PyTorch, MLX, and MPS are therefore assessed as non-canonical for this gate.
  CPU is the canonical backend. No new parallel experiment stage is introduced.
- If future measurements justify a CPU-parallel implementation, it must use the
  recorded logical-CPU-minus-one policy with inner numerical threads fixed to
  one and prove identical coordinates before adoption.

## Verification

Focused tests must prove:

1. the exact frozen inventory passes;
2. a changed count, all-missing row, or unrepaired nonfinite value fails before
   MiniRocket starts;
3. one parity score beyond tolerance or one candidate mismatch fails readiness;
4. no TEST labels, evaluation files, r3 artifacts, or r4 paths are opened;
5. repeated runs on the same inputs produce byte-identical canonical results.

After the focused tests pass, Codex independently checks the diff, code-map
fingerprints, provenance bindings, and a bounded real-data execution. The
official r4 launch remains a separate user-approved action.

## Success boundary

This task ends when the readiness gate is implemented and verified. It does not
create or launch r4, evaluate labels, compute AP/AUROC, modify public baselines,
or change any detector score.
