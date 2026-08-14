# Exp449 Reproducible Equipment-Anomaly Rebaseline

## Document status

- Date: 2026-08-14
- Experiment: Exp449
- Status: approved design, implementation not started
- Purpose: establish a new reproducible paper result from the current frozen algorithm without requiring coordinate parity with historical Exp448 artifacts
- Scope: label-free proposal generation, deterministic replay, failure recovery, immutable freeze, and later AUROC/AP evaluation

## Decision

Exp449 is a new scientific object. It preserves the current Task4 through Task7 algorithm and Candidate C policy, but it does not claim to reproduce historical Exp137, Exp236, Exp291, Exp404, or Exp448 coordinates and aggregate counts.

Historical artifacts remain immutable diagnostic evidence. They may be inspected after current scoring to explain differences, but they are not scorer inputs, selector inputs, calibration inputs, thresholds, or pass/fail authorities for Exp449.

Differences between Exp449 and historical outputs are acceptable when the algorithm definition, preprocessing, configuration, seed, environment, and data identities are those preregistered for Exp449. Repeated Exp449 execution must satisfy its own reproducibility contract.

## Goals

1. Freeze the current Task4, Task5, Task6, Task7, and Candidate C algorithm without tuning it toward historical coordinates.
2. Execute a clean, values-only smoke twice and demonstrate deterministic coordinate identity.
3. Execute all 1,117 datasets from a new clean official root with resumable, context-bound per-dataset checkpoints.
4. Continue after individual dataset failures, then retry only the failed and never-attempted datasets.
5. Block population-global stages until all 1,117 Task4 and Task5 results are verified successes.
6. Freeze an immutable proposal manifest before any TEST label access.
7. Evaluate Exp449 and public anomaly-detection baselines with the same AUROC/AP evaluation implementation.

## Non-goals

- Reconstructing unavailable contemporaneous Exp89, Exp90, or Exp93 source code.
- Forcing Exp449 to match historical counts or coordinates.
- Using historical prediction artifacts as runtime scientific inputs.
- Tuning thresholds, seeds, routes, or policies after inspecting TEST labels.
- Reusing any Exp448, paper-r4, failed-smoke, or failed-official scientific checkpoint.
- Running official scoring, creating run roots, or opening labels during design or implementation.

## Frozen scientific definition

The implementation plan must inventory and hash every scientific module that contributes to:

- Task4 TRAIN-only preprocessing and alignment;
- Task5's frozen label-free transductive value-length rule, which derives its route alignment length from TRAIN and TEST values without reading TEST labels;
- Task4 hard-source scores, conformal p-values, eligibility, and native-pass decisions;
- Task5 route replay and Automatic Hard construction;
- Task6 low-frequency scoring;
- Task7 stability classification;
- Candidate C (`hard_plus_stable`) construction;
- proposal shard serialization and immutable freeze.

The Task4 TRAIN-only path and the Task5 TRAIN+TEST-value-transductive path are distinct frozen scientific behaviors. Exp449 must hash, test, and disclose both; it must not describe the Task5 path as TRAIN-only.

The following are immutable inside Exp449:

- all feature definitions and transforms;
- all score definitions and conformal rules;
- all thresholds and alpha values;
- all route and promotion rules;
- all random seeds;
- dataset catalog order and raw TRAIN/TEST row identities;
- dependency versions and interpreter identity;
- backend selection decision;
- Candidate C definition.

Any scientific change requires a new experiment version. Operational repairs that provably do not alter scientific bytes may remain in Exp449 only after focused regression evidence and independent review.

## Version and lineage boundary

Exp448 remains the immutable record of the historical-parity recovery attempt and its diagnosis. Exp449 receives a new versioned codemap triplet and a new research-state section before implementation changes begin.

The Exp449 closure must bind:

- Git commit and clean-worktree status;
- every scientific source SHA-256;
- lineage and codemap lock hashes;
- database path, SHA-256, size, and row-identity catalog hash;
- exact Python executable and runtime identity;
- package inventory and normalized package-inventory hash;
- CPU count, worker count, backend facts, and acceleration assessment;
- seeds, thresholds, alpha values, and schema/verifier versions.

## Execution roots

Implementation must preregister new, absent, non-symlink roots. The final plan will verify exact names immediately before creation. The intended namespaces are:

- smoke A scientific root: `paper_proposal_exp449_smoke_a_antigravity_20260814`;
- smoke A operations root: `paper_proposal_exp449_smoke_a_antigravity_20260814_ops_antigravity`;
- smoke B scientific root: `paper_proposal_exp449_smoke_b_antigravity_20260814`;
- smoke B operations root: `paper_proposal_exp449_smoke_b_antigravity_20260814_ops_antigravity`;
- official scientific root: `paper_proposal_exp449_official_antigravity_20260814`;
- official operations root: `paper_proposal_exp449_official_antigravity_20260814_ops_antigravity`.

No prior root or record may be copied, migrated, hard-linked, or treated as an Exp449 success. Smoke A and smoke B are also isolated from one another: neither run may read or reuse the other's checkpoint, shard, manifest, journal, or operations state.

## Smoke design

The smoke inventory must be frozen before execution and must cover, at minimum:

- a normal small dataset;
- a dataset with TRAIN rows greater than 1,000;
- all known nonfinite-input shapes, including TRAIN-only and TEST-only cases;
- the canonical pycatch22 path;
- Exp84/Hydra dependency availability;
- the large-length imaging branch;
- route tie behavior and count-cap behavior;
- at least one dataset producing a nonempty Candidate C.

The same smoke is executed twice in independent clean smoke namespaces with identical code, environment, data, seed, and backend.

Smoke passes only when:

- every preregistered dataset succeeds in both executions;
- every selected anomaly coordinate is identical between executions;
- identical-CPU continuous scores have maximum absolute difference at most `1e-12`;
- all scores are finite;
- dataset and row identities match the catalog exactly;
- no TEST label query occurs;
- create-only artifacts and manifests verify independently.

Historical coordinate parity is not a smoke criterion.

## Full execution and failure recovery

The full run uses the frozen 1,117-dataset catalog and a clean official root.

CPU-parallel stages use detected logical CPU count minus one worker, with a minimum of one. On the currently observed 12-logical-CPU host, the expected worker count is 11; the launcher must detect and record the actual count at launch.

Task4 and Task5 are dataset-local. Their successful results are published by the parent process only after typed round-trip validation, semantic validation, and content hashing. Each success checkpoint is bound to the exact Exp449 context and raw dataset identity.

Failure policy:

- a dataset failure is written to an append-only attempt ledger;
- remaining datasets continue processing;
- the scheduler does not cancel healthy pending work because another dataset failed;
- the initial wave ends with a deterministic queue of failed plus never-attempted datasets;
- retries process only that queue;
- the parent publishes each success atomically and create-only only after complete validation;
- a verified success is never silently overwritten;
- every resume reopens and fully revalidates every on-disk success against the current context, catalog identity, payload schema, semantics, and content hash before it can be counted;
- the verified success keys must be an exact bijection with the frozen 1,117-dataset catalog: no missing dataset, duplicate dataset, unknown extra dataset, or duplicate file is accepted;
- attempt and failure records never count as successes;
- code, configuration, database, environment, seed, backend, or closure drift invalidates all checkpoints and requires a new clean root;
- Task6, Task7, Candidate C, shard freeze, and journal stage completion remain blocked until exactly 1,117 verified dataset-local successes exist.

The final population mapping is rebuilt in catalog order, independent of risk-first scheduling or retry order.

After the preregistered retry wave, the run has exactly two terminal states. It is `dataset_local_complete` only when the exact catalog bijection above is satisfied. Otherwise it is `proposal_score_incomplete`, records the remaining failed and never-attempted queue, and cannot enter any population-global stage.

## Acceleration assessment

Exp449 must explicitly assess PyTorch, MLX, and MPS before official scoring.

The assessment records:

- whether each exact stage is tensor-eligible;
- backend and device availability;
- package and driver versions;
- compilation, transfer, and synchronization overhead;
- deterministic CPU-versus-accelerated parity;
- candidate-coordinate parity at every preregistered alpha;
- continuous-score maximum absolute difference;
- end-to-end timing including transfer and synchronization;
- canonical CPU fallback and fallback reason.

An accelerated path may become canonical only when it preserves the frozen algorithm, produces finite outputs, has identical selected coordinates, has score error at most `1e-5`, and is at least 1.1 times faster end to end. Otherwise CPU remains canonical. Only one MLX/MPS-heavy process may run at a time, and it is not combined with outer multiprocessing without a separate deterministic speed benchmark.

## Proposal freeze

After all 1,117 dataset-local successes are verified:

1. rebuild Task4 and Task5 mappings in exact catalog order;
2. execute Task6 once over the complete population;
3. execute Task7 once over the verified Task6 result;
4. build all frozen policies, including Candidate C, without historical-coordinate gates;
5. write typed create-only proposal shards;
6. write an immutable manifest binding every shard path, size, SHA-256, schema, context, and policy coordinate digest;
7. run an independent label-free verifier over the entire frozen root.

No result count is preregistered as a success target. The newly observed Exp449 counts become reportable outputs only after freeze verification.

## Reproducibility contract

Within the exact Exp449 environment:

- selected coordinates must be identical;
- route memberships and policy masks must be identical;
- continuous scores must differ by at most `1e-12` on repeated CPU smoke;
- serialized canonical artifacts must be byte-identical when timestamps and explicitly non-scientific runtime fields are excluded by schema;
- all scientific hashes and identities must be equal.

For an approved accelerated backend, CPU-versus-accelerated coordinates must be identical and score difference must be at most `1e-5`.

If repeated Exp449 executions violate coordinate identity, Exp449 is not reproducible and official evaluation is blocked. A similarity of final AUROC/AP does not override this gate.

## Label firewall and evaluation

TEST labels remain inaccessible through preparation, acceleration assessment, smoke, full Task4 through Task7 scoring, proposal freeze, and independent proposal verification.

Only after the immutable proposal and every preregistered public-baseline score manifest pass verification may the evaluation command load labels. The existing Exp448-coupled evaluation authority and its historical worker settings are not valid Exp449 authorities; Exp449 requires a separately versioned adapter and score/config manifests with no Exp448 prediction or label-capability dependency.

The Exp449 proposal's preregistered continuous evaluation field is the serialized float64 `PolicyRoute.scores` array for policy `hard_plus_stable` (Candidate C). It is consumed directly from each verified proposal shard without recomputation. Higher values mean more anomalous. Candidate alerts are exactly `score >= 0.5`; a value exactly equal to `0.5` is an alert. Missing or abstained evidence uses the already frozen `route_preserving_scores` behavior and receives no evaluation-time substitution. Equal scores remain ties and are passed unchanged to the metric implementation.

The binary positive class is the exact database TEXT label `'1'`; TEXT `'0'` is normal. Any other label representation is rejected. Dataset-level AP and AUROC are computed only for datasets containing both classes. A single-class dataset receives explicit `null` AP/AUROC and is excluded from the corresponding macro denominator; the excluded dataset identities and counts are reported. The primary continuous estimands are:

- `macro_average_precision`: the arithmetic mean of valid dataset-level AP values;
- `macro_auroc`: the arithmetic mean of valid dataset-level AUROC values.

Pooled instance-level AP/AUROC are not confirmatory Exp449 estimands because proposal and public-baseline raw scores do not share a common calibrated scale. Micro TP/FP/FN/TN and their derived threshold metrics may still be reported from each frozen method's preregistered alerts.

Before any label access, every public baseline must have a label-free immutable manifest binding its exact method identifier, implementation/source hashes, preprocessing, seed, continuous score direction, alert calibration, abstention behavior, dataset/row identities, environment, and worker policy. The preregistered set is the four existing classical methods plus Deep SVDD and DROCC unless a later design revision is approved before scoring. Classical CPU-parallel scoring uses detected logical CPU count minus one worker (11 on the currently observed host). A deep method may remain a deliberately non-parallel one-process scientific stage only when its Exp449 manifest states that fact and records why it is not a CPU-parallel stage; an old seven-worker Exp448 baseline manifest is never reused.

The evaluation must:

- bind the exact verified proposal-manifest SHA-256;
- bind every verified public-baseline score/config manifest SHA-256 before importing the label capability;
- evaluate Exp449 and preregistered public anomaly-detection baselines with one shared metric implementation;
- compute dataset-level AUROC and Average Precision from each method's frozen higher-is-more-anomalous continuous score;
- report the two primary dataset-macro estimands and explicit undefined-metric identities;
- preserve dataset-level results for paired statistical analysis;
- avoid any threshold, route, or policy modification after labels are opened.

Evaluation failure does not authorize mutation or retry of the frozen proposal. The failure remains immutable evidence; a scientific correction requires a new version.

## Verification gates

Implementation is not authorized to launch smoke until all of the following pass:

1. Exp449 codemap, lineage, closure, schema, and root-safety validation.
2. Dependency preflight, including exact pycatch22 and Exp84/Hydra requirements.
3. Dataset catalog and row-identity verification for all 1,117 datasets.
4. Typed checkpoint red/green tests, including interruption and tamper cases.
5. Failure batching and retry-queue tests.
6. Proof that Task6 and Task7 cannot run with fewer than 1,117 verified successes.
7. Label-firewall tests.
8. PyTorch/MLX/MPS assessment and canonical-backend decision.
9. Independent read-only review focused on result correctness and false-pass prevention.

Smoke is not authorized to promote to full execution until both independent smoke runs satisfy the reproducibility contract.

## Logging and status

Progress reporting is operational and must not alter scientific results. It records timestamped events for:

- total, completed, failed, pending, and retry counts;
- current dataset and stage;
- elapsed time and estimated remaining time;
- worker count and backend;
- latest verified checkpoint;
- terminal failure reason.

Status and verification commands are read-only. A logging failure must not convert an unverified scientific result into a success.

## Acceptance criteria

Exp449 implementation is ready for smoke only when:

- the current algorithm is fully hash-bound and no historical oracle is a scientific input or equality gate;
- new roots are absent and root-safety checks pass;
- all focused and relevant regression tests pass;
- failure recovery cannot publish partial global results;
- acceleration is assessed under the project policy;
- independent review returns no result-integrity blocker, high, or medium finding.

Exp449 is ready for label evaluation only when:

- both smoke executions pass coordinate and score reproducibility;
- all 1,117 dataset-local results are verified successes;
- Task6, Task7, and Candidate C complete once over the complete population;
- the proposal root and manifest pass independent verification;
- all six preregistered public-baseline score/config manifests are frozen and independently verified, including typed abstentions where scientifically required;
- label-access count remains zero through proposal verification.

## Implementation boundary

This design authorizes a future implementation plan, not code changes or experiment execution. The plan must separate:

1. Exp449 version/codemap/lineage preregistration;
2. removal of historical-oracle equality gates from Exp449 only;
3. deterministic smoke and its two-run verifier;
4. full-run failure batching and context-bound checkpointing;
5. population-global freeze and independent verification;
6. launch readiness review;
7. smoke execution;
8. official full execution;
9. post-freeze label evaluation and paper metrics.

Each execution authorization is a separate gate. Implementation completion alone does not authorize smoke, full scoring, or label evaluation.
