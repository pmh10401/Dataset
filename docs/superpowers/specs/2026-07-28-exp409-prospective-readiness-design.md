# Exp409 Prospective-Readiness Shadow Pipeline Design

Date: 2026-07-28  
Status: approved design, not implemented  
Audience: research and implementation reviewers

## 1. Purpose

Exp409 prepares a fixed, label-blind shadow pipeline that can be applied
without configuration changes when genuinely new time-series data becomes
available.

No new data is currently available. Exp409 therefore evaluates readiness,
not anomaly-detection performance. Its success criteria are configuration
freeze, label isolation, review-budget enforcement, lane immutability,
determinism, and complete component-level provenance.

Exp409 does not alter or reinterpret Exp403-408. In particular, it does not
repair or bypass the unreproducible checkpoint input hashes documented in
`docs/EXP403_408_CHECKPOINT_PROVENANCE_AUDIT_KO.md`.

## 2. Frozen Experiment Contract

The following values are fixed before implementation:

| Field | Frozen value |
|---|---|
| Experiment | `Exp409` |
| Mode | prospective-readiness shadow |
| Consensus variant | `two_new_plus_existing_independent` |
| Alpha | `0.005` |
| Global review budget | 100 candidates per publication batch |
| Candidate disposition | review-only Shadow |
| Automatic Hard action | unchanged |
| Standard/Priority action | unchanged |
| TEST labels during prediction | forbidden |
| Performance evaluation | not evaluated |
| Deployment eligibility | false |

The fixed variant requires agreement from two new evidence kinds and an
independent existing-evidence component. Alpha `0.005` is the most
conservative member of the already frozen Exp408 alpha grid. These choices
define a governance contract; Exp409 does not use Exp408 TEST outcomes as an
optimization objective.

The budget is a publication gate, not a ranking instruction. If a batch
contains more than 100 candidates, Exp409 must not select a TEST-informed or
score-informed top 100. It fails closed and publishes no candidate file for
that batch.

## 3. Claim Boundary

Exp409 may claim only that:

- the label-blind generation contract is executable;
- the frozen lanes remain byte-for-byte or value-for-value unchanged;
- the configured candidate package is deterministic for identical inputs;
- the publication budget and fail-closed checks work;
- component-level provenance is complete and verifiable.

Exp409 must not claim:

- prospective anomaly-detection performance;
- improved precision, recall, F1, or false-alarm rate;
- production or deployment validation;
- automatic Hard, Standard, or Priority promotion;
- safe direct reuse of Exp403-408 checkpoints;
- superiority of the frozen variant or alpha.

All performance fields are recorded as `not_evaluated`, and deployment
eligibility is recorded as `false`.

## 4. Inputs and Data Flow

Prediction may consume:

- normal TRAIN values;
- unlabeled TEST values;
- verified Exp291/292 frozen lane artifacts;
- current source artifacts needed to recompute the Exp408 TRAIN-only evidence;
- the frozen Exp409 configuration.

Prediction must not consume:

- TEST labels or label tables;
- actual anomaly positions;
- TP, FP, FN, precision, recall, F1, or other TEST-derived metrics;
- historical family-level TEST performance;
- Exp408 candidate evaluation CSV files;
- Exp403-408 data-stage checkpoints.

The execution flow is:

1. Resolve and hash the database, lane artifacts, source artifacts, code, and
   frozen configuration.
2. Verify the Git revision and require a clean worktree for publication.
3. Load normal TRAIN values and unlabeled TEST values under the label-access
   guard.
4. Recompute all source outputs required by the frozen consensus contract.
5. Record exactly thirteen source statuses per dataset and fail closed on
   corrupt or unrecognized states.
6. Collapse evidence by lineage and TRAIN OOF correlation using the frozen
   Exp408 independence logic.
7. Produce only `two_new_plus_existing_independent` candidates at alpha
   `0.005`.
8. Remove frozen Hard indices and verify that Hard, Standard, and Priority
   lanes are unchanged.
9. Enforce the global budget of 100 candidates.
10. Write the candidate package and manifests atomically only after every
    gate passes.

## 5. Component Design

### 5.1 `Exp409FrozenConfig`

An immutable configuration object contains:

- experiment identifier;
- variant;
- alpha;
- review budget;
- seed;
- independence correlation threshold;
- minimum aligned TRAIN OOF rows;
- allowed source identities;
- output schema versions.

The serialized frozen configuration is itself a hashed input. Command-line
options may select input and output locations but may not override frozen
research parameters.

### 5.2 `LabelAccessGuard`

The guard combines:

- forbidden field and key validation;
- SQLite trace validation that rejects TEST label table, column, value, blob,
  population, and TEST-derived ordering access;
- output schema validation that rejects label-derived and evaluation fields;
- dependency validation that rejects Exp408 evaluation CSVs as prediction
  inputs.

Any violation terminates the batch before candidate publication.

### 5.3 `ReadinessRunner`

The runner is an adapter around the verified TRAIN-only source and consensus
functions. It does not duplicate model implementations and does not accept
Exp403-408 checkpoints.

The runner:

- selects smoke datasets using only canonical dataset identity and normal
  TRAIN counts;
- recomputes source results;
- exposes only the frozen variant and alpha;
- preserves source abstentions;
- validates candidate indices, support components, source markers, and lane
  exclusion;
- delegates publication to the budget and provenance gates.

### 5.4 `ProvenanceManifest`

The manifest records individual values rather than only a composite hash:

- Git commit;
- Git dirty status;
- serialized configuration SHA256;
- every execution code file SHA256;
- database snapshot artifact SHA256 values;
- Exp291/292 lane artifact and manifest SHA256 values;
- each source artifact SHA256;
- per-dataset values SHA256;
- per-dataset source-output SHA256 values;
- upstream artifact SHA256 values;
- candidate artifact SHA256;
- lane parity result;
- budget result;
- label-access result;
- deterministic rerun result;
- runtime package versions and platform.

A composite closure hash is also recorded, but it never replaces the
individual entries.

## 6. Fail-Closed Rules

Candidate publication is prohibited when any of the following occurs:

- TEST label or actual anomaly-position access;
- a dirty Git worktree;
- a DB snapshot change between preflight and publication, or an unexpected
  pinned upstream artifact identity;
- missing component-level provenance;
- source identity, dataset identity, or TRAIN-count mismatch;
- corrupt source payload or unrecognized source state;
- candidate index duplication or range violation;
- overlap with frozen Hard;
- any Hard, Standard, or Priority lane change;
- more than 100 batch candidates;
- any dataset execution error;
- non-atomic or unverifiable artifact write.

An aborted run writes only a status/failure record that contains no partial
candidate publication.

## 7. Outputs

The canonical output set is:

- `exp409_frozen_config.json`
- `exp409_shadow_candidates.jsonl`
- `exp409_source_status.jsonl`
- `exp409_provenance_manifest.json`
- `exp409_readiness_summary.json`
- `EXP409_READINESS_REPORT_KO.md`

Candidate rows contain only:

- dataset identifier;
- candidate index;
- frozen variant;
- frozen alpha;
- supporting independent components;
- supporting source markers;
- `review_only` disposition.

Candidate rows must not contain TEST labels, anomaly truth, TP/FP/FN, F1,
precision, recall, evaluation metrics, or any lane-promotion instruction.

The readiness summary explicitly states:

- `performance_evaluation: "not_evaluated"`
- `prospective_validated: false`
- `deployment_eligible: false`
- `operational_policy_changed: false`

## 8. Smoke Validation

No full 1,117-dataset execution is authorized by this design.

The initial smoke uses a small deterministic set selected only from normal
TRAIN dataset names and counts. It covers every TRAIN eligibility tier:

- abstain;
- robust distance;
- classical committee;
- learned methods.

The smoke validates pipeline behavior, not performance. It must not read TEST
labels or produce TP/FP/F1 reports.

## 9. Test Strategy

### Unit tests

- frozen configuration rejects parameter overrides;
- output schemas reject label-derived fields;
- candidate validation rejects duplicates, range errors, and Hard overlap;
- budget 100 passes and budget 101 fails without partial publication;
- dirty Git state blocks publication;
- missing individual provenance entries block publication;
- source identity and TRAIN-count mismatches fail closed;
- unknown source status and corrupt payload fail closed.

### Integration tests

- SQLite trace proves no TEST label access;
- all four TRAIN eligibility tiers appear in the deterministic smoke;
- every dataset has exactly thirteen source statuses;
- Hard, Standard, and Priority parity is exact;
- identical inputs produce byte-identical candidate payloads and an identical
  canonical provenance closure; timestamps and run IDs live in a separate
  runtime envelope and are not part of the determinism comparison;
- a forced budget overflow leaves no candidate artifact;
- a forced source corruption leaves no candidate artifact.

### Regression tests

- existing Exp403-408 tests continue to pass;
- Exp409 cannot load Exp408 evaluation CSVs or checkpoints;
- no automatic lane mutation API is imported or emitted;
- all output files validate against their declared schemas.

## 10. Error and Status Semantics

Run status is one of:

- `ready`: every readiness gate passed and the review-only package was
  published;
- `abstained`: the frozen source contract produced no eligible evidence, with
  explicit reasons and no candidate rows;
- `blocked`: integrity, leakage, provenance, lane, or budget gate failed;
- `error`: an unexpected execution failure occurred.

`blocked` and `error` runs are not partially successful publications.

## 11. Non-Goals

Exp409 does not:

- develop a new model or feature;
- search variants, alpha values, thresholds, or budgets;
- evaluate TEST performance;
- rerun or rewrite Exp403-408;
- bypass checkpoint validation;
- promote Shadow candidates;
- deploy an operational service;
- execute the full historical benchmark.

## 12. Future Evaluation

When genuinely new labeled data becomes available, the frozen Exp409
configuration may be applied without modification. Labels may be opened only
in a separate evaluation step after the prediction artifact and its
provenance manifest are frozen.

If any research parameter must change, the change receives a new experiment
number and a new preregistration. Exp409 itself remains immutable.
