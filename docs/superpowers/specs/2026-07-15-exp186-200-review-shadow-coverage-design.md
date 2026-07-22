# Exp186-200 Review, Shadow Promotion, and Coverage Design

## Goal

Run fifteen sequential retrospective experiments that extend the Exp177-185
findings without changing Exp137 autonomous Hard alerts. Five experiments
improve review quality, five measure shadow promotion safety, and five measure
the coverage/workload frontier.

## Roles

- Gemini produces the first implementation from this frozen design.
- Codex reviews the implementation, repairs contract violations before TEST
  evaluation, runs tests, registers the queue, and starts the sequential run.
- Neither agent may select a policy from TEST TP, FP, precision, or F1.

## Frozen Scientific Contract

- Exp137 autonomous Hard alerts remain unchanged.
- All new candidates are review-only unless explicitly labelled
  `shadow_hard_diagnostic`; shadow candidates are not operational alerts.
- Prediction reads frozen TRAIN-normal score references and TEST values or
  frozen prediction indices, but never TEST labels.
- Evaluation reads TEST labels only after prediction JSONL and its SHA256 are
  frozen.
- Dataset and family names are lookup keys only and are not policy inputs.
- No TEST-length top-N budget, tail replacement, retrospective family routing,
  or Priority-review promotion is permitted.
- No alpha, feature source, agreement rule, stability rule, or workload guard
  is selected from TEST performance.
- Every preregistered experiment is reported, including negative results.
- Results are retrospective and are not prospective or deployment validation.
- The inherited four-source configuration still has unresolved end-to-end
  provenance and must not be described as strict TRAIN-only.

## Inputs

- Exp137 frozen lane CSV.
- Exp177 frozen TRAIN-OOF policy.
- Exp178 frozen conservative and expanded review predictions.
- Exp180 canonical score predictions and score cache.
- Exp181 existing/canonical fusion predictions.
- Exp183 wavelet prediction artifacts for negative-control comparisons only.
- Exp185 independently reaggregated summary.
- The frozen Exp153-157 four-source TRAIN OOF and TEST score cache.
- `/Users/minho/Documents/Dataset/univariate_ts.db`, with labels unavailable to
  prediction code.

The primary evaluation universe is the same 639 datasets with
`train_normal_count >= 20`. Missing source coverage is a stage failure; a
dataset may not be silently dropped or represented by an empty candidate set.

## Group A: Review Quality, Exp186-190

### Exp186: Conservative plus Canonical KNN agreement

Use the intersection of Exp178 conservative candidates and canonical KNN
alpha 0.005 candidates. Output is a high-confidence review lane.

### Exp187: Conservative plus dual canonical backend agreement

Require Exp178 conservative support and both canonical KNN local-gap and
canonical robust-diagonal support at alpha 0.005. Output remains review-only.

### Exp188: TRAIN-OOF stability guarded review

Apply a preregistered five-fold TRAIN-OOF stability guard to Exp178
conservative candidates. A dataset passes only when the two-source agreement
rule satisfies the frozen TRAIN risk contract in at least four of five folds.
Fold construction and tie-breaking are seed-fixed before prediction.

### Exp189: TRAIN-normal workload guarded review

Derive a dataset-level review count guard only from cross-fitted TRAIN-normal
candidate counts. Use the one-sided 99% conformal count quantile with the
finite-sample correction already used by this project. Report raw and
minimum-one counts as sensitivity columns, but keep the raw rule as the frozen
primary row. TEST length and TEST candidate count are prohibited inputs.

### Exp190: Three-tier review routing

Create mutually exclusive tiers:

1. high-confidence review: Exp178 conservative and canonical KNN agree;
2. standard review: Exp178 conservative only;
3. exploratory review: canonical KNN only.

The tiers are workload labels, not autonomous decisions.

## Group B: Shadow Promotion Safety, Exp191-195

All five experiments produce `shadow_hard_diagnostic` rows in a namespace
separate from Exp137. They never replace or union into Exp137 Hard indices.

### Exp191: Conservative-canonical shadow agreement

Shadow-promote only the Exp186 intersection.

### Exp192: Conservative-dual-backend shadow agreement

Shadow-promote only the Exp187 intersection.

### Exp193: Three-existing-source plus canonical confirmation

Require support from at least three of the inherited four score sources at the
Exp177 conservative alpha and canonical KNN confirmation at alpha 0.005.

### Exp194: Fold-stable shadow agreement

Require the Exp191 candidate rule plus the Exp188 four-of-five TRAIN-OOF
dataset stability condition.

### Exp195: Zero-normal-candidate shadow guard

Require the Exp191 candidate rule and zero cross-fitted TRAIN-normal candidates
under the same agreement rule for that dataset. This is a strict shadow safety
probe, not an operating rule.

## Group C: Coverage and Workload, Exp196-200

### Exp196: Expanded TRAIN-risk review

Use the frozen Exp178 risk-target 0.02 candidates as exploratory review.

### Exp197: Conservative plus canonical-unique union

Union Exp178 conservative candidates with canonical KNN alpha 0.005 candidates,
preserving an evidence-origin field for each candidate.

### Exp198: Existing-one plus canonical alpha 0.01

Require at least one inherited source and canonical KNN agreement at alpha
0.01. This is review-only and does not replace Exp181 historical artifacts.

### Exp199: Expanded two-of-six with TRAIN workload guard

Use the frozen six-source two-agreement candidate rule at alpha 0.01, then
apply the Exp189 TRAIN-normal workload guard. No TEST count may affect the cap.

### Exp200: Frozen frontier reaggregation

Independently reaggregate Exp186-199 into conservative, balanced, and expanded
workload views. Exp200 does not name a best or recommended policy. It reports
the full workload/TP/FP frontier and exact overlaps.

## Prediction and Evaluation Boundary

Prediction payloads contain:

- experiment ID and frozen configuration name;
- dataset lookup key;
- candidate indices and evidence flags;
- review or shadow-diagnostic lane;
- TRAIN-only guard metadata;
- `contains_test_labels=false`;
- `contains_evaluation_metrics=false`.

Prediction payloads must not contain labels, TP, FP, FN, precision, recall, F1,
or combined F1. Evaluation verifies the prediction SHA256 before opening labels.

## Metrics

Report each experiment independently:

- dataset coverage and errors;
- candidate count, TP, FP, precision, and recall;
- dataset mean candidate-only F1, clearly labelled as such;
- active dataset count and zero-candidate dataset count;
- workload median, p90, p95, and maximum;
- remaining anomaly recovery relative to the frozen Exp137 lanes;
- exact candidate overlap and unique contribution;
- full-1117 human-assisted counterfactual only as a separate diagnostic;
- shadow metrics separated from autonomous Hard metrics.

No metric is used to alter a frozen rule after evaluation.

## Runtime and Queue

- Register Exp186 through Exp200 as fifteen individual queue items.
- Run one stage at a time with seven CPU workers where parallel work is valid.
- Use MPS only if a deterministic smoke benchmark demonstrates at least 1.1x
  speedup for the exact stage; otherwise record CPU.
- Each stage writes a progress log, frozen prediction and SHA256, result CSV,
  summary CSV, completion marker, and manifest.
- Retry missing datasets once. Any remaining omission fails that stage and is
  visible on the dashboard.
- The dashboard must show all fifteen queued stages and the active stage.
- The batch runner exits after the queue is empty. No agent monitor remains.

## Required Tests

- Prediction functions do not accept `y_test`.
- Prediction artifacts contain no evaluation fields.
- Label and anomaly-position mutation leaves predictions unchanged.
- Dataset and family rename leaves policy results unchanged after key remapping.
- TEST length is not passed to a workload-guard function.
- TRAIN fold assignment and guard outputs are deterministic.
- Candidate indices are in range and lane sets are mutually exclusive.
- Source coverage is exactly 639/639 with zero silent exclusions.
- Frozen input and prediction hashes are verified.
- Exp137 Hard indices are byte-for-byte unchanged.
- Queue order is exactly Exp186 through Exp200.

## Interpretation

Exp186-190 can identify a review-lane candidate for later prospective testing.
Exp191-195 can only estimate whether strong agreement deserves future shadow
validation; they cannot authorize automatic promotion. Exp196-200 describe the
cost of wider detection coverage. Any deployment choice requires a new locked
benchmark or prospective shadow data.
