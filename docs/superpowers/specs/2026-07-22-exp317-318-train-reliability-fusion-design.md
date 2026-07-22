# Exp317-318 TRAIN-normal reliability fusion design

## 1. Purpose

Exp316 counted independent method groups but treated every source vote as equally trustworthy. Its best completed Shadow variant reached only 20% precision, so it is not eligible for autonomous alert expansion.

Exp317-318 test whether a source that is stable on TRAIN-normal pseudo-tests should carry more evidence than a source that frequently raises normal false alerts. The experiments remain retrospective research Shadow evaluations. Exp291 autonomous Hard alerts remain unchanged.

## 2. Plain-language meaning

Exp316 asked, "How many detectors voted anomaly?"

Exp317-318 ask, "How many independent detectors voted anomaly, and how reliable were those detectors when they saw only normal TRAIN runs?"

Two weak votes must not automatically outrank two stable votes. A detector receives less influence when its TRAIN-normal audit has a high or uncertain false-alert rate.

## 3. Fixed inputs

- Evaluation population: the frozen Exp291 1,117-dataset population.
- Source predictions: frozen Exp309, Exp310, Exp311, Exp313, Exp314, and Exp315 prediction JSONL files and manifests.
- Source names and method groups:
  - multi-scale patch discord: local shape
  - multi-window change point: transition
  - wavelet phase coherence: frequency
  - predictive residual: prediction
  - contrastive normal memory: representation
  - Deep SVDD normal boundary: representation
- Alpha grid: `(0.005, 0.01, 0.02, 0.05)`.
- Seed: `20260722`.
- Existing guards: remove Exp291 autonomous Hard indices and Exp290 TRAIN raw exact-normal indices from new Shadow candidates.

Input SHA256 values are recorded before prediction. A source coverage mismatch, manifest mismatch, missing dataset, or prediction hash mismatch stops the experiment.

## 4. Exp317 TRAIN-normal reliability audit

Exp317 never reads TEST labels or anomaly positions.

For each dataset, source, and alpha:

1. Read only the source's stored TRAIN-normal out-of-fold or held-out calibration scores.
2. Require at least 20 scores. Otherwise mark the source `insufficient_train_reference` and abstain.
3. Create five deterministic 50/50 reference-audit splits using seeds `20260722 + split_index`.
4. For every audit score, calculate an empirical upper-tail p-value from the disjoint reference scores:

   `p = (1 + count(reference_score >= audit_score)) / (1 + n_reference)`

5. Count audit false alerts where `p <= alpha`.
6. Calculate the one-sided 95% Wilson upper bound with `z = 1.6448536269514722`.
7. Use the maximum Wilson upper bound across the five splits as the conservative normal-risk estimate `U`.
8. Let `n_audit_min` be the smallest audit split size across the five splits and define source reliability weight:

   `weight = max(0, -log10(max(U, 1 / (n_audit_min + 1))))`

The formula is fixed before TEST evaluation. Small or uncertain TRAIN references naturally receive small weights. Exp317 reports weights and abstentions but produces no TEST candidates and no TP, FP, precision, recall, or F1.

### Exp317 outputs

- Dataset/source/alpha reliability audit CSV
- Split-level TRAIN-normal false-alert audit CSV
- Source coverage and abstention summary
- Frozen reliability JSONL and SHA256 manifest
- Invariance and contract report

## 5. Exp318 frozen reliability-weighted fusion

Exp318 prediction reads the frozen Exp317 reliability artifact and the frozen source prediction artifacts. It does not read TEST labels.

For each dataset and alpha:

1. A source is available only when Exp317 did not abstain and its reliability weight is greater than zero.
2. Convert the six available source weights into five independent method-group weights.
3. For the representation group, use the maximum available reliability weight from contrastive memory and Deep SVDD. This prevents two correlated representation sources from being counted twice.
4. A group supports a TEST candidate when at least one available source in the group marked that candidate at the same alpha. Its supporting weight is the maximum weight among the supporting sources in that group, not the maximum weight of a non-supporting source.
5. Calculate:

   `weighted_support_share = supporting_group_weight_sum / available_group_weight_sum`

6. If the sum of available group weights is zero, abstain.
7. Require at least three available method groups and at least two supporting method groups.
8. Produce all three preregistered sensitivity variants:
   - `share_0.50`
   - `share_0.67`
   - `share_0.75`
9. Remove Exp291 Hard and Exp290 exact-normal indices.
10. Freeze prediction JSONL and its SHA256 before loading TEST labels.

No share threshold or alpha is selected as primary from TEST performance.

## 6. Evaluation

After prediction hash verification, TEST labels are loaded only by the evaluation phase.

Each alpha and share threshold reports:

- candidate count
- TP and FP
- micro precision
- TP outside Exp291 Hard
- TP outside all existing Exp291 lanes
- zero-Hard-F1 datasets recovered
- candidate workload mean, median, p90, p95, and maximum
- datasets with insufficient available method groups
- candidate exact overlap with Exp316

An 80% precision line is descriptive only. Reaching it does not automatically promote a variant.

## 7. Interpretation rules

- Exp317 is a TRAIN-normal reliability audit, not a performance experiment.
- Exp318 is a retrospective counterfactual Shadow evaluation.
- `human_assisted_only = true` and `automatic_policy_changed = false` for every Exp318 variant.
- Exp291 remains the autonomous Hard-alert policy.
- TEST results cannot modify the alpha grid, Wilson confidence level, split count, minimum reference size, method groups, share thresholds, or guards.
- Family name, dataset name, TEST length, TEST labels, anomaly positions, and past family TEST performance are prohibited prediction inputs.
- Tail replacement is prohibited.
- Priority review is not promoted to autonomous alert.

## 8. Failure and abstention behavior

- Missing or invalid source manifest: stop with nonzero exit.
- Source coverage not exactly 1,117 datasets: stop with nonzero exit.
- Reliability artifact hash mismatch: stop before prediction.
- Fewer than 20 TRAIN-normal scores: source-level abstention.
- Fewer than three available independent groups: dataset/alpha-level abstention.
- Empty candidate set: valid result, not an error.
- Any TEST evaluation field in prediction JSONL: contract failure.

## 9. Tests

- Deterministic split reproducibility
- Reference and audit split disjointness
- Empirical p-value formula
- One-sided Wilson upper-bound calculation
- Reliability decreases when TRAIN-normal false alerts increase
- Representation sources count as one method group
- Minimum available and supporting group rules
- Hard and exact-normal guards
- Prediction contains no labels or evaluation metrics
- Prediction hash tamper detection
- 1,117-dataset source coverage contract
- Independent result reaggregation

## 10. Execution

1. Unit and structure tests
2. Ten-dataset Exp317 smoke audit
3. Ten-dataset Exp318 prediction/evaluation smoke
4. Full Exp317 audit on 1,117 datasets
5. Full Exp318 frozen prediction
6. Prediction SHA256 verification
7. Exp318 TEST-label evaluation
8. Independent reaggregation
9. Dashboard result registration

Only one experiment runs at a time. These experiments reuse frozen score artifacts and do not retrain the neural sources, so MPS is not expected to improve runtime.

## 11. Completion criteria

- Exp317 has 1,117-dataset coverage with explicit source abstentions.
- Exp318 has 1,117 dataset rows for every preregistered alpha/share variant.
- Prediction and evaluation are separated by a frozen hash.
- All source, config, code, and result hashes are recorded.
- Independent reaggregation matches generated summaries.
- Results are reported even when precision or coverage is worse than Exp316.
- No operational policy is selected from the retrospective TEST results.
