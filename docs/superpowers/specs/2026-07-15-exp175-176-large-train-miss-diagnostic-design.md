# Exp175/176 Large-TRAIN Miss Diagnostic Design

## 1. Purpose

Exp137 misses 2,227 anomaly instances in its autonomous Hard-alert lane. Of
those misses, 2,107 occur in the 639 datasets with at least 20 TRAIN-normal
instances. Exp175/176 determines whether those misses are mainly caused by a
candidate/routing policy that is too restrictive or by score representations
that do not rank the anomalies highly enough.

This is a retrospective diagnostic. It does not create a new autonomous alert
policy and does not select a threshold, score source, or budget from TEST
performance.

## 2. Inputs

- Exp137 frozen lane results:
  `results/exp137/experiment_137_operational_triage_results.csv`
- Full score cache manifest:
  `outputs/exp153_157_train_only_improvements/01_score_cache_manifest.csv`
- Read-only score cache files from the primary workspace:
  `/Users/minho/Documents/timesries project/outputs/exp153_157_train_only_improvements/cache/`
- Labels and dataset metadata from:
  `/Users/minho/Documents/Dataset/univariate_ts.db`

The score cache contains four frozen score vectors: temporal random-kernel,
spectrogram-image, texture-image, and interval-statistics scores. TRAIN OOF
scores and TEST scores are loaded before TEST labels are read.

## 3. Exp175 Dataset Diagnostic

Exp175 covers exactly the Exp137 datasets with `train_normal_count >= 20`.
For each dataset it records:

- TEST size and true anomaly multiplicity;
- Exp137 Hard, Standard-review, and Priority-review counts;
- Hard TP, FP, and FN;
- missed-anomaly share;
- score-rank and TRAIN-reference visibility summaries;
- concentration groups by anomaly multiplicity, TEST size, and miss count.

No policy is modified. Exp175 only explains where the 2,107 Hard misses are
concentrated.

## 4. Exp176 Cause Separation

For every true anomaly missed by the Exp137 Hard lane, Exp176 computes for each
frozen score source:

1. descending TEST rank and rank percentile;
2. finite-sample upper-tail p-value against the source's TRAIN OOF scores;
3. support at the pre-registered diagnostic grid
   `alpha = 0.005, 0.01, 0.02, 0.05`;
4. the best single-source rank and four-source mean-rank position.

The diagnostic categories are:

- `already_in_standard_review`: missed by autonomous Hard alert but already in
  Exp137 Standard review;
- `already_in_priority_review`: already in retrospective Priority review;
- `score_visible_policy_miss`: not in an Exp137 review lane, but at least two
  frozen sources support it at alpha 0.05;
- `single_source_only`: only one source supports it at alpha 0.05;
- `representation_blind`: no frozen source supports it at alpha 0.05.

Alpha 0.05 is the most permissive member of the already frozen diagnostic grid.
It is not selected as an operating threshold. Results for every alpha are
reported so the conclusion does not depend on a hidden best-alpha choice.

## 5. Budget-versus-Representation Counterfactual

Exp176 reports fixed rank-depth sensitivity for `K = 1, 2, 3, 5, 8, 10`.
For each source and the mean-rank ensemble it counts how many missed anomalies
would be visible within the first K positions. No K is called best, optimal, or
recommended.

Interpretation:

- many anomalies visible at shallow K indicates candidate budget or routing is
  a plausible bottleneck;
- anomalies visible only at deep K indicate a workload problem;
- anomalies unsupported even at alpha 0.05 indicate a representation/score
  bottleneck that cannot be repaired by merely increasing the candidate cap.

## 6. Safety and Integrity

- Scores and ranks are frozen before labels are loaded.
- TEST labels are used only for retrospective row selection and evaluation.
- Dataset and family names are not used for prediction or policy branching.
- Exp137 outputs are never overwritten.
- Exp174 is not merged into the pipeline; its overlap audit showed no new TP
  relative to the complete Exp137 lanes.
- Autonomous Hard metrics and human-assisted review metrics stay separate.
- Tail replacement is not used.

## 7. Outputs

All outputs are written to
`outputs/exp175_176_large_train_miss_diagnostic/`:

- dataset-level Exp175 diagnostic CSV;
- anomaly-level Exp176 diagnostic CSV;
- alpha and rank-depth summaries;
- cause summary and concentration report;
- independent reaggregation JSON;
- execution log, contract, and manifest.

## 8. Completion Criteria

- exactly 639 eligible datasets;
- exactly 3,440 true anomalies in the eligible group;
- exactly 2,107 Exp137 Hard misses accounted for;
- score cache coverage 639/639 with no length mismatch;
- each missed anomaly assigned to exactly one cause category;
- independent reaggregation matches every published total;
- no existing experiment output is changed.
