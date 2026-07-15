# Exp161 Codex Verification Report

## Verdict

Exp161 full read-only provenance/coverage audit completed. The frozen Exp160 target population was reproduced exactly, but none of the three historical alternative-source artifacts is eligible for policy replay. The Exp162 policy experiment is therefore blocked until alternative scores are recomputed with one frozen, family-independent, TEST-metric-independent configuration over the required coverage.

This is not a model performance experiment. No TP, FP, F1, TEST label, or anomaly-position field was used to select a source or route a candidate.

## Frozen target integrity

- Frozen Exp160 prediction SHA256: `ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d`
- Target rule: `n_train >= 5 and candidate_count == 0`
- Dataset/alpha target rows: `1,412`
- Unique target datasets: `537`
- Duplicate dataset/alpha keys: `0`
- Independent target-set comparison: exact match, missing `0`, extra `0`

Target rows by alpha:

| Alpha | Target rows |
| ---: | ---: |
| 0.005 | 537 |
| 0.01 | 425 |
| 0.02 | 305 |
| 0.05 | 145 |

## Historical source coverage

Counts below are coverage evidence only. They are not evidence that the source can be used in a new policy.

| Source | Computed dataset/alpha rows | Computed unique datasets | Policy-eligible rows |
| --- | ---: | ---: | ---: |
| 미분·스펙트럼 통계 점수 | 487 | 178 | 0 |
| 부분 형태 일치 점수 | 439 | 161 | 0 |
| 구간 분포 점수 | 14 | 7 | 0 |

- Pairwise common computed support: `0` rows for every source pair.
- Three-source common computed support: `0` rows.
- Full-coverage policy-eligible sources: `0`.
- Legacy selected-index overlap rows are retrospective diagnostics only and were not used for eligibility or routing.

## Why each historical artifact is ineligible

- Exp99 미분·스펙트럼 통계 점수: actual computation was restricted using family/research target provenance. Non-target Exp93 passthrough rows were excluded.
- Exp101 부분 형태 일치 점수: actual computation was restricted using shapelet/research target provenance. Non-target Exp93 passthrough rows were excluded.
- Exp46 구간 분포 점수: the historical run covered 13 datasets selected by family/difficulty diagnostics. Its partial results are research evidence only.

## Codex review fixes

1. Marked the historical Exp46 artifact policy-ineligible instead of partially eligible.
2. Added a distinct `--output-dir` so smoke and full outputs cannot be mixed.
3. Added direct-script import bootstrapping and a CLI regression test.
4. Resolved relative output directories against the fixed repository root.
5. Tightened Exp99/101 computed detection to explicit target flags; selector prose is no longer accepted as computation evidence.
6. Added unique-dataset counts and an input-file SHA256 manifest.
7. Corrected Smoke/Full labels in generated reports.

## Verification commands

```bash
python3 -m py_compile experiments/exp161/*.py
python3 -m unittest \
  experiments/exp161/test_exp161_alternative_source_audit.py \
  experiments/exp161/test_exp161_structure.py
python3 experiments/exp161/run_experiment_161_alternative_source_audit.py \
  --dataset-limit 10 \
  --output-dir outputs/exp161_alternative_source_audit_smoke
python3 experiments/exp161/run_experiment_161_alternative_source_audit.py \
  --output-dir outputs/exp161_alternative_source_audit
```

- Unit/structure tests: `13 passed`.
- Smoke audit: completed with blocker.
- Full audit: completed with blocker.
- Output CSV forbidden performance fields: none.
- `git diff --check` for Exp161 code/docs: passed.

## Next gate

Do not route candidates using these legacy result files. The next experiment must first recompute at least two alternative score sources using only TRAIN-normal fitting/calibration, one preregistered configuration, and complete required coverage. Any later TEST evaluation remains retrospective and must not select the source, threshold, or agreement rule.
