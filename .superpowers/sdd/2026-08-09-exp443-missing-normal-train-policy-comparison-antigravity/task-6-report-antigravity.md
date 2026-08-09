# Exp443 Task 6 — Policy A control gate report

## Scope

Task 6 adds the fail-closed, label-free control gate in
`control_antigravity.py` and its behavior/real-run test.  The gate compares
Policy A (`linear_edge_nearest_control`) with the verified Exp249 cache for
all affected datasets, both kernels, both frozen seeds, and then replays the
complete Standard-gated route before any alternative policy can be admitted.

## TDD evidence

- RED: `python3 -m unittest discover -s experiments/exp443_missing_normal_train_policy_comparison_antigravity/tests -p 'test_control_antigravity.py'` produced `Ran 8 tests ... FAILED (errors=7, skipped=1)` while the control module was absent.
- GREEN focused run: `Ran 8 tests ... OK (skipped=1)`; seven fixture checks pass and the real run is opt-in.
- The gate rejects shape changes, non-finite values, differences greater than `1e-6`, and candidate mismatches even when score values match.  Exactly `1e-6` passes.

## Pinned full-suite verification

Using the frozen Python 3.14.6 environment at
`/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314`,
with all six numerical inner-thread variables set to `1`:

```text
Ran 94 tests in 19.204s
OK (skipped=2)
```

The two skips are the existing opt-in real source/preprocessing fixtures; the
Task 6 real control was run separately with its marker enabled.

## Real 19-dataset control

Command marker: `EXP443_REAL_CONTROL_ANTIGRAVITY=1`.

Datasets were scored in frozen order, with progress emitted for every unit:

```text
DodgerLoopDay_normal_1, DodgerLoopDay_normal_2, DodgerLoopDay_normal_4,
DodgerLoopDay_normal_5, DodgerLoopDay_normal_6, DodgerLoopDay_normal_7,
DodgerLoopGame_normal_1, DodgerLoopGame_normal_2,
DodgerLoopWeekend_normal_1, DodgerLoopWeekend_normal_2,
MelbournePedestrian_normal_1, MelbournePedestrian_normal_2,
MelbournePedestrian_normal_3, MelbournePedestrian_normal_4,
MelbournePedestrian_normal_5, MelbournePedestrian_normal_6,
MelbournePedestrian_normal_7, MelbournePedestrian_normal_8,
MelbournePedestrian_normal_9
```

The real run completed with `1 test`, `OK`, `63.787s`.  A second control-only
evidence run printed the result fields directly:

```text
status=control_reproduction_passed passed=True reason=None
score_evidence_count=76 finite_all=True candidate_equal_all=True
max_abs_difference=4.5397329895280336e-08 candidate_mismatch_count=0
label_access_count_before_close=0
label_access_count_snapshot=0
```

Per-dataset maximum absolute differences (four kernel/seed pairs each) were:

```text
DodgerLoopDay_normal_1  1.1151056446279117e-08
DodgerLoopDay_normal_2  1.0162466257401803e-08
DodgerLoopDay_normal_4  7.392397638472659e-09
DodgerLoopDay_normal_5  6.870671365177827e-09
DodgerLoopDay_normal_6  6.925724493900276e-09
DodgerLoopDay_normal_7  9.982962734245149e-09
DodgerLoopGame_normal_1 8.918171062743951e-09
DodgerLoopGame_normal_2 1.7478214986432762e-08
DodgerLoopWeekend_normal_1 8.460983913494147e-09
DodgerLoopWeekend_normal_2 1.179951338026175e-08
MelbournePedestrian_normal_1 2.4762231320973882e-08
MelbournePedestrian_normal_2 4.5397329895280336e-08
MelbournePedestrian_normal_3 2.906214585651767e-08
MelbournePedestrian_normal_4 4.030236488983263e-08
MelbournePedestrian_normal_5 2.0480152185253786e-08
MelbournePedestrian_normal_6 2.8579490329860846e-08
MelbournePedestrian_normal_7 2.079253486053645e-08
MelbournePedestrian_normal_8 3.299564826919621e-08
MelbournePedestrian_normal_9 1.99948521073523e-08
```

Routing evidence is exact at coordinate level, not only by count:

```text
rows/promotions/pre_exact/exact_normal_removals/final_hard = 1117/137/2253/35/2218
promotions_coordinate_equal=True
pre_exact_coordinate_equal=True
exact_normal_removals_coordinate_equal=True
exp291_coordinate_equal=True
exp404_coordinate_equal=True
recomputed_rows/imported_rows=19/1098
candidate_sources=('exp249_frozen_seed_intersection', 'recomputed_policy')
```

Only the official aeon CPU scorer was called (four calls per affected
dataset), with `inner_threads=1`, `outer_workers=11`, and acceleration
recorded as `inapplicable_official_cpu_transformer`; no MLX/MPS rewrite or
MLX-heavy process was used.  The control test has no Policy B/C scorer or
TEST evaluator call, and the gate's failure status is the terminal
`control_reproduction_failed` value.
