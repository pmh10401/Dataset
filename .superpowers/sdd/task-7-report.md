# Task 7 Report: Pipeline, Queue, And Dashboard Integration

## Source And Scope

- Required starting code SHA: `c01ae5cd` (`docs(exp264): record report preflight validation`).
- Task 7 files:
  - `experiments/exp263_264_full_missing_aware_integration/run_pipeline.py`
  - `experiments/exp263_264_full_missing_aware_integration/tests/test_run_pipeline.py`
  - `scratch/run_exp263_264_full_missing_aware_integration.py`
  - `scratch/test_run_exp263_264_full_missing_aware_integration.py`
  - Exp263/264 catalog hunks only in `serve_rank_dashboard.py`
- The unrelated uncommitted dashboard, Vercel, Exp256-262, and documentation work was preserved and excluded from Task 7 staging.

## Implementation

- Added the `audit`, `predict`, `evaluate`, and `report` phase CLI.
- Every phase writes `status.json` atomically with experiment ID, PID, phase,
  `done`/`expected` aliases used by the dashboard, current dataset, error list,
  error count, and all output paths.
- Exceptions retain prior artifacts, write `status="failed"`, and return nonzero.
- The scratch runner atomically updates the canonical state, removes duplicate
  or stale Exp263/264 queue entries, records phase and return code in history,
  and removes Task 7 queue entries after a failure.
- Dashboard catalog entries for Exp263 and Exp264 both retain the production
  expectation of 1,117 datasets.
- Smoke subsets are selected only after the full frozen source contract passes.
  The production population and anomaly contracts remain 1,117 and 3,918;
  smoke evaluation requires an explicit anomaly override.

## Commands And Results

- `python3 -m unittest discover -s experiments/exp263_264_full_missing_aware_integration/tests -v`
  - Passed: 51 tests.
- `python3 -m unittest scratch.test_run_exp263_264_full_missing_aware_integration scratch.test_dashboard_state_contract -v`
  - Passed: 16 tests.
- `python3 -m py_compile experiments/exp263_264_full_missing_aware_integration/run_pipeline.py scratch/run_exp263_264_full_missing_aware_integration.py serve_rank_dashboard.py`
  - Passed.
- `git diff --check`
  - Passed.
- Real failure contract:
  - `python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase evaluate --dataset-limit 10 --output-root /tmp/exp263_264_failure_contract_task7_20260721`
  - Exited `1`; status recorded `failed`, `error_count=1`, and the explicit missing anomaly-override error.

## Smoke Run

- Output root: `/tmp/exp263_264_smoke_task7_20260721`.
- Audit command:
  - `python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase audit --dataset-limit 10 --output-root /tmp/exp263_264_smoke_task7_20260721`
- Prediction command:
  - `python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase predict --dataset-limit 10 --output-root /tmp/exp263_264_smoke_task7_20260721`
- Evaluation command:
  - `python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase evaluate --dataset-limit 10 --expected-total-anomalies 10 --output-root /tmp/exp263_264_smoke_task7_20260721`
- Report command:
  - `python3 -m experiments.exp263_264_full_missing_aware_integration.run_pipeline --phase report --dataset-limit 10 --output-root /tmp/exp263_264_smoke_task7_20260721`
- All four commands exited `0`.
- Selected datasets: `ACSF1_normal_0` through `ACSF1_normal_9`.
- Audit coverage: baseline/statistical/structural/change-point `10/10`;
  reconstruction `0/10`; missing override `0/10`; errors `0`.
- Prediction rows: `10`; SHA256:
  `dde03dc489040eaaf691123709fa71a815aafea9f3df770ae0bbe42d4606a90c`.
- Evaluation: expected anomalies `10`, actual anomalies `10`, result rows `10`.
- Final status: `completed`, phase `report`, `10/10`, current dataset
  `ACSF1_normal_9`, error count `0`.

## Dashboard And Isolation

- In-process dashboard contract check presented the smoke status as Exp264,
  phase `report`, `10/10`, while both catalog entries remained at `1117`.
- The canonical queue state mtime remained `2026-07-20 23:19:33`, before the
  smoke run, confirming the direct phase commands did not mutate the full queue.
- The existing dashboard server PID `32416` remained alive. No server restart,
  signal, port bind, or request was performed.
- The full production output root was not used by smoke commands.

## Concerns

- The first ten sorted datasets contain no missing-override or reconstruction
  rows, so those branches remain covered by the existing focused Task 1-6 tests,
  not this live smoke subset.
- The smoke anomaly total is intentionally explicit and smoke-only. Task 8 must
  omit both smoke arguments so production evaluation enforces 1,117 datasets and
  3,918 anomalies.
- The dashboard has unrelated user changes in the same file. Only the two new
  Exp263/264 catalog dictionaries belong to this Task 7 commit.
