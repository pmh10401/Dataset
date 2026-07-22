# Exp206-210 Residual Coverage Implementation Review

This review compares the implementation of `experiments/exp206_210` against the specifications in:
* `docs/superpowers/specs/2026-07-16-exp206-210-residual-coverage-design.md`
* `docs/superpowers/plans/2026-07-16-exp206-210-residual-coverage.md`

---

## Summary of Findings by Severity

| Severity | Category | Focus Area | File / Function | Description |
| :--- | :--- | :--- | :--- | :--- |
| **High (Blocker)** | Bug | Dashboard Progress | `serve_rank_dashboard.py` (Line 4477)<br>`run_experiment_206_210.py` (Line 83, 458, 699, 888) | **Dashboard progress parser regex mismatch.** The dashboard regex expects `(\d+)/(\d+)` (no spaces around `/`), but the runner prints `[{done} / {expected}]` (with spaces). |
| **High** | Missing Feature | Missing Required Outputs | `experiments/exp206_210/independent_reaggregate.py`<br>`independently_reaggregate()` | **Recall, Candidate F1, and Risk Lanes are missing in the Stage 210 Summary.** The spec requires stratified views of candidate recall, candidate F1, and separating shadow/review/research metrics, but the summary CSV lacks these fields. |
| **Medium** | Security / Integrity | Hash Freeze | `experiments/exp206_210/run_experiment_206_210.py` (Line 895) | **Empty input hashes in Stage 209.** Unlike Stage 208 which freezes the rules config hash, Stage 209 passes an empty `{}` to `freeze_predictions`. Also, input caches are not hashed/verified in either stage. |
| **Low** | Architecture | Queue Deadline | `run_rank_experiments_sequential.py` (Line 1630) | **No deadline handling for Exp206-210 batch.** The queue runner does not register `exp206_210_batch` in its hard-deadline check loop. (Likely minor if no hard runtime limit is expected). |
| **None (Pass)** | Conformance | Label Leakage | `experiments/exp206_210/artifacts.py`<br>`load_dataset_series_only()` | **Strict label isolation.** Series features are loaded without labels. Labels are only loaded after prediction freeze and hash verification. |
| **None (Pass)** | Conformance | Fold Safety | `experiments/exp206_210/alternative_oof.py`<br>`crossfit_alternative_sources()` | **No data leakage across folds.** Prototype selection and scoring use fit fold statistics only. |
| **None (Pass)** | Conformance | OOF Calibration | `experiments/exp206_210/core.py`<br>`leave_one_out_pvalues()` | **Leakage-free p-values.** Reference distribution excludes the target query point. |
| **None (Pass)** | Conformance | Queue Order | `run_rank_experiments_sequential.py` | **Queue order is correct.** Runs sequentially in order 206 $\rightarrow$ 207 $\rightarrow$ 208 $\rightarrow$ 209 $\rightarrow$ 210. |

---

## Detailed Findings

### 1. [Blocker] Dashboard Progress Regex Mismatch
* **File References**: 
  * [serve_rank_dashboard.py:L4477](file:///private/tmp/timeseries-exp177-185-batch/serve_rank_dashboard.py#L4477)
  * [run_experiment_206_210.py:L83](file:///private/tmp/timeseries-exp177-185-batch/experiments/exp206_210/run_experiment_206_210.py#L83)
  * [run_experiment_206_210.py:L458](file:///private/tmp/timeseries-exp177-185-batch/experiments/exp206_210/run_experiment_206_210.py#L458)
* **Details**: The dashboard logs parser searches for progress lines using this regex:
  ```python
  plain_pattern = re.compile(r"^Progress:\s*\[\s*(\d+)/(\d+)\]\s*rows=(\d+)", re.IGNORECASE)
  ```
  However, the runner prints progress with spaces around the slash:
  ```python
  f"Progress: [{done} / {expected}] rows={rows} ..."
  ```
  Because the regex expects `(\d+)/(\d+)` (no spaces around the `/`), the match fails.
* **Impact**: The dashboard will not show execution progress for Exp206-210; it will appear permanently stuck at 0% or pending.
* **Suggested Fix**: Update `serve_rank_dashboard.py` to allow optional spaces around the slash, e.g.:
  ```python
  plain_pattern = re.compile(r"^Progress:\s*\[\s*(\d+)\s*/\s*(\d+)\]\s*rows=(\d+)", re.IGNORECASE)
  ```

---

### 2. [High] Missing Required Outputs in Stage 210 Stratified Summary
* **File References**: 
  * [independent_reaggregate.py:L99-L124](file:///private/tmp/timeseries-exp177-185-batch/experiments/exp206_210/independent_reaggregate.py#L99-L124)
* **Details**: The design spec (Section 10, "Exp210: Independent Frontier Reaggregation") mandates the following outputs in the stratified summary:
  * "added candidate count, TP, FP, precision, **recall**, and **mean candidate F1**"
  * "**shadow, review, and research-only metrics separated**"
  
  In the current implementation of `independent_reaggregate.py`:
  1. `recall` and `candidate_f1` are computed per dataset in the detail file but are completely absent from `summary_rows` (only `candidate_precision` is computed).
  2. The risk classification (e.g. `final_lane` or `metric_scope` like `shadow_hard_diagnostic`) is not included in `summary_rows`, making it impossible to separate or filter the summary rows by risk tier.
* **Impact**: Stage 210 does not fully comply with the scientific and reporting requirements of the design specification.
* **Suggested Fix**: Update `independent_reaggregate.py` to calculate and record `mean_candidate_recall`, `mean_candidate_f1`, and include a `final_lane` or `metric_scope` column in `summary_rows`.

---

### 3. [Medium] Incomplete Hash Freeze in Stage 209
* **File References**: 
  * [run_experiment_206_210.py:L895](file:///private/tmp/timeseries-exp177-185-batch/experiments/exp206_210/run_experiment_206_210.py#L895)
* **Details**: In Stage 209 (Seven-Source Joint Normal Novelty), the runner freezes predictions via:
  ```python
  pred_path, manifest_path = freeze_predictions(
      "209", STAGE_META["209"][0], prediction_rows, output_root, {}
  )
  ```
  Passing an empty dict `{}` for `input_hashes` means that Stage 209 does not freeze or audit its inputs. Additionally, the hashes of the input OOF cache files (Stage 206 cache and Exp153 cache) are never verified before prediction in Stage 208 and Stage 209.
* **Impact**: If input OOF cache files are tampered with or modified between Stage 206 and Stage 208/209, the runner will proceed without detecting the changes, violating the Scientific Contract (Section 4: "Evaluation verifies a frozen prediction SHA256 before loading TEST labels").
* **Suggested Fix**: Record the SHA256 hashes of the OOF cache files loaded in `input_hashes` during the prediction freeze step.

---

### 4. [Low] Missing Hard-Deadline Enforcement for Exp206-210 Batch
* **File References**: 
  * [run_rank_experiments_sequential.py:L1630](file:///private/tmp/timeseries-exp177-185-batch/run_rank_experiments_sequential.py#L1630)
* **Details**: The sequential queue runner checks 12-hour hard deadlines for `exp177_185_batch` and `exp186_200_batch`, but does not list `exp206_210_batch`.
* **Impact**: The runner will not enforce a hard deadline for the Exp206-210 batch. Since the queue runs sequentially, if any dataset takes too long or hangs in CPU multiprocessing (e.g., shapelet feature extraction), the queue runner will run indefinitely without timing out.
* **Suggested Fix**: If a batch-level timeout is desired, add `"exp206_210_batch"` to the loop list in `cmd_run_queue`.

---

## Verified Valid Components (Conformance Checklist)

* **Label Leakage Protection (Pass)**: Verified that prediction steps only read series values via `load_dataset_series_only()`. `validate_prediction_payload()` enforces the absolute absence of evaluation fields (`y_test`, `tp`, `fp`, etc.) before freezing.
* **Fold Safety (Pass)**: Verified that prototype selection in `choose_shapelets()` and scaling/centering in `robust_score_pair()` are calculated purely on the fit-fold indices, preventing query fold data from leaking into the fitting process.
* **OOF Calibration (Pass)**: Verified that `leave_one_out_pvalues()` correctly excludes each query instance from its own reference set.
* **Queue Sequence (Pass)**: Sequential batch correctly registered in order 206 $\rightarrow$ 207 $\rightarrow$ 208 $\rightarrow$ 209 $\rightarrow$ 210.
