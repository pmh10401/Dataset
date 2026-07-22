# Exp301 TRAIN Risk And Stability Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate whether the Exp300 independent-evidence candidates are sufficiently rare on TRAIN-normal data and stable under resampling before any automatic-alert promotion is considered.

**Architecture:** Reuse the frozen score caches from official MultiROCKET, structural-pattern, and change-point sources. Prediction reads TRAIN OOF and TEST score vectors without TEST labels, computes TRAIN-normal risk and deterministic bootstrap stability, and freezes mutually exclusive shadow lanes. Evaluation verifies the prediction hash before loading TEST labels.

**Tech Stack:** Python 3, NumPy, unittest, existing JSONL freeze/verify helpers, sequential experiment runner, Vercel dashboard snapshot exporter.

## Global Constraints

- Current Exp292 autonomous Hard alerts remain unchanged.
- `alpha=0.01`, 3-of-3 is an automatic-qualification shadow candidate, not an autonomous alert.
- `alpha=0.02`, 3-of-3 is priority review only.
- `alpha=0.005`, 2-of-3 is general review only.
- Automatic qualification requires pooled TRAIN-normal 3-of-3 rate at most 0.005, one-sided 95% Wilson upper bound at most 0.01, and candidate bootstrap stability at least 0.80.
- Bootstrap replicates are fixed at 32 with seed 20260722.
- TEST labels, anomaly positions, family names, dataset-name policy branches, and TEST-length budgets are forbidden during prediction.
- Alpha choices are retrospective research choices informed by Exp300 and are not prospective validation.
- Local dashboard server must not be started; Vercel is the only user-facing dashboard.

---

### Task 1: Core TRAIN-only statistics and routing

**Files:**
- Create: `experiments/exp301_train_risk_stability_routing/core.py`
- Create: `experiments/exp301_train_risk_stability_routing/test_exp301.py`

**Interfaces:**
- Produces: `loo_upper_pvalues`, `empirical_upper_pvalues`, `wilson_upper`, `build_train_risk_audit`, `build_routing_row`, `evaluate_predictions`, `summarize_evaluated`.

- [ ] Write tests for tie-safe leave-one-out p-values, TEST empirical p-values, three-source TRAIN risk, deterministic bootstrap stability, mutually exclusive lanes, and label-free prediction rows.
- [ ] Run `python3 -m unittest experiments.exp301_train_risk_stability_routing.test_exp301` and confirm the missing module/functions fail.
- [ ] Implement the minimum core functions.
- [ ] Re-run the unit tests and confirm they pass.

### Task 2: Frozen prediction and evaluation pipeline

**Files:**
- Create: `experiments/exp301_train_risk_stability_routing/__init__.py`
- Create: `experiments/exp301_train_risk_stability_routing/run_pipeline.py`
- Create: `scratch/run_exp301_train_risk_stability_routing.py`

**Interfaces:**
- Consumes: Exp250, Exp254, Exp255 score caches; Exp298 baseline; Exp300 frozen candidates.
- Produces: frozen prediction JSONL, prediction manifest, TRAIN-risk CSV, evaluated CSV, summary CSV, report, and status JSON.

- [ ] Verify all frozen input hashes and 251-dataset cache coverage before prediction.
- [ ] Build TRAIN-normal 3-of-3 risk for alpha 0.01 and 0.02 without TEST labels.
- [ ] Run 32 deterministic TRAIN-reference bootstrap replicates and calculate per-candidate stability.
- [ ] Freeze automatic-qualification shadow, priority-review, and general-review lanes with zero overlap.
- [ ] Verify the prediction hash before loading TEST labels and write retrospective evaluation artifacts.

### Task 3: Queue and Vercel dashboard integration

**Files:**
- Modify: `run_rank_experiments_sequential.py`
- Modify: `serve_rank_dashboard.py`
- Modify: `scratch/test_rank_dashboard_operational.py`

**Interfaces:**
- Produces: `queue-exp301` and one Exp301 completed-result card in the Vercel snapshot.

- [ ] Add a failing dashboard registration test.
- [ ] Register Exp301 in the runner and dashboard metadata.
- [ ] Run the dashboard operational tests.

### Task 4: Execute and verify

**Files:**
- Outputs: `/Users/minho/Documents/Dataset/exp301_train_risk_stability_routing/`

- [ ] Run all Exp301 and dashboard tests.
- [ ] Queue and execute Exp301 once, sequentially.
- [ ] Verify 251 datasets, zero missing caches, zero lane overlap, no evaluation fields in frozen prediction, and all artifact hashes.
- [ ] Confirm the Vercel snapshot contains Exp301 without starting a local dashboard.
- [ ] Report TRAIN-normal risk, stability-qualified counts, and separate autonomous/review evaluation metrics without promoting any candidate.
