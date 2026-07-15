# Exp161 Alternative Source Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether previously computed spectral/derivative, shapelet, and interval-distribution artifacts can safely support a label-free candidate-zero recovery experiment after Exp160.

**Architecture:** Build a read-only provenance and coverage audit over the frozen Exp160 prediction and historical Exp99, Exp101, and Exp46 result artifacts. Separate “a row exists” from “the alternative source was actually computed,” and separate both from “the source is policy-eligible.” Do not generate a new alert or evaluate TP/FP in Exp161.

**Tech Stack:** Python 3 standard library, CSV/JSONL, SHA256, `unittest`.

## Global Constraints

- Experiment ID: `experiment_161_alternative_source_candidate_zero_audit`.
- Output directory: `outputs/exp161_alternative_source_audit/`.
- Exp160 prediction SHA256 must equal `ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d`.
- Audit all `ALPHAS = (0.005, 0.01, 0.02, 0.05)`; do not select one.
- Target population is the frozen Exp160 rows with `n_train >= 5` and `candidate_count == 0`.
- User-facing source names: `미분·스펙트럼 통계 점수`, `부분 형태 일치 점수`, `구간 분포 점수`.
- Historical result metric columns (`tp`, `fp`, `f1`, labels, anomaly counts) must never be used to determine coverage, eligibility, source choice, or recommendation.
- A CSV row existing does not mean its source was computed; passthrough rows must be detected.
- Exp99/101 family-target and Exp97 research-target calculations are not policy-eligible for full-coverage replay.
- Exp46 family/difficulty-limited coverage is not policy-eligible for full-coverage replay.
- No Hard, Standard, Priority, candidate, threshold, alpha, or source-selection policy is changed.
- No performance conclusion is allowed. Exp161 is a retrospective provenance/coverage audit only.

---

### Task 1: Add Pure Provenance Rules with TDD

**Files:**
- Create: `experiments/exp161/__init__.py`
- Create: `experiments/exp161/alternative_source_audit_core.py`
- Create: `experiments/exp161/test_exp161_alternative_source_audit.py`

**Interfaces:**
- Consumes: allowlisted historical row metadata and frozen Exp160 prediction rows.
- Produces: target classification, actual-computation detection, provenance classification, and policy-eligibility decisions.

- [ ] **Step 1:** Write failing tests for target selection, passthrough detection, family/research-limited provenance, partial coverage, and forbidden metric independence.
- [ ] **Step 2:** Run tests and verify expected failures.
- [ ] **Step 3:** Implement minimal pure functions without DB or label access.
- [ ] **Step 4:** Run tests and verify they pass.

### Task 2: Build Read-Only Audit Runner

**Files:**
- Create: `experiments/exp161/run_experiment_161_alternative_source_audit.py`
- Create: `experiments/exp161/test_exp161_structure.py`
- Create: `docs/EXP161_PREREGISTRATION_KO.md`
- Create: `docs/EXP161_RUNBOOK_KO.md`

**Interfaces:**
- Consumes: frozen Exp160 prediction, Exp99 result CSV, Exp101 result CSV, Exp46 result CSV.
- Produces: source inventory, target-level coverage, source-pair overlap availability, summary, blocker report, and manifest.

- [ ] **Step 1:** Verify all input hashes and record them.
- [ ] **Step 2:** Read only an explicit metadata/index allowlist from historical rows; do not read metric values.
- [ ] **Step 3:** Write one target row per dataset/alpha with source actual-computation and policy-eligibility flags.
- [ ] **Step 4:** Calculate only label-free counts: targets, actual source coverage, common support, and legacy index overlap where provenance is clearly marked research-only.
- [ ] **Step 5:** If fewer than two policy-eligible sources have full target coverage, create `BLOCKER_REPORT.md` and do not create an Exp162 policy recommendation.

### Task 3: Codex Review and Full Audit

**Files:**
- Create: `outputs/exp161_alternative_source_audit/`
- Create: `outputs/exp161_alternative_source_audit/VERIFICATION_REPORT.md`

**Interfaces:**
- Consumes: reviewed implementation and 10-dataset smoke.
- Produces: full 1,117-dataset provenance/coverage audit.

- [ ] **Step 1:** Review code and tests for metric/label access and passthrough misclassification.
- [ ] **Step 2:** Run compile, unit tests, and 10-target smoke.
- [ ] **Step 3:** Run full audit after review passes.
- [ ] **Step 4:** Independently verify target counts against Exp160 and source coverage against original files.
- [ ] **Step 5:** State whether Exp162 requires new full-coverage score computation without choosing a feature source from TEST performance.
