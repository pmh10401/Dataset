# Exp292 Standard Review Evidence Tiering Implementation Plan

**Goal:** Keep Exp291 decisions unchanged while freezing a label-free evidence-based ordering of its 806 Standard review candidates.

**Architecture:** Hash-verified frozen source predictions are collapsed to one stable vote per physical source family. A pure router partitions Standard review into P1/P2/P3, freezes prediction, and evaluates labels only afterward.

### Task 1: Contracts and source loading

- [x] Test input hash, alpha, coverage, and per-family variant validation.
- [x] Implement immutable source specifications and stable family votes.

### Task 2: Review routing

- [x] Test P1/P2/P3 partition, source unavailability, and forbidden fields.
- [x] Implement pure label-free review tier routing and prediction freeze.

### Task 3: Evaluation and execution

- [x] Test prediction hash verification and review-only metric separation.
- [x] Run all 1,117 datasets and independently verify counts and overlaps.

### Task 4: Dashboard and report

- [x] Register Exp292 in the sequential runner and dashboard.
- [x] Record retrospective tier results without promoting any tier to Hard.
