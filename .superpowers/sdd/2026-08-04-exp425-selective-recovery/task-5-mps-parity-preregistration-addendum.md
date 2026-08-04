# Task 5 Architecture A addendum — 2026-08-04

Commit `1213768` made an unsatisfiable single-`BackendAudit` assumption: a
truthful MPS review speedup of 0.6709759390840965 cannot construct an adopted
MPS `BackendAudit`, while the former session required one. This was found
before any TRAIN, TEST, label, or native-MPS operation; no smoke artifact was
created.

Architecture A freezes three separate claims before the first actual run:

- the authenticated Task7 CPU `BackendAudit` remains the scientific/model and
  frozen-contract audit;
- an immutable MPS adoption assessment records the same audit provenance,
  requested MPS reproduction, 0.6709759390840965 versus the 1.1 threshold,
  `adopted=false`, and `speedup_below_threshold`;
- each native record carries actual MPS execution facts (device availability,
  model devices, source/current commits and exact source bytes, timing/sync
  semantics and zero explicit synchronizes).

This separation changes neither frozen model/scoring algorithm nor thresholds.
It prevents falsely reporting an unadopted review reproduction as a CPU run or
inventing a passing MPS speed. The Architecture A implementation and tests are
committed before the sole fixture invocation.
