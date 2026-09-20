# Exp422-IR Design: interrupted Exp419 predecessor diagnostic

## Context

The official Exp419 attempt was intentionally stopped after its runtime became
unproductive. It recorded exit code 143 and no completed dataset. The original
Exp422 guard correctly rejects that state because it requires a successful
Exp419. This document registers a separate diagnostic boundary rather than
weakening the original guard.

## Contract

Exp422-IR accepts only a dead Exp419 PID and an explicit exit code 143. It uses
the existing frozen Exp422 numerical core and bounded normal-TRAIN score path,
with the exact five batch sizes and three paired model seeds. It never reads
TEST values, labels, locations, historical TEST metrics, or human review.

The output is diagnostic evidence only. It cannot authorize a backend,
threshold, source, lane, budget, or production alert change.

## Operational evidence

The runner writes a durable JSONL progress stream and a sidecar execution
context before the first MPS-heavy stage. A launcher redirects stdout/stderr to
a separate run log and records the Python PID. One MPS-heavy process is allowed
at a time; the Exp419 checkpoint and official output are read-only boundaries.
