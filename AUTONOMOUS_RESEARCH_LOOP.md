# Autonomous label-free anomaly-detection research loop

## Objective

Use only the frozen local semiconductor time-series data to produce a label-free anomaly
detector that is demonstrably better than verified references. Improvement is not a single
mean score: it must account for newly rescued datasets, false alarms, regressions on strong
datasets, paired whole-population performance, and preregistered bootstrap uncertainty.

## Mandatory resume order

After context compaction or on a new Codex host, read these files before acting:

1. `AUTONOMOUS_RESEARCH_LOOP.md`
2. `RESEARCH_STATE.json`
3. `EXPERIMENT_CATALOG.json`
4. `FAILURE_LEDGER.json`
5. `CHAMPION_LEDGER.json`
6. the active experiment preregistration, runbook, and progress log named by
   `RESEARCH_STATE.json`

Never reconstruct current state from conversation memory when these files disagree with it.
Verify Git, checkpoint, PID, and artifact bytes before resuming.

## Scientific firewall

- Prediction may use normal TRAIN data and preregistered fixed metadata only.
- Prediction must not use TEST labels, anomaly positions, past family-level TEST performance,
  or a human-assisted metric.
- Freeze and hash prediction coordinates before opening TEST labels.
- TEST labels are retrospective evaluation only. Never use a result to reselect alpha,
  threshold, score source, family, budget, or routing inside the same experiment.
- Every adaptive attempt receives a new experiment number, preregistration, checkpoint/output
  roots, and manifest. Failed and unfavorable attempts remain in the ledgers.
- `mean_combined_f1` remains human-assisted evidence and cannot authorize prediction changes.
- Priority/Shadow/review-only evidence is not an automatic operational alarm.
- Do not use tail replacement.

## Cycle

1. **Recover:** validate the five control files, Git HEAD/cleanliness, DB/checkpoint hashes,
   process state, and the active log without mutating an experiment root.
2. **Diagnose:** use only frozen predictions plus retrospective evaluation to identify broad
   failure modes. Do not expose family TEST performance to the next prediction function.
3. **Choose a base:** continue from the current experiment when its architecture is sound, or
   reuse a verified component from another experiment. The historical reference is a
   comparator, not a mandatory code base.
4. **Preregister:** allocate a new experiment number and freeze hypothesis, source, features,
   thresholds, alphas, budgets, routing, compute backend, seeds, success margins, bootstrap,
   abort rules, logs, and artifact paths before prediction.
5. **Implement and verify:** use TDD, label-access guards, deterministic CPU/device parity,
   checkpoint crash tests, and an independent review. Apply the repository MLX/MPS policy.
6. **Run label-free:** use explicit `new` or `resume`, atomic checkpointing, and process-written
   logs. Do not spend Codex tokens repeatedly polling an unchanged long process.
7. **Freeze:** verify completion, canonicalize predictions, record their SHA256, and close the
   prediction phase.
8. **Evaluate retrospectively:** load TEST labels only now. Produce per-dataset paired evidence,
   confusion counts, rescued/missed/regressed datasets, false alarms, global metrics, and the
   preregistered bootstrap interval.
9. **Decide:** update `CHAMPION_LEDGER.json` only if the preregistered decision rule passes.
   Otherwise append `FAILURE_LEDGER.json`, update `RESEARCH_STATE.json`, allocate the next
   experiment, and continue without requesting routine approval.

## Improvement decision

Each experiment preregistration must name its parent and comparison references and freeze its
own quantitative margins before execution. At minimum the evaluator reports:

- datasets with zero/failed detection before and nonzero correct detection now;
- datasets that regress, especially previously strong datasets;
- TP, FP, FN, TN; precision, recall, specificity, FPR, and F1 at micro and dataset-balanced
  levels where defined;
- paired per-dataset deltas and a preregistered clustered/bootstrap confidence interval;
- candidate-coordinate/prediction hashes and population conservation.

Some regressions are acceptable only when the experiment's preregistered global paired rule
passes and the rescued-dataset/false-alarm guardrails pass. Changing the rule after seeing TEST
results is prohibited.

## Long-run observability

The process, not Codex polling, writes and fsyncs an atomic JSON state plus append-only JSONL
and text log. Required fields include PID, current dataset, TRAIN/TEST counts and lengths,
label-free byte/value sizes, stage/backend, completed/total, import/recompute/success/error,
throughput, elapsed time, ETA, checkpoint, heartbeat, and latest stable error. The runbook and
`RESEARCH_STATE.json` record absolute paths and the exact resume command.

## Authority boundary

Routine research, local implementation, local execution, evaluation after prediction freeze,
and new preregistered iterations are approved until verified improvement. Stop only for actions
outside that scope, including material data deletion, external publication/messages, paid
resources, credential/security expansion, or an ambiguity that could invalidate the scientific
firewall. If forced to stop, atomically update `RESEARCH_STATE.json` with the last good commit,
PID/checkpoint/log paths, blocker, and exact next command.
