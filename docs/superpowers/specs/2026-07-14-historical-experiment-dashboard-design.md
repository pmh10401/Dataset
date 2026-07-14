# Historical Experiment Dashboard Design

## Goal

Show the complete experiment history without turning invalid, blocked, or
human-assisted results into apparent operational winners. The dashboard must
make missing data explicit instead of displaying an unexplained dash.

## Scope

- Include every known experiment through Exp152 and the active Exp153-157
  bundle.
- Preserve existing experiment files and sequential queue state.
- Do not rerun experiments or modify their result CSV files.
- Keep autonomous Hard alert metrics separate from Standard/Priority review
  and combined human-assisted metrics.

## Experiment Status Model

Each experiment receives one history status:

- `running`: an active process is detected.
- `queued`: registered for sequential execution.
- `completed_valid`: completed and eligible for like-for-like performance
  comparison.
- `completed_research`: completed retrospective audit or sensitivity result;
  visible, but excluded from the default operational ranking.
- `invalid`: completed artifact with a confirmed implementation or evaluation
  defect. Exp149/150 belongs here and remains visible for traceability.
- `blocked`: preregistered but not executable because a required contract or
  data assumption was not established.
- `not_run`: known experiment with no valid result artifact.

Status is derived from an explicit catalog plus artifact existence. Absence of
a file is never treated as a zero-valued result.

## Metric Adapters

The dashboard reads historical schemas through explicit adapters:

1. Legacy classification rows: `f1`, `fp`, `tp`, `fn`, `train_exceed_rate`.
2. Hard-alert rows: `hard_f1`, `hard_fp`, `hard_tp`, `hard_fn`,
   `hard_precision`.
3. Summary-only sensitivity rows: `mean_hard_f1`, `hard_alert_precision`, and
   total Hard alert counts where available.
4. Audit-only outputs: coverage, binding count, abstention, or workload fields;
   these do not receive fabricated F1 values.

For Exp137 and later, the primary F1 shown in operational comparison is the
autonomous Hard alert F1. `mean_combined_f1` is displayed only in a separate
human-assisted field.

## Historical Coverage

- Existing completed experiments retain their current entries.
- Exp138-152 receive explicit catalog entries and artifact paths.
- Exp140/142/145/148 are shown according to their actual run or blocker state,
  not inferred as completed.
- Exp149/150 are marked invalid because their routing defect was corrected by
  Exp151/152.
- Exp151/152 are displayed as corrected retrospective experiments with all four
  alpha variants, while no alpha is labelled optimal or recommended.

## Dashboard Changes

### Experiment Compare

- Default view contains only `completed_valid` experiments with comparable
  autonomous metrics.
- A status filter can reveal research, invalid, blocked, and not-run entries.
- Rows show experiment status, evidence scope, Hard F1, Hard FP, Hard
  precision, and review workload when available.

### Completed Results / History

- Rename conceptually to experiment history while preserving the existing UI
  section location.
- Show every cataloged experiment with a status badge.
- Replace ambiguous dashes with a short reason such as `audit only`,
  `artifact missing`, `blocked`, or `metric not recorded`.
- Keep links to the actual result or report artifact.

### Detail View

- Display autonomous Hard metrics and human-assisted review metrics in
  separate groups.
- Show retrospective/prospective scope and known limitations.
- For multi-variant experiments, show every variant without selecting a winner
  from TEST results.

## Data Flow

1. Load the explicit experiment catalog.
2. Reconcile queue state, active process state, and artifact existence.
3. Select the schema adapter from available columns.
4. Produce normalized autonomous, review, provenance, and status fields.
5. Apply comparison eligibility only after normalization.
6. Return both the complete history and the filtered operational comparison to
   the browser.

## Error Handling

- A missing or malformed artifact yields a visible reason and does not remove
  the experiment from history.
- Conflicting summary/detail values are marked `metric conflict` and excluded
  from ranking until reconciled.
- Unsupported schemas are marked `unsupported schema`; no silent zero filling.
- The dashboard remains read-only and never writes experiment outputs.

## Verification

- Catalog coverage test for every known experiment through Exp152.
- Adapter tests for legacy, Hard-alert, summary-only, and audit-only schemas.
- Tests proving combined F1 cannot populate autonomous F1.
- Tests proving invalid/blocked experiments are visible but excluded from the
  default ranking.
- API reconciliation checks for status counts and artifact paths.
- Browser checks at desktop and mobile widths after restarting only the
  dashboard server; the active experiment PID must remain unchanged.

## Non-Goals

- Re-evaluating or rerunning historical experiments.
- Choosing an optimal alpha, threshold, budget, or selector from TEST results.
- Promoting Priority review to an autonomous alert.
- Claiming retrospective experiments are prospective equipment validation.
