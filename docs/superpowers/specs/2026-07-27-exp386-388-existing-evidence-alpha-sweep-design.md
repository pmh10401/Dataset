# Exp386-388 Existing Evidence Alpha Sweep Design

## Goal

Reproduce the original Exp357 evidence code and evaluate the fixed alpha grid
`(0.005, 0.01, 0.02, 0.05, 0.10, 0.20)` without changing the existing Hard
alert policy.

## Provenance

- Historical source commit: `930b8ce8927d18a0101c647d26e21c2cd20be800`
- Existing Exp357 alpha: `0.005`
- Database: `/Users/minho/Documents/Dataset/univariate_ts.db`
- Population: frozen Exp291/Exp357 1,117 datasets

The historical source closure is extracted to an isolated temporary directory
at runtime. Every extracted source is checked against the SHA256 mapping stored
in the original Exp356 baseline contract. The current worktree does not restore
or overwrite the removed historical packages.

## Experiments

### Exp386: Exact alpha 0.005 replay

Run the original Exp357-359 prediction and evaluation code with alpha 0.005.
The frozen Exp357 prediction must match the original SHA256 exactly. A mismatch
stops the remaining queue.

### Exp387: Pre-registered alpha sweep

Run alpha `0.01`, `0.02`, `0.05`, `0.10`, and `0.20`. The first four values
including Exp386's 0.005 are the established calibration range. Values 0.10 and
0.20 are coverage sensitivities and remain review/confidence research only.

`n_train < 5` always abstains. No cross-dataset pooling is used.

### Exp388: Read-only comparison

Aggregate coverage and the frozen Hard-confidence/review results across all
alphas. Existing Hard indices remain unchanged. Report:

- source availability and resolution-blocked counts
- strong/general confidence counts and precision
- review high/partial/unconfirmed counts and precision
- dataset-level lane changes between alpha values

No alpha is selected for deployment from TEST metrics.

## Safety

- Prediction functions receive no TEST labels.
- Evaluation starts after each prediction SHA256 is frozen.
- TEST length is not used to choose alpha or budget.
- Family and dataset names do not select an alpha.
- Priority/review candidates are not promoted to Hard.
- Existing Exp357-359 outputs are read-only.
- All results are retrospective counterfactual, not prospective validation.

## Output

`/Users/minho/Documents/Dataset/exp386_388_existing_evidence_alpha_sweep`
