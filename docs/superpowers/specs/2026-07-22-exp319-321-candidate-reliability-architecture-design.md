# Exp319-321 candidate reliability architecture design

## Objective

Exp291 autonomous Hard alert remains frozen. Exp319 improves Exp317/318 at candidate level, Exp320 reorders only the existing Exp291 default-review set, and Exp321 records a dual-lane architecture simulation without promoting research candidates.

## Frozen inputs

- Exp291 automatic Hard and review lanes
- Exp290 exact-normal guard
- Exp309, 310, 311, 313, 314, 315 frozen source scores and candidates
- Exp317 TRAIN-normal source reliability

No TEST label, anomaly position, dataset name, family name, or TEST length may affect prediction.

## Exp319 candidate confidence

For every source candidate and alpha:

1. Source reliability is the frozen Exp317 weight.
2. Calibration stability is the fraction of five deterministic TRAIN OOF reference splits in which the TEST score exceeds the alpha threshold.
3. Margin strength is the bounded robust excess above the TRAIN-only alpha quantile.
4. Candidate source evidence is `reliability * (0.5 * stability + 0.5 * margin_strength)`.
5. Sources in the same method group are reduced by maximum evidence.
6. Candidate confidence is supporting group evidence divided by available group reliability.

At least three available method groups and two supporting method groups are required. All alpha and confidence thresholds 0.50, 0.67, 0.75 are reported; no primary variant is selected.

## Exp320 review ranking

The existing Exp291 default-review indices are unchanged. They are sorted by Exp319 candidate confidence, then dataset and index for deterministic ties. Metrics are reported at global K = 20, 50, 100 and for the complete fixed review set.

## Exp321 architecture simulation

- Automatic Hard: unchanged Exp291 indices
- Existing review: unchanged indices, Exp320 ordering only
- Research Shadow: Exp319 candidates outside all existing Exp291 lanes
- No candidate is promoted automatically

Exp321 is retrospective and cannot choose a deployment policy from TEST metrics.
