# Exp322-324 Exp291 reliability-weighting design

## Scope

This sequence verifies whether TRAIN-only reliability weighting already computed in Exp116 changes the score-source selection that ultimately feeds Exp291.

- Exp322 audits the complete Exp291 lineage and distinguishes numeric weights from gates and set consensus.
- Exp323 freezes a label-free replay of Exp116 baseline versus reliability-weighted selected indices.
- Exp324 propagates the replay result to a frozen Exp291 routing counterfactual only when deterministic index parity permits it.

## Known boundary

Exp93 assigns fixed weights to five rank sources. Exp116 recomputed TRAIN-only reliability weights for the three primary sources only. The two Exp84 sources retain TRAIN-exceed safety gates, and Exp236/253 use consensus/guard rules rather than numeric score weights. Therefore this sequence must not claim that all Exp291 upstream evidence has been reliability weighted.

No TEST metric selects a weight or route. Existing Exp291 outputs are never overwritten.
