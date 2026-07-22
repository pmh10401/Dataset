# Exp306-307 Componentized Routing and Expanded MoE Design

## Objective

Separate the current Exp291 benchmark Hard set into a guarded Exp137 core and
two retrospective research extensions, then evaluate a six-expert MoE over the
frozen Exp292 Standard candidate universe. Existing artifacts remain unchanged.

## Frozen preprocessing contract

Every newly computed score source must use this exact order:

1. preserve the raw missing mask and repair NaN/Inf/missing values on the original grid;
2. align TRAIN and query rows to a target length derived only from TRAIN normal rows;
3. normalize only after alignment, without TEST-wide statistics.

The machine-readable order is `missing_repair -> train_length_alignment ->
normalization`. Exp306/307 replay existing frozen scores and therefore record,
rather than retroactively alter, the provenance of those inputs.

## Exp306 lanes

- `core_hard_indices`: Exp137 portion of Exp236 Hard after the Exp290 exact-normal guard.
- `exp236_research_shadow_indices`: Exp236 three-source additions after the guard.
- `exp253_research_shadow_indices`: Exp253 MiniROCKET promotions after the guard.
- `benchmark_union_hard_indices`: the disjoint union of the three lanes; must equal Exp291 Hard.
- Existing Standard, Priority, Shadow, and recovery lanes remain frozen.

The candidate operational policy uses only `core_hard_indices`; it is not
deployment activated. Exp236/253 remain visible as high-quality retrospective
Shadow evidence.

## Exp307 experts

Existing experts:

1. statistical
2. structural
3. change point
4. official MultiROCKET

New experts:

5. ordinal-pattern distribution shift (Exp240 default, alpha 0.005)
6. masked-patch context reconstruction (Exp247 seed intersection, alpha 0.005)

The new experts must have 1,117-row cache and prediction coverage, verified
hashes, no TEST labels, and TRAIN OOF scores. A single TRAIN variant does not
establish cross-variant stability; `stability_observable` is false and the Gate
must not treat the numerical fallback as evidence.

## Method groups and Gate

- distribution: statistical, ordinal pattern
- structure: structural, change point
- random kernel: official MultiROCKET
- reconstruction: masked context

Each Gate selects at most three eligible experts from distinct method groups.
Candidate ordering uses selected-expert support, independent method-group
support, total verified support, TRAIN-only risk rank, then stable dataset/index
ordering. Dataset/family names, TEST length, TEST labels, and TEST performance
are prohibited policy inputs.

## Outputs

Exp307 preserves the 806 Standard candidates and reports fixed workloads 17,
74, and 806. It also freezes review-only Shadow sets for at least two and at
least three independent method groups. No Shadow candidate becomes autonomous
Hard and no Gate is selected from TEST metrics.

All evaluations are retrospective and labels are loaded only after prediction
SHA256 is frozen.
