# Exp328 Large-dataset FN Reachability Audit

## Scope

- Population: 251 datasets with at least 100 TRAIN-normal instances.
- Frozen automatic policy: Exp291, unchanged.
- Prediction phase reads no TEST labels and recomputes the three primary score rankings through top 64.
- Independent frozen evidence: statistical, structural, change-point, masked reconstruction, and official MultiROCKET.
- Evaluation is retrospective diagnostic only.

## Integrity

- Dataset coverage: 251/251.
- Prediction errors: 0.
- Hard FN count independently reconciled: 1,635.
- Every FN is assigned to exactly one category.

## FN classification

| Category | FN | Share |
|---|---:|---:|
| Outside all audited top-64 rankings | 338 | 20.7% |
| Already in review lane | 327 | 20.0% |
| Ranked 13-32 but below threshold | 269 | 16.5% |
| One score source candidate only | 261 | 16.0% |
| Ranked top-12 but below threshold | 205 | 12.5% |
| Ranked 33-64 but below threshold | 199 | 12.2% |
| Multiple-source candidate but not routed | 36 | 2.2% |

## Candidate quality warning

The following values evaluate the full corresponding candidate universes, not only their true-positive members.

| Frozen candidate rule outside Hard | Candidates | TP | FP | Precision |
|---|---:|---:|---:|---:|
| Existing review lane | 664 | 327 | 337 | 49.2% |
| Outside review, support from at least two sources | 144 | 36 | 108 | 25.0% |
| Outside review, one source | 1,783 | 261 | 1,522 | 14.6% |
| Outside review, top-12 but below threshold | 3,592 | 205 | 3,387 | 5.7% |
| Outside review, rank 13-32 below threshold | 8,910 | 269 | 8,641 | 3.0% |
| Outside review, rank 33-64 below threshold | 13,214 | 199 | 13,015 | 1.5% |

No listed group is suitable for direct autonomous promotion. Loosening rank or threshold gates would materially damage the current large-dataset Hard precision of 87.1%.

## Decision

1. Keep Exp291 automatic Hard unchanged.
2. Build a TRAIN-only second-stage verifier for existing review and multi-source candidate universes. Candidate evidence should include source margin, rank percentile, cross-fold stability, and normal-only risk.
3. Build a separate learned representation experiment for the 338 FN outside all audited top-64 rankings. Large TRAIN populations make normal-only representation learning feasible.
4. Do not route by dataset or family name, and do not choose verifier thresholds from TEST performance.

The next experiments should separate `candidate verification` from `representation recovery`; combining both would hide which mechanism provides any gain.
