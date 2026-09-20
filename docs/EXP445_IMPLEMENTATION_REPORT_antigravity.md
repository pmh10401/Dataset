# Exp445 implementation governance record

## Current status

`Exp445` is `implementation_verified_awaiting_official_readiness`. A fresh Sol
High clean-HEAD review shipped the reviewed candidate; official readiness is the
next gated action. This is not official execution evidence: readiness, `new`,
publication, evaluation, TEST-label access, and promotion remain unperformed.

## Frozen authority

- Design SHA256: `099c0d412d2375764a7e30440feca17d70a2a6bbfdd14e3e53999fe3c458d70d`
- Plan SHA256: `4b369f044456ce9e3cca0abad141beb28b5b3431691677f19a2900997dc5ac9a`
- Run identity: `exp445-exp444-publication-source-authority-repair-antigravity-20260810`
- Fixed run root: `/Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810`
- Exp444 immutable predecessor commit: `f4ae53c46422a46cc880d5d204f68a6a3971e92d`
- Exp444 ordered 30-record payload SHA256: `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`
- Backend: inherited CPU only; CPU/11 outer workers on 12 logical CPUs and one inner numerical thread.

The pending-candidate field remains empty and the verified champion remains
unchanged. Prediction, publication, and evaluation hashes are null. Exp444 and
Exp442 external artifacts are not touched by this governance record.

## Implementation verification evidence

- Functional implementation commit: `c8a0ad351d701b11df69fac66bd7c95284e644be`
- Functional map commit: `477399062818b49dcc01101e3a83e723c8ec7d44`
- Fresh Sol High clean-HEAD review: candidate
  `09903c608fb78508ae8fdd343594e3d99dfee660`, verdict `ship`.
- Controller-captured full regression: Exp444 `205` tests in `195.955` seconds
  and Exp445 `39` tests in `1.397` seconds; both `OK`, sentinels
  `exp444=0 exp445=0`, outer exit `0` (total `244`). This includes the fixed
  profile, real predecessor, real provenance-to-values-only composition, closed
  CLI, and parity coverage.
- Read-only CLI smoke: help exit `0`, forbidden `resume` exit `2`, `0.133967`
  seconds, with the fixed root absent.
- Temporary 30-record sequential/11-worker parity smoke: canonical SHA256
  `267449fa3a4d00e3d214ca3fa5b5f38c1f73ecd07ec3d33545003e0b92a76b30`,
  `0.341842` seconds.
- A standalone real-composition attempt had a lost output handle, so it is
  inconclusive and is not claimed as success; its unowned temporary DB pins
  were removed. The captured full regression is the evidence used for that
  exact test method.
- Post-check DB inode is `1048138` with link count `1`; the fixed Exp445 root
  is absent. No TEST label was opened.

## Implementation boundary

The implementation exposes exactly four commands: `readiness`,
single-use `new`, read-only Exp445-only `inspect`, and post-double-inspection
`evaluate`. It has no resume or override surface. Exact record parity is required
before publication; record, failure, publication, and evaluation artifacts after
root allocation remain create-only evidence. The only same-run exception is that
checkpoint state, progress, and PID retain the existing frozen atomic-replacement
behavior. Evaluation is descriptive-only and cannot promote a champion.

```text
same-run mutable artifacts = checkpoint state, progress, PID (atomic replacement only)
create-only artifacts = record, failure, publication, evaluation
```
