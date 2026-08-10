# Exp445 implementation governance record

## Current status

`Exp445` is `design_approved_implementation_planned`. This record freezes the
publication-source-authority recovery governance and operator contract only. It
does not authorize or evidence official readiness, `new`, `inspect`, `evaluate`,
publication, evaluation, TEST-label access, or promotion.

## Frozen authority

- Design SHA256: `099c0d412d2375764a7e30440feca17d70a2a6bbfdd14e3e53999fe3c458d70d`
- Plan SHA256: `4b369f044456ce9e3cca0abad141beb28b5b3431691677f19a2900997dc5ac9a`
- Run identity: `exp445-exp444-publication-source-authority-repair-antigravity-20260810`
- Fixed run root: `/Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810`
- Exp444 immutable predecessor commit: `f4ae53c46422a46cc880d5d204f68a6a3971e92d`
- Exp444 ordered 30-record payload SHA256: `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`
- Backend: inherited CPU only; CPU/11 outer workers on 12 logical CPUs and one inner numerical thread.

The pending-candidate field remains empty and the verified champion remains
unchanged. Prediction and evaluation hashes are null. Exp444 and Exp442 external
artifacts are not touched by this governance record.

## Implementation boundary

The future implementation must expose exactly four commands: `readiness`,
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
