# Task 5 pre-data retry addendum — 2026-08-04

The first no-clobber root was
`/Users/mac/Documents/Dataset/exp425_task5_mps_parity_20260804_architecture_a`.
Its PID was 32415. It failed in immutable golden preflight with
`golden qualification is not a no-candidate abstention`, before database
snapshot construction, any TRAIN/TEST load, MPS initialization, model work, or
scientific computation. Loader counts are therefore TRAIN=0, TEST values=0,
TEST labels=0.

The preserved failure artifacts are: `parent.log` SHA256
`35686f22ba4875bb6e3e6deab8209675c7d36064c4bad22b0cd778dc68a74ef8`,
`events.jsonl` SHA256
`981b231d429ec20c542701aff30fa601ef14304566c13c88312e4db4fd29af78`, and
`progress.json` SHA256
`0523b34d67e338b67c8b86236549c18d18997d22465e53e8bb22a43099dee2b1`.
They are not modified.

The proposed retry root is
`/Users/mac/Documents/Dataset/exp425_task5_mps_parity_20260804_architecture_a_retry`.
A retry is authorized only because the failure was pre-data and exposed no
adaptive/model result. This addendum authorizes no invocation by itself; a
fresh review must approve the corrected harness first.
