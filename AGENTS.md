# Project Execution Rules

## MLX acceleration policy

For every new score-source experiment, including non-learning feature and score
pipelines, assess MLX acceleration before the full run.

Use MLX by default only when all of the following are true:

1. The exact operation can be expressed with MLX without changing the frozen
   algorithm, feature definition, score definition, threshold, or routing rule.
2. A deterministic CPU-versus-MLX smoke uses the same input and seed.
3. Both paths produce finite outputs and identical candidate indices at every
   preregistered alpha. Numerical score differences must also remain within the
   preregistered tolerance.
4. MLX is at least 1.1x faster for the exact stage, including device transfer and
   synchronization time.
5. The manifest records MLX and mlx-metal versions, device, benchmark timing,
   parity result, and fallback reason.

If any condition fails, retain the existing CPU implementation and use up to
seven dataset workers where valid. Do not rewrite an official third-party CPU
transformer merely to claim GPU use because that changes the experiment.

Run only one MLX/MPS-heavy process at a time. Do not combine outer multi-process
parallelism with concurrent MLX GPU jobs unless a separate benchmark proves it
is faster and deterministic.

Current Exp254 RandomInterval, MatrixProfile, SAX, and SFA transforms use the
official sktime CPU implementations. Exp254 remains CPU-only for comparability.
Future MLX-eligible stages include dense distance matrices, batched projections,
tensor feature fusion, and neural embeddings when parity and speed gates pass.
