# Project Execution Rules

## CPU worker policy

CPU-parallel experiment stages must always use the detected logical CPU count
minus one worker, with a minimum of one worker. Record the detected CPU count
and resulting worker count in the run provenance. Historical frozen experiments
keep their recorded worker counts; apply this policy to new or not-yet-started
official runs.

## PyTorch, MLX, and MPS acceleration assessment policy

Every experiment design must explicitly assess PyTorch, MLX, and MPS. Record
whether the workload is tensor-eligible, current backend availability,
transfer/compilation overhead, deterministic and numerical parity
requirements, the canonical CPU fallback, and the benchmark required before
an accelerated path becomes canonical.

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

If any condition fails, retain the existing CPU implementation and use the CPU
worker policy above where valid. Do not rewrite an official third-party CPU
transformer merely to claim GPU use because that changes the experiment.

Run only one MLX/MPS-heavy process at a time. Do not combine outer multi-process
parallelism with concurrent MLX GPU jobs unless a separate benchmark proves it
is faster and deterministic.

Current Exp254 RandomInterval, MatrixProfile, SAX, and SFA transforms use the
official sktime CPU implementations. Exp254 remains CPU-only for comparability.
Future MLX-eligible stages include dense distance matrices, batched projections,
tensor feature fusion, and neural embeddings when parity and speed gates pass.

## Native Delegation Policy

For native delegated implementation, use `sol_advisor_terra_implementer`
(GPT-5.6 Terra / High). For design, commitment-boundary review, and final review,
use a fresh `sol_advisor_sol_reviewer` (GPT-5.6 Sol / High).

GPT-5.6 Luna / Max is not used for native delegation. Use it only when the user
explicitly opts into the separate Luna task lane for that specific request.

## File Naming Policy

All newly created files across all tasks and experiments must include `antigravity` in their file name (e.g., `antigravity_<name>.<ext>`, `<name>_antigravity.<ext>`).

## Human-readable document policy

Every newly created human-readable document is self-contained .html and has no Markdown duplicate; JSON, lock, and code files are unaffected; Historical documents are not mass-converted.

In HTML implementation plans, mark a task or step checkbox checked immediately
after its required implementation and review/verification boundary is complete;
never pre-check pending work.

## Code Map Synchronization

At the start of every code-changing task, resolve the current experiment version
from the active research state or the user-declared version, then compare the
repo with its versioned triplet:
`docs/codemap/codemap_<version>_antigravity.html`,
`docs/codemap/codemap_<version>_antigravity.json`, and
`docs/codemap/codemap_<version>_antigravity.lock`. For Exp444 the canonical
triplet is `docs/codemap/codemap_exp444_antigravity.html`,
`docs/codemap/codemap_exp444_antigravity.json`, and
`docs/codemap/codemap_exp444_antigravity.lock`.

Before modifying a module, use the current version's `codemap_<version>_antigravity.json`
to answer three questions:

1. What calls it?
2. What does it affect?
3. Which tests cover it?

If the map is stale or cannot answer those questions, regenerate all three
versioned files together before changing the code. A new experiment/version
creates its own versioned triplet; it never overwrites a prior version map.

Whenever module boundaries, dependencies, routes, databases, queues, or major data flows change, update that version's complete triplet in the same commit as the code.
