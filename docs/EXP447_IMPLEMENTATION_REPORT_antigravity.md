implementation_status=official_readiness_passed_evaluation_failed_immutable_no_retry

# Exp447 implementation report

## Frozen authority

- approved_design_sha256=`043009941619335b658af99911b1463f66ea73e93d600fe1cfb884391afea868`
- approved_plan_sha256=`d137020a319ec004a6054744a215ffbce9ecef4b6bfb3f2bab5f6f71f0a921d3`
- preregistration_sha256=`a3400c9222c5f3d8e5f5a15a8e38222d62471e0d5d2f2a73a43072df9cb0e6bd`
- runbook_sha256=`2d5110b82322797fb01060db2e24f118cd0ab06aecb9dc9c8fd7fdb4c82dd86d`
- functional_commit=`8f5dbec4e87e093a255d1f065de930b85b141e0c`
- reviewed_implementation_candidate=`f0d5f696b23abb68097f9e1911ce197f3949066e`
- fresh_sol_high_verdict=`ship` (critical=0, important=0)
- evaluation_sha256=`null`
- champion_sha256=`null`
- promotion_sha256=`null`
- test_labels_opened=`false`
- official_label_access_count=`0` completed rows (one label-query attempt opened a session but returned no valid row)
- official_readiness=`true` (one successful invocation)
- run_root_exists=`false`

Exp447 implementation is verified and has a bounded source, evaluator, launcher, and test suite.
The one allowed official readiness invocation passed; the one allowed evaluation invocation then failed
immutably before a valid TEST-label row or result artifact existed. The immutable Exp446 publication is input-only;
its prediction SHA is `602ed1b09da630a8eaf4caa9829c01ae1906450a3edff5c5cbd7cc4453866100` and its
authorized label-free inspect output SHA is
`937b46960ed0b611fc9d3c74bffa7b494ff306af6d78a92129217b0da691fdf0`.

## Governance decision

The planned evaluator is review-only and descriptive. No success criterion is preregistered;
no champion, promotion, operational alert, scientific winner, or same-experiment TEST-informed
reselection is allowed. The exact clean candidate received a fresh Sol High `ship` review with
zero Critical and zero Important findings. The official lifecycle is now permanently closed:
`readiness` ran once and passed, `evaluate` ran once and failed, and `inspect` ran zero times.
No Exp447 command may be rerun.

## Implementation verification evidence

The functional commit passed the Exp447 suite twice: `66/66` in 75.485 seconds (real 75.56)
and `66/66` in 76.280 seconds (real 76.36), each with exit 0. The Exp447-only repair preserved
the previously recorded predecessor regressions: Exp444 `206/206`, Exp445 `39/39`, and Exp446
`50/50`, all exit 0. No TEST labels were opened (`official_label_access_count=0`), the source
database SHA256 remained `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
with inode `1048138` and link count `1`, and the official Exp447 root is absent and not a symlink.

## Official immutable failure evidence

On 2026-08-12, the exact reviewed commit
`33e7dc9c18c179a64bc0020b0bb939c5e4caa394` ran the official lifecycle exactly as follows:

- `readiness --reviewed-commit 33e7dc9c18c179a64bc0020b0bb939c5e4caa394` ran once and exited 0.
  Its detached output reports 1,117 datasets, 2,218 Hard coordinates, label access 0, and an
  absent Exp447 root.
- `evaluate --reviewed-commit 33e7dc9c18c179a64bc0020b0bb939c5e4caa394` ran once and exited 1.
  The TEST-label boundary was entered, its SQLite session opened and later closed, and the first
  query attempt completed no query and returned zero label rows because it raised exactly
  `ValueError: TEST label row is invalid`.
- `inspect` ran zero times and is permanently prohibited: there is no eleven-leaf evaluation
  artifact to inspect, and no evaluation retry is permitted.

The failure audit is a sibling of the absent run root, not a leaf inside it:

```text
/Users/mac/Documents/Dataset/evaluation_audit_89aff47cdd7c43cd919a34f81982bc8b_antigravity.json
```

It is a regular, single-link file with inode `19812443`, link count `1`, and SHA-256
`9f8f7d3d9b92670988ca1d0bb689711d12ef1536777881f18991143a090a2b97`.
Its canonical `exp447_evaluation_audit` payload records `phase=label_snapshot`,
`query_attempts=1`, `query_completions=0`, `label_rows=0`,
`label_session_state=closed`, `label_session_closed=true`, `destination_state=absent`,
`error_type=ValueError`, and `error=TEST label row is invalid`; it also binds the immutable
Exp446 publication identity and all recorded leaf hashes. No valid evaluation, prediction,
champion, or promotion hash exists.

The one post-freeze failure-analysis aggregate query on 2026-08-12 is preserved as a storage
fact, not prediction/readiness evidence: raw SQLite TEST labels are exact `TEXT` values with
`'0' = 221284` and `'1' = 4516`, while `instance_index` is an integer. It used no dataset names
or positions and did not authorize any Exp447 rerun.

Evaluation, champion, and promotion hashes remain `null`; no scientific winner, pending
candidate, promotion, operational alert, or dashboard action was created. The only next action
is a separately numbered Exp448 design and preregistration. It must not run Exp447
`readiness`, `evaluate`, or `inspect` again.

## Deferred private dashboard boundary

No site repository was read, edited, built, committed, deployed, or contacted in Task 1 because
`/Users/mac/.codex/visualizations/2026/08/10/019fedc4-df12-73b2-9812-a5f8f387ac78/research-observatory-antigravity-site`
is dirty for different work. After immutable evaluation plus label-free inspect and independent
truth closeout, a separate exporter may emit only `exp447.public-evaluation.v1` aggregate fields
to the private owner-only observatory. The evaluator imports no dashboard/network code; live status
is a separate exact 12-field `PublicExperimentStatus` allowlist sync with `total=1117`.

No raw TEST labels, instance/anomaly indices, dataset names/lists, local paths, hashes, logs,
checkpoints, DB names, secrets, or per-dataset rows may cross this boundary. Dashboard failure is
presentation-only and cannot change scientific artifacts or authorize evaluation retry/reevaluation.
