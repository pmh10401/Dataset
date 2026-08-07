# Exp417 final whole-branch review fix

Status: complete on top of `d348124`.

The official 1,117-dataset population was not run. No TEST-label artifact was
opened, hashed, stated, or queried. All evidence below is synthetic or
repository-local.

## Original ten findings

### 1 and 5 — frozen perturbation and primary-gate semantics

Exact RED reproducers:

- `test_secondary_spike_cannot_rescue_failing_primary_pooled_sensitivity`
  showed that the diagnostic-only spike could make a failing primary pooled
  gate pass.
- `test_secondary_spike_disagreement_is_diagnostic_only` showed that spike
  disagreement incorrectly vetoed otherwise complete primary evidence.
- `test_amplitude_is_centered_and_spike_covers_exact_one_percent` showed
  raw-origin amplitude scaling and the one-sample long-series spike.
- `test_each_perturbation_requires_every_series_seed_trial` showed that an
  incomplete series-by-seed perturbation grid was accepted.
- `test_production_qualification_executes_every_frozen_perturbation_seed`
  showed that production used only the first perturbation seed.

GREEN:

- Only the four frozen primary perturbations enter the primary qualification
  gate; spike is retained as diagnostic evidence only.
- Amplitude scaling operates on the centered waveform.
- Spike covers one deterministic contiguous `ceil(0.01 * length)` segment.
- Every series executes all three frozen perturbation seeds.

### 2 — lane and SQLite descriptor pinning

Exact RED reproducers:

- `test_path_mutation_after_pinned_capture_cannot_change_parsed_evidence`
- `test_lane_hash_and_parse_share_one_pinned_descriptor_across_aba_swap`
- `test_database_query_is_bound_to_verified_inode_across_path_aba`
- `test_sqlite_queries_use_an_immutable_descriptor_uri`

The old path-based sequence accepted bytes or rows from a replacement inode.
GREEN binds lane hash and parsing to one `O_NOFOLLOW` descriptor and opens
SQLite through a verified immutable descriptor URI with stable `fstat`
identity.

### 3 — complete checkpoint trust root

Exact RED reproducers:

- `test_checkpoint_material_digest_covers_qualification_and_backend`
- `test_canonical_result_digest_matches_the_full_persisted_result_payload`
- `test_fully_coherent_qualification_rewrite_fails_checkpoint_payload_root`
- `test_fully_coherent_result_rewrite_fails_external_checkpoint_root`
- `test_trusted_checkpoint_snapshot_cannot_mix_two_complete_roots`

The reduced result digest allowed coherent qualification/backend rewrites.
GREEN anchors the complete canonical `DatasetResult`, the checkpoint
`payload_sha256`, and one locked immutable result/digest snapshot. Schema
version 3 invalidates pre-fix records.

### 4 — diagnostic and abstention state machine

Exact RED reproducers:

- `test_group_states_preserve_undefined_diagnostics_as_none`
- `test_failed_group_is_a_measured_candidate_group_that_misses_a_gate`
- `test_decision_evidence_scores_only_qualified_length_identities`
- `test_diagnostic_preprocessing_is_terminal_before_oof_or_test`
- `test_preprocessing_abstentions_are_accounted_without_test`
- `test_undefined_group_metrics_render_as_blank_and_round_trip`
- `test_legitimate_dataset_workload_abstention_round_trips_officially`
- `test_mixed_diagnostic_checkpoint_accepts_oof_without_a_final_model`
- `test_checkpoint_stage_hashes_exactly_match_each_group_state`

GREEN distinguishes unmeasured preprocessing abstention, 50–99-series
diagnostic OOF, measured qualification failure, and qualified scoring.
Diagnostic groups never load TEST values or fit a final model. Undefined
metrics remain `None`/blank instead of fabricated zeroes, and legitimate
dataset all-or-none abstention remains valid accounting.

### 6 — explicit Git provenance

`test_context_uses_the_explicitly_verified_head_without_cwd_git_lookup` was RED
when context freezing performed a CWD-dependent `git rev-parse HEAD`. GREEN
passes the already verified clean HEAD into the frozen context and binds Git
commands to `REPO_ROOT`. The CLI model-contract hash also covers runner routing
and seed loops.

### 7 — lock-before-snapshot checkpoint writes

`test_writer_lock_precedes_progress_snapshot_and_prevents_stale_rollback`
reproduced the real `_atomic_write_exclusive` interleaving: the second writer
could take a stale progress snapshot and rewind the first commit. GREEN takes
the root exclusive lock before duplicate checks and metadata snapshots,
revalidates under that lock, and makes readers wait for one coherent commit.

### 8 — `BaseException` publication rollback

Exact RED reproducers:

- `test_control_flow_interrupt_after_atomic_rename_rolls_back_output`
- `test_cleanup_base_exception_is_grouped_with_original_interrupt`

A post-rename `KeyboardInterrupt`/`SystemExit` could leave official output.
GREEN catches `BaseException`, continues all cleanup attempts, and preserves
the original interruption plus cleanup errors in `BaseExceptionGroup`.

### 9 — crash quarantine and recovery

Exact RED reproducers cover owned crash markers, temporary files,
record/progress mismatch, missing `datasets`, pathname ABA, and symlink
substitution:

- `test_reopen_quarantines_owned_crash_artifacts_and_preserves_every_byte`
- `test_reopen_quarantines_progress_record_mismatch_and_preserves_record`
- `test_reopen_quarantines_an_owned_root_missing_datasets`
- `test_constructor_validation_cannot_quarantine_from_path_aba`
- `test_quarantine_never_chmods_a_substituted_foreign_path`

GREEN establishes ownership through pinned provenance, atomically quarantines
only the verified owned root inode, preserves every byte, and rejects foreign
provenance without moving or modifying it.

### 10 — disjoint official paths

`test_checkpoint_and_output_ancestry_must_be_disjoint` and
`test_programmatic_publication_rejects_overlapping_checkpoint_ancestry_before_mutation`
were RED for equal and nested checkpoint/output paths. GREEN rejects equality
and both nesting directions before filesystem mutation.

## Additional final-review checkpoint hardening

Every item below was introduced as an exact failing adversarial test against
the then-current implementation and was made GREEN before the next review:

- locked root/child directory ABA and immutable snapshot binding;
- record, progress, provenance, and final-basename substitution before a
  successful writer return;
- cleanup after `BaseException`, stat failure, rollback-probe failure, missing
  temporary source, and competing canonical target;
- preservation of foreign replacements instead of unlinking by basename;
- owned-corruption quarantine for missing datasets and stale metadata;
- metadata-only append reconciliation, eliminating prior-result reconstruction
  and quadratic payload reads;
- initialization through an unpredictable staging root with parent inode-delta
  ownership and no-replace publication.

The last three exact RED/GREEN cycles produced these observed failures:

```text
test_root_initialization_populates_only_its_staging_inode
RED: AssertionError: ValueError not raised
GREEN: Ran 1 test in 0.004s — OK

test_root_publication_rolls_back_a_substituted_staging_basename
RED: AssertionError: True is not false
     (the foreign inode remained at canonical checkpoint)
GREEN: canonical checkpoint absent; foreign inode and bytes preserved under
       a recovery name

test_root_publication_revalidates_staging_payload_after_rename
RED: AssertionError: BaseException not raised
     (corrupt-after-preflight progress was published successfully)
GREEN: constructor fails; canonical checkpoint absent; corrupt root preserved
       under a recovery name
```

Final root publication now revalidates the pinned root inode, exact root
inventory, provenance/progress bytes and inode identities, datasets binding and
empty inventory after rename. Any post-publish failure atomically captures the
canonical target as
`.exp417-checkpoint.rollback-root-publish-target.*.entry`.

The runbook records the staging ownership rule, recovery capsules, and the
portable POSIX threat boundary: an arbitrary malicious same-UID process able to
watch and move entries across unrelated directories is outside the operating
model, so the checkpoint parent must be private to the execution account.

## Focused GREEN evidence

```text
contracts + qualification + inputs + publication + CLI:
Ran 106 tests in 458.097s — OK

reporting:
Ran 12 tests in 153.428s — OK

checkpoint after final publication fixes:
Ran 50 tests in 1.756s — OK
```

The final independent reviewer returned:

```text
CLEAN — no remaining Critical or Important checkpoint findings.
Checkpoint suite: 50/50 passed
git diff --check: passed
py_compile for checkpoint implementation/tests: passed
No TEST labels or official population accessed
```

## Final required regressions

```text
Exp417:
Ran 239 tests in 913.351s — OK

Exp414:
Ran 102 tests in 2.148s — OK

Exp416:
Ran 89 tests in 0.170s — OK (skipped=1, pre-existing platform skip)

compileall: exit 0
git diff --check: exit 0
```

No official checkpoint, prediction, metrics, report, or eight-file
publication was created by this fix wave.
