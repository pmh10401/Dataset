# Exp437 Task 7 implementation report

## Reviewed implementation and scope

- Worktree: `/Users/mac/Documents/Time Series/.worktrees/exp425-selective-recovery`
- Reviewed implementation commit: `d977dee17e5d870ff5f84078647fb93b5231a725`
- Task 7 adds this report and readiness-handoff evidence only; no product code,
  readiness artifact, prediction output, publication, evaluation, or Exp437 run
  root was created.
- The reviewed commit is the clean implementation base. The eventual handoff
  commit is intentionally not written into `last_verified_commit`; state points
  to the implementation commit under review, not to a self-referential future
  documentation commit.

## Fix Round 1: provenance validation

The first handoff review found a trailing `d` in the recorded base hash. A
focused validator now checks every recorded reviewed/last-verified/provenance
commit field for exactly 40 lowercase hexadecimal characters and verifies the
object with `git cat-file -e <hash>^{commit}` before accepting the handoff:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 - <<'PY'
import json, re, subprocess
from pathlib import Path

expected = "d977dee17e5d870ff5f84078647fb93b5231a725"
state = json.loads(Path("RESEARCH_STATE.json").read_text())
catalog = json.loads(Path("EXPERIMENT_CATALOG.json").read_text())
exp437 = next(item for item in catalog["experiments"] if item["id"] == "Exp437")
values = {
    "state.last_verified_commit": state["last_verified_commit"],
    "state.exp437_task7_handoff.reviewed_implementation_commit": state["exp437_task7_handoff"]["reviewed_implementation_commit"],
    "catalog.Exp437.reviewed_implementation_commit": exp437["reviewed_implementation_commit"],
    "codemap.generated_from_commit": json.loads(Path("docs/codemap/codemap.json").read_text())["generated_from_commit"],
    "codemap.lock.current_commit": json.loads(Path("docs/codemap/codemap.lock").read_text())["current_commit"],
}
assert all(re.fullmatch(r"[0-9a-f]{40}", value) for value in values.values())
assert all(value == expected for value in values.values())
assert all(subprocess.run(["git", "cat-file", "-e", f"{value}^{{commit}}"], check=False).returncode == 0 for value in values.values())
print("provenance validation: PASS", len(values), "recorded commit fields")
PY
```

Result: `provenance validation: PASS 5 recorded commit fields`. The tracked
implementation report and ignored SDD report both now carry the exact 40-byte
base hash; abbreviated prose references (`d977dee1`) are descriptive only.

## Regression evidence

The exact commands from the Task 7 brief were run with the existing
`PYTHONDONTWRITEBYTECODE=1` governance.

### Complete Exp436 shared-engine suite

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests \
  -p 'test_*_antigravity.py' -v
```

Result: `Ran 299 tests in 122.880s`, `OK`; exit 0, failures 0, errors 0,
skips 0.

### Predecessor boundary regressions

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_checkpoint_antigravity \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_progress_antigravity \
  experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_runtime_antigravity \
  experiments.exp425_selective_recovery.tests.test_database -v
```

Result: `Ran 68 tests in 4.571s`, `OK`; exit 0, failures 0, errors 0,
skips 0. The database benchmark line was
`256MiB snapshot=0.136s verified_opens=2.587s`.

## Immutable Exp436 failure proof

The exact readiness hash command returned:

```text
5def8dbe785163dc69e80e0bb6a5a1cdba1d3663058f4d73625a6deda048c43a  /Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/readiness_antigravity.json
```

The exact read-only parser proof opened
`checkpoint_antigravity/records_antigravity/1f10eac76bc530cbb48ab776cf8fe1817d08acab7ce4976ccf2c58d0d4c04af0.json`, parsed
`AllGestureWiimoteX_normal_1`, printed `AllGestureWiimoteX_normal_1 59498`,
and asserted that the bytes before and after parsing were identical. The
lexicographic histogram object-key order is now treated as semantic map data,
not numeric insertion order; no failed-root file was written, repaired, or
deleted. The Exp436 root, 47-record authority, orphan record, and readiness
artifact remain immutable historical evidence.

## Exp437 root, wrapper, and forbidden-term firewall

```text
test ! -e /Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807  -> PASS
```

The exact wrapper help output exposes only the five fixed commands:
`readiness`, `new`, `resume`, `inspect`, and `evaluate`; no override flags are
available. The forbidden-term grep returned only the following deliberate
evidence fields and zero-valued delayed-evaluation guards:

| Source symbol | Classification |
| --- | --- |
| `cli_antigravity._READINESS_KEYS` | Schema names `test_values_opened`, `test_labels_opened`, and `label_access_count`; counters are evidence fields, not label reads. |
| `cli_antigravity._validate_readiness_payload` | Rejects readiness unless evaluation is absent and all three access counters are zero. |
| `cli_antigravity._production_readiness` | Emits zero-valued access counters in the pre-prediction readiness binding. |
| `runtime_antigravity._READINESS_KEYS` | Mirrors the same readiness evidence schema. |
| `runtime_antigravity._canonical_readiness` | Delayed-evaluation guard rejects nonzero value/label access before runtime execution. |

There were no matching terms in the Exp437 wrapper. No grep hit is a query for
TEST labels, anomaly locations, ground truth, or family-level TEST performance.

## Frozen configuration, topology, and accelerator policy

The profile projection comparison produced exactly these changed fields:

```text
experiment_id: 'Exp436' -> 'Exp437'
run_identity: 'exp436-missing-safe-strict-consensus-antigravity-20260806' -> 'exp437-metadata-roundtrip-antigravity-20260807'
```

The changed-key set is exactly `{experiment_id, run_identity}`; all scientific
fields are equal. CPU topology is `logical_cpus=12`, `workers=11`,
`inner_threads=1`, matching `max(1, logical_cpus - 1)`.

Official MLX/MPS readiness remains pending. Task 7 did not invoke readiness or
start an accelerator-heavy process, and makes no accelerator adoption claim.
The existing local MLX probe remains the recorded CPU fallback because its best
observed speedup was below the 1.1x gate; the frozen CPU implementation remains
canonical until parity, tolerance, provenance, and end-to-end speed gates all
pass. Prediction started is `false`; TEST labels opened is `false`.

## State, catalog, and code-map handoff

`RESEARCH_STATE.json` and `EXPERIMENT_CATALOG.json` retain
`implementation_complete_readiness_pending`, `prediction_phase_started=false`,
`test_labels_opened=false`, and `official_readiness=false`. They record the
fresh suite counts, immutable hash/record proof, exact two-field config diff,
12/11/1 topology, absent Exp437 root, and this report. The exact Task 7 command
is:

```bash
python3 scratch/run_exp437_metadata_roundtrip_antigravity.py readiness
```

The state continuation command preserves the repository governance prefix:
`PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp437_metadata_roundtrip_antigravity.py readiness`.
No readiness invocation was made in this task.

The 20-node, 50-edge, 11-flow codemap triplet was regenerated together from
the same pre-handoff implementation commit and now includes the report/state
handoff evidence under the existing research-control and Exp436 boundary nodes.
