# Exp403-408 Portable Data Root Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the restored research data to `/Users/mac/Documents/Dataset` and let Exp403-408 consume it through one portable, explicitly configurable data-root contract without changing frozen predictions or checkpoints.

**Architecture:** Add a focused `experiments.exp403_408.paths` module that resolves `TIMESERIES_DATA_ROOT` once and derives every Exp403-408 input/output path from it. Existing Exp403-408 modules import these constants instead of independently hardcoding the former Mac username. Preserve `/Users/minho/Documents/Dataset` as the unset default so the historical environment remains reproducible.

**Tech Stack:** Python 3.14, `pathlib`, `os.environ`, `unittest`, SQLite, SHA256, Dulwich-backed local Git.

## Global Constraints

- The restored data location is exactly `/Users/mac/Documents/Dataset`.
- `TIMESERIES_DATA_ROOT` is the only new path override.
- When the environment variable is unset, the default remains `/Users/minho/Documents/Dataset`.
- Existing prediction, consensus, database, artifact, and checkpoint contents must not be modified.
- Do not run the full Exp403-408 pipeline.
- Do not add TEST labels, actual anomaly locations, or historical family TEST performance to prediction logic.
- Recover `docs/EXP403_408_PREREGISTRATION_KO.md` only from a verifiable original; never reconstruct preregistration claims from later results.

---

## File Structure

- Create `experiments/exp403_408/paths.py`: sole owner of portable Exp403-408 root and derived path constants.
- Create `experiments/exp403_408/tests/test_paths.py`: subprocess-based tests that prove unset and overridden imports without contaminating the parent test process.
- Modify `experiments/exp403_408/foundation.py`: consume lane and DB paths from `paths.py`.
- Modify `experiments/exp403_408/feature_registry.py`: build every frozen feature source from the shared root.
- Modify `experiments/exp403_408/pipeline.py`: derive its default output directory from the shared root.
- Modify `docs/HANDOFF_20260728_EXP403_408_KO.md`: document the environment variable and this Mac's restored path after implementation verifies it.
- Conditionally restore `docs/EXP403_408_PREREGISTRATION_KO.md`: only if an exact historical Git object or GitHub revision is found.

### Task 1: Move and verify the restored dataset

**Files:**
- Move: `/Users/mac/Documents/Time Series/Dataset`
- Create target: `/Users/mac/Documents/Dataset`

**Interfaces:**
- Consumes: restored DB and artifact tree currently under the repository.
- Produces: immutable input root `/Users/mac/Documents/Dataset`.

- [ ] **Step 1: Prove the source and target state**

Run:

```bash
test -d "/Users/mac/Documents/Time Series/Dataset"
test ! -e "/Users/mac/Documents/Dataset"
shasum -a 256 "/Users/mac/Documents/Time Series/Dataset/univariate_ts.db"
sqlite3 "/Users/mac/Documents/Time Series/Dataset/univariate_ts.db" "PRAGMA quick_check;"
```

Expected: source exists, target does not exist, DB SHA256 is
`8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`,
and SQLite prints `ok`.

- [ ] **Step 2: Move the exact directory**

Run with filesystem approval:

```bash
mv "/Users/mac/Documents/Time Series/Dataset" "/Users/mac/Documents/Dataset"
```

Expected: the target exists and the old source path no longer exists.

- [ ] **Step 3: Verify moved inputs and frozen publications**

Run:

```bash
shasum -a 256 \
  "/Users/mac/Documents/Dataset/univariate_ts.db" \
  "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl" \
  "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp408_consensus_results.jsonl"
sqlite3 "/Users/mac/Documents/Dataset/univariate_ts.db" "PRAGMA quick_check;"
```

Expected SHA256 values in order:

```text
8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70
588b246a7aae3a2cc00fd684731657c816f7a680d22e6899f35b43715b7760da
```

SQLite must print `ok`.

### Task 2: Add one portable Exp403-408 path contract

**Files:**
- Create: `experiments/exp403_408/paths.py`
- Create: `experiments/exp403_408/tests/test_paths.py`
- Modify: `experiments/exp403_408/foundation.py:15,36-52`
- Modify: `experiments/exp403_408/feature_registry.py:19`
- Modify: `experiments/exp403_408/pipeline.py:31,41`

**Interfaces:**
- Consumes: optional environment variable `TIMESERIES_DATA_ROOT`.
- Produces:
  - `resolve_data_root() -> pathlib.Path`
  - `DATA_ROOT: pathlib.Path`
  - `DB_PATH: pathlib.Path`
  - `EXP291_PREDICTION_PATH: pathlib.Path`
  - `EXP291_MANIFEST_PATH: pathlib.Path`
  - `EXP292_PREDICTION_PATH: pathlib.Path`
  - `EXP292_MANIFEST_PATH: pathlib.Path`
  - `DEFAULT_OUTPUT_DIR: pathlib.Path`

- [ ] **Step 1: Write the failing subprocess tests**

Create `experiments/exp403_408/tests/test_paths.py`:

```python
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


REPOSITORY = Path(__file__).resolve().parents[3]
PROBE = """
import json
from experiments.exp403_408 import feature_registry, foundation, paths, pipeline
print(json.dumps({
    "data_root": str(paths.DATA_ROOT),
    "db_path": str(foundation.DB_PATH),
    "exp291": str(foundation.EXP291_PREDICTION_PATH),
    "exp292": str(foundation.EXP292_PREDICTION_PATH),
    "feature_roots": sorted({str(source.root) for source in feature_registry.FEATURE_SOURCES}),
    "output_dir": str(pipeline.DEFAULT_OUTPUT_DIR),
}))
"""


class PortablePathTests(unittest.TestCase):
    def _probe(self, data_root: str | None) -> dict[str, object]:
        environment = dict(os.environ)
        if data_root is None:
            environment.pop("TIMESERIES_DATA_ROOT", None)
        else:
            environment["TIMESERIES_DATA_ROOT"] = data_root
        completed = subprocess.run(
            [sys.executable, "-c", PROBE],
            cwd=REPOSITORY,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(completed.stdout)

    def test_unset_environment_preserves_historical_default(self) -> None:
        result = self._probe(None)
        self.assertEqual(result["data_root"], "/Users/minho/Documents/Dataset")
        self.assertEqual(
            result["db_path"], "/Users/minho/Documents/Dataset/univariate_ts.db"
        )

    def test_override_rebinds_every_exp403_408_path(self) -> None:
        root = "/Users/mac/Documents/Dataset"
        result = self._probe(root)
        self.assertEqual(result["data_root"], root)
        self.assertEqual(result["db_path"], f"{root}/univariate_ts.db")
        self.assertEqual(
            result["exp291"],
            f"{root}/exp291_final_operational_routing_freeze/"
            "exp291_frozen_operational_routing.jsonl",
        )
        self.assertEqual(
            result["exp292"],
            f"{root}/exp292_standard_review_evidence_tiering/"
            "exp292_frozen_review_tiers.jsonl",
        )
        self.assertTrue(
            all(str(path).startswith(f"{root}/") for path in result["feature_roots"])
        )
        self.assertEqual(
            result["output_dir"],
            f"{root}/exp403_408_feature_ml_validation",
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m unittest \
  experiments.exp403_408.tests.test_paths -v
```

Expected: error or failure because `experiments.exp403_408.paths` does not exist
and existing modules still use `/Users/minho/Documents/Dataset`.

- [ ] **Step 3: Implement the minimal path module**

Create `experiments/exp403_408/paths.py`:

```python
"""Portable filesystem paths for the frozen Exp403-408 workflow."""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "TIMESERIES_DATA_ROOT"
HISTORICAL_DATA_ROOT = Path("/Users/minho/Documents/Dataset")


def resolve_data_root() -> Path:
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured is None:
        return HISTORICAL_DATA_ROOT
    return Path(os.path.expandvars(configured)).expanduser().resolve(strict=False)


DATA_ROOT = resolve_data_root()
DB_PATH = DATA_ROOT / "univariate_ts.db"
EXP291_PREDICTION_PATH = (
    DATA_ROOT
    / "exp291_final_operational_routing_freeze"
    / "exp291_frozen_operational_routing.jsonl"
)
EXP291_MANIFEST_PATH = EXP291_PREDICTION_PATH.with_name(
    "exp291_prediction_manifest.json"
)
EXP292_PREDICTION_PATH = (
    DATA_ROOT
    / "exp292_standard_review_evidence_tiering"
    / "exp292_frozen_review_tiers.jsonl"
)
EXP292_MANIFEST_PATH = EXP292_PREDICTION_PATH.with_name(
    "exp292_prediction_manifest.json"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "exp403_408_feature_ml_validation"
```

Modify `foundation.py` to import `DB_PATH`, `DATA_ROOT`, and the four Exp291/292
paths from `.paths`; remove its local `DATA_ROOT` and derived path definitions.

Modify `feature_registry.py` to import `DATA_ROOT` from `.paths`; remove its
local hardcoded `DATA_ROOT`.

Modify `pipeline.py` to import `DEFAULT_OUTPUT_DIR` from `.paths`; remove its
local hardcoded definition. Continue importing the lane and DB constants from
`foundation.py` so existing test patches remain compatible.

- [ ] **Step 4: Run the new tests and verify GREEN**

Run:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m unittest \
  experiments.exp403_408.tests.test_paths -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Run focused existing path consumers**

Run:

```bash
TIMESERIES_DATA_ROOT=/Users/mac/Documents/Dataset \
PATH="$PWD/.venv/bin:$PATH" \
.venv/bin/python -m unittest \
  experiments.exp403_408.tests.test_foundation \
  experiments.exp403_408.tests.test_consensus \
  experiments.exp403_408.tests.test_prediction_boundary -v
```

Expected: all focused tests pass and no `FileNotFoundError` mentions
`/Users/minho/Documents/Dataset`.

- [ ] **Step 6: Commit the path contract**

```bash
git add \
  experiments/exp403_408/paths.py \
  experiments/exp403_408/foundation.py \
  experiments/exp403_408/feature_registry.py \
  experiments/exp403_408/pipeline.py \
  experiments/exp403_408/tests/test_paths.py
git commit -m "fix: make Exp403-408 data root portable"
```

### Task 3: Diagnose and recover the missing preregistration source

**Files:**
- Conditionally create: `docs/EXP403_408_PREREGISTRATION_KO.md`

**Interfaces:**
- Consumes: Git object history, GitHub branch history, PR 12 metadata/files.
- Produces: exact original document, or a verified finding that no original is available.

- [ ] **Step 1: Query GitHub history for the exact path**

Run:

```bash
curl -fsSL \
  'https://api.github.com/repos/pmh10401/Dataset/commits?path=docs%2FEXP403_408_PREREGISTRATION_KO.md&per_page=100'
curl -fsSL \
  'https://api.github.com/repos/pmh10401/Dataset/pulls/12/files?per_page=100'
```

Expected: structured GitHub JSON. Record whether any commit or PR file has the
exact path.

- [ ] **Step 2: Search local Git objects for the exact filename**

Run a read-only Dulwich tree walk over all fetched refs and print matching
commit IDs and blob IDs. Expected: either at least one exact match or an empty
result; do not infer content from similarly named documents.

- [ ] **Step 3: Restore only a verified blob**

If an exact blob is found, use the GitHub contents endpoint or Dulwich object
store to restore its bytes unchanged to
`docs/EXP403_408_PREREGISTRATION_KO.md`, then verify its blob SHA against the
source. If no exact blob exists, do not create the file and record the missing
source in the final report.

- [ ] **Step 4: Commit only when an original was restored**

```bash
git add docs/EXP403_408_PREREGISTRATION_KO.md
git commit -m "docs: restore Exp403-408 preregistration"
```

Expected: skip this step when no verified original exists.

### Task 4: Document the portable resume command

**Files:**
- Modify: `docs/HANDOFF_20260728_EXP403_408_KO.md`

**Interfaces:**
- Consumes: `TIMESERIES_DATA_ROOT` contract from Task 2.
- Produces: exact activation and verification commands for this Mac.

- [ ] **Step 1: Add the new environment contract**

Add this block immediately before “복원 검증”:

```markdown
## 다른 사용자 계정의 데이터 경로

데이터를 다른 위치에 복원한 경우 Exp403-408 명령 전에 데이터 루트를
명시한다. 환경변수가 없으면 역사적 기본 경로
`/Users/minho/Documents/Dataset`를 유지한다.

```bash
export TIMESERIES_DATA_ROOT="/Users/mac/Documents/Dataset"
```
```

Update the verification command examples in the handoff section to use:

```bash
shasum -a 256 "$TIMESERIES_DATA_ROOT/univariate_ts.db"
sqlite3 "$TIMESERIES_DATA_ROOT/univariate_ts.db" "PRAGMA quick_check;"
```

- [ ] **Step 2: Add a documentation regression assertion**

Extend `test_override_rebinds_every_exp403_408_path` to read the handoff
document and assert that it contains both `TIMESERIES_DATA_ROOT` and
`/Users/mac/Documents/Dataset`.

- [ ] **Step 3: Run the documentation/path test**

Run:

```bash
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m unittest \
  experiments.exp403_408.tests.test_paths -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit the handoff update**

```bash
git add \
  docs/HANDOFF_20260728_EXP403_408_KO.md \
  experiments/exp403_408/tests/test_paths.py
git commit -m "docs: document portable Exp403-408 restore path"
```

### Task 5: Full verification without running the pipeline

**Files:**
- Verify only: `/Users/mac/Documents/Dataset`
- Verify only: `experiments/exp403_408/tests`

**Interfaces:**
- Consumes: completed Tasks 1-4.
- Produces: evidence-backed restore and checkpoint reuse report.

- [ ] **Step 1: Run syntax and full unit verification**

Run outside the restricted process sandbox so Python process semaphores work:

```bash
TIMESERIES_DATA_ROOT=/Users/mac/Documents/Dataset \
PATH="$PWD/.venv/bin:$PATH" \
.venv/bin/python -m py_compile \
  experiments/exp403_408/*.py \
  scratch/run_exp403_408_feature_ml_validation.py

TIMESERIES_DATA_ROOT=/Users/mac/Documents/Dataset \
PATH="$PWD/.venv/bin:$PATH" \
.venv/bin/python -m unittest discover \
  -s experiments/exp403_408/tests -p "test_*.py"
```

Expected: syntax command exits 0; the full suite exits 0 with no failures or
errors. The existing MPS availability skip is acceptable.

- [ ] **Step 2: Verify handoff hashes**

Run:

```bash
shasum -a 256 \
  "/Users/mac/Documents/Dataset/univariate_ts.db" \
  "/Users/mac/Downloads/univariate_ts.db.zst" \
  "/Users/mac/Downloads/exp403_408_required_artifacts.tar.zst" \
  "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl" \
  "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp408_consensus_results.jsonl"
sqlite3 "/Users/mac/Documents/Dataset/univariate_ts.db" "PRAGMA quick_check;"
```

Expected hashes:

```text
8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
8c0e342c6e0dc8d372544ee7140d8e465585617d420bf932ba68b62f18619441
c503a37e052808e2ad13c52247cfe004500aafe99f66a7bdc690bb669847f8bf
437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70
588b246a7aae3a2cc00fd684731657c816f7a680d22e6899f35b43715b7760da
```

SQLite must print `ok`.

- [ ] **Step 3: Verify checkpoint reuse evidence read-only**

For each stage below, count non-AppleDouble JSON checkpoint files:

```text
audit-features       1117
classical-predict    1117
ssl-predict          1117
synthetic-predict    1117
consensus-evaluate   1117
```

Also verify `MANIFEST.json` reports:

```text
prediction_sha256 = 437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70
independent_reaggregation.mismatch_count = 0
```

Do not invoke `scratch/run_exp403_408_feature_ml_validation.py --stage all`.

- [ ] **Step 4: Verify repository state**

Run:

```bash
git status --short --branch
git diff --check
git log -4 --oneline
```

Expected: no uncommitted source changes; recent commits contain the design,
portable path fix, and handoff documentation update. A preregistration commit
appears only if Task 3 found an exact original.
