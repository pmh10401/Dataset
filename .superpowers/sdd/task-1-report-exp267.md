# Task 1 Report: Exp267-273 Frozen Configuration, Dependencies, And Contracts

## Status

DONE

The original dependency concern was resolved in the reviewer-fix pass below.

## Implementation

Created the new `experiments.exp267_273_zero_support_fn_recovery` package with:

- Frozen policy constants: `SEED`, `SECOND_SEED`, `ALPHAS`, `PRIMARY_ALPHA`,
  `RFF_MIN_TRAIN`, `TCN_MIN_TRAIN`, `REVIEW_RISK_LIMIT`, `HARD_RISK_LIMIT`,
  and `MIN_RETROSPECTIVE_PRECISION`.
- Immutable `SourceSpec`, `SourceScore`, `FrozenSourcePrediction`,
  `RecoveryLanes`, and `PromotionAudit` dataclasses.
- A label-free `FrozenSourcePrediction` contract with exactly the requested
  freeze fields and `contains_test_labels=False` enforcement.
- Immutable existing Hard, Shadow, Priority, and Standard lane indices plus
  Recovery Shadow and Recovery Priority indices.
- `dependency_report()` using package metadata and PyTorch MPS capability
  reporting, without importing labels or database rows.

## Files

Created by this task:

- `experiments/exp267_273_zero_support_fn_recovery/__init__.py`
- `experiments/exp267_273_zero_support_fn_recovery/config.py`
- `experiments/exp267_273_zero_support_fn_recovery/contracts.py`
- `experiments/exp267_273_zero_support_fn_recovery/dependencies.py`
- `experiments/exp267_273_zero_support_fn_recovery/tests/test_contracts.py`
- `.superpowers/sdd/task-1-report-exp267.md`

No Exp253 or Exp266 artifacts were changed. Existing dirty `.superpowers`
files were not edited or staged.

## RED/GREEN Evidence

### RED

Command:

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_contracts -v
```

Result before implementation: exit 1, with the expected missing-package
failure:

```text
ModuleNotFoundError: No module named 'experiments.exp267_273_zero_support_fn_recovery.config'
Ran 1 test in 0.000s
FAILED (errors=1)
```

### GREEN

Command:

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_contracts -v
```

Fresh final result:

```text
Ran 5 tests in 0.000s
OK
```

The five passing tests cover the exact policy constants, forbidden evaluation
fields, label-free prediction default, frozen dataclass records, and lane
field contract.

Additional verification:

```bash
python3 -m py_compile experiments/exp267_273_zero_support_fn_recovery/__init__.py experiments/exp267_273_zero_support_fn_recovery/config.py experiments/exp267_273_zero_support_fn_recovery/contracts.py experiments/exp267_273_zero_support_fn_recovery/dependencies.py experiments/exp267_273_zero_support_fn_recovery/tests/test_contracts.py
```

Result: exit 0.

The unscoped `git diff --check` also inspected unrelated pre-existing dirty
content and reported only:

```text
.superpowers/sdd/progress.md:6: new blank line at EOF.
```

The package files themselves have no whitespace errors; the unrelated file was
left untouched.

## Dependency Status

Environment before the requested installation:

```text
Python 3.14.6
kymatio=MISSING (No package metadata was found for kymatio)
iisignature=MISSING (No package metadata was found for iisignature)
torch=2.13.0
mps_built=True
mps_available=True
```

Requested command:

```bash
python3 -m pip install --upgrade kymatio iisignature
```

Exact result: exit 1 while building `iisignature` requirements. The build
environment raised:

```text
ModuleNotFoundError: No module named 'numpy'
ERROR: Failed to build 'iisignature' when getting requirements to build wheel
```

After the failed install, both `kymatio` and `iisignature` remained missing.
Calling the implemented audit produced:

```text
PackageNotFoundError: No package metadata was found for kymatio
```

The audit contract was kept as specified; no alternate algorithm or dependency
substitution was introduced. A separate `BLOCKER_REPORT.md` was not created
because the brief restricts edits to the listed files and this report path.

## Self-Review

- The RED failure was observed before production implementation.
- The focused GREEN suite and Python compilation passed after implementation.
- `FrozenSourcePrediction` has no `y_test`, `tp`, `fp`, `fn`, `f1`, or
  `precision` fields and rejects `contains_test_labels=True`.
- The dependency module imports only `importlib.metadata` and `torch`; it does
  not read labels, SQLite state, or experiment-local modules.
- The package contains only the five brief-listed source/test files, apart from
  the required report.
- Existing dirty `.superpowers` content and Exp253/Exp266 artifacts remain
  outside the change set.

## Concerns

- Python 3.14 dependency installation failed for `iisignature`; the downstream
  scattering/signature stages cannot run until the dependency environment is
  repaired.
- `dependency_report()` therefore cannot return a complete version mapping in
  this environment, but it preserves the exact requested audit behavior.
- This task freezes contracts only. Source algorithms, population loading,
  scoring, routing, evaluation, and promotion audit logic remain for later
  tasks.

## Reviewer Fixes

### Fix Scope

The reviewer-requested Task 1 fixes were applied without changing any Exp253 or
Exp266 artifact or implementing downstream algorithms:

- Resolved the Python 3.14 dependency blocker. `kymatio==0.3.0` and
  `iisignature==0.24` are installed; the latter was built with
  `--no-build-isolation` and the macOS 26.5 SDK C++ include path. No algorithm
  substitution was made, and no `BLOCKER_REPORT.md` is needed.
- Added explicit calibration eligibility constants for `n_train < 5`
  abstention: `MIN_TRAIN_FOR_CALIBRATION=5`, `CALIBRATION_MIN_TRAIN=5`, and
  `CALIBRATION_ABSTENTION_REASON="n_train_below_5"`.
- Made every `RecoveryLanes` field required. Construction now defensively
  converts all index collections to sorted tuples, rejects duplicate or
  negative indices, and rejects pairwise lane overlap.
- Deeply freezes `SourceScore.provenance` mappings and nested sequences, and
  defensively normalizes `FrozenSourcePrediction.candidate_indices` to a sorted
  tuple.
- Added focused tests for the exact DB path, installed dependency versions,
  eligibility constants, deep read-only provenance, required lane fields,
  index normalization, and overlap rejection.

### Fix RED

Command after adding the reviewer regression tests and before the production
fixes:

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_contracts -v
```

Result: exit 1 at the expected missing calibration contract import:

```text
ImportError: cannot import name 'CALIBRATION_ABSTENTION_REASON'
Ran 1 test in 0.000s
FAILED (errors=1)
```

After the implementation was added, the first fix run reached the intended
immutability assertion and exposed an over-specific test expectation:

```text
ERROR: test_recovery_lanes_require_all_fields_and_normalize_indices
dataclasses.FrozenInstanceError: cannot assign to field 'hard_indices'
Ran 11 tests in 0.020s
FAILED (errors=1)
```

The test was corrected to assert `dataclasses.FrozenInstanceError`, matching
the frozen dataclass contract.

### Fix GREEN

Focused command:

```bash
python3 -m unittest experiments.exp267_273_zero_support_fn_recovery.tests.test_contracts -v
```

Result:

```text
Ran 11 tests in 0.023s
OK
```

Compilation command:

```bash
python3 -m py_compile experiments/exp267_273_zero_support_fn_recovery/__init__.py experiments/exp267_273_zero_support_fn_recovery/config.py experiments/exp267_273_zero_support_fn_recovery/contracts.py experiments/exp267_273_zero_support_fn_recovery/dependencies.py experiments/exp267_273_zero_support_fn_recovery/tests/test_contracts.py
```

Result: exit 0.

### Resolved Dependency Verification

Command:

```bash
python3 - <<'PY'
from experiments.exp267_273_zero_support_fn_recovery.dependencies import dependency_report
print(dependency_report())
PY
```

Result:

```text
{'kymatio': '0.3.0', 'iisignature': '0.24', 'torch': '2.13.0', 'mps_built': True, 'mps_available': True}
```

The focused dependency test independently compared the reported versions with
`importlib.metadata.version(...)` and passed. Exact frozen policy verification
also returned:

```text
db_path=/Users/minho/Documents/Dataset/univariate_ts.db
calibration_min_train=5
min_train_for_calibration=5
calibration_abstention_reason=n_train_below_5
```

The dependency concern in the original report is resolved. The remaining
concern is limited to later tasks consuming these structural contracts; no
production source algorithm has been substituted or run by Task 1.
