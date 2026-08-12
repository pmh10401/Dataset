# Task 2 Report — Values-only data boundary and lazy label loader

## Implementation and files

- Added `data_paper.py` with frozen-path/SHA256 verification, a `/dev/fd/<descriptor>` read-only immutable SQLite URI, pre/post-read descriptor identity checks, sorted population loading, normal-only TRAIN loading, and label-free TEST loading.
- Added `label_access_paper.py` as the separate lazy label capability; it accepts only exact TEXT `'0'` and `'1'`, and returns integer tuples through a read-only mapping.
- Extended `contracts_paper.py` with immutable `RawDataset`, retaining contiguous indices, exact blobs, and copied little-endian float32 rows.
- Added real SQLite fixture coverage in `tests_paper/test_data_boundary_paper.py`; updated the complete `codemap_paper` HTML/JSON/lock triplet.

## TDD evidence

- RED: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_data_boundary_paper -v` produced 6 expected errors because `data_paper` and `label_access_paper` did not yet exist (`ImportError: cannot import name ...`).
- GREEN: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_data_boundary_paper experiments.paper_public_baselines.tests.test_data_boundary experiments.paper_public_baselines.tests.test_evaluation_boundary -v` passed 21 tests.
- Additional codemap check: `python3 -m unittest paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper -v` passed 9 tests; `py_compile` of the changed Python files passed.

## Codemap and self-review

- The JSON inventory now exactly includes both production boundary modules and the new test; HTML was regenerated with the established renderer and its SHA256 rebound in the lock.
- Reviewed the query boundary: TEST SQL selects only `instance_index` and `values_blob`; `data_paper.py` contains no `label_access_paper` import; TRAIN is constrained to `label='0'`; bytes decode as copied `<f4` arrays.
- The task adds no score-source experiment or tensor workload, so no CPU worker or PyTorch/MLX/MPS benchmark was run; those assessments remain due when a scoring stage is introduced.

## Concerns

- The required inherited public-baseline suite emitted three pre-existing `ResourceWarning: unclosed database` messages while still passing; this task does not modify that package.
