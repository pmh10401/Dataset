STATUS: complete
OBJECTIVE: Restore the frozen Exp249 MiniRocket-only nonfinite preprocessing before later batching/checkpoint work.
CHANGES:
- `preprocessing_paper.py`: added float32 original-grid linear interpolation with nearest finite edge extension and fail-closed empty/all-nonfinite validation.
- `minirocket_paper.py`: repairs every scored TRAIN and TEST row before alignment/z-normalization and raises a dataset/split/row-local typed input exception when repair fails.
- `test_minirocket_paper.py`: added Exp443 byte parity fixtures and a MiniRocket boundary regression proving repaired finite matrices reach scoring.
- `source_provenance_paper.json`: bound changed helper hashes and the Exp443 repair origin.
- Both Exp448 and paper codemap triplets: regenerated caller/effect/test coverage, embedded HTML, and hash locks.
VERIFIED:
- RED: `python3 -m unittest ...test_linear_edge_nearest_repair_is_byte_identical_to_exp443_policy_a ...test_confirmations_repair_train_and_test_before_alignment` failed with missing `repair_linear_edge_nearest_paper` and observed nonfinite scoring matrices.
- GREEN: the same focused command passed 2 tests; default and capable MiniRocket suites passed 6 tests each.
- Capable bounded smoke: `DodgerLoopDay_normal_1` loaded values-only and emitted finite 1024/4096 MiniRocket outputs for 19 TRAIN and 50 TEST rows without an aeon missing-value exception.
- Paper provenance/codemap and relevant hard-source/config-lineage validation passed; final structural/hash checks are recorded with the Task 1 commit.
JUDGMENT CALLS:
- The repair itself preserves Exp443 `ValueError` semantics for byte-level formula parity; the MiniRocket caller wraps invalid rows in `MiniRocketNonfiniteInputPaper` so failure is typed and dataset-local.
- The `< MINIMUM_TRAIN_PAPER` empty-result branch remains untouched to preserve frozen Exp249 control flow.
GAPS: none

ROUND 1 REVIEW FIX EVIDENCE:
- Added focused public-boundary tests for `MiniRocketNonfiniteInputPaper` dataset/split/row context and the frozen small-TRAIN short circuit before repair.
- Corrected the Exp448 map with all AST callers of `preprocessing_paper.py` and an explicit Exp443 repair-origin provenance node, edge, and flow.
- Corrected the paper map with the Exp443 preprocessing origin and expanded MiniRocket test coverage description; regenerated both HTML/JSON/lock triplets.
- `python3 -m unittest -v` over the seven non-capable MiniRocket boundary tests: 7 passed.
- `python3 -m unittest -v paper.equipment_anomaly_detection.tests_paper.test_config_lineage_paper`: 15 passed.
- Explicit Exp448 HTML-payload/node/edge/flow/fingerprint validator printed `codemap-preflight 28 34 9`; `git diff --cached --check` was clean.
