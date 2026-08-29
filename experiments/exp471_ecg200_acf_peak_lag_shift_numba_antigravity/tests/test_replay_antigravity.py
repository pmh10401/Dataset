from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.tests.test_artifacts_antigravity import _published_inputs_antigravity


def _canonical_bytes_antigravity(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def tamper_and_rehash_antigravity(root: Path, leaf: str, mutate) -> None:
    """Test-only helper: refresh hashes to prove semantic replay is not hash-only."""
    path = root / leaf
    if leaf.endswith(".jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        mutate(rows)
        path.write_bytes(b"".join(_canonical_bytes_antigravity(row) for row in rows))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_bytes(_canonical_bytes_antigravity(payload))
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.artifacts_antigravity import LEAF_NAMES_ANTIGRAVITY

    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in LEAF_NAMES_ANTIGRAVITY[1:]}
    (root / LEAF_NAMES_ANTIGRAVITY[0]).write_bytes(_canonical_bytes_antigravity({"hashes": hashes, "schema": "exp471_artifact_sha256_v1"}))


def published_result_root_antigravity(directory: str) -> Path:
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.artifacts_antigravity import publish_result_antigravity

    records, qualification, decision = _published_inputs_antigravity()
    root = Path(directory) / "result_antigravity"
    publish_result_antigravity(root, records, qualification, decision)
    return root


class ResultReplayTestAntigravity(unittest.TestCase):
    def test_inspect_replays_stored_inputs_without_data_file_access(self) -> None:
        """Break caught: replay trusts stored scores instead of regenerating all evidence."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.artifacts_antigravity import publish_result_antigravity
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.replay_antigravity import inspect_result_antigravity

        with tempfile.TemporaryDirectory() as directory:
            records, qualification, decision = _published_inputs_antigravity()
            root = published_result_root_antigravity(directory)
            inspected = inspect_result_antigravity(root)
        self.assertEqual(len(inspected.records), 10)
        self.assertEqual(inspected.qualification, qualification)
        self.assertEqual(inspected.decision, decision)
        self.assertEqual(tuple(inspected.hashes), (
            "context_antigravity.json", "dashboard_snapshot_antigravity.json", "decision_antigravity.json",
            "failed_folds_antigravity.jsonl", "method_fold_metrics_antigravity.jsonl", "method_summary_antigravity.jsonl",
        ))
        for identity, replayed in inspected.records.items():
            original = records[identity]
            self.assertEqual(replayed.backend, "numba")
            self.assertEqual(replayed.input_train140.tobytes(), original.input_train140.tobytes())
            self.assertEqual(replayed.fit_raw_shift.tobytes(), original.fit_raw_shift.tobytes())
            self.assertEqual(replayed.fit_normal_max_raw_shift, original.fit_normal_max_raw_shift)
            scopes = ("normal", "level_shift", "gradual_drift", "amplitude_scaling", "period_change")
            for scope in scopes:
                self.assertEqual(replayed.query_raw_shift[scope].tobytes(), original.query_raw_shift[scope].tobytes())
                self.assertEqual(replayed.confirmed[scope].tobytes(), original.confirmed[scope].tobytes())

    def test_inspect_rejects_byte_canonical_and_leaf_set_tamper_antigravity(self) -> None:
        """Break caught: damaged publication reaches semantic replay."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.replay_antigravity import inspect_result_antigravity

        with tempfile.TemporaryDirectory() as directory:
            root = published_result_root_antigravity(directory)
            (root / "context_antigravity.json").write_bytes(b' {"bad":1}\n')
            with self.assertRaises(ValueError):
                inspect_result_antigravity(root)
        with tempfile.TemporaryDirectory() as directory:
            root = published_result_root_antigravity(directory)
            (root / "extra_antigravity.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leaf"):
                inspect_result_antigravity(root)
        with tempfile.TemporaryDirectory() as directory:
            root = published_result_root_antigravity(directory)
            (root / "method_summary_antigravity.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "leaf"):
                inspect_result_antigravity(root)

    def test_inspect_rejects_rehashed_semantic_tampering_across_evidence_leaves(self) -> None:
        """Break caught: attacker changes evidence then makes all six hashes consistent."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.replay_antigravity import inspect_result_antigravity

        cases = (
            ("decision_antigravity.json", lambda value: value.__setitem__("period_fixed", {"20260721": 999, "20261721": 0})),
            ("method_fold_metrics_antigravity.jsonl", lambda rows: rows[0]["fold_result"]["pair_repairs"]["period_change"].__setitem__("fixed", 999)),
            ("method_summary_antigravity.jsonl", lambda rows: rows[0].__setitem__("fold_count", 999)),
            ("dashboard_snapshot_antigravity.json", lambda value: value.__setitem__("status", "forged")),
            ("context_antigravity.json", lambda value: value["shared_train140"][0].__setitem__(0, 99.0)),
            ("context_antigravity.json", lambda value: value["records"][0]["query_shifts"]["normal"].__setitem__(0, -99)),
            ("context_antigravity.json", lambda value: value["records"][0]["query_raw_shift"]["period_change"].__setitem__(0, 99.0)),
            ("context_antigravity.json", lambda value: value["records"][0]["query_ranks"]["normal"].__setitem__(0, 99.0)),
            ("context_antigravity.json", lambda value: value["records"][0]["query_ranks"]["period_change"].__setitem__(0, 99.0)),
        )
        for leaf, mutate in cases:
            with self.subTest(leaf=leaf):
                with tempfile.TemporaryDirectory() as directory:
                    root = published_result_root_antigravity(directory)
                    tamper_and_rehash_antigravity(root, leaf, mutate)
                    with self.assertRaises(ValueError):
                        inspect_result_antigravity(root)

    def test_inspect_rejects_exact_numeric_type_drift_antigravity(self) -> None:
        """Break caught: JSON int or bool impersonates exact float/int evidence."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.replay_antigravity import inspect_result_antigravity

        cases = (("query_raw_shift", "normal", 0), ("query_shifts", "normal", True), ("query_early_peak_lags", "normal", True))
        for key, scope, replacement in cases:
            def mutate(value, key=key, scope=scope, replacement=replacement):
                value["records"][0][key][scope].__setitem__(0, replacement)
            with self.subTest(key=key, scope=scope, replacement=replacement):
                with tempfile.TemporaryDirectory() as directory:
                    root = published_result_root_antigravity(directory)
                    tamper_and_rehash_antigravity(root, "context_antigravity.json", mutate)
                    with self.assertRaises(ValueError):
                        inspect_result_antigravity(root)


if __name__ == "__main__":
    unittest.main()
