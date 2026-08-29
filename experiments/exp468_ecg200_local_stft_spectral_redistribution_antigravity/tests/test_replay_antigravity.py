"""Task 5 independent replay contract tests, written before production modules."""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.artifacts_antigravity import (
    LEAF_NAMES_ANTIGRAVITY,
    _canonical_json_bytes_antigravity,
    publish_result_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.evaluation_antigravity import (
    evaluate_qualification_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.predecessor_antigravity import (
    capture_exp465_authority_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.qualification_antigravity import (
    blocked_fold_indices_antigravity,
    qualify_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.replay_antigravity import (
    InspectedResultAntigravity,
    inspect_result_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.scoring_antigravity import (
    FoldMethodScoresAntigravity,
    score_fold_methods_antigravity,
)


@lru_cache(maxsize=1)
def records_fixture_antigravity() -> MappingProxyType:
    x = np.arange(140, dtype=np.float64)
    base = 0.01 * x + np.sin(2.0 * np.pi * x / 21.0) + 0.3 * np.cos(2.0 * np.pi * x / 9.0)
    train = np.asarray(
        base[None, :] + np.random.default_rng(46850).normal(scale=0.01, size=(69, 140)),
        dtype=np.float64,
        order="C",
    )
    train.setflags(write=False)
    authority = capture_exp465_authority_antigravity()
    records: dict[tuple[int, int], FoldMethodScoresAntigravity] = {}
    for seed in (20260721, 20261721):
        for fold_index, (fit, query) in enumerate(blocked_fold_indices_antigravity()):
            records[(seed, fold_index)] = score_fold_methods_antigravity(
                train,
                fit,
                query,
                seed=seed,
                fold_index=fold_index,
                authority=authority,
            )
    return MappingProxyType(records)


def _refresh_hashes_antigravity(root: Path) -> None:
    leaves = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.iterdir()
        if path.name != "artifact_sha256_antigravity.json"
    }
    (root / "artifact_sha256_antigravity.json").write_text(
        json.dumps(leaves, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


@lru_cache(maxsize=1)
def qualification_fixture_antigravity():
    return qualify_antigravity(records_fixture_antigravity())


@lru_cache(maxsize=1)
def decision_fixture_antigravity():
    return evaluate_qualification_antigravity(qualification_fixture_antigravity())


@lru_cache(maxsize=1)
def publication_bytes_fixture_antigravity() -> MappingProxyType:
    """Build one canonical root; individual tamper cases copy its bytes."""

    with tempfile.TemporaryDirectory(prefix="exp468-task5-fixture-") as directory:
        published = publish_result_antigravity(
            Path(directory) / "root",
            records_fixture_antigravity(),
            qualification_fixture_antigravity(),
            decision_fixture_antigravity(),
        )
        return MappingProxyType(
            {
                name: (published.root / name).read_bytes()
                for name in LEAF_NAMES_ANTIGRAVITY
            },
        )


def _materialize_publication_antigravity(parent: Path, name: str = "root") -> Path:
    root = parent / name
    root.mkdir()
    for leaf_name, content in publication_bytes_fixture_antigravity().items():
        (root / leaf_name).write_bytes(content)
    return root


def _rewrite_context_antigravity(root: Path, mutate) -> None:
    path = root / "context_antigravity.json"
    context = json.loads(path.read_text(encoding="utf-8"))
    mutate(context)
    path.write_bytes(_canonical_json_bytes_antigravity(context))
    _refresh_hashes_antigravity(root)


def _rewrite_json_leaf_antigravity(root: Path, leaf_name: str, mutate) -> None:
    path = root / leaf_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_bytes(_canonical_json_bytes_antigravity(payload))
    _refresh_hashes_antigravity(root)


def _mutate_array_envelope_antigravity(envelope: dict[str, object], operation) -> None:
    raw = base64.b64decode(str(envelope["data"]).encode("ascii"), validate=True)
    dtype = np.dtype(str(envelope["dtype"]))
    array = np.frombuffer(raw, dtype=dtype).copy()
    operation(array)
    changed = array.tobytes(order="C")
    envelope["data"] = base64.b64encode(changed).decode("ascii")
    envelope["sha256"] = hashlib.sha256(changed).hexdigest()


def _rewrite_jsonl_antigravity(root: Path, leaf_name: str, mutate) -> None:
    path = root / leaf_name
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    path.write_bytes(b"".join(_canonical_json_bytes_antigravity(row) for row in rows))
    _refresh_hashes_antigravity(root)


class ReplayAntigravityTest(unittest.TestCase):
    def test_inspect_recomputes_exact_decision_without_ecg_loader(self) -> None:
        records = records_fixture_antigravity()
        qualification = qualification_fixture_antigravity()
        decision = decision_fixture_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp468-task5-") as directory:
            root = _materialize_publication_antigravity(Path(directory))
            with patch(
                "experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.sources_antigravity.load_ecg200_normal_train_antigravity",
                side_effect=AssertionError("replay must not load ECG200"),
            ):
                inspected = inspect_result_antigravity(root)
            self.assertIs(type(inspected), InspectedResultAntigravity)
            self.assertEqual(tuple(inspected.records), tuple(records))
            for identity in records:
                self.assertEqual(
                    inspected.records[identity].input_train140.tobytes(),
                    records[identity].input_train140.tobytes(),
                )
            self.assertEqual(inspected.qualification, qualification)
            self.assertEqual(inspected.decision, decision)

    def test_semantic_context_mutation_is_rejected_even_after_hash_refresh(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp468-task5-") as directory:
            root = _materialize_publication_antigravity(Path(directory))

            def mutate(context):
                context["records"][0]["original"]["C0"]["data"] = (
                    context["records"][0]["original"]["C0"]["data"][:-4] + "AAAA"
                )

            _rewrite_context_antigravity(root, mutate)
            with self.assertRaisesRegex(ValueError, "replay|C0|digest"):
                inspect_result_antigravity(root)

    def test_stale_hash_duplicate_key_and_extra_leaf_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp468-task5-") as directory:
            root = _materialize_publication_antigravity(Path(directory))
            digest_path = root / "artifact_sha256_antigravity.json"
            digest_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest|leaf"):
                inspect_result_antigravity(root)

            root = _materialize_publication_antigravity(Path(directory), "root-2")
            (root / "extra_antigravity.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seven|leaf"):
                inspect_result_antigravity(root)

    def test_rehashed_semantic_tamper_matrix_is_rejected(self) -> None:
        """Every brief-listed scalar and evidence leaf must fail independent replay."""

        cases = (
            (
                "raw_js",
                lambda root: _rewrite_context_antigravity(
                    root,
                    lambda context: _mutate_array_envelope_antigravity(
                        context["records"][0]["query_raw_js"]["period_change"],
                        lambda values: values.__setitem__(0, values[0] + 0.125),
                    ),
                ),
            ),
            (
                "fit_max",
                lambda root: _rewrite_context_antigravity(
                    root,
                    lambda context: context["records"][0].__setitem__(
                        "fit_normal_max_raw_js",
                        context["records"][0]["fit_normal_max_raw_js"] + 0.125,
                    ),
                ),
            ),
            (
                "rank",
                lambda root: _rewrite_context_antigravity(
                    root,
                    lambda context: _mutate_array_envelope_antigravity(
                        context["records"][0]["original"]["S10"],
                        lambda values: values.__setitem__(0, values[0] + 0.125),
                    ),
                ),
            ),
            (
                "confirmation",
                lambda root: _rewrite_context_antigravity(
                    root,
                    lambda context: _mutate_array_envelope_antigravity(
                        context["records"][0]["confirmed"]["normal"],
                        lambda values: values.__setitem__(0, not bool(values[0])),
                    ),
                ),
            ),
            (
                "C10",
                lambda root: _rewrite_context_antigravity(
                    root,
                    lambda context: _mutate_array_envelope_antigravity(
                        context["records"][0]["original"]["C10"],
                        lambda values: values.__setitem__(0, values[0] + 0.125),
                    ),
                ),
            ),
            (
                "AUC",
                lambda root: _rewrite_jsonl_antigravity(
                    root,
                    "method_fold_metrics_antigravity.jsonl",
                    lambda rows: rows[0]["per_type_auc"].__setitem__(
                        0,
                        rows[0]["per_type_auc"][0] + 0.001,
                    ),
                ),
            ),
            (
                "C0_delta",
                lambda root: _rewrite_jsonl_antigravity(
                    root,
                    "method_fold_metrics_antigravity.jsonl",
                    lambda rows: rows[0]["delta_per_type_auc"].__setitem__(
                        0,
                        rows[0]["delta_per_type_auc"][0] + 0.001,
                    ),
                ),
            ),
            (
                "normal_confirmation",
                lambda root: _rewrite_json_leaf_antigravity(
                    root,
                    "dashboard_snapshot_antigravity.json",
                    lambda payload: payload["normal_confirmations_by_seed"][0].__setitem__(
                        "count",
                        payload["normal_confirmations_by_seed"][0]["count"] + 1,
                    ),
                ),
            ),
            (
                "decision",
                lambda root: _rewrite_json_leaf_antigravity(
                    root,
                    "decision_antigravity.json",
                    lambda payload: payload.__setitem__("status", "tampered_status"),
                ),
            ),
            (
                "context",
                lambda root: _rewrite_context_antigravity(
                    root,
                    lambda context: context["qualification"].__setitem__(
                        "label_query_count",
                        context["qualification"]["label_query_count"] + 1,
                    ),
                ),
            ),
            (
                "summary",
                lambda root: _rewrite_jsonl_antigravity(
                    root,
                    "method_summary_antigravity.jsonl",
                    lambda rows: rows[0]["mean_per_type_auc"].__setitem__(
                        0,
                        rows[0]["mean_per_type_auc"][0] + 0.001,
                    ),
                ),
            ),
            (
                "failed_fold",
                lambda root: _rewrite_jsonl_antigravity(
                    root,
                    "failed_folds_antigravity.jsonl",
                    lambda rows: rows.append(
                        {
                            "failure_reasons": ["tampered"],
                            "fold_index": 0,
                            "passed": False,
                            "seed": 20260721,
                        },
                    ),
                ),
            ),
            (
                "dashboard_snapshot",
                lambda root: _rewrite_json_leaf_antigravity(
                    root,
                    "dashboard_snapshot_antigravity.json",
                    lambda payload: payload.__setitem__(
                        "qualification_passed",
                        not payload["qualification_passed"],
                    ),
                ),
            ),
            (
                "backend",
                lambda root: _rewrite_context_antigravity(
                    root,
                    lambda context: context.__setitem__("backend", "tampered_backend"),
                ),
            ),
            (
                "approval_sha",
                lambda root: _rewrite_context_antigravity(
                    root,
                    lambda context: context["approved_fallback"].__setitem__(
                        "approval_document_sha256",
                        "0" * 64,
                    ),
                ),
            ),
            (
                "pre_approval_benchmark_sha",
                lambda root: _rewrite_context_antigravity(
                    root,
                    lambda context: context["approved_fallback"].__setitem__(
                        "pre_approval_benchmark_sha256",
                        "0" * 64,
                    ),
                ),
            ),
        )
        for label, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                prefix=f"exp468-task5-{label}-",
            ) as directory:
                root = _materialize_publication_antigravity(Path(directory))
                mutate(root)
                with self.assertRaises(ValueError):
                    inspect_result_antigravity(root)

    def test_malformed_envelopes_and_leaf_layout_are_rejected(self) -> None:
        cases = ("bad_base64", "bad_shape", "bad_digest", "duplicate_key", "nan", "missing_leaf")
        for label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                prefix=f"exp468-task5-{label}-",
            ) as directory:
                root = _materialize_publication_antigravity(Path(directory))
                context_path = root / "context_antigravity.json"
                if label == "bad_base64":
                    _rewrite_context_antigravity(
                        root,
                        lambda context: context["records"][0]["input_train140"].__setitem__(
                            "data",
                            "not base64!",
                        ),
                    )
                elif label == "bad_shape":
                    _rewrite_context_antigravity(
                        root,
                        lambda context: context["records"][0]["input_train140"].__setitem__(
                            "shape",
                            [69, 141],
                        ),
                    )
                elif label == "bad_digest":
                    _rewrite_context_antigravity(
                        root,
                        lambda context: context["records"][0]["input_train140"].__setitem__(
                            "sha256",
                            "0" * 64,
                        ),
                    )
                elif label == "duplicate_key":
                    raw = context_path.read_bytes()
                    duplicate = raw.replace(
                        b'"backend":"numpy"',
                        b'"backend":"numpy","backend":"numpy"',
                        1,
                    )
                    context_path.write_bytes(duplicate)
                    _refresh_hashes_antigravity(root)
                elif label == "nan":
                    context = json.loads(context_path.read_text(encoding="utf-8"))
                    value = context["records"][0]["fit_normal_max_raw_js"]
                    raw = context_path.read_bytes()
                    needle = f'"fit_normal_max_raw_js":{value!r}'.encode("ascii")
                    self.assertIn(needle, raw)
                    context_path.write_bytes(raw.replace(needle, b'"fit_normal_max_raw_js":NaN', 1))
                    _refresh_hashes_antigravity(root)
                elif label == "missing_leaf":
                    (root / "failed_folds_antigravity.jsonl").unlink()
                with self.assertRaises(ValueError):
                    inspect_result_antigravity(root)

    def test_repaired_hash_metric_provenance_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp468-task5-metric-provenance-") as directory:
            root = _materialize_publication_antigravity(Path(directory))
            metrics_path = root / "method_fold_metrics_antigravity.jsonl"
            rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            self.assertIn("backend", rows[0])
            rows[0]["backend"] = "mlx"
            metrics_path.write_bytes(
                b"".join(_canonical_json_bytes_antigravity(row) for row in rows),
            )
            _refresh_hashes_antigravity(root)
            with self.assertRaisesRegex(ValueError, "metric|provenance|replay"):
                inspect_result_antigravity(root)


if __name__ == "__main__":
    unittest.main()
