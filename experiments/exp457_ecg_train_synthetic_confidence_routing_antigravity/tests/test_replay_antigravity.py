from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
import unittest

from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.tests.test_artifacts_antigravity import (
    publication_evidence_antigravity,
)


def _canonical_antigravity(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _rehash_antigravity(root: Path) -> None:
    names = tuple(
        name
        for name in (
            "context_antigravity.json",
            "train_qualification_antigravity.json",
            "features_antigravity.jsonl",
            "route_decisions_antigravity.jsonl",
            "comparison_metrics_antigravity.json",
            "family_metrics_antigravity.jsonl",
            "dashboard_snapshot_antigravity.json",
            "decision_antigravity.json",
        )
    )
    artifact_map = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }
    (root / "artifact_sha256_antigravity.json").write_bytes(
        _canonical_antigravity(artifact_map)
    )


class ReplayAntigravityTest(unittest.TestCase):
    def test_rejects_rehashed_passing_gate_values_with_false_seed_and_aggregate_flags_antigravity(self) -> None:
        """Break caught: stored Gate 1 booleans override Wilson and sensitivity evidence."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.replay_antigravity import (
            _inspect_result_with_authority_antigravity,
        )

        evidence = publication_evidence_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            publish_result_antigravity(root, evidence)
            path = root / "train_qualification_antigravity.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            qualification = payload["qualifications"][0]
            for seed in qualification["seed_evidence"]:
                seed["normal_trials"] = 100
                seed["normal_alerts"] = 0
                seed["normal_wilson_upper"] = 0.03699349820698568
                seed["pooled_sensitivity"] = 0.75
                seed["per_type_sensitivity"] = {
                    "level_shift": 0.75,
                    "gradual_drift": 0.75,
                    "amplitude_scaling": 0.75,
                    "period_change": 0.75,
                }
                seed["passed"] = False
            qualification["passed"] = False
            path.write_bytes(_canonical_antigravity(payload))
            _rehash_antigravity(root)
            with self.assertRaises(ValueError):
                _inspect_result_with_authority_antigravity(
                    root, evidence.exp456_authority
                )

    def test_inspect_recomputes_routes_and_all_comparisons_antigravity(self) -> None:
        """Break caught: inspect trusts stored route or metric leaves instead of semantic replay."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.replay_antigravity import (
            _inspect_result_with_authority_antigravity,
        )

        evidence = publication_evidence_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            publish_result_antigravity(root, evidence)
            self.assertEqual(
                _inspect_result_with_authority_antigravity(root, evidence.exp456_authority),
                evidence.result,
            )

    def test_rejects_extra_missing_or_wrong_hash_leaf_antigravity(self) -> None:
        """Break caught: inventory/hash drift survives bounded replay."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.replay_antigravity import (
            _inspect_result_with_authority_antigravity,
        )

        evidence = publication_evidence_antigravity()
        for mutation in ("extra", "missing", "wrong_hash"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "result"
                publish_result_antigravity(root, evidence)
                if mutation == "extra":
                    (root / "extra_antigravity.json").write_text("{}\n", encoding="utf-8")
                elif mutation == "missing":
                    (root / "features_antigravity.jsonl").unlink()
                else:
                    (root / "features_antigravity.jsonl").write_bytes(
                        (root / "features_antigravity.jsonl").read_bytes() + b"\n"
                    )
                with self.assertRaises(ValueError):
                    _inspect_result_with_authority_antigravity(
                        root, evidence.exp456_authority
                    )

    def test_rejects_every_rehashed_semantic_leaf_tamper_antigravity(self) -> None:
        """Break caught: recomputing the artifact map makes rewritten result evidence authoritative."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.replay_antigravity import (
            _inspect_result_with_authority_antigravity,
        )

        def mutate_json(root: Path, name: str, mutate) -> None:
            path = root / name
            value = json.loads(path.read_text(encoding="utf-8"))
            mutate(value)
            path.write_bytes(_canonical_antigravity(value))

        def mutate_jsonl(root: Path, name: str, mutate) -> None:
            path = root / name
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            mutate(rows)
            path.write_bytes(b"".join(_canonical_antigravity(row) for row in rows))

        mutations = (
            (
                "train_qualification_antigravity.json",
                lambda root: mutate_json(root, "train_qualification_antigravity.json", lambda value: value["qualifications"][0].__setitem__("passed", True)),
            ),
            (
                "features_antigravity.jsonl",
                lambda root: mutate_jsonl(root, "features_antigravity.jsonl", lambda rows: rows[0].__setitem__("added_count", 99)),
            ),
            (
                "route_decisions_antigravity.jsonl",
                lambda root: mutate_jsonl(root, "route_decisions_antigravity.jsonl", lambda rows: rows[0].__setitem__("route", "candidate")),
            ),
            (
                "comparison_metrics_antigravity.json",
                lambda root: mutate_json(root, "comparison_metrics_antigravity.json", lambda value: value["comparisons"][-1]["confusion"].__setitem__("fn", 830)),
            ),
            (
                "family_metrics_antigravity.jsonl",
                lambda root: mutate_jsonl(root, "family_metrics_antigravity.jsonl", lambda rows: rows[0].__setitem__("fn_delta", -7)),
            ),
            (
                "dashboard_snapshot_antigravity.json",
                lambda root: mutate_json(root, "dashboard_snapshot_antigravity.json", lambda value: value.__setitem__("passed", False)),
            ),
            (
                "decision_antigravity.json",
                lambda root: mutate_json(root, "decision_antigravity.json", lambda value: value.__setitem__("passed", False)),
            ),
        )
        evidence = publication_evidence_antigravity()
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "result"
                publish_result_antigravity(root, evidence)
                mutate(root)
                _rehash_antigravity(root)
                with self.assertRaises(ValueError):
                    _inspect_result_with_authority_antigravity(
                        root, evidence.exp456_authority
                    )


if __name__ == "__main__":
    unittest.main()
