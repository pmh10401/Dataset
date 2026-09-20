from __future__ import annotations

from dataclasses import fields, replace
from hashlib import sha256
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def _canonical_antigravity(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _synthetic_leaves_antigravity() -> dict[str, bytes]:
    datasets = [f"Other__{index}" for index in range(40)]
    datasets[7:7] = [f"ECG5000__{index}" for index in range(5)]
    datasets[19:19] = [f"ChlorineConcentration__{index}" for index in range(3)]
    families = {
        **{f"ECG5000__{index}": "ECG5000" for index in range(5)},
        **{f"ChlorineConcentration__{index}": "ChlorineConcentration" for index in range(3)},
    }
    evidence = []
    prediction_rows = []
    sequence = 0
    for dataset in datasets:
        family = families.get(dataset, "Other")
        methods = (
            ("multirocket_local_gap", "hydra_local_gap")
            if family == "ECG5000"
            else (("minirocket_local_gap", "multirocket_local_gap") if family == "ChlorineConcentration" else ("base_a", "base_b"))
        )
        candidates = {}
        for method in methods:
            indices = [sequence % 11, (sequence + 3) % 11]
            prediction_rows.append({
                "sequence": sequence,
                "candidate_name": f"{dataset}__{method}",
                "dataset": dataset,
                "family": family,
                "method": method,
                "indices": indices,
                "target_length": 11,
            })
            candidates[method] = {"indices": indices, "confusion": {"tp": 1, "fp": 2, "fn": 3, "tn": 4}}
            sequence += 1
        evidence.append({
            "dataset": dataset,
            "family": family,
            "target_length": 11,
            "baseline_confusion": {"tp": 5, "fp": 6, "fn": 7, "tn": 8},
            "candidates": candidates,
        })
    payloads = {
        "context_antigravity.json": _canonical_antigravity({"schema": "exp454"}),
        "compatibility_antigravity.json": _canonical_antigravity({"schema": "exp454.task4"}),
        "candidate_registry_antigravity.json": _canonical_antigravity([]),
        "frozen_predictions_antigravity.jsonl": b"".join(_canonical_antigravity(row) for row in prediction_rows),
        "prediction_manifest_antigravity.json": _canonical_antigravity({"rows": 96}),
        "evaluation_antigravity.json": _canonical_antigravity({"dataset_evidence": evidence}),
        "candidate_metrics_antigravity.jsonl": b"", "family_metrics_antigravity.jsonl": b"",
        "bootstrap_antigravity.json": _canonical_antigravity({}),
        "decision_antigravity.json": _canonical_antigravity({}),
        "dashboard_snapshot_antigravity.json": _canonical_antigravity({}),
    }
    payloads["artifact_sha256_antigravity.json"] = _canonical_antigravity(
        {name: sha256(payload).hexdigest() for name, payload in payloads.items()}
    )
    return payloads


class RoutingAuthorityAntigravityTests(unittest.TestCase):
    def test_capture_rejects_changed_frozen_target_or_leaf_antigravity(self) -> None:
        """Break caught: a wrong target binding or rehashed Exp454 leaf reaches Exp455."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity import authority_antigravity as authority

        leaves = _synthetic_leaves_antigravity()
        identity = SimpleNamespace(stable_coordinates=(("ECG5000__0", 2), ("ChlorineConcentration__0", 6)))
        hard = SimpleNamespace(coordinates=frozenset({("ECG5000__0", 1), ("ChlorineConcentration__0", 5)}))
        config = SimpleNamespace(exp454_sha256=(
            ("artifact_map", sha256(leaves["artifact_sha256_antigravity.json"]).hexdigest()),
            ("evaluation", sha256(leaves["evaluation_antigravity.json"]).hexdigest()),
            ("decision", sha256(leaves["decision_antigravity.json"]).hexdigest()),
        ))
        captured = authority._capture_routing_authority_with_dependencies_antigravity(config, identity, hard, leaves)
        self.assertEqual(len(captured.rows), 48)
        self.assertEqual(captured.label_query_count, 0)
        self.assertEqual(tuple(name for name, _digest in captured.exp454_leaf_sha256), tuple(sorted(leaves)))
        self.assertEqual(captured.rows[7].dataset, "ECG5000__0")
        self.assertEqual(captured.rows[7].method, "hydra_local_gap")
        self.assertEqual(captured.rows[19].dataset, "ChlorineConcentration__0")
        self.assertEqual(captured.rows[19].method, "multirocket_local_gap")
        self.assertEqual(captured.rows[7].baseline_indices, frozenset({1, 2}))
        self.assertEqual(captured.rows[19].baseline_indices, frozenset({5, 6}))

        changed = dict(leaves)
        changed["decision_antigravity.json"] = _canonical_antigravity({"changed": True})
        with self.assertRaisesRegex(ValueError, "leaf"):
            authority._capture_routing_authority_with_dependencies_antigravity(config, identity, hard, changed)

        changed_coordinate = dict(leaves)
        prediction_rows = [json.loads(line) for line in changed_coordinate["frozen_predictions_antigravity.jsonl"].splitlines()]
        prediction_rows[14]["indices"] = [9]
        changed_coordinate["frozen_predictions_antigravity.jsonl"] = b"".join(_canonical_antigravity(row) for row in prediction_rows)
        with self.assertRaisesRegex(ValueError, "leaf"):
            authority._capture_routing_authority_with_dependencies_antigravity(config, identity, hard, changed_coordinate)

        wrong_family = dict(leaves)
        evaluation = json.loads(wrong_family["evaluation_antigravity.json"])
        evaluation["dataset_evidence"][7]["family"] = "Wrong"
        wrong_family["evaluation_antigravity.json"] = _canonical_antigravity(evaluation)
        wrong_family["artifact_sha256_antigravity.json"] = _canonical_antigravity({
            name: sha256(payload).hexdigest() for name, payload in wrong_family.items() if name != "artifact_sha256_antigravity.json"
        })
        changed_config = SimpleNamespace(exp454_sha256=(
            ("artifact_map", sha256(wrong_family["artifact_sha256_antigravity.json"]).hexdigest()),
            ("evaluation", sha256(wrong_family["evaluation_antigravity.json"]).hexdigest()),
            ("decision", sha256(wrong_family["decision_antigravity.json"]).hexdigest()),
        ))
        with self.assertRaisesRegex(ValueError, "target"):
            authority._capture_routing_authority_with_dependencies_antigravity(changed_config, identity, hard, wrong_family)

    def test_capture_uses_only_the_two_frozen_source_authorities_antigravity(self) -> None:
        """Break caught: Exp455 recreates predictions instead of using Exp446 and Exp404 authorities."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity import authority_antigravity as authority

        leaves = _synthetic_leaves_antigravity()
        identity = SimpleNamespace(stable_coordinates=())
        hard = SimpleNamespace(coordinates=frozenset())
        config = SimpleNamespace(exp454_sha256=(
            ("artifact_map", sha256(leaves["artifact_sha256_antigravity.json"]).hexdigest()),
            ("evaluation", sha256(leaves["evaluation_antigravity.json"]).hexdigest()),
            ("decision", sha256(leaves["decision_antigravity.json"]).hexdigest()),
        ))
        with (
            patch.object(authority, "FROZEN_CONFIG_ANTIGRAVITY", config),
            patch.object(authority, "capture_exp446_publication_antigravity", return_value=identity) as capture_identity,
            patch.object(authority, "capture_hard_predictions_antigravity", return_value=hard) as capture_hard,
            patch.object(authority, "_capture_exp454_leaves_antigravity", return_value=leaves),
        ):
            captured = authority.capture_routing_authority_antigravity()
        self.assertEqual(len(captured.rows), 48)
        capture_identity.assert_called_once_with()
        capture_hard.assert_called_once_with(identity)

    def test_authority_contract_is_immutable_and_exposes_no_labels_antigravity(self) -> None:
        """Break caught: mutable source coordinates or label outcomes leak through the public authority."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity import authority_antigravity as authority
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.contracts_antigravity import DatasetRoutingAuthorityAntigravity

        leaves = _synthetic_leaves_antigravity()
        config = SimpleNamespace(exp454_sha256=(
            ("artifact_map", sha256(leaves["artifact_sha256_antigravity.json"]).hexdigest()),
            ("evaluation", sha256(leaves["evaluation_antigravity.json"]).hexdigest()),
            ("decision", sha256(leaves["decision_antigravity.json"]).hexdigest()),
        ))
        captured = authority._capture_routing_authority_with_dependencies_antigravity(
            config, SimpleNamespace(stable_coordinates=(("ECG5000__0", 2),)), SimpleNamespace(coordinates=frozenset({("ECG5000__0", 1)})), leaves,
        )
        row = captured.rows[7]
        self.assertIs(type(row), DatasetRoutingAuthorityAntigravity)
        self.assertEqual(row.baseline_indices, frozenset({1, 2}))
        self.assertEqual(replace(row, baseline_indices=frozenset()).baseline_indices, frozenset())
        self.assertEqual(captured.rows[7].baseline_indices, frozenset({1, 2}))
        names = {field.name for field in fields(DatasetRoutingAuthorityAntigravity)}
        self.assertFalse({"tp", "fp", "fn", "tn", "test_label", "variant_number"} & names)


if __name__ == "__main__":
    unittest.main()
