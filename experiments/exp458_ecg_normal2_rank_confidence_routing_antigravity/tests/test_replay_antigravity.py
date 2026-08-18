from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.tests.test_artifacts_antigravity import (
    publication_evidence_antigravity,
)


def canonical_antigravity(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def rehash_antigravity(root: Path) -> None:
    names = (
        "context_antigravity.json",
        "train_qualification_antigravity.jsonl",
        "features_antigravity.jsonl",
        "route_decisions_antigravity.jsonl",
        "comparison_metrics_antigravity.json",
        "family_metrics_antigravity.json",
        "dashboard_snapshot_antigravity.json",
        "decision_antigravity.json",
    )
    artifact_map = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    (root / "artifact_sha256_antigravity.json").write_bytes(canonical_antigravity(artifact_map))


class ReplayAntigravityTest(unittest.TestCase):
    def test_inspect_recomputes_routes_and_all_result_leaves_antigravity(self) -> None:
        """Break caught: inspect trusts stored metric leaves instead of semantic reconstruction."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.replay_antigravity import (
            _inspect_result_with_authority_antigravity,
        )

        evidence = publication_evidence_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            publish_result_antigravity(root, evidence)
            self.assertEqual(_inspect_result_with_authority_antigravity(root, evidence.exp456_authority), evidence.result)

    def test_rejects_inventory_hash_canonical_duplicate_and_bool_number_drift_antigravity(self) -> None:
        """Break caught: byte-level artifact validation misses common forged JSON shapes."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.replay_antigravity import (
            _inspect_result_with_authority_antigravity,
        )

        evidence = publication_evidence_antigravity()
        mutations = (
            lambda root: (root / "extra_antigravity.json").write_text("{}\n", encoding="utf-8"),
            lambda root: (root / "features_antigravity.jsonl").unlink(),
            lambda root: (root / "features_antigravity.jsonl").write_bytes((root / "features_antigravity.jsonl").read_bytes() + b"\n"),
            lambda root: (root / "decision_antigravity.json").write_text('{"passed":true,"passed":false,"status":"routing_candidate_not_qualified_for_external_validation"}\n', encoding="utf-8"),
            lambda root: (root / "decision_antigravity.json").write_text('{"passed":1,"status":"routing_candidate_qualified_for_external_validation"}\n', encoding="utf-8"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "result"
                publish_result_antigravity(root, evidence)
                mutate(root)
                if (root / "artifact_sha256_antigravity.json").exists() and not (root / "extra_antigravity.json").exists():
                    try:
                        rehash_antigravity(root)
                    except FileNotFoundError:
                        pass
                with self.assertRaises(ValueError):
                    _inspect_result_with_authority_antigravity(root, evidence.exp456_authority)

    def test_rejects_every_rehashed_semantic_result_tamper_antigravity(self) -> None:
        """Break caught: rehashed artifacts can rewrite AUC, structure, route, metrics, dashboard, or decision."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.replay_antigravity import (
            _inspect_result_with_authority_antigravity,
        )

        def mutate_json(root: Path, name: str, mutate) -> None:
            path = root / name
            value = json.loads(path.read_text(encoding="utf-8"))
            mutate(value)
            path.write_bytes(canonical_antigravity(value))

        def mutate_jsonl(root: Path, name: str, mutate) -> None:
            path = root / name
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            mutate(rows)
            path.write_bytes(b"".join(canonical_antigravity(row) for row in rows))

        mutations = (
            lambda root: mutate_jsonl(root, "train_qualification_antigravity.jsonl", lambda rows: rows[0]["seed_evidence"][0].__setitem__("pooled_auc", 0.91)),
            lambda root: mutate_jsonl(root, "train_qualification_antigravity.jsonl", lambda rows: rows[0]["seed_evidence"][0]["per_type_auc"].__setitem__("level_shift", 0.86)),
            lambda root: mutate_jsonl(root, "train_qualification_antigravity.jsonl", lambda rows: rows[0]["seed_evidence"][0].__setitem__("normal_wilson_upper", 0.04)),
            lambda root: mutate_jsonl(root, "train_qualification_antigravity.jsonl", lambda rows: rows[0]["seed_evidence"][0].__setitem__("passed", False)),
            lambda root: mutate_jsonl(root, "train_qualification_antigravity.jsonl", lambda rows: rows[0].__setitem__("passed", False)),
            lambda root: mutate_jsonl(root, "features_antigravity.jsonl", lambda rows: rows[0].__setitem__("added_count", 5)),
            lambda root: mutate_jsonl(root, "features_antigravity.jsonl", lambda rows: rows[0].__setitem__("removed_count", 14)),
            lambda root: mutate_jsonl(root, "route_decisions_antigravity.jsonl", lambda rows: rows[0].__setitem__("route", "baseline")),
            lambda root: mutate_json(root, "comparison_metrics_antigravity.json", lambda value: value["comparisons"][0]["confusion"].__setitem__("tp", 610)),
            lambda root: mutate_json(root, "comparison_metrics_antigravity.json", lambda value: value["comparisons"][-1]["f1"].__setitem__ if False else value["comparisons"][-1].__setitem__("f1", 0.58)),
            lambda root: mutate_json(root, "family_metrics_antigravity.json", lambda value: value["families"][0].__setitem__("fn_delta", -6)),
            lambda root: mutate_json(root, "dashboard_snapshot_antigravity.json", lambda value: value.__setitem__("passed", False)),
            lambda root: mutate_json(root, "decision_antigravity.json", lambda value: value.__setitem__("status", "routing_candidate_not_qualified_for_external_validation")),
        )
        evidence = publication_evidence_antigravity()
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "result"
                publish_result_antigravity(root, evidence)
                mutate(root)
                rehash_antigravity(root)
                with self.assertRaises(ValueError):
                    _inspect_result_with_authority_antigravity(root, evidence.exp456_authority)

    def test_rejects_coordinated_rehashed_auc_wilson_dashboard_tamper_antigravity(self) -> None:
        """Break caught: all-AUC=.99/Wilson=0 evidence and matching downstream payloads can replay cleanly."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.replay_antigravity import (
            _inspect_result_with_authority_antigravity,
        )

        evidence = publication_evidence_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result"
            publish_result_antigravity(root, evidence)

            qualification_path = root / "train_qualification_antigravity.jsonl"
            qualification = json.loads(qualification_path.read_text(encoding="utf-8").splitlines()[0])
            for seed in qualification["seed_evidence"]:
                seed["normal_wilson_upper"] = 0.0
                seed["per_type_auc"] = {name: 0.99 for name in seed["per_type_auc"]}
                seed["pooled_auc"] = 0.99
                seed["passed"] = True
            qualification["passed"] = True
            qualification_path.write_bytes(canonical_antigravity(qualification))

            dashboard_path = root / "dashboard_snapshot_antigravity.json"
            dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
            for seed in dashboard["rank_evidence"]:
                seed["normal_wilson_upper"] = 0.0
                seed["per_type_auc"] = {name: 0.99 for name in seed["per_type_auc"]}
                seed["pooled_auc"] = 0.99
                seed["passed"] = True
            dashboard_path.write_bytes(canonical_antigravity(dashboard))
            rehash_antigravity(root)

            with self.assertRaises(ValueError):
                _inspect_result_with_authority_antigravity(root, evidence.exp456_authority)

    def test_rejects_forged_expected_authority_indices_inventory_and_label_count_antigravity(self) -> None:
        """Break caught: replay can be pointed at a rehashed but internally forged Exp456 authority object."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.artifacts_antigravity import (
            publish_result_antigravity,
        )
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.replay_antigravity import (
            _inspect_result_with_authority_antigravity,
        )

        evidence = publication_evidence_antigravity()
        row = evidence.exp456_authority.authority.rows[0]
        forged_baseline = frozenset((*row.baseline_indices, max(row.baseline_indices) + 10_000))
        inventory = evidence.exp456_authority.authority.exp454_leaf_sha256
        cases = (
            replace(
                evidence.exp456_authority,
                authority=replace(
                    evidence.exp456_authority.authority,
                    rows=(replace(row, baseline_indices=forged_baseline),) + evidence.exp456_authority.authority.rows[1:],
                ),
            ),
            replace(
                evidence.exp456_authority,
                authority=replace(evidence.exp456_authority.authority, exp454_leaf_sha256=inventory[:-1]),
            ),
            replace(
                evidence.exp456_authority,
                authority=replace(evidence.exp456_authority.authority, label_query_count=1),
            ),
        )
        for authority in cases:
            with self.subTest(label_count=authority.authority.label_query_count), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "result"
                publish_result_antigravity(root, evidence)
                rehash_antigravity(root)
                with self.assertRaises(ValueError):
                    _inspect_result_with_authority_antigravity(root, authority)


if __name__ == "__main__":
    unittest.main()
