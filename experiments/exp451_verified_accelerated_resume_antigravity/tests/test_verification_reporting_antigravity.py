"""Post-completion gates and frozen Exp451 reporting definitions."""

from __future__ import annotations

from types import MappingProxyType
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import io
from dataclasses import replace
import unittest
from unittest.mock import patch


def _verification_antigravity(*, total: int = 4468, manifests: tuple[str, ...] = ("A", "B", "C", "D"), native_c: int = 117):
    """Test-only full 4,468-slot authority; production has no fixture issuer."""
    from experiments.exp451_verified_accelerated_resume_antigravity import verifier_antigravity as verifier
    from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import FROZEN_EXP451_CONFIG
    datasets = [f"fixture_{index:04d}_antigravity" for index in range(1117)]
    tests = [{"dataset": name, "count": 600 if index == 1116 else 175} for index, name in enumerate(datasets)]
    entries = [{"dataset": name, "test": test} for name, test in zip(datasets, tests, strict=True)]
    imported = {"A": 1117, "B": 1117, "C": 1000, "D": 1000}
    native = {"A": 0, "B": 0, "C": native_c, "D": 117}
    context = type("FixtureContextAntigravity", (), {
        "database_sha256": FROZEN_EXP451_CONFIG.database_sha256, "sha256": "a" * 64,
        "catalog_sha256": "b" * 64,
        "backend": type("FixtureBackendAntigravity", (), {"artifact_record": lambda self: {}})(),
    })()
    records = {
        condition: verifier._condition_manifest_antigravity(
            condition=condition, entries=entries, context=context,
            checkpoint_hashes=[{"dataset": name, "sha256": __import__("hashlib").sha256(f"{condition}:{name}".encode()).hexdigest()} for name in datasets],
            imported=imported[condition], native=native[condition],
        ) for condition in manifests
    }
    return verifier.Exp451VerificationAntigravity(
        total, MappingProxyType(imported), MappingProxyType(native), MappingProxyType(records),
        context.sha256, context.catalog_sha256, context.database_sha256,
        verifier._inventory_sha256_antigravity(records), verifier._VERIFICATION_CAPABILITY_ANTIGRAVITY,
    )


class VerificationGateTests(unittest.TestCase):
    def test_evaluation_refuses_4467_successes_before_label_loader(self) -> None:
        """A partial coordinate set cannot reach the TEST-label boundary."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            require_exp451_evaluation_ready_antigravity,
        )

        with self.assertRaisesRegex(ValueError, "4468"):
            require_exp451_evaluation_ready_antigravity(_verification_antigravity(total=4467))

    def test_evaluation_refuses_missing_condition_manifest(self) -> None:
        """All four independently frozen condition manifests are required."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            require_exp451_evaluation_ready_antigravity,
        )

        with self.assertRaisesRegex(ValueError, "four"):
            require_exp451_evaluation_ready_antigravity(
                _verification_antigravity(manifests=("A", "B", "C"))
            )

    def test_provenance_keeps_imported_and_native_counts_separate(self) -> None:
        """The evidence summary never collapses the two execution origins."""
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
            provenance_summary_antigravity,
        )

        summary = provenance_summary_antigravity(_verification_antigravity())
        self.assertEqual(summary["imported_by_condition"], {"A": 1117, "B": 1117, "C": 1000, "D": 1000})
        self.assertEqual(summary["native_by_condition"], {"A": 0, "B": 0, "C": 117, "D": 117})
        self.assertEqual(summary["imported_total"], 4234)
        self.assertEqual(summary["native_total"], 234)

    def test_curve_axis_contracts_keep_roc_and_pr_coordinates_correct(self) -> None:
        """Paper figure coordinates are delegated to the frozen Exp450 definitions."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            curve_tables_exp451_antigravity,
        )

        methods = ("A", "B", "C", "D", "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d")
        scores = {method: {"fixture_antigravity": (0.1, 0.9)} for method in methods}
        curves = curve_tables_exp451_antigravity(
            {"fixture_antigravity": (0, 1)}, scores
        )
        self.assertEqual(curves["roc"]["x_name"], "false_positive_rate")
        self.assertEqual(curves["roc"]["y_name"], "true_positive_rate")
        self.assertEqual(curves["precision_recall"]["x_name"], "recall")
        self.assertEqual(curves["precision_recall"]["y_name"], "precision")
        self.assertEqual(
            {row["method"] for row in curves["roc"]["rows"]}, set(methods)
        )

    def test_four_verified_condition_manifests_publish_create_only(self) -> None:
        """The label gate is backed by four durable, byte-stable manifest leaves."""
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
            write_exp451_condition_manifests_antigravity,
        )
        from experiments.exp451_verified_accelerated_resume_antigravity import verifier_antigravity

        with TemporaryDirectory(prefix="exp451_manifest_fixture_antigravity_") as temporary:
            root = Path(temporary).resolve()
            with patch.object(verifier_antigravity, "OFFICIAL_ROOT", root):
                first = write_exp451_condition_manifests_antigravity(root, _verification_antigravity())
                second = write_exp451_condition_manifests_antigravity(root, _verification_antigravity())
        self.assertEqual(tuple(first), ("A", "B", "C", "D"))
        self.assertEqual(first, second)

    def test_forged_public_dataclass_does_not_open_label_boundary(self) -> None:
        """Only an authority issued by checkpoint verification can cross the gate."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            require_exp451_evaluation_ready_antigravity,
        )
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
            Exp451VerificationAntigravity,
        )
        forged = Exp451VerificationAntigravity(
            4468, {"A": 1117, "B": 1117, "C": 1000, "D": 1000},
            {"A": 0, "B": 0, "C": 117, "D": 117}, MappingProxyType({}),
        )
        with self.assertRaisesRegex(ValueError, "authority"):
            require_exp451_evaluation_ready_antigravity(forged)

    def test_gate_rejects_one_coordinate_origin_distribution_drift(self) -> None:
        """C/D native counts are fixed at 117 each; total alone is insufficient."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            require_exp451_evaluation_ready_antigravity,
        )
        with self.assertRaisesRegex(ValueError, "origin"):
            require_exp451_evaluation_ready_antigravity(_verification_antigravity(native_c=116))

    def test_cli_routes_verify_evaluate_and_report_commands(self) -> None:
        """Public documented commands no longer end at the former Task 6 placeholder."""
        script = Path(__file__).resolve().parents[3] / "scratch" / "run_exp451_verified_accelerated_resume_antigravity.py"
        specification = importlib.util.spec_from_file_location("exp451_task6_cli_antigravity", script)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        for command, function in (("verify", "_verify_antigravity"), ("evaluate", "_evaluate_antigravity"), ("report", "_report_antigravity")):
            output = io.StringIO()
            with patch.object(module, function, return_value={"experiment": "Exp451", "command": command}), patch("sys.stdout", output):
                self.assertEqual(module.main([command, "--mode", "full"]), 0)
            self.assertIn(f'"command":"{command}"', output.getvalue())

    def test_evaluation_does_not_call_label_loader_when_completion_is_partial(self) -> None:
        """The public evaluator reaches no label callback when its first gate fails."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            evaluate_exp451_antigravity,
        )
        from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import EVALUATION_ROOT
        called = []
        with self.assertRaisesRegex(ValueError, "4468"):
            evaluate_exp451_antigravity(
                EVALUATION_ROOT, verification=_verification_antigravity(total=4467),
                score_manifests={}, label_loader=lambda _: called.append(True),
            )
        self.assertEqual(called, [])

    def test_tampered_manifest_and_wrong_database_are_rejected(self) -> None:
        """A capability alone cannot authorize altered manifest bytes or database identity."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            require_exp451_evaluation_ready_antigravity,
        )
        verification = _verification_antigravity()
        altered = {key: dict(value) for key, value in verification.condition_manifests.items()}
        altered["A"]["origin_counts"] = dict(altered["A"]["origin_counts"])
        altered["A"]["origin_counts"]["imported_exp450_mps"] = 1116
        with self.assertRaisesRegex(ValueError, "manifest"):
            require_exp451_evaluation_ready_antigravity(
                replace(verification, condition_manifests=MappingProxyType(altered))
            )
        with self.assertRaisesRegex(ValueError, "database"):
            require_exp451_evaluation_ready_antigravity(
                replace(verification, database_sha256="f" * 64)
            )

    def test_economical_fixture_represents_all_4468_checkpoint_hashes_and_195900_test_rows(self) -> None:
        """The private fixture is full-population evidence, not a reduced production rule."""
        verification = _verification_antigravity()
        manifests = verification.condition_manifests
        self.assertEqual(sum(len(manifests[condition]["checkpoint_hashes"]) for condition in manifests), 4468)
        self.assertEqual(sum(item["count"] for item in manifests["A"]["test_identities"]), 195900)

    def test_score_manifest_tamper_and_wrong_root_stop_before_labels(self) -> None:
        """Changed score bytes and an unconfigured evaluation root cannot open labels."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import evaluate_exp451_antigravity
        verification = _verification_antigravity()
        called = []
        with self.assertRaisesRegex(ValueError, "root"):
            evaluate_exp451_antigravity(
                Path("/tmp/wrong_exp451_eval_antigravity"), verification=verification,
                score_manifests={}, label_loader=lambda _: called.append(True),
            )
        from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import EVALUATION_ROOT
        tampered = {condition: {"score_rows": {}} for condition in ("A", "B", "C", "D")}
        with self.assertRaisesRegex(ValueError, "score manifest"):
            evaluate_exp451_antigravity(
                EVALUATION_ROOT, verification=verification, score_manifests=tampered,
                label_loader=lambda _: called.append(True),
            )
        self.assertEqual(called, [])

    def test_missing_persisted_manifests_and_fabricated_evaluation_are_rejected(self) -> None:
        """Neither absent manifest leaves nor a copied evaluation object can reach reporting."""
        from experiments.exp451_verified_accelerated_resume_antigravity import verifier_antigravity
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import read_persisted_exp451_condition_manifests_antigravity
        verification = _verification_antigravity()
        with TemporaryDirectory(prefix="exp451_missing_manifest_antigravity_") as temporary:
            root = Path(temporary).resolve()
            with patch.object(verifier_antigravity, "OFFICIAL_ROOT", root):
                with self.assertRaises(ValueError):
                    read_persisted_exp451_condition_manifests_antigravity(root, verification)
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import Exp451EvaluationAntigravity
        from experiments.exp451_verified_accelerated_resume_antigravity.reporting_antigravity import write_exp451_report_antigravity
        from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import REPORT_ROOT
        forged = Exp451EvaluationAntigravity(None, verification.inventory_sha256, verification.context_sha256)
        with self.assertRaisesRegex(ValueError, "evaluation authority"):
            write_exp451_report_antigravity(forged, REPORT_ROOT, verification=verification)
