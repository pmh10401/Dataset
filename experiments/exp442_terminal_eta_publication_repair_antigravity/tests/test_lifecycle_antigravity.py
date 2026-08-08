"""Contract tests for the thin Exp442 engineering-repair lifecycle."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from inspect import signature
import json
from pathlib import Path
import subprocess
import sys
import unittest

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import BackendAssessmentAntigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.lifecycle_profile_antigravity import (
    EXP441_PROFILE_ANTIGRAVITY,
    EXP442_PROFILE_ANTIGRAVITY,
    shared_scientific_config_sha256_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.readiness_antigravity import (
    _readiness_payload_antigravity,
    _validate_readiness_payload_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity import (
    _selection_payload_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import (
    CompletedAttemptAntigravity,
    SelectionResultAntigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity import parse_canonical_json_bytes_antigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.reporting_antigravity import _publication_payloads_antigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity import _validate_selection_artifact_antigravity
from experiments.exp442_terminal_eta_publication_repair_antigravity.profile_antigravity import PROFILE_ANTIGRAVITY


class Exp442LifecycleAntigravityTests(unittest.TestCase):
    def test_frozen_profile_hashes_match_the_actual_reviewed_documents(self) -> None:
        # Break caught: readiness can claim a design or preregistration digest that no longer matches the reviewed bytes.
        repository = Path(__file__).resolve().parents[3]
        design = repository / "docs/superpowers/specs/2026-08-08-exp442-terminal-eta-publication-repair-antigravity-design.md"
        preregistration = repository / "docs/EXP442_PREREGISTRATION_antigravity_KO.md"
        self.assertEqual(
            EXP442_PROFILE_ANTIGRAVITY.reviewed_design_sha256,
            sha256(design.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            EXP442_PROFILE_ANTIGRAVITY.reviewed_preregistration_sha256,
            sha256(preregistration.read_bytes()).hexdigest(),
        )

    def test_generic_production_lifecycle_has_no_path_override_surface(self) -> None:
        # Break caught: a fixed Exp442 identity can write or read another root while still claiming the canonical root hash.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.evaluation_antigravity import _evaluate_for_profile_antigravity
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.readiness_antigravity import (
            _run_readiness_for_profile_antigravity,
            _validate_readiness_for_profile_antigravity,
        )
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.reporting_antigravity import (
            _inspect_for_profile_antigravity,
            _publish_for_profile_antigravity,
        )
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity import _run_new_for_profile_antigravity
        forbidden_calls = (
            (_run_readiness_for_profile_antigravity, (EXP442_PROFILE_ANTIGRAVITY, "a" * 40), {"run_root": Path("/tmp/changed_antigravity")}),
            (_validate_readiness_for_profile_antigravity, (EXP442_PROFILE_ANTIGRAVITY, "a" * 40), {"run_root": Path("/tmp/changed_antigravity")}),
            (_run_new_for_profile_antigravity, (EXP442_PROFILE_ANTIGRAVITY,), {"run_root": Path("/tmp/changed_antigravity"), "reviewed_commit": "a" * 40}),
            (_inspect_for_profile_antigravity, (EXP442_PROFILE_ANTIGRAVITY,), {"run_root": Path("/tmp/changed_antigravity")}),
            (_publish_for_profile_antigravity, (EXP442_PROFILE_ANTIGRAVITY, object()), {"run_root": Path("/tmp/changed_antigravity")}),
            (_evaluate_for_profile_antigravity, (EXP442_PROFILE_ANTIGRAVITY,), {"run_root": Path("/tmp/changed_antigravity")}),
            (_evaluate_for_profile_antigravity, (EXP442_PROFILE_ANTIGRAVITY,), {"evaluation_destination": Path("/tmp/changed_antigravity")}),
        )
        for function, args, kwargs in forbidden_calls:
            with self.subTest(function=function.__name__, keyword=next(iter(kwargs))):
                with self.assertRaises(TypeError):
                    signature(function).bind(*args, **kwargs)

    def test_profiles_differ_only_in_lifecycle_identity_and_bind_one_scientific_config(self) -> None:
        self.assertIs(PROFILE_ANTIGRAVITY, EXP442_PROFILE_ANTIGRAVITY)
        self.assertEqual(EXP441_PROFILE_ANTIGRAVITY.scientific_config_sha256, EXP442_PROFILE_ANTIGRAVITY.scientific_config_sha256)
        self.assertEqual(EXP442_PROFILE_ANTIGRAVITY.scientific_config_sha256, shared_scientific_config_sha256_antigravity())
        self.assertEqual(FROZEN_CONFIG.candidate_budget, 30)
        self.assertEqual(FROZEN_CONFIG.bootstrap_seed, 44120260808)
        with self.assertRaises(ValueError):
            replace(EXP442_PROFILE_ANTIGRAVITY, run_root="/tmp/changed_antigravity")

    def test_exp442_readiness_and_attempt_identity_contain_no_exp441_lifecycle_literal(self) -> None:
        source = {
            "context_dataset_count": 1117, "record_digest_count": 1117,
            "hard_coordinate_count": 2218, "provisional_candidate_count": 840,
            "label_access_count": 0,
            "source_hashes": {
                "context": FROZEN_CONFIG.checkpoint_context_sha256,
                "state": FROZEN_CONFIG.checkpoint_state_sha256,
                "retained_inventory": FROZEN_CONFIG.checkpoint_retained_inventory_sha256,
                "publication_prediction": FROZEN_CONFIG.publication_prediction_sha256,
                "publication_manifest": FROZEN_CONFIG.publication_manifest_sha256,
                "publication_artifact_map": FROZEN_CONFIG.publication_artifact_map_sha256,
                "hard_prediction": FROZEN_CONFIG.hard_prediction_sha256,
                "hard_manifest": FROZEN_CONFIG.hard_manifest_sha256,
                "hard_coordinate_identity": FROZEN_CONFIG.hard_coordinate_identity_sha256,
            },
        }
        selection_counts = {
            "provisional_candidate_count": 840, "outside_hard_candidate_count": 394,
            "eligible_candidate_count": 73, "winner_count": 49,
            "selected_candidate_count": 30,
        }
        backend = BackendAssessmentAntigravity(
            "cpu", 11, False, "not_run_inapplicable", "inapplicable_no_tensor_stage",
            "inapplicable_no_tensor_stage", "inapplicable_no_tensor_stage", {},
        )
        readiness = _readiness_payload_antigravity(
            "a" * 40, source, selection_counts, backend, EXP442_PROFILE_ANTIGRAVITY,
        )
        _validate_readiness_payload_antigravity(readiness, EXP442_PROFILE_ANTIGRAVITY)
        attempt = _selection_payload_antigravity(
            SelectionResultAntigravity((), (), (), (), (), frozenset()),
            reviewed_commit="a" * 40, source_snapshot_sha256="b" * 64,
            profile=EXP442_PROFILE_ANTIGRAVITY,
        )
        identity_text = json.dumps({
            key: readiness[key] for key in ("schema", "experiment_id", "run_identity")
        }) + json.dumps({
            key: attempt[key] for key in ("experiment_id", "run_identity")
        })
        self.assertNotIn("Exp441", identity_text)
        self.assertNotIn("exp441", identity_text)
        self.assertIn("Exp442", identity_text)
        self.assertIn("exp442", identity_text)

    def test_cli_has_exact_four_commands_and_no_override_surface(self) -> None:
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.cli_antigravity import _build_parser_for_profile_antigravity
        parser = _build_parser_for_profile_antigravity(EXP442_PROFILE_ANTIGRAVITY)
        self.assertEqual(parser.prog, "run_exp442_terminal_eta_publication_repair_antigravity")
        for command in ("readiness", "new", "inspect", "evaluate"):
            self.assertEqual(parser.parse_args([command]).command, command)
        for flag in ("--path", "--workers", "--backend", "--seed", "--budget", "--threshold"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["new", flag, "x"])

    def test_exp441_attempt_identity_is_rejected_and_exp442_publication_schemas_are_isolated(self) -> None:
        selection = SelectionResultAntigravity((), (), (), (), (), frozenset())
        exp441_payload = _selection_payload_antigravity(
            selection, reviewed_commit="a" * 40, source_snapshot_sha256="b" * 64,
            profile=EXP441_PROFILE_ANTIGRAVITY,
        )
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            _validate_selection_artifact_antigravity(exp441_payload, EXP442_PROFILE_ANTIGRAVITY)
        exp442_payload = _selection_payload_antigravity(
            selection, reviewed_commit="a" * 40, source_snapshot_sha256="b" * 64,
            profile=EXP442_PROFILE_ANTIGRAVITY,
        )
        _validate_selection_artifact_antigravity(exp442_payload, EXP442_PROFILE_ANTIGRAVITY)
        attempt = CompletedAttemptAntigravity(0, "a" * 40, "b" * 64, selection, "c" * 64, "d" * 64)
        files = _publication_payloads_antigravity(attempt, {}, selection, EXP442_PROFILE_ANTIGRAVITY)
        manifest = parse_canonical_json_bytes_antigravity(files["manifest_antigravity.json"])
        status = parse_canonical_json_bytes_antigravity(files["status_antigravity.json"])
        artifact_map = parse_canonical_json_bytes_antigravity(files["artifact_sha256_antigravity.json"])
        self.assertEqual(
            (manifest["schema"], manifest["experiment_id"], manifest["run_identity"]),
            ("exp442.publication.v1", "Exp442", EXP442_PROFILE_ANTIGRAVITY.run_identity),
        )
        self.assertEqual(status["schema"], "exp442.status.v1")
        self.assertEqual(artifact_map["schema"], "exp442.artifact-map.v1")
        governing_text = b"".join((
            files["manifest_antigravity.json"], files["status_antigravity.json"],
            files["artifact_sha256_antigravity.json"],
        )).decode("utf-8")
        self.assertNotIn("Exp441", governing_text)
        self.assertNotIn("exp441", governing_text)

    def test_outside_cwd_launcher_help_imports_without_evaluator_or_sqlite(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        launcher = repository / "scratch" / "run_exp442_terminal_eta_publication_repair_antigravity.py"
        command = [sys.executable, str(launcher), "--help"]
        completed = subprocess.run(command, cwd="/tmp", check=True, capture_output=True, text=True)
        self.assertIn("readiness", completed.stdout)
        self.assertNotIn("evaluation_antigravity", completed.stdout)


if __name__ == "__main__":
    unittest.main()
