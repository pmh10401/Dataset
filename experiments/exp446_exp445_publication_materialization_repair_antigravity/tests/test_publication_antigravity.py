"""Closed publication boundary tests for Exp446."""
from __future__ import annotations

import tempfile
import unittest
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class Exp446PublicationAntigravityTests(unittest.TestCase):
    def _real_terminal_root_antigravity(self) -> tuple[Path, object, object, object]:
        """Materialize only Task5's tiny 30-record terminal fixture, never an official root."""
        from experiments.exp446_exp445_publication_materialization_repair_antigravity.tests import test_lifecycle_antigravity as lifecycle_tests

        case = lifecycle_tests.Exp446LifecycleAntigravityTests("test_all_thirty_wrappers_require_exp444_and_exp445_parity_before_success_checkpoint_antigravity")
        root, outcome = case._run_antigravity()
        self.addCleanup(case.doCleanups)
        self.assertEqual(getattr(outcome, "status", None), "complete")
        attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
        marker = json.loads((attempt / "terminal_parity_antigravity.json").read_text())
        state = json.loads((attempt / "checkpoint_antigravity/state_antigravity.json").read_text())
        first = SimpleNamespace(record_sha256_prefix=tuple(state["record_sha256_prefix"]), record_prefix_payload_sha256=marker["authorities"][0])
        second = SimpleNamespace(record_sha256_prefix=tuple(state["record_sha256_prefix"]), record_prefix_payload_sha256=marker["authorities"][1])
        return root, attempt, first, second

    def test_facade_publishes_marker_inclusive_terminal_fixture_antigravity(self) -> None:
        """Facade-to-reporter publication must accept a real Task5 marker without generic capture."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp444_train_only_stability_precision_gate_antigravity import provenance_antigravity
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import predecessor_antigravity as exp444_predecessor
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import predecessor_antigravity as exp445_predecessor

        root, attempt, exp444, exp445 = self._real_terminal_root_antigravity()
        base = lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY
        profile = object.__new__(type(base))
        for name in ("experiment_id", "schema", "run_identity", "reviewed_design_path", "preregistration_path", "predecessor_required"):
            object.__setattr__(profile, name, getattr(base, name))
        object.__setattr__(profile, "run_root", root)
        expected = {name: f"{index}".encode() for index, name in enumerate(reporting._PUBLICATION_FILES_ANTIGRAVITY)}
        with (
            mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile),
            mock.patch.object(reporting, "EXP446_PROFILE_ANTIGRAVITY", profile),
            mock.patch.object(exp444_predecessor, "verify_exp444_predecessor_antigravity", return_value=exp444),
            mock.patch.object(exp445_predecessor, "verify_exp445_predecessor_antigravity", return_value=exp445),
            mock.patch.object(provenance_antigravity, "verify_exp444_provenance_antigravity", return_value=SimpleNamespace(dataset_summary_bytes=b"", backend_authority={})),
            mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=object()),
            mock.patch.object(reporting, "_payloads_antigravity", return_value=({}, object())),
            mock.patch.object(reporting, "_final_payloads_antigravity", return_value=expected),
            mock.patch.object(reporting, "_inspect_publication_payloads_antigravity", return_value=({}, object())),
            mock.patch.object(reporting, "FrozenPublicationAntigravity", return_value=mock.sentinel.publication),
        ):
            self.assertIs(lifecycle_antigravity.publish_prediction_antigravity(), mock.sentinel.publication)
        self.assertTrue((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).is_dir())

    def test_marker_removal_or_replacement_between_terminal_and_publication_capture_preserves_failure_antigravity(self) -> None:
        """A facade-to-reporter marker bridge cannot survive removal or same-path replacement."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity

        for action in ("remove", "replace"):
            with self.subTest(action=action):
                root, attempt, _exp444, _exp445 = self._real_terminal_root_antigravity()
                base = lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY
                profile = object.__new__(type(base))
                for name in ("experiment_id", "schema", "run_identity", "reviewed_design_path", "preregistration_path", "predecessor_required"):
                    object.__setattr__(profile, name, getattr(base, name))
                object.__setattr__(profile, "run_root", root)
                marker = attempt / "terminal_parity_antigravity.json"
                def mutate(_attempt: Path) -> None:
                    if action == "remove":
                        marker.unlink()
                    else:
                        marker.write_bytes(b"{}\n")
                with (
                    mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile),
                    mock.patch.object(reporting, "EXP446_PROFILE_ANTIGRAVITY", profile),
                    mock.patch.object(reporting, "_EXP446_TERMINAL_CAPTURE_HOOK_ANTIGRAVITY", mutate),
                ):
                    with self.assertRaisesRegex(ValueError, "terminal|marker|inventory"):
                        lifecycle_antigravity.publish_prediction_antigravity()
                self.assertFalse((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())
                failure = json.loads((attempt / reporting._EXP446_PUBLICATION_FAILURE_FILE_ANTIGRAVITY).read_text())
                self.assertEqual((failure["schema"], failure["stage"]), ("exp446.publication-failure.v1", "publication"))

    def test_capture_namespace_replacement_preserves_retained_failure_antigravity(self) -> None:
        """An identical replacement cannot bridge descriptor capture into source work."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity

        root, attempt, _exp444, _exp445 = self._real_terminal_root_antigravity()
        base = lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY
        profile = object.__new__(type(base))
        for name in ("experiment_id", "schema", "run_identity", "reviewed_design_path", "preregistration_path", "predecessor_required"):
            object.__setattr__(profile, name, getattr(base, name))
        object.__setattr__(profile, "run_root", root)
        retained = attempt.with_name("retained_attempt_antigravity")

        def replace(_snapshot: object) -> None:
            attempt.rename(retained)
            shutil.copytree(retained, attempt)

        with (
            mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile),
            mock.patch.object(reporting, "EXP446_PROFILE_ANTIGRAVITY", profile),
            mock.patch.object(reporting, "_EXP446_PUBLICATION_CAPTURE_HOOK_ANTIGRAVITY", replace),
        ):
            with self.assertRaisesRegex(ValueError, "canonical-path drift|namespace"):
                lifecycle_antigravity.publish_prediction_antigravity()
        self.assertFalse((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())
        self.assertFalse((retained / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())
        self.assertTrue((retained / reporting._EXP446_PUBLICATION_FAILURE_FILE_ANTIGRAVITY).is_file())

    def test_checkpoint_directory_replacement_after_capture_preserves_failure_antigravity(self) -> None:
        """An identical checkpoint replacement must not bridge retained nested descriptors."""
        self._assert_nested_capture_replacement_antigravity("checkpoint_antigravity")

    def test_records_directory_replacement_after_capture_preserves_failure_antigravity(self) -> None:
        """An identical records replacement must not bridge retained nested descriptors."""
        self._assert_nested_capture_replacement_antigravity("checkpoint_antigravity/records_antigravity")

    def _assert_nested_capture_replacement_antigravity(self, relative: str) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity

        root, attempt, _exp444, _exp445 = self._real_terminal_root_antigravity()
        base = lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY
        profile = object.__new__(type(base))
        for name in ("experiment_id", "schema", "run_identity", "reviewed_design_path", "preregistration_path", "predecessor_required"):
            object.__setattr__(profile, name, getattr(base, name))
        object.__setattr__(profile, "run_root", root)
        target = attempt / relative
        retained = target.with_name(f"retained_{target.name}")

        def replace(_snapshot: object) -> None:
            target.rename(retained)
            shutil.copytree(retained, target)

        with (
            mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile),
            mock.patch.object(reporting, "EXP446_PROFILE_ANTIGRAVITY", profile),
            mock.patch.object(reporting, "_EXP446_PUBLICATION_CAPTURE_HOOK_ANTIGRAVITY", replace),
        ):
            with self.assertRaisesRegex(ValueError, "checkpoint|records|authority"):
                lifecycle_antigravity.publish_prediction_antigravity()
        self.assertFalse((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())
        self.assertTrue((attempt / reporting._EXP446_PUBLICATION_FAILURE_FILE_ANTIGRAVITY).is_file())

    def test_pre_rename_namespace_replacement_quarantines_retained_destination_antigravity(self) -> None:
        """A replacement between staging and rename cannot claim the canonical publication."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp444_train_only_stability_precision_gate_antigravity import provenance_antigravity
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import predecessor_antigravity as exp444_predecessor
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import predecessor_antigravity as exp445_predecessor

        root, attempt, exp444, exp445 = self._real_terminal_root_antigravity()
        base = lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY
        profile = object.__new__(type(base))
        for name in ("experiment_id", "schema", "run_identity", "reviewed_design_path", "preregistration_path", "predecessor_required"):
            object.__setattr__(profile, name, getattr(base, name))
        object.__setattr__(profile, "run_root", root)
        retained = attempt.with_name("retained_before_rename_antigravity")
        expected = {name: f"{index}".encode() for index, name in enumerate(reporting._PUBLICATION_FILES_ANTIGRAVITY)}

        def replace(_snapshot: object) -> None:
            attempt.rename(retained)
            shutil.copytree(retained, attempt)

        with (
            mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile),
            mock.patch.object(reporting, "EXP446_PROFILE_ANTIGRAVITY", profile),
            mock.patch.object(exp444_predecessor, "verify_exp444_predecessor_antigravity", return_value=exp444),
            mock.patch.object(exp445_predecessor, "verify_exp445_predecessor_antigravity", return_value=exp445),
            mock.patch.object(provenance_antigravity, "verify_exp444_provenance_antigravity", return_value=SimpleNamespace(dataset_summary_bytes=b"", backend_authority={})),
            mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=object()),
            mock.patch.object(reporting, "_payloads_antigravity", return_value=({}, object())),
            mock.patch.object(reporting, "_final_payloads_antigravity", return_value=expected),
            mock.patch.object(reporting, "_EXP446_STAGING_BEFORE_RENAME_HOOK_ANTIGRAVITY", replace),
        ):
            with self.assertRaisesRegex(ValueError, "canonical-path drift|namespace"):
                lifecycle_antigravity.publish_prediction_antigravity()
        self.assertFalse((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())
        self.assertFalse((retained / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())
        self.assertTrue((retained / reporting._EXP446_PUBLICATION_FAILURE_FILE_ANTIGRAVITY).is_file())
        self.assertTrue(any("quarantine_antigravity" in item.name for item in retained.iterdir()))

    def test_cleanup_never_quarantines_swapped_publication_name_antigravity(self) -> None:
        """Cleanup must retain its publication identity and leave a replacement untouched."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp444_train_only_stability_precision_gate_antigravity import provenance_antigravity
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import predecessor_antigravity as exp444_predecessor
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import predecessor_antigravity as exp445_predecessor

        root, attempt, exp444, exp445 = self._real_terminal_root_antigravity()
        base = lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY
        profile = object.__new__(type(base))
        for name in ("experiment_id", "schema", "run_identity", "reviewed_design_path", "preregistration_path", "predecessor_required"):
            object.__setattr__(profile, name, getattr(base, name))
        object.__setattr__(profile, "run_root", root)
        expected = {name: f"{index}".encode() for index, name in enumerate(reporting._PUBLICATION_FILES_ANTIGRAVITY)}
        swapped = attempt / "swapped_publication_antigravity"

        def replace(_snapshot: object, _identity: object) -> None:
            publication = attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY
            publication.rename(swapped)
            publication.mkdir()
            (publication / "replacement_antigravity.txt").write_text("foreign")

        with (
            mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile),
            mock.patch.object(reporting, "EXP446_PROFILE_ANTIGRAVITY", profile),
            mock.patch.object(exp444_predecessor, "verify_exp444_predecessor_antigravity", return_value=exp444),
            mock.patch.object(exp445_predecessor, "verify_exp445_predecessor_antigravity", return_value=exp445),
            mock.patch.object(provenance_antigravity, "verify_exp444_provenance_antigravity", return_value=SimpleNamespace(dataset_summary_bytes=b"", backend_authority={})),
            mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=object()),
            mock.patch.object(reporting, "_payloads_antigravity", return_value=({}, object())),
            mock.patch.object(reporting, "_final_payloads_antigravity", return_value=expected),
            mock.patch.object(reporting, "_inspect_publication_payloads_antigravity", side_effect=ValueError("frozen audit failure")),
            mock.patch.object(reporting, "_EXP446_BEFORE_CLEANUP_HOOK_ANTIGRAVITY", replace),
        ):
            with self.assertRaisesRegex(ValueError, "frozen audit failure"):
                lifecycle_antigravity.publish_prediction_antigravity()
        replacement = attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY
        self.assertTrue((replacement / "replacement_antigravity.txt").is_file())
        self.assertTrue(swapped.is_dir())
        self.assertTrue((attempt / reporting._EXP446_PUBLICATION_FAILURE_FILE_ANTIGRAVITY).is_file())
    def test_exp446_has_marker_inclusive_publication_capture_antigravity(self) -> None:
        """Generic marker-free capture cannot bridge Task5 terminal authority into publication."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting

        self.assertTrue(callable(reporting._capture_exp446_publication_authority_antigravity))
    def test_exp446_reporting_entrypoints_are_private_profile_bound_antigravity(self) -> None:
        """The repaired publisher must not reuse the public Exp444 wrapper."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting

        self.assertTrue(callable(reporting._publish_exp446_prediction_antigravity))
        self.assertTrue(callable(reporting._inspect_exp446_publication_antigravity))

    def test_finalization_failure_rejects_before_publisher_import_antigravity(self) -> None:
        """A terminal closeout failure is immutable ineligibility, not publishable evidence."""
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting

        with tempfile.TemporaryDirectory(prefix="exp446_finalization_failure_antigravity_") as directory:
            root = Path(directory) / "exp446_exp445_publication_materialization_repair_antigravity_20260811"
            attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
            attempt.mkdir(parents=True)
            (attempt / "terminal_finalization_failure_antigravity.json").write_text("{}\n")
            profile = SimpleNamespace(run_root=root)
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(reporting, "EXP446_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(lifecycle_antigravity, "_require_terminal_parity_antigravity", side_effect=ValueError("Exp446 terminal finalization failure evidence is present")) as require,
            ):
                with self.assertRaisesRegex(ValueError, "finalization failure"):
                    lifecycle_antigravity.publish_prediction_antigravity()
            require.assert_not_called()

    def test_terminal_capture_rejects_forged_design_or_preregistration_antigravity(self) -> None:
        """Task5 context document digests are semantic authority, not decorative hashes."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity

        base = lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY
        for field in ("reviewed_design_path", "preregistration_path"):
            with self.subTest(field=field), tempfile.TemporaryDirectory(prefix="exp446_forged_document_antigravity_") as directory:
                root, attempt, _exp444, _exp445 = self._real_terminal_root_antigravity()
                profile = object.__new__(type(base))
                for name in ("experiment_id", "schema", "run_identity", "reviewed_design_path", "preregistration_path", "predecessor_required"):
                    object.__setattr__(profile, name, getattr(base, name))
                forged = Path(directory) / f"forged_{field}_antigravity.md"
                forged.write_text("forged\n")
                object.__setattr__(profile, field, forged)
                with self.assertRaisesRegex(ValueError, "document"):
                    reporting._capture_exp446_publication_authority_antigravity(attempt, profile)

    def test_terminal_capture_rejects_regenerated_forged_prefix_digest_antigravity(self) -> None:
        """A syntactically regenerated marker cannot bless a prefix that disagrees with records."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity

        _root, attempt, _exp444, _exp445 = self._real_terminal_root_antigravity()
        state_path = attempt / "checkpoint_antigravity" / "state_antigravity.json"
        state = json.loads(state_path.read_text())
        state["record_sha256_prefix"][0] = "0" * 64
        state["payload_sha256"] = reporting._sha256_antigravity(reporting.canonical_json_bytes_antigravity(state["record_sha256_prefix"]))
        state_path.write_bytes(reporting.canonical_json_bytes_antigravity(state))
        marker_path = attempt / "terminal_parity_antigravity.json"
        marker = json.loads(marker_path.read_text())
        marker["record_prefix_payload_sha256"] = state["payload_sha256"]
        marker["authorities"][0] = state["payload_sha256"]
        terminal: dict[str, str] = {}
        for path in (attempt / "checkpoint_antigravity").glob("*.json"):
            terminal[f"checkpoint_antigravity/{path.name}"] = reporting._sha256_antigravity(path.read_bytes())
        for path in (attempt / "checkpoint_antigravity" / "records_antigravity").glob("*.json"):
            terminal[f"checkpoint_antigravity/records_antigravity/{path.name}"] = reporting._sha256_antigravity(path.read_bytes())
        for name in reporting._ATTEMPT_FILES_ANTIGRAVITY:
            terminal[name] = reporting._sha256_antigravity((attempt / name).read_bytes())
        marker["terminal_sha256"] = terminal
        marker_path.write_bytes(reporting.canonical_json_bytes_antigravity(marker))
        with self.assertRaisesRegex(ValueError, "terminal parity"):
            reporting._capture_exp446_publication_authority_antigravity(attempt, lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY)

    def test_exp446_pre_staging_order_authenticates_both_predecessors_then_materializes_control_antigravity(self) -> None:
        """Moving source capture before either failed-root parity gate must fail this order check."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp444_train_only_stability_precision_gate_antigravity import provenance_antigravity
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import predecessor_antigravity as exp444_predecessor
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import predecessor_antigravity as exp445_predecessor

        with tempfile.TemporaryDirectory(prefix="exp446_publication_order_antigravity_") as directory:
            root = Path(directory) / "exp446_exp445_publication_materialization_repair_antigravity_20260811"
            attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
            attempt.mkdir(parents=True)
            profile = SimpleNamespace(run_root=root)
            snapshot = SimpleNamespace(payloads={}, attempt_fd=-1, close_antigravity=lambda: None)
            authority = SimpleNamespace(record_sha256_prefix=("a" * 64,) * 30, record_prefix_payload_sha256="b" * 64)
            order: list[str] = []

            def parity(_snapshot, checked):
                order.append("parity-444" if checked is exp444 else "parity-445")

            exp444 = object()
            exp445 = object()
            with (
                mock.patch.object(reporting, "EXP446_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(reporting, "_capture_exp446_publication_authority_antigravity", return_value=object()),
                mock.patch.object(reporting, "_capture_exp446_publication_snapshot_antigravity", return_value=snapshot),
                mock.patch.object(reporting, "_capture_attempt_antigravity", return_value=snapshot),
                mock.patch.object(reporting, "_assert_exp446_publication_snapshot_antigravity"),
                mock.patch.object(reporting, "_complete_checkpoint_results_antigravity", side_effect=lambda *_args, **_kwargs: order.append("terminal") or ()),
                mock.patch.object(exp444_predecessor, "verify_exp444_predecessor_antigravity", side_effect=lambda: order.append("exp444") or exp444),
                mock.patch.object(exp445_predecessor, "verify_exp445_predecessor_antigravity", side_effect=lambda: order.append("exp445") or exp445),
                mock.patch.object(reporting, "_assert_exp445_predecessor_parity_antigravity", side_effect=parity),
                mock.patch.object(provenance_antigravity, "verify_exp444_provenance_antigravity", side_effect=lambda: order.append("provenance") or SimpleNamespace(dataset_summary_bytes=b"", backend_authority={})),
                mock.patch.object(reporting, "_publish_source_authority_antigravity", side_effect=lambda *_args: order.append("source") or authority),
                mock.patch.object(reporting, "_payloads_antigravity", side_effect=lambda *_args, **_kwargs: order.append("materialize") or (_ for _ in ()).throw(ValueError("control invalid"))),
                mock.patch.object(reporting, "_write_staging_antigravity", side_effect=AssertionError("staging opened")),
            ):
                with self.assertRaisesRegex(ValueError, "control invalid"):
                    reporting._publish_exp446_prediction_antigravity(profile)
            self.assertEqual(order, ["terminal", "exp444", "exp445", "parity-444", "parity-445", "provenance", "source", "materialize"])
            self.assertTrue((attempt / reporting._EXP446_PUBLICATION_FAILURE_FILE_ANTIGRAVITY).is_file())

    def test_exp446_postrename_audit_quarantines_canonical_publication_antigravity(self) -> None:
        """A rejected frozen-leaf audit must preserve bytes off the canonical publication path."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp444_train_only_stability_precision_gate_antigravity import provenance_antigravity
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import predecessor_antigravity as exp444_predecessor
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import predecessor_antigravity as exp445_predecessor

        with tempfile.TemporaryDirectory(prefix="exp446_publication_postrename_antigravity_") as directory:
            root = Path(directory) / "exp446_exp445_publication_materialization_repair_antigravity_20260811"
            attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
            records = attempt / "checkpoint_antigravity" / "records_antigravity"
            records.mkdir(parents=True)
            for name in reporting._ATTEMPT_FILES_ANTIGRAVITY:
                (attempt / name).write_bytes(b"{}\n")
            for name in reporting._CHECKPOINT_FILES_ANTIGRAVITY:
                (attempt / "checkpoint_antigravity" / name).write_bytes(b"{}\n")
            for name in reporting._RECORD_NAMES_ANTIGRAVITY:
                (records / name).write_bytes(b"{}\n")
            profile = SimpleNamespace(run_root=root)
            predecessor = SimpleNamespace(record_sha256_prefix=("a" * 64,) * 30, record_prefix_payload_sha256="b" * 64)
            expected = {name: f"{index}".encode() for index, name in enumerate(reporting._PUBLICATION_FILES_ANTIGRAVITY)}
            with (
                mock.patch.object(reporting, "EXP446_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(reporting, "_capture_exp446_publication_authority_antigravity", return_value=object()),
                mock.patch.object(reporting, "_capture_exp446_publication_snapshot_antigravity", side_effect=lambda *_args: reporting._capture_attempt_antigravity(attempt)),
                mock.patch.object(reporting, "_assert_exp446_terminal_authority_unchanged_antigravity"),
                mock.patch.object(reporting, "_assert_exp446_publication_snapshot_antigravity"),
                mock.patch.object(reporting, "_complete_checkpoint_results_antigravity", return_value=()),
                mock.patch.object(exp444_predecessor, "verify_exp444_predecessor_antigravity", return_value=predecessor),
                mock.patch.object(exp445_predecessor, "verify_exp445_predecessor_antigravity", return_value=predecessor),
                mock.patch.object(reporting, "_assert_exp445_predecessor_parity_antigravity"),
                mock.patch.object(provenance_antigravity, "verify_exp444_provenance_antigravity", return_value=SimpleNamespace(dataset_summary_bytes=b"", backend_authority={})),
                mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=object()),
                mock.patch.object(reporting, "_payloads_antigravity", return_value=({}, object())),
                mock.patch.object(reporting, "_final_payloads_antigravity", return_value=expected),
                mock.patch.object(reporting, "_inspect_publication_payloads_antigravity", side_effect=ValueError("frozen audit failure")),
            ):
                with self.assertRaisesRegex(ValueError, "frozen audit failure"):
                    reporting._publish_exp446_prediction_antigravity(profile)
            self.assertFalse((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())
            self.assertTrue((attempt / reporting._EXP446_PUBLICATION_FAILURE_FILE_ANTIGRAVITY).is_file())
            self.assertEqual(len([item for item in attempt.iterdir() if "quarantine_antigravity" in item.name]), 1)
