"""TDD coverage for Exp445's create-only parity-gated publication boundary."""
from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock


class Exp445PublicationAntigravityTests(unittest.TestCase):
    def test_profile_publication_boundary_exists_before_operator_exposure_antigravity(self) -> None:
        """Task 6 needs a closed Exp445-only publication implementation."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting

        self.assertTrue(callable(reporting._publish_exp445_prediction_antigravity))

    @staticmethod
    def _profile_antigravity(root: Path) -> SimpleNamespace:
        return SimpleNamespace(
            experiment_id="Exp445", schema="exp445",
            run_identity="exp445-exp444-publication-source-authority-repair-antigravity-20260810",
            run_root=root,
        )

    @staticmethod
    def _snapshot_antigravity() -> SimpleNamespace:
        return SimpleNamespace(
            payloads={}, attempt_fd=-1,
            close_antigravity=lambda: None,
        )

    def _attempt_antigravity(self, directory: str) -> tuple[Path, Path, SimpleNamespace]:
        root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"
        attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
        attempt.mkdir(parents=True)
        return root, attempt, self._snapshot_antigravity()

    def test_terminal_failure_stops_before_predecessor_and_staging_antigravity(self) -> None:
        """An incomplete checkpoint cannot open predecessor/source or publication paths."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import predecessor_antigravity

        with tempfile.TemporaryDirectory(prefix="exp445_publication_terminal_antigravity_") as directory:
            root, attempt, snapshot = self._attempt_antigravity(directory)
            profile = self._profile_antigravity(root)
            with (
                mock.patch.object(reporting, "EXP445_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(reporting, "_capture_attempt_antigravity", return_value=snapshot),
                mock.patch.object(reporting, "_complete_checkpoint_results_antigravity", side_effect=ValueError("not complete")),
                mock.patch.object(predecessor_antigravity, "verify_exp444_predecessor_antigravity", side_effect=AssertionError("predecessor opened")),
                mock.patch.object(reporting, "_write_staging_antigravity", side_effect=AssertionError("staging opened")),
            ):
                with self.assertRaisesRegex(ValueError, "not complete"):
                    reporting._publish_exp445_prediction_antigravity(profile)
            self.assertFalse((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())
            self.assertTrue((attempt / reporting._EXP445_PUBLICATION_FAILURE_FILE_ANTIGRAVITY).is_file())

    def test_each_pre_staging_gate_preserves_no_complete_publication_antigravity(self) -> None:
        """Fresh predecessor, parity, and Exp440 provenance fail in their fixed order."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp444_train_only_stability_precision_gate_antigravity import provenance_antigravity
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import predecessor_antigravity

        cases = (
            ("predecessor", ValueError("predecessor")),
            ("parity", ValueError("parity")),
            ("provenance", ValueError("provenance")),
        )
        for name, failure in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix=f"exp445_publication_{name}_antigravity_") as directory:
                root, attempt, snapshot = self._attempt_antigravity(directory)
                profile = self._profile_antigravity(root)
                order: list[str] = []

                def complete(*_args, **_kwargs):
                    order.append("terminal")
                    return ()

                def predecessor():
                    order.append("predecessor")
                    if name == "predecessor":
                        raise failure
                    return SimpleNamespace(record_sha256_prefix=("a" * 64,) * 30, record_prefix_payload_sha256="b" * 64)

                def parity(*_args, **_kwargs):
                    order.append("parity")
                    if name == "parity":
                        raise failure

                def provenance():
                    order.append("provenance")
                    if name == "provenance":
                        raise failure
                    return SimpleNamespace(dataset_summary_bytes=b"", backend_authority={})

                with (
                    mock.patch.object(reporting, "EXP445_PROFILE_ANTIGRAVITY", profile),
                    mock.patch.object(reporting, "_capture_attempt_antigravity", return_value=snapshot),
                    mock.patch.object(reporting, "_complete_checkpoint_results_antigravity", side_effect=complete),
                    mock.patch.object(predecessor_antigravity, "verify_exp444_predecessor_antigravity", side_effect=predecessor),
                    mock.patch.object(reporting, "_assert_exp445_predecessor_parity_antigravity", side_effect=parity),
                    mock.patch.object(provenance_antigravity, "verify_exp444_provenance_antigravity", side_effect=provenance),
                    mock.patch.object(reporting, "_publish_source_authority_antigravity", side_effect=AssertionError("source opened")),
                    mock.patch.object(reporting, "_write_staging_antigravity", side_effect=AssertionError("staging opened")),
                ):
                    with self.assertRaisesRegex(ValueError, str(failure)):
                        reporting._publish_exp445_prediction_antigravity(profile)
                self.assertEqual(tuple(order), ("terminal", "predecessor") if name == "predecessor" else ("terminal", "predecessor", "parity") if name == "parity" else ("terminal", "predecessor", "parity", "provenance"))
                self.assertFalse((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())

    def test_staging_failure_is_create_only_and_same_root_retry_is_rejected_antigravity(self) -> None:
        """A staging fault leaves no complete destination, and an existing destination is never reused."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp444_train_only_stability_precision_gate_antigravity import provenance_antigravity
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import predecessor_antigravity

        with tempfile.TemporaryDirectory(prefix="exp445_publication_staging_antigravity_") as directory:
            root, attempt, snapshot = self._attempt_antigravity(directory)
            profile = self._profile_antigravity(root)
            predecessor = SimpleNamespace(record_sha256_prefix=("a" * 64,) * 30, record_prefix_payload_sha256="b" * 64)
            with (
                mock.patch.object(reporting, "EXP445_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(reporting, "_capture_attempt_antigravity", return_value=snapshot),
                mock.patch.object(reporting, "_complete_checkpoint_results_antigravity", return_value=()),
                mock.patch.object(predecessor_antigravity, "verify_exp444_predecessor_antigravity", return_value=predecessor),
                mock.patch.object(reporting, "_assert_exp445_predecessor_parity_antigravity"),
                mock.patch.object(provenance_antigravity, "verify_exp444_provenance_antigravity", return_value=SimpleNamespace(dataset_summary_bytes=b"", backend_authority={})),
                mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=object()),
                mock.patch.object(reporting, "_payloads_antigravity", return_value=({}, object())),
                mock.patch.object(reporting, "_final_payloads_antigravity", return_value={}),
                mock.patch.object(reporting, "_assert_attempt_namespace_antigravity"),
                mock.patch.object(reporting, "_write_staging_antigravity", side_effect=OSError("staging fault")),
            ):
                with self.assertRaisesRegex(OSError, "staging fault"):
                    reporting._publish_exp445_prediction_antigravity(profile)
            self.assertFalse((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())
            self.assertTrue((attempt / reporting._EXP445_PUBLICATION_FAILURE_FILE_ANTIGRAVITY).is_file())
            (attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).mkdir()
            with (
                mock.patch.object(reporting, "EXP445_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(reporting, "_capture_attempt_antigravity", side_effect=AssertionError("attempt reopened")),
            ):
                with self.assertRaisesRegex(ValueError, "same-root retry"):
                    reporting._publish_exp445_prediction_antigravity(profile)

    def test_post_rename_audit_failure_keeps_exp445_canonical_destination_absent_antigravity(self) -> None:
        """A writer-reported post-rename rejection cannot be returned as publication success."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
        from experiments.exp444_train_only_stability_precision_gate_antigravity import provenance_antigravity
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import predecessor_antigravity

        with tempfile.TemporaryDirectory(prefix="exp445_publication_postrename_antigravity_") as directory:
            root, attempt, snapshot = self._attempt_antigravity(directory)
            profile = self._profile_antigravity(root)
            predecessor = SimpleNamespace(record_sha256_prefix=("a" * 64,) * 30, record_prefix_payload_sha256="b" * 64)

            def rejected_after_rename(destination: Path, *_args, **_kwargs) -> None:
                destination.mkdir()
                destination.rename(destination.with_name(".publication_antigravity.audit_quarantine_antigravity"))
                raise OSError("post-rename audit failure")

            with (
                mock.patch.object(reporting, "EXP445_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(reporting, "_capture_attempt_antigravity", return_value=snapshot),
                mock.patch.object(reporting, "_complete_checkpoint_results_antigravity", return_value=()),
                mock.patch.object(predecessor_antigravity, "verify_exp444_predecessor_antigravity", return_value=predecessor),
                mock.patch.object(reporting, "_assert_exp445_predecessor_parity_antigravity"),
                mock.patch.object(provenance_antigravity, "verify_exp444_provenance_antigravity", return_value=SimpleNamespace(dataset_summary_bytes=b"", backend_authority={})),
                mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=object()),
                mock.patch.object(reporting, "_payloads_antigravity", return_value=({}, object())),
                mock.patch.object(reporting, "_final_payloads_antigravity", return_value={}),
                mock.patch.object(reporting, "_assert_attempt_namespace_antigravity"),
                mock.patch.object(reporting, "_write_staging_antigravity", side_effect=rejected_after_rename),
            ):
                with self.assertRaisesRegex(OSError, "post-rename audit failure"):
                    reporting._publish_exp445_prediction_antigravity(profile)
            self.assertFalse((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())
            self.assertTrue((attempt / reporting._EXP445_PUBLICATION_FAILURE_FILE_ANTIGRAVITY).is_file())

    def test_lifecycle_facade_has_no_profile_or_source_override_antigravity(self) -> None:
        """The only lifecycle handoff passes the closed singleton without user inputs."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity

        with mock.patch.object(lifecycle_antigravity, "_publish_exp445_prediction_antigravity", return_value=mock.sentinel.publication) as publish:
            self.assertIs(lifecycle_antigravity.publish_prediction_antigravity(), mock.sentinel.publication)
        publish.assert_called_once_with(lifecycle_antigravity._EXP445_PROFILE_ANTIGRAVITY)

    def test_exp445_inspection_replays_frozen_leaves_without_live_sources_antigravity(self) -> None:
        """Read-only inspection owns only publication leaves and no source authority."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting

        @contextmanager
        def frozen_payloads(*_args, **_kwargs):
            yield {}

        with tempfile.TemporaryDirectory(prefix="exp445_publication_inspect_antigravity_") as directory:
            root = Path(directory) / "publication_antigravity"
            root.mkdir()
            with (
                mock.patch.object(reporting, "_pinned_publication_payloads_antigravity", frozen_payloads),
                mock.patch.object(reporting, "_inspect_publication_payloads_antigravity", return_value=({}, object())) as inspect,
                mock.patch.object(reporting, "FrozenPublicationAntigravity", return_value=mock.sentinel.frozen) as frozen,
                mock.patch.object(reporting, "_publish_source_authority_antigravity", side_effect=AssertionError("source opened")),
            ):
                self.assertIs(reporting._inspect_exp445_publication_antigravity(root), mock.sentinel.frozen)
            inspect.assert_called_once_with({}, profile=reporting.EXP445_PROFILE_ANTIGRAVITY)
            frozen.assert_called_once()
