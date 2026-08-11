"""Closed publication boundary tests for Exp446."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class Exp446PublicationAntigravityTests(unittest.TestCase):
    def test_exp446_reporting_entrypoints_are_private_profile_bound_antigravity(self) -> None:
        """The repaired publisher must not reuse the public Exp444 wrapper."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting

        self.assertTrue(callable(reporting._publish_exp446_prediction_antigravity))
        self.assertTrue(callable(reporting._inspect_exp446_publication_antigravity))

    def test_finalization_failure_rejects_before_publisher_import_antigravity(self) -> None:
        """A terminal closeout failure is immutable ineligibility, not publishable evidence."""
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity

        with tempfile.TemporaryDirectory(prefix="exp446_finalization_failure_antigravity_") as directory:
            root = Path(directory) / "exp446_exp445_publication_materialization_repair_antigravity_20260811"
            attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
            attempt.mkdir(parents=True)
            (attempt / "terminal_finalization_failure_antigravity.json").write_text("{}\n")
            profile = SimpleNamespace(run_root=root)
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(lifecycle_antigravity, "_require_terminal_parity_antigravity", side_effect=ValueError("Exp446 terminal finalization failure evidence is present")) as require,
            ):
                with self.assertRaisesRegex(ValueError, "finalization failure"):
                    lifecycle_antigravity.publish_prediction_antigravity()
            require.assert_not_called()

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
                mock.patch.object(reporting, "_capture_attempt_antigravity", return_value=snapshot),
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
