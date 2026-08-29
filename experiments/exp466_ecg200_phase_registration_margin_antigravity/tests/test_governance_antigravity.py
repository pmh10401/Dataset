from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.exp466_ecg200_phase_registration_margin_antigravity.tests.test_cli_antigravity import (
    _DecisionAntigravity,
    inspected_antigravity,
)


_COMMIT_ANTIGRAVITY = "b" * 40
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_PACKAGE_ANTIGRAVITY = "experiments.exp466_ecg200_phase_registration_margin_antigravity"
_LAUNCHER_PATH_ANTIGRAVITY = _REPOSITORY_ANTIGRAVITY / (
    "scratch/run_exp466_ecg200_phase_registration_margin_antigravity.py"
)
_RUN1_ANTIGRAVITY = Path("/private/tmp/exp466_task6_private_run1_antigravity")
_RUN2_ANTIGRAVITY = Path("/private/tmp/exp466_task6_private_run2_antigravity")
_OFFICIAL_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp466_ecg200_phase_registration_margin_antigravity_20260823",
)


def _launcher_module_antigravity():
    spec = importlib.util.spec_from_file_location(
        "_exp466_task6_launcher_unit_antigravity",
        _LAUNCHER_PATH_ANTIGRAVITY,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("launcher module spec is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CliGovernanceAntigravityTest(unittest.TestCase):
    def test_parser_has_only_two_commands_and_one_option_antigravity(self) -> None:
        """Break caught: a hidden action becomes a tuning or routing control."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.cli_antigravity import (
            parser_antigravity,
        )

        parser = parser_antigravity()
        choices = parser._subparsers._group_actions[0].choices
        self.assertEqual(tuple(choices), ("evaluate", "inspect"))
        self.assertEqual(len(choices["evaluate"]._actions), 2)
        self.assertEqual(
            choices["evaluate"]._actions[1].option_strings,
            ["--reviewed-commit"],
        )
        self.assertEqual(len(choices["inspect"]._actions), 1)

    def test_create_only_root_rejects_existing_symlink_and_official_antigravity(self) -> None:
        """Break caught: publication overwrites evidence or enters official scope."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity import (
            cli_antigravity as cli,
        )

        self.assertEqual(cli.OFFICIAL_ROOT_ANTIGRAVITY, _OFFICIAL_ANTIGRAVITY)
        with tempfile.TemporaryDirectory(prefix="exp466_create_only_antigravity_") as temporary:
            base = Path(temporary)
            unused = base / "unused"
            cli._validate_create_only_root_antigravity(unused)
            existing = base / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "create-only"):
                cli._validate_create_only_root_antigravity(existing)
            symlink = base / "symlink"
            symlink.symlink_to(existing)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                cli._validate_create_only_root_antigravity(symlink)
        with self.assertRaisesRegex(ValueError, "non-official"):
            cli._validate_create_only_root_antigravity(_OFFICIAL_ANTIGRAVITY)

    def test_clean_commit_official_absence_and_duplicate_process_antigravity(self) -> None:
        """Break caught: dirty, wrong, official-present, or concurrent work starts."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity import (
            cli_antigravity as cli,
        )

        with (
            patch.object(cli.subprocess, "check_output", side_effect=["c" * 40 + "\n", ""]),
            self.assertRaisesRegex(ValueError, "exact clean current commit"),
        ):
            cli._validate_clean_current_commit_antigravity(_COMMIT_ANTIGRAVITY)
        with (
            patch.object(
                cli.subprocess,
                "check_output",
                side_effect=[_COMMIT_ANTIGRAVITY + "\n", "dirty\n"],
            ),
            self.assertRaisesRegex(ValueError, "exact clean current commit"),
        ):
            cli._validate_clean_current_commit_antigravity(_COMMIT_ANTIGRAVITY)

        official = Mock()
        official.exists.return_value = True
        official.is_symlink.return_value = False
        with (
            patch.object(cli, "OFFICIAL_ROOT_ANTIGRAVITY", official),
            patch.object(
                cli.subprocess,
                "check_output",
                side_effect=[_COMMIT_ANTIGRAVITY + "\n", ""],
            ),
            self.assertRaisesRegex(ValueError, "official result root must be absent"),
        ):
            cli._validate_clean_current_commit_antigravity(_COMMIT_ANTIGRAVITY)

        duplicate = f" 501 1 scratch/{cli._LAUNCHER_NAME_ANTIGRAVITY} evaluate\n"
        with (
            patch.object(
                cli.subprocess,
                "check_output",
                side_effect=[_COMMIT_ANTIGRAVITY + "\n", "", duplicate],
            ),
            patch.object(cli.os, "getpid", return_value=99),
            self.assertRaisesRegex(ValueError, "duplicate Exp466 process"),
        ):
            cli._validate_clean_current_commit_antigravity(_COMMIT_ANTIGRAVITY)
        with patch.object(
            cli.subprocess,
            "check_output",
            side_effect=[_COMMIT_ANTIGRAVITY + "\n", "", " 99 1 ignored\n"],
        ):
            cli._validate_clean_current_commit_antigravity(_COMMIT_ANTIGRAVITY)

    def test_inspect_rejects_official_root_before_loading_replay_antigravity(self) -> None:
        """Break caught: official output appears between private evaluate and inspect."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity import (
            cli_antigravity as cli,
        )

        official = Mock()
        official.exists.return_value = True
        official.is_symlink.return_value = False
        with (
            patch.object(cli, "OFFICIAL_ROOT_ANTIGRAVITY", official),
            patch.object(cli, "_private_test_root_antigravity", return_value=_RUN1_ANTIGRAVITY),
            patch.object(
                cli,
                "_load_inspect_antigravity",
                side_effect=AssertionError("official guard must run before replay"),
            ),
            self.assertRaisesRegex(ValueError, "official result root must be absent"),
        ):
            cli.main_antigravity(["inspect"])

    def test_launcher_ast_binds_runtime_roots_and_sequence_guard_antigravity(self) -> None:
        """Break caught: launcher gains a third root or drops run ordering."""
        source = _LAUNCHER_PATH_ANTIGRAVITY.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignments: dict[str, ast.AST] = {}
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                assignments[node.targets[0].id] = node.value
        roots = assignments["_PRIVATE_ROOTS_ANTIGRAVITY"]
        self.assertIsInstance(roots, ast.Dict)
        self.assertEqual(
            [ast.literal_eval(key) for key in roots.keys],
            ["run1", "run2"],
        )

        def path_value_antigravity(node: ast.AST) -> Path:
            self.assertIsInstance(node, ast.Call)
            return Path(ast.literal_eval(node.args[0]))  # type: ignore[union-attr]

        self.assertEqual(
            [path_value_antigravity(value) for value in roots.values],
            [_RUN1_ANTIGRAVITY, _RUN2_ANTIGRAVITY],
        )
        self.assertIn("detection-coverage-antigravity-aeon140-py314/bin/python", source)
        self.assertIn("os.execve", source)
        self.assertIn("run1 inspect must complete before run2", source)
        self.assertIn("_validate_private_sequence_antigravity", source)

    def test_launcher_rejects_skipped_repeated_and_commit_drift_phases_antigravity(self) -> None:
        """Break caught: run2 starts before run1 inspect or at another commit."""
        launcher = _launcher_module_antigravity()
        run1_commit = "b" * 40
        state = {
            "schema": "exp466_task6_private_sequence_antigravity_v1",
            "stage": "run1_evaluated",
            "reviewed_commit": run1_commit,
        }
        with (
            patch.object(launcher, "_load_sequence_state_antigravity", return_value=state),
            self.assertRaisesRegex(RuntimeError, "run1 inspect"),
        ):
            launcher._validate_private_sequence_antigravity(
                "run2",
                ("evaluate", "--reviewed-commit", run1_commit),
            )
        repeated = dict(state, stage="run1_inspected")
        with (
            patch.object(launcher, "_load_sequence_state_antigravity", return_value=repeated),
            self.assertRaisesRegex(RuntimeError, "expected stage run1_evaluated"),
        ):
            launcher._validate_private_sequence_antigravity("run1", ("inspect",))
        with (
            patch.object(launcher, "_load_sequence_state_antigravity", return_value=repeated),
            patch.object(launcher, "_has_exact_inventory_antigravity", return_value=True),
            patch.object(launcher, "_is_absent_antigravity", return_value=True),
            patch.object(launcher, "_current_clean_commit_antigravity", return_value=run1_commit),
            self.assertRaisesRegex(RuntimeError, "same reviewed commit"),
        ):
            launcher._validate_private_sequence_antigravity(
                "run2",
                ("evaluate", "--reviewed-commit", "c" * 40),
            )

    def test_launcher_advances_sequence_only_after_success_antigravity(self) -> None:
        """Break caught: failed evaluate/inspect is recorded as completed."""
        launcher = _launcher_module_antigravity()
        main = Mock(return_value=0)
        with (
            patch.object(launcher, "_ensure_fixed_runtime_antigravity"),
            patch.object(launcher, "_install_private_capability_antigravity", return_value="run1"),
            patch.object(launcher, "_load_main_antigravity", return_value=main),
            patch.object(launcher, "_advance_private_sequence_antigravity") as advance,
            patch.object(launcher.sys, "argv", [str(_LAUNCHER_PATH_ANTIGRAVITY), "inspect"]),
        ):
            self.assertEqual(launcher._run_antigravity(), 0)
        advance.assert_called_once_with("run1", ("inspect",))

        failed = Mock(side_effect=ValueError("failed command"))
        with (
            patch.object(launcher, "_ensure_fixed_runtime_antigravity"),
            patch.object(launcher, "_install_private_capability_antigravity", return_value="run1"),
            patch.object(launcher, "_load_main_antigravity", return_value=failed),
            patch.object(launcher, "_advance_private_sequence_antigravity") as advance,
            patch.object(launcher.sys, "argv", [str(_LAUNCHER_PATH_ANTIGRAVITY), "inspect"]),
            self.assertRaisesRegex(ValueError, "failed command"),
        ):
            launcher._run_antigravity()
        self.assertEqual(advance.call_count, 0)

    def test_launcher_sequence_state_is_canonical_and_replaceable_antigravity(self) -> None:
        """Break caught: phase evidence cannot be replayed after a successful command."""
        launcher = _launcher_module_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp466_sequence_state_antigravity_") as temporary:
            state = Path(temporary) / "sequence_state_antigravity.json"
            next_state = Path(temporary) / "sequence_next_antigravity.json"
            with (
                patch.object(launcher, "_SEQUENCE_STATE_PATH_ANTIGRAVITY", state),
                patch.object(launcher, "_SEQUENCE_NEXT_PATH_ANTIGRAVITY", next_state),
            ):
                self.assertIsNone(launcher._load_sequence_state_antigravity())
                launcher._write_sequence_state_antigravity("run1_evaluated", _COMMIT_ANTIGRAVITY)
                self.assertEqual(
                    launcher._load_sequence_state_antigravity(),
                    {
                        "reviewed_commit": _COMMIT_ANTIGRAVITY,
                        "schema": "exp466_task6_private_sequence_antigravity_v1",
                        "stage": "run1_evaluated",
                    },
                )
                launcher._write_sequence_state_antigravity("run1_inspected", _COMMIT_ANTIGRAVITY)
                self.assertEqual(
                    launcher._load_sequence_state_antigravity()["stage"],
                    "run1_inspected",
                )
                self.assertFalse(next_state.exists())

    def test_evaluate_pipeline_is_seed_major_and_calls_each_stage_once_antigravity(self) -> None:
        """Break caught: loader, folds, scoring, publication, or replay is rerouted."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity import (
            artifacts_antigravity,
            cli_antigravity as cli,
            evaluation_antigravity,
            interpolation_antigravity,
            predecessor_antigravity,
            qualification_antigravity,
            replay_antigravity,
            scoring_antigravity,
            sources_antigravity,
        )

        normal96 = np.arange(69 * 96, dtype=np.float64).reshape(69, 96)
        train140 = np.arange(69 * 140, dtype=np.float64).reshape(69, 140)
        authority = object()
        scored = [object() for _ in range(10)]
        qualification = types.SimpleNamespace()
        decision = _DecisionAntigravity()
        inspected = inspected_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp466_pipeline_antigravity_") as temporary:
            root = Path(temporary) / "result"
            publication = types.SimpleNamespace(
                root=root,
                leaf_sha256=inspected.leaf_sha256,
            )
            with (
                patch.object(sources_antigravity, "load_ecg200_normal_train_antigravity", return_value=normal96) as loader,
                patch.object(interpolation_antigravity, "interpolate_to_140_antigravity", return_value=train140) as interpolate,
                patch.object(predecessor_antigravity, "capture_exp465_authority_antigravity", return_value=authority) as predecessor,
                patch.object(scoring_antigravity, "score_fold_methods_antigravity", side_effect=scored) as scorer,
                patch.object(qualification_antigravity, "qualify_antigravity", return_value=qualification) as qualify,
                patch.object(evaluation_antigravity, "evaluate_qualification_antigravity", return_value=decision) as evaluate,
                patch.object(artifacts_antigravity, "publish_result_antigravity", return_value=publication) as publish,
                patch.object(replay_antigravity, "inspect_result_antigravity", return_value=inspected) as inspect,
            ):
                payload = cli._evaluate_at_root_antigravity(root)

        self.assertEqual(loader.call_count, 1)
        self.assertEqual(loader.call_args.args[0], cli._TRAIN_PATH_ANTIGRAVITY)
        interpolate.assert_called_once_with(normal96)
        self.assertEqual(predecessor.call_count, 1)
        self.assertEqual(scorer.call_count, 10)
        qualify.assert_called_once()
        evaluate.assert_called_once_with(qualification)
        publish.assert_called_once()
        inspect.assert_called_once_with(root)
        records = qualify.call_args.args[0]
        expected_keys = tuple(
            (seed, fold)
            for seed in (20260721, 20261721)
            for fold in range(5)
        )
        self.assertEqual(tuple(records), expected_keys)
        self.assertEqual(tuple(records.values()), tuple(scored))
        self.assertEqual(
            tuple(call.kwargs["seed"] for call in scorer.call_args_list),
            (20260721,) * 5 + (20261721,) * 5,
        )
        self.assertEqual(
            tuple(call.kwargs["fold_index"] for call in scorer.call_args_list),
            (0, 1, 2, 3, 4) * 2,
        )
        self.assertEqual(
            tuple(len(call.args[2]) for call in scorer.call_args_list),
            (14, 14, 14, 14, 13) * 2,
        )
        self.assertTrue(all(call.kwargs["authority"] is authority for call in scorer.call_args_list))
        self.assertEqual(
            payload,
            cli._result_payload_from_inspected_antigravity(inspected),
        )

    def test_inspect_delegates_only_to_source_free_replay_antigravity(self) -> None:
        """Break caught: inspect loads data, scores, qualifies, or publishes."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity import (
            artifacts_antigravity,
            cli_antigravity as cli,
            qualification_antigravity,
            replay_antigravity,
            scoring_antigravity,
            sources_antigravity,
        )

        inspected = inspected_antigravity()
        root = Path("/private/tmp/exp466_nonexistent_inspect_test_antigravity")
        with (
            patch.object(replay_antigravity, "inspect_result_antigravity", return_value=inspected) as inspect,
            patch.object(sources_antigravity, "load_ecg200_normal_train_antigravity", side_effect=AssertionError("inspect loaded data")),
            patch.object(scoring_antigravity, "score_fold_methods_antigravity", side_effect=AssertionError("inspect scored")),
            patch.object(qualification_antigravity, "qualify_antigravity", side_effect=AssertionError("inspect qualified")),
            patch.object(artifacts_antigravity, "publish_result_antigravity", side_effect=AssertionError("inspect published")),
        ):
            payload = cli._inspect_at_root_antigravity(root)
        inspect.assert_called_once_with(root)
        self.assertEqual(payload, cli._result_payload_from_inspected_antigravity(inspected))

    def test_cli_source_has_no_test_label_or_routing_controls_antigravity(self) -> None:
        """Break caught: CLI introduces TEST, retry, promotion, or routing knobs."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity import (
            cli_antigravity as cli,
        )

        for name in (
            "TEST_PATH_ANTIGRAVITY",
            "LABEL_ACCESS_ANTIGRAVITY",
            "RETRY_CONTROL_ANTIGRAVITY",
            "ROUTING_CONTROL_ANTIGRAVITY",
        ):
            self.assertIsNone(getattr(cli, name, None))
        source = Path(cli.__file__).read_text(encoding="utf-8")
        self.assertIn("test_labels_opened", source)
        self.assertNotIn("ECG200_TEST", source)


if __name__ == "__main__":
    unittest.main()
