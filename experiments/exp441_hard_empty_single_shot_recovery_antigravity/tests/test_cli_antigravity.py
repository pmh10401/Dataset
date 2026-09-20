"""Behavior tests for Exp441's deliberately small CLI surface."""
from __future__ import annotations

import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import types
from types import MappingProxyType
import unittest
from unittest.mock import patch


class CliAntigravityTests(unittest.TestCase):
    """The production changes caught here are route expansion and non-JSON success output."""

    def test_public_cli_has_exact_four_commands(self) -> None:
        module = importlib.import_module(
            "experiments.exp441_hard_empty_single_shot_recovery_antigravity.cli_antigravity"
        )
        parser = module.build_parser_antigravity()
        self.assertEqual(set(parser._subparsers._group_actions[0].choices), {"readiness", "new", "inspect", "evaluate"})
        with self.assertRaises(SystemExit):
            parser.parse_args(["resume"])

    def test_readiness_route_returns_zero_and_prints_one_canonical_json_document(self) -> None:
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli

        result = {"readiness_sha256": "a" * 64}
        stream = io.StringIO()
        with patch.object(cli, "current_clean_head_antigravity", return_value="b" * 40), patch(
            "experiments.exp441_hard_empty_single_shot_recovery_antigravity.readiness_antigravity.run_readiness_antigravity",
            return_value=result,
        ):
            with contextlib.redirect_stdout(stream):
                self.assertEqual(cli.main_antigravity(["readiness"]), 0)
        self.assertEqual(json.loads(stream.getvalue()), {"command": "readiness", "result": result})
        self.assertEqual(stream.getvalue().encode(), json.dumps(json.loads(stream.getvalue()), sort_keys=True, separators=(",", ":")).encode() + b"\n")

    def test_parser_rejects_every_override_surface(self) -> None:
        """A route accepting path/worker/backend/seed/budget flags would unfreeze Exp441."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli

        parser = cli.build_parser_antigravity()
        for flag in ("--path", "--workers", "--backend", "--seed", "--budget", "--source", "--threshold", "--bootstrap"):
            with self.assertRaises(SystemExit, msg=flag): parser.parse_args(["readiness", flag, "x"])

    def test_evaluate_is_the_sole_lazy_evaluator_route(self) -> None:
        """The implemented evaluator stays lazy and emits the same one-document CLI form."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli

        module_name = "experiments.exp441_hard_empty_single_shot_recovery_antigravity.evaluation_antigravity"
        fake = types.ModuleType(module_name)
        fake.evaluate_exp441_antigravity = lambda: {"evaluation_sha256": "a" * 64}  # type: ignore[attr-defined]
        stream = io.StringIO()
        with patch.dict(sys.modules, {module_name: fake}):
            with contextlib.redirect_stdout(stream):
                self.assertEqual(cli.main_antigravity(["evaluate"]), 0)
        self.assertEqual(json.loads(stream.getvalue()), {
            "command": "evaluate", "result": {"evaluation_sha256": "a" * 64},
        })

    def test_outside_cwd_wrapper_help_needs_only_the_wrapper_and_cli(self) -> None:
        """Resolving imports relative to CWD would break the documented outside-CWD command."""
        wrapper = Path(__file__).resolve().parents[3] / "scratch" / "run_exp441_hard_empty_single_shot_recovery_antigravity.py"
        completed = subprocess.run([sys.executable, str(wrapper), "--help"], cwd="/tmp", check=True, capture_output=True, text=True)
        self.assertIn("{readiness,new,inspect,evaluate}", completed.stdout)

    def test_real_readiness_binding_serializes_directly_and_through_main(self) -> None:
        """A real MappingProxyType environment must not fail before canonical readiness stdout."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import readiness_antigravity as readiness
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import (
            BackendAssessmentAntigravity, ReadinessBindingAntigravity,
        )

        backend = readiness._backend_antigravity()
        self.assertIs(type(backend.environment), MappingProxyType)
        binding = ReadinessBindingAntigravity("a" * 40, "b" * 64, backend, "c" * 64)
        expected_result = {
            "reviewed_commit": "a" * 40,
            "source_snapshot_sha256": "b" * 64,
            "backend": backend.to_payload(),
            "readiness_sha256": "c" * 64,
        }
        expected_stdout = json.dumps(
            {"command": "readiness", "result": expected_result},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ) + "\n"

        self.assertEqual(cli._json_safe_antigravity(binding), expected_result)
        self.assertEqual(cli.canonical_cli_status_antigravity("readiness", binding), expected_stdout)
        reordered_backend = BackendAssessmentAntigravity(
            backend.canonical_backend, backend.workers, backend.tensor_eligible,
            backend.acceleration_benchmark_status, backend.pytorch, backend.mlx, backend.mps,
            dict(reversed(tuple(backend.environment.items()))),
        )
        reordered_binding = ReadinessBindingAntigravity(
            binding.reviewed_commit, binding.source_snapshot_sha256,
            reordered_backend, binding.readiness_sha256,
        )
        self.assertEqual(
            cli.canonical_cli_status_antigravity("readiness", reordered_binding),
            expected_stdout,
        )
        stream = io.StringIO()
        with patch.object(cli, "current_clean_head_antigravity", return_value=binding.reviewed_commit), patch.object(
            readiness, "run_readiness_antigravity", return_value=binding,
        ), contextlib.redirect_stdout(stream):
            result = cli.main_antigravity(["readiness"])
        self.assertIs(type(result), int)
        self.assertEqual(result, 0)
        self.assertEqual(stream.getvalue(), expected_stdout)

    def test_real_frozen_publication_is_byte_deterministic_across_set_insertion_order(self) -> None:
        """Frozen hard coordinates must use canonical JSON order, never set hash order."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import (
            FrozenPublicationAntigravity,
            SelectionResultAntigravity,
        )

        def publication(coordinates: list[tuple[str, int]]) -> FrozenPublicationAntigravity:
            selection = SelectionResultAntigravity((), (), (), (), (), frozenset(coordinates))
            return FrozenPublicationAntigravity("a" * 64, "b" * 64, "c" * 64, "d" * 64, selection)

        first = publication([("Zulu", 2), ("Alpha", 1)])
        second = publication([("Alpha", 1), ("Zulu", 2)])
        first_bytes = cli.canonical_cli_status_antigravity("inspect", first)
        self.assertEqual(first_bytes, cli.canonical_cli_status_antigravity("inspect", first))
        self.assertEqual(first_bytes, cli.canonical_cli_status_antigravity("inspect", second))
        self.assertEqual(
            json.loads(first_bytes)["result"]["selection"]["hard_coordinates"],
            [["Alpha", 1], ["Zulu", 2]],
        )

    def test_serializer_supports_json_containers_and_rejects_unsupported_values(self) -> None:
        """Unsupported values or non-string mapping keys must not leak through the JSON boundary."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli

        value = MappingProxyType({
            "containers": (frozenset({("Zulu", 2), "Alpha", 3}), [None, True, 4, 1.5]),
        })
        self.assertEqual(
            cli._json_safe_antigravity(value),
            {"containers": [["Alpha", 3, ["Zulu", 2]], [None, True, 4, 1.5]]},
        )
        for unsupported in (object(), 1 + 2j, {1: "coerced-key"}, frozenset({object()})):
            with self.subTest(unsupported=type(unsupported).__name__), self.assertRaises(TypeError):
                cli._json_safe_antigravity(unsupported)
