from __future__ import annotations

import tempfile
import shlex
import unittest
from pathlib import Path
from types import MappingProxyType

from experiments.exp266_fn_recovery_shadow.config import EVIDENCE_GROUPS
from experiments.exp266_fn_recovery_shadow.contracts import (
    FrozenDatasetInputs,
    FrozenRecoveryInputs,
)
from experiments.exp266_fn_recovery_shadow.invariance import (
    CHECKS,
    command_for_check,
    run_test_length_mutation_check,
)
from experiments.exp266_fn_recovery_shadow.prediction import (
    _route_prediction_row,
    freeze_rows,
)
from experiments.exp266_fn_recovery_shadow.inputs import sha256_file


class InvarianceTests(unittest.TestCase):
    def test_commands_shell_quote_spaces_and_unicode_without_json_escapes(self) -> None:
        command = command_for_check(
            "label-permutation",
            baseline_root=Path("/tmp/results with spaces"),
            source_database=Path("/tmp/데이터.db"),
            temp_root=Path("/tmp/temp root"),
            workers=7,
        )

        self.assertNotIn("\\u", command)
        arguments = shlex.split(command)
        self.assertEqual(arguments[5], "/tmp/results with spaces")
        self.assertEqual(arguments[7], "/tmp/데이터.db")
        self.assertEqual(arguments[9], "/tmp/temp root")

    def test_complete_check_set_includes_test_length_mutation(self) -> None:
        self.assertEqual(
            CHECKS,
            (
                "label-permutation",
                "anomaly-position-mutation",
                "display-name-remapping",
                "test-length-mutation",
                "input-row-permutation",
                "repeat-prediction-hash",
            ),
        )

    def test_test_length_mutation_preserves_prediction_hash(self) -> None:
        item = FrozenDatasetInputs(
            dataset="demo",
            n_train=10,
            n_test=3,
            target_length=20,
            hard=frozenset({0}),
            shadow=frozenset(),
            priority=frozenset(),
            standard=frozenset(),
            evidence=MappingProxyType(
                {group: frozenset() for group in EVIDENCE_GROUPS}
            ),
            available_groups=frozenset(EVIDENCE_GROUPS),
            quality_flags=frozenset(),
        )
        recovery = FrozenRecoveryInputs(
            population=("demo",),
            datasets=MappingProxyType({"demo": item}),
            input_sha256=MappingProxyType({}),
            lane_totals=MappingProxyType({}),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = freeze_rows(
                root / "baseline",
                [_route_prediction_row(item, recovery)],
            )
            evidence = run_test_length_mutation_check(
                recovery,
                root / "mutated",
                baseline_sha256=sha256_file(baseline),
            )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["details"]["mutated_datasets"], 1)
        self.assertEqual(evidence["details"]["added_test_slots"], 1)
        self.assertEqual(evidence["output_sha256"], evidence["baseline_sha256"])


if __name__ == "__main__":
    unittest.main()
