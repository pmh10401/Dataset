from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.exp465_ecg200_c8_external_validation_antigravity.sources_antigravity import (
    load_ecg200_normal_train_antigravity as exp465_load_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.sources_antigravity import (
    load_ecg200_normal_train_antigravity as exp466_load_antigravity,
)


class SourcesAntigravityTest(unittest.TestCase):
    def test_export_is_exact_exp465_loader_antigravity(self) -> None:
        """Break caught: Exp466 wraps or copies the frozen Exp465 TRAIN loader."""
        self.assertIs(exp466_load_antigravity, exp465_load_antigravity)

    def test_test_name_is_rejected_before_any_file_open_antigravity(self) -> None:
        """Break caught: a forbidden ECG200 TEST name reaches a file-open operation."""
        with patch.object(Path, "open", side_effect=AssertionError("TEST file was opened")) as open_spy:
            with self.assertRaisesRegex(ValueError, "TEST paths are forbidden"):
                exp466_load_antigravity(Path("local_fixture_ECG200_TEST_antigravity.ts"))
        open_spy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
