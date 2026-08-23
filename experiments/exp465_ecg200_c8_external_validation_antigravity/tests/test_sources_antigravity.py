from __future__ import annotations

import hashlib
import inspect
import unittest.mock
import tempfile
import unittest
from pathlib import Path

import numpy as np


TRAIN_MD5_ANTIGRAVITY = "808875aeac1c358a90c630fe369327c4"

HEADER_LINES_ANTIGRAVITY = (
    "#comment",
    "@problemName ECG200",
    "@timeStamps false",
    "@missing false",
    "@univariate true",
    "@equalLength true",
    "@seriesLength 96",
    "@classLabel true -1 1",
    "@data",
)


def _row_text_antigravity(seed: int, label: int) -> str:
    values = np.random.default_rng(seed).standard_normal(96)
    joined = ",".join(format(float(value), ".17g") for value in values)
    return f"{joined}:{label}"


def _train_text_antigravity(*, normal_rows: int = 69, infarct_rows: int = 31) -> str:
    lines = list(HEADER_LINES_ANTIGRAVITY)
    for row_index in range(normal_rows):
        lines.append(_row_text_antigravity(4650000 + row_index, +1))
    for row_index in range(infarct_rows):
        lines.append(_row_text_antigravity(4650200 + row_index, -1))
    return "\n".join(lines) + "\n"


def _write_fixture_antigravity(text: str, name: str) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="exp465_source_antigravity_"))
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def _valid_fixture_path_antigravity() -> Path:
    text = _train_text_antigravity()
    path = _write_fixture_antigravity(text, "ECG200_TRAIN_antigravity.ts")
    assert hashlib.md5(path.read_bytes()).hexdigest() != TRAIN_MD5_ANTIGRAVITY
    return path


class _PatchedMd5Antigravity:
    """Patch the sources module md5 so synthetic fixtures pass the frozen gate."""

    def __init__(self) -> None:
        self.calls = 0

    def __enter__(self) -> "_PatchedMd5Antigravity":
        import experiments.exp465_ecg200_c8_external_validation_antigravity.sources_antigravity as module

        self._module = module
        self._original = module.hashlib.md5

        outer = self

        def patched_md5(*args, **kwargs):
            outer.calls += 1
            return _FrozenDigestAntigravity()

        module.hashlib.md5 = patched_md5  # type: ignore[assignment]
        return self

    def __exit__(self, *exc_info) -> None:
        self._module.hashlib.md5 = self._original  # type: ignore[assignment]


class _FrozenDigestAntigravity:
    def hexdigest(self) -> str:
        return TRAIN_MD5_ANTIGRAVITY

    def update(self, data: object) -> None:
        return None


class SourcesAntigravityTest(unittest.TestCase):
    maxDiff = None

    def test_loader_exports_only_the_train_entrypoint_antigravity(self) -> None:
        """Break caught: source module grows TEST, label, or interpolation entrypoints."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity import sources_antigravity

        self.assertEqual(
            tuple(sources_antigravity.__all__),
            ("load_ecg200_normal_train_antigravity",),
        )
        signature = inspect.signature(sources_antigravity.load_ecg200_normal_train_antigravity)
        self.assertEqual(tuple(signature.parameters), ("path",))
        self.assertFalse(hasattr(sources_antigravity, "load_ecg200_test_antigravity"))
        self.assertFalse(hasattr(sources_antigravity, "load_labels_antigravity"))
        self.assertEqual(sources_antigravity.TRAIN_LABEL_QUERY_COUNT_ANTIGRAVITY, 0)

    def test_loader_returns_exact_normal_train_contract_antigravity(self) -> None:
        """Break caught: normal selection, dtype, ownership, layout, or immutability drifts."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity.sources_antigravity import (
            load_ecg200_normal_train_antigravity,
        )

        path = _valid_fixture_path_antigravity()
        with _PatchedMd5Antigravity():
            first = load_ecg200_normal_train_antigravity(path)
        with _PatchedMd5Antigravity():
            second = load_ecg200_normal_train_antigravity(path)

        expected = np.array(
            [
                np.random.default_rng(4650000 + row_index).standard_normal(96)
                for row_index in range(69)
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(first, expected)
        self.assertEqual(first.shape, (69, 96))
        self.assertIs(first.dtype.type, np.float64)
        self.assertTrue(first.flags.owndata)
        self.assertTrue(first.flags.c_contiguous)
        self.assertFalse(first.flags.writeable)
        with self.assertRaises(ValueError):
            first[0, 0] = 1.0
        self.assertFalse(np.shares_memory(first, second))
        self.assertNotEqual(id(first), id(second))

    def test_loader_rejects_test_named_path_before_reading_antigravity(self) -> None:
        """Break caught: TEST-named paths are read or parsed before rejection."""
        import experiments.exp465_ecg200_c8_external_validation_antigravity.sources_antigravity as sources

        opened = {"count": 0}
        original_open = Path.open

        def guarded_open(self: Path, *args: object, **kwargs: object):
            opened["count"] += 1
            return original_open(self, *args, **kwargs)  # type: ignore[arg-type]

        nested_test_directory = Path(tempfile.mkdtemp(prefix="exp465_source_nested_TEST_"))
        nested_original_test_path = nested_test_directory / "ECG200_TRAIN_antigravity.ts"
        nested_original_test_path.write_text("not-read", encoding="utf-8")

        safe_directory = Path(tempfile.mkdtemp(prefix="exp465_source_safe_"))
        forbidden_target = safe_directory / "ECG200_TEST.ts"
        forbidden_target.write_text("not-read", encoding="utf-8")
        symlink_train_to_test = safe_directory / "ECG200_TRAIN_antigravity.ts"
        symlink_train_to_test.symlink_to(forbidden_target)

        with unittest.mock.patch.object(Path, "open", guarded_open):
            with self.assertRaisesRegex(ValueError, "TEST"):
                sources.load_ecg200_normal_train_antigravity(nested_original_test_path)
            with self.assertRaisesRegex(ValueError, "TEST"):
                sources.load_ecg200_normal_train_antigravity(symlink_train_to_test)
        self.assertEqual(opened["count"], 0)

    def test_loader_rejects_wrong_md5_shape_label_count_and_nonfinite_antigravity(self) -> None:
        """Break caught: frozen MD5, shape, labels, counts, or finiteness gates weaken."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity.sources_antigravity import (
            load_ecg200_normal_train_antigravity,
        )

        wrong_md5 = _valid_fixture_path_antigravity()
        with self.assertRaisesRegex(ValueError, "MD5"):
            load_ecg200_normal_train_antigravity(wrong_md5)

        wrong_width_lines = _train_text_antigravity().splitlines()
        first_data_line = wrong_width_lines[9]
        wrong_width_values, wrong_width_label = first_data_line.rsplit(":", 1)
        wrong_width_lines[9] = ",".join(wrong_width_values.split(",")[:-1]) + ":" + wrong_width_label
        wrong_width_text = "\n".join(wrong_width_lines) + "\n"
        wrong_width = _write_fixture_antigravity(wrong_width_text, "ECG200_TRAIN_antigravity.ts")
        with _PatchedMd5Antigravity(), self.assertRaisesRegex(ValueError, "width"):
            load_ecg200_normal_train_antigravity(wrong_width)

        bad_label_text = _train_text_antigravity().rsplit(":", 1)[0] + ":2\n"
        bad_label = _write_fixture_antigravity(bad_label_text, "ECG200_TRAIN_antigravity.ts")
        with _PatchedMd5Antigravity(), self.assertRaisesRegex(ValueError, "label"):
            load_ecg200_normal_train_antigravity(bad_label)

        bool_label_text = _train_text_antigravity().rsplit(":", 1)[0] + ":True\n"
        bool_label = _write_fixture_antigravity(bool_label_text, "ECG200_TRAIN_antigravity.ts")
        with _PatchedMd5Antigravity(), self.assertRaisesRegex(ValueError, "label"):
            load_ecg200_normal_train_antigravity(bool_label)

        wrong_counts = _write_fixture_antigravity(_train_text_antigravity(normal_rows=68, infarct_rows=32), "ECG200_TRAIN_antigravity.ts")
        with _PatchedMd5Antigravity(), self.assertRaisesRegex(ValueError, "69"):
            load_ecg200_normal_train_antigravity(wrong_counts)

        missing_row = _write_fixture_antigravity("\n".join(_train_text_antigravity().splitlines()[:-1]) + "\n", "ECG200_TRAIN_antigravity.ts")
        with _PatchedMd5Antigravity(), self.assertRaisesRegex(ValueError, "100"):
            load_ecg200_normal_train_antigravity(missing_row)

    def test_loader_rejects_nonfinite_and_alias_values_antigravity(self) -> None:
        """Break caught: nonfinite values or float32/int alias outputs slip through."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity.sources_antigravity import (
            load_ecg200_normal_train_antigravity,
        )

        lines = _train_text_antigravity().splitlines()
        target_index = len(lines) - 1
        parts = lines[target_index].split(",")
        parts[-1] = parts[-1].split(":")[0]
        lines[target_index] = ",".join(parts[:-1]) + ",nan:-1"
        nonfinite = _write_fixture_antigravity("\n".join(lines) + "\n", "ECG200_TRAIN_antigravity.ts")
        with _PatchedMd5Antigravity(), self.assertRaisesRegex(ValueError, "finite"):
            load_ecg200_normal_train_antigravity(nonfinite)

        text_lines = _train_text_antigravity().splitlines()
        data_line = text_lines[9].rsplit(":", 1)[0]
        text_lines[9] = "1e400" + data_line[data_line.index(","):] + ":1"
        overflow = _write_fixture_antigravity("\n".join(text_lines) + "\n", "ECG200_TRAIN_antigravity.ts")
        with _PatchedMd5Antigravity(), self.assertRaisesRegex(ValueError, "finite"):
            load_ecg200_normal_train_antigravity(overflow)
