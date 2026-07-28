import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp410_family_robustness.aggregation import (
    aggregate_family_robustness,
)
from experiments.exp410_family_robustness.publication import publish_audit
from experiments.exp410_family_robustness.tests.test_aggregation import (
    aggregation_fixture,
    fixture_config,
)


def publication_fixture():
    metrics, profiles = aggregation_fixture()
    config = fixture_config()
    return aggregate_family_robustness(metrics, profiles, config=config), config


def provenance_fixture() -> dict[str, object]:
    return {
        "inputs": {"fixture": "a" * 64},
        "code": {"commit": "fixture-commit", "sha256": "b" * 64},
        "runtime": {
            "python": "fixture-python",
            "canonical_path": "cpu_numpy",
        },
    }


class PublicationTests(unittest.TestCase):
    def test_publication_is_complete_deterministic_and_manifested(self) -> None:
        result, config = publication_fixture()
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            one = parent / "one"
            hashes = publish_audit(
                result,
                one,
                provenance_fixture(),
                config=config,
            )
            expected = {
                "exp410_frozen_config.json",
                "exp410_dataset_metrics.csv",
                "exp410_family_metrics.csv",
                "exp410_weight_sensitivity.csv",
                "exp410_family_influence.csv",
                "exp410_stratified_metrics.csv",
                "exp410_summary.json",
                "EXP410_RESULT_REPORT_KO.md",
                "MANIFEST.json",
            }
            self.assertEqual({path.name for path in one.iterdir()}, expected)
            manifest = json.loads(
                (one / "MANIFEST.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["artifact_sha256"],
                {
                    name: digest
                    for name, digest in hashes.items()
                    if name != "MANIFEST.json"
                },
            )

            two = parent / "two"
            second_hashes = publish_audit(
                result,
                two,
                provenance_fixture(),
                config=config,
            )
            self.assertEqual(hashes, second_hashes)
            for name in expected:
                self.assertEqual((one / name).read_bytes(), (two / name).read_bytes())

    def test_failed_publication_leaves_no_partial_directory(self) -> None:
        result, config = publication_fixture()
        with TemporaryDirectory() as raw:
            output = Path(raw) / "blocked"
            with patch(
                "experiments.exp410_family_robustness.publication._write_payload",
                side_effect=OSError("forced"),
            ):
                with self.assertRaisesRegex(OSError, "forced"):
                    publish_audit(
                        result,
                        output,
                        provenance_fixture(),
                        config=config,
                    )
            self.assertFalse(output.exists())
            self.assertEqual(tuple(Path(raw).iterdir()), ())

    def test_pre_publish_verifier_blocks_atomic_rename(self) -> None:
        result, config = publication_fixture()
        with TemporaryDirectory() as raw:
            output = Path(raw) / "changed"

            def reject_changed_state() -> None:
                raise ValueError("execution state changed")

            with self.assertRaisesRegex(ValueError, "execution state changed"):
                publish_audit(
                    result,
                    output,
                    provenance_fixture(),
                    config=config,
                    pre_publish_verify=reject_changed_state,
                )
            self.assertFalse(output.exists())
            self.assertEqual(tuple(Path(raw).iterdir()), ())


if __name__ == "__main__":
    unittest.main()
