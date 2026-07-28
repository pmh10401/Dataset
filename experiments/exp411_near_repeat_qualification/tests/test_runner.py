from dataclasses import replace
import csv
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp411_near_repeat_qualification.config import FROZEN_CONFIG
from experiments.exp411_near_repeat_qualification.runner import (
    load_population_manifest,
    run_representative_smoke,
)


class RunnerTests(unittest.TestCase):
    def test_representative_smoke_publishes_complete_manifest(self) -> None:
        with TemporaryDirectory() as raw:
            output = Path(raw) / "smoke"
            hashes = run_representative_smoke(output)
            manifest = json.loads(
                (output / "MANIFEST.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (output / "exp411_summary.json").read_text(encoding="utf-8")
            )
            backend = json.loads(
                (output / "exp411_backend_assessment.json").read_text(
                    encoding="utf-8"
                )
            )
            with (output / "exp411_dataset_qualification.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                dataset_rows = tuple(csv.DictReader(handle))
        self.assertFalse(summary["full_qualification_executed"])
        self.assertFalse(summary["operational_policy_changed"])
        self.assertFalse(summary["exp412_prediction_created"])
        self.assertEqual(summary["dataset_count"], 32)
        self.assertEqual(summary["family_count"], 4)
        self.assertIn("perturbation_contract", manifest["provenance"])
        self.assertEqual(
            backend["canonical_backend"],
            "numpy_cpu",
        )
        self.assertIn("assessments", backend)
        self.assertEqual(
            set(backend["assessments"]),
            {"numba_cpu", "torch_cpu", "torch_mps"},
        )
        self.assertEqual(
            backend["baseline_tag_sha256"],
            backend["canonical_tag_sha256"],
        )
        self.assertIn("query_identity", backend["trace_contract"])
        self.assertIn("numba_cold_stage_seconds", backend["timing"])
        self.assertIn("numba_warm_stage_seconds", backend["timing"])
        for assessment in backend["assessments"].values():
            if "candidate_trace_sha256" in assessment:
                self.assertEqual(
                    assessment["candidate_trace_sha256"],
                    backend["baseline_tag_sha256"],
                )
        self.assertEqual(
            set(summary["provenance"]["smoke_coverage"]["identity_regimes"]),
            {"abstain_lt5", "logo_5_19", "five_fold_20_plus"},
        )
        self.assertEqual(
            {int(row["identity_group_count"]) for row in dataset_rows},
            {4, 6, 20},
        )
        self.assertEqual(set(hashes), set(manifest["artifact_sha256"]))
        self.assertEqual(len(hashes), 7)

    def test_existing_output_is_rejected(self) -> None:
        with TemporaryDirectory() as raw:
            output = Path(raw) / "existing"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "must not exist"):
                run_representative_smoke(output)

    def test_population_manifest_is_hash_verified(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "population.json"
            path.write_text(
                json.dumps(
                    {"frozen_hard_lanes": {"Alpha_normal_0": []}},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            digest = sha256(path.read_bytes()).hexdigest()
            config = replace(
                FROZEN_CONFIG,
                expected_datasets=1,
                expected_families=1,
                population_manifest_sha256=digest,
            )
            self.assertEqual(
                load_population_manifest(path, config=config),
                ("Alpha_normal_0",),
            )
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_population_manifest(
                    path,
                    config=replace(config, population_manifest_sha256="0" * 64),
                )


if __name__ == "__main__":
    unittest.main()
