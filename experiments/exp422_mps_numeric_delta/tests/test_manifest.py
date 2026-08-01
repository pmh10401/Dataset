"""Canonical Exp422 result-manifest and source-trust boundaries."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from experiments.exp422_mps_numeric_delta.capability import DeviceCapability
from experiments.exp422_mps_numeric_delta.config import FROZEN_CONFIG, frozen_config_sha256
from experiments.exp422_mps_numeric_delta.contracts import (
    BatchKey,
    CoreParityReport,
    StageResult,
    TensorDelta,
    TimingEvidence,
)
from experiments.exp422_mps_numeric_delta.core_parity import CoreStageResult
from experiments.exp422_mps_numeric_delta.manifest import (
    SOURCE_MANIFEST_RELATIVE,
    build_manifest,
    validate_manifest,
    verify_source_manifest,
    write_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _capability() -> DeviceCapability:
    return DeviceCapability(
        torch_version="2.13.0",
        mps_built=True,
        mps_available=False,
        device_name="Darwin arm64 CPU",
        mlx_version="unavailable",
        mlx_metal_version="unavailable",
        mlx_availability="unavailable",
        fallback_reason="mps_unavailable",
    )


def _core(state: StageResult | None = None) -> CoreStageResult:
    stage = state or StageResult.failed("mps_unavailable")
    delta = TensorDelta(
        "prediction",
        True,
        0.0,
        0.0,
        0.0,
        0,
        "1" * 64,
        "1" * 64,
        None,
    )
    reports = tuple(
        CoreParityReport(
            BatchKey(batch, FROZEN_CONFIG.model_seeds[0], FROZEN_CONFIG.perturbation_seeds[0]),
            (delta,),
            TimingEvidence("cpu", 0.0, 0.02, 0.0, 0.02),
            TimingEvidence("mps", 0.0, 0.01, 0.0, 0.01),
            stage.state == "passed",
            1.0,
        )
        for batch in FROZEN_CONFIG.candidate_batch_sizes
    )
    return CoreStageResult(_capability(), reports, stage)


def _provenance() -> dict[str, object]:
    capability = _capability()
    return {
        "attempt_root": "/tmp/exp422-attempt",
        "config_sha256": frozen_config_sha256(),
        "database_sha256": FROZEN_CONFIG.exp417_config.database_sha256,
        "device_name": capability.device_name,
        "fresh_no_clobber_attempt": True,
        "git_commit": "a" * 40,
        "mlx_availability": capability.mlx_availability,
        "mlx_eligibility_reason": "package_absent",
        "mlx_metal_version": capability.mlx_metal_version,
        "mlx_version": capability.mlx_version,
        "mps_available": capability.mps_available,
        "mps_built": capability.mps_built,
        "repeats": 1,
        "source_manifest_sha256": "b" * 64,
        "sqlite_quick_check": "ok",
        "torch_version": capability.torch_version,
        "train_only": True,
    }


class Exp422ManifestTests(unittest.TestCase):
    def test_failed_core_builds_complete_immutable_manifest(self) -> None:
        manifest = build_manifest(_core(), None, _provenance())

        self.assertEqual(manifest.state, "failed")
        self.assertEqual(manifest.candidate_batch_sizes, (16, 32, 64, 128, 256))
        self.assertEqual(manifest.fallback_reasons, ("mps_unavailable",))
        self.assertEqual(manifest.provenance["mlx_eligibility_reason"], "package_absent")
        with self.assertRaises(TypeError):
            manifest.provenance["repeats"] = 2
        self.assertEqual(validate_manifest(manifest.to_payload()), manifest)

    def test_validation_rejects_missing_provenance_and_changed_batch_schedule(self) -> None:
        payload = build_manifest(_core(), None, _provenance()).to_payload()
        missing = deepcopy(payload)
        del missing["provenance"]["git_commit"]
        changed = deepcopy(payload)
        changed["candidate_batch_sizes"] = [16, 32, 64]

        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_manifest(missing)
        with self.assertRaisesRegex(ValueError, "batch"):
            validate_manifest(changed)

    def test_validation_rejects_noncanonical_digest_and_test_token(self) -> None:
        payload = build_manifest(_core(), None, _provenance()).to_payload()
        digest = deepcopy(payload)
        digest["provenance"]["source_manifest_sha256"] = "B" * 64
        forbidden = deepcopy(payload)
        forbidden["provenance"]["attempt_root"] = "/tmp/TEST-exp422"

        with self.assertRaisesRegex(ValueError, "SHA-256|digest"):
            validate_manifest(digest)
        with self.assertRaisesRegex(ValueError, "TEST|evaluation"):
            validate_manifest(forbidden)

    def test_passed_core_requires_score_and_failed_core_forbids_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "score"):
            build_manifest(_core(StageResult.passed()), None, _provenance())

        payload = build_manifest(_core(), None, _provenance()).to_payload()
        payload["score_stage"] = {"state": "passed"}
        with self.assertRaisesRegex(ValueError, "score|stage"):
            validate_manifest(payload)

    def test_failed_core_cannot_hide_all_passing_reports(self) -> None:
        passing = _core(StageResult.passed())
        inconsistent = CoreStageResult(
            passing.capability,
            passing.reports,
            StageResult.failed("core_parity_gate_failed"),
        )
        with self.assertRaisesRegex(ValueError, "failed core|inconsistent"):
            build_manifest(inconsistent, None, _provenance())

    def test_mlx_eligibility_reason_is_canonical(self) -> None:
        provenance = _provenance()
        provenance["mlx_eligibility_reason"] = "use_mlx_anyway"
        with self.assertRaisesRegex(ValueError, "MLX|mlx"):
            build_manifest(_core(), None, provenance)

    def test_write_manifest_is_canonical_and_exclusive(self) -> None:
        manifest = build_manifest(_core(), None, _provenance())
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            written = write_manifest(path, manifest)
            raw = written.read_bytes()

            self.assertEqual(raw, json.dumps(manifest.to_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n")
            with self.assertRaises(FileExistsError):
                write_manifest(path, manifest)

    def test_checked_in_source_manifest_matches_declared_sources(self) -> None:
        digest = verify_source_manifest(REPO_ROOT)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_declared_source_tampering_is_rejected(self) -> None:
        source_manifest = REPO_ROOT / SOURCE_MANIFEST_RELATIVE
        source_payload = json.loads(source_manifest.read_bytes())
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for entry in source_payload["sources"]:
                source = REPO_ROOT / entry["path"]
                target = root / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            manifest_copy = root / SOURCE_MANIFEST_RELATIVE
            manifest_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_manifest, manifest_copy)
            declared = root / source_payload["sources"][0]["path"]
            declared.write_bytes(declared.read_bytes() + b"\n# tampered\n")

            with self.assertRaisesRegex(ValueError, "digest|source"):
                verify_source_manifest(root)


if __name__ == "__main__":
    unittest.main()
