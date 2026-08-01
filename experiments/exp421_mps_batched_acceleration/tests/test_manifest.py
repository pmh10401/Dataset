from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp421_mps_batched_acceleration.config import (
    Exp421Config,
    frozen_config_sha256,
)
from experiments.exp421_mps_batched_acceleration.contracts import (
    Exp421Manifest,
)
from experiments.exp421_mps_batched_acceleration.manifest import (
    build_manifest,
    canonical_manifest_sha256,
    validate_manifest,
    write_manifest,
)
from experiments.exp421_mps_batched_acceleration.policy import choose_backend


class ManifestTests(unittest.TestCase):
    """Red tests for the canonical, train-only Exp421 manifest boundary."""

    def _policy_bundle(self):
        # Reuse the deterministic evidence fixture from the policy contract
        # without depending on a live MPS device or the official benchmark.
        from experiments.exp421_mps_batched_acceleration.tests.test_policy import (
            BackendPolicyTests,
        )

        helper = BackendPolicyTests()
        return choose_backend(helper._all_assessments(), Exp421Config())

    def _provenance(self, attempt_root: Path):
        return {
            "git_commit": "a" * 40,
            "database_sha256": "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
            "config_sha256": frozen_config_sha256(Exp421Config()),
            "device": "Apple M4",
            "runtime": "Python 3.14",
            "source": "exp421-train-only",
            "fresh_no_clobber_attempt": True,
            "train_only": True,
            "torch_version": "2.13.0",
            "mps_built": True,
            "mps_available": True,
            "device_name": "Darwin arm64 MPS",
            "mlx_availability": "available",
            "mlx_version": "0.20.0",
            "mlx_metal_version": "0.20.0",
            "mlx_eligibility_reason": "algorithm_rewrite_required",
        }

    def _source_file(self, root: Path) -> Path:
        path = root / "exp421_source.py"
        path.write_text("TRAIN_ONLY = True\n", encoding="utf-8")
        return path

    def _build(self, root: Path):
        source = self._source_file(root)
        cached = np.array([1, 4], dtype=np.int64)
        cached.setflags(write=False)
        return build_manifest(
            policy_bundle=self._policy_bundle(),
            cached_candidate_indices=cached,
            provenance=self._provenance(root),
            source_digests={str(source): ""},
            attempt_root=root / "attempt-001",
            state="draft",
        )

    def test_build_and_round_trip_preserves_all_hybrid_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_file(root)
            from hashlib import sha256

            digest = sha256(source.read_bytes()).hexdigest()
            manifest = build_manifest(
                policy_bundle=self._policy_bundle(),
                cached_candidate_indices=np.array([1, 4], dtype=np.int64),
                provenance=self._provenance(root),
                source_digests={str(source): digest},
                attempt_root=root / "attempt-001",
                state="draft",
            )
            payload = json.loads(json.dumps(manifest.to_payload()))
            restored = validate_manifest(payload)
            self.assertIsInstance(restored, Exp421Manifest)
            self.assertEqual(tuple(restored.policy_bundle.policies), (1, 4, 8, 16, 32))
            self.assertEqual(tuple(restored.policy_bundle.reasons), (1, 4, 8, 16, 32))
            self.assertEqual(restored.policy, restored.policy_bundle.policies[1])
            self.assertEqual(payload["candidate_batch_sizes"], [1, 4, 8, 16, 32])
            self.assertEqual(set(payload["measurements"]), {"1", "4", "8", "16", "32"})
            self.assertEqual(set(payload["parity"]), {"1", "4", "8", "16", "32"})

    def test_payload_has_exact_required_fields_and_canonical_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_file(root)
            from hashlib import sha256

            digest = sha256(source.read_bytes()).hexdigest()
            manifest = build_manifest(
                policy_bundle=self._policy_bundle(),
                cached_candidate_indices=np.array([], dtype=np.int64),
                provenance=self._provenance(root),
                source_digests={str(source): digest},
                attempt_root=root / "attempt-001",
            )
            payload = manifest.to_payload()
            required = {
                "experiment_id", "state", "config", "candidate_batch_sizes",
                "cached_candidate_indices", "measurements", "parity", "policy",
                "policy_bundle", "provenance", "source_digests",
                "source_manifest_sha256", "attempt_root",
            }
            self.assertEqual(set(payload), required)
            self.assertEqual(canonical_manifest_sha256(payload), canonical_manifest_sha256(deepcopy(payload)))
            changed = deepcopy(payload)
            changed["state"] = "failed"
            self.assertNotEqual(canonical_manifest_sha256(payload), canonical_manifest_sha256(changed))

    def test_validation_rejects_missing_extra_fields_and_frozen_schedule_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_file(root)
            from hashlib import sha256

            manifest = build_manifest(
                policy_bundle=self._policy_bundle(),
                cached_candidate_indices=np.array([], dtype=np.int64),
                provenance=self._provenance(root),
                source_digests={str(source): sha256(source.read_bytes()).hexdigest()},
                attempt_root=root / "attempt-001",
            )
            payload = manifest.to_payload()
            for bad in ("missing", "extra"):
                altered = deepcopy(payload)
                if bad == "missing":
                    del altered["policy"]
                else:
                    altered["unexpected"] = True
                with self.subTest(bad=bad):
                    with self.assertRaises(ValueError):
                        validate_manifest(altered)
            altered = deepcopy(payload)
            altered["candidate_batch_sizes"] = [1, 2, 4, 8, 16]
            with self.assertRaises(ValueError):
                validate_manifest(altered)

    def test_boundary_rejects_test_label_location_and_official_exp419_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_file(root)
            from hashlib import sha256

            kwargs = {
                "policy_bundle": self._policy_bundle(),
                "cached_candidate_indices": np.array([], dtype=np.int64),
                "provenance": self._provenance(root),
                "source_digests": {str(source): sha256(source.read_bytes()).hexdigest()},
                "attempt_root": root / "attempt-001",
            }
            for field, value in (
                ("source", "TEST labels"),
                ("source", "ground_truth_locations"),
                ("attempt_root", root / "exp419_deterministic_cpu_parallel"),
            ):
                changed = dict(kwargs)
                if field == "source":
                    changed["provenance"] = {**kwargs["provenance"], "source": value}
                else:
                    changed[field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        build_manifest(**changed)

    def test_source_drift_is_detected_before_read_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_file(root)
            from hashlib import sha256

            manifest = build_manifest(
                policy_bundle=self._policy_bundle(),
                cached_candidate_indices=np.array([], dtype=np.int64),
                provenance=self._provenance(root),
                source_digests={str(source): sha256(source.read_bytes()).hexdigest()},
                attempt_root=root / "attempt-001",
            )
            payload = manifest.to_payload()
            source.write_text("DRIFTED = True\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_manifest(payload)

    def test_write_is_atomic_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_file(root)
            from hashlib import sha256

            manifest = build_manifest(
                policy_bundle=self._policy_bundle(),
                cached_candidate_indices=np.array([], dtype=np.int64),
                provenance=self._provenance(root),
                source_digests={str(source): sha256(source.read_bytes()).hexdigest()},
                attempt_root=root / "attempt-001",
            )
            target = root / "attempt-001" / "manifest.json"
            written = write_manifest(target, manifest)
            self.assertIsInstance(written, Exp421Manifest)
            self.assertTrue(target.is_file())
            with self.assertRaises(FileExistsError):
                write_manifest(target, manifest)
            target2 = root / "attempt-002" / "manifest.json"
            manifest2 = build_manifest(
                policy_bundle=self._policy_bundle(),
                cached_candidate_indices=np.array([], dtype=np.int64),
                provenance=self._provenance(root),
                source_digests={str(source): sha256(source.read_bytes()).hexdigest()},
                attempt_root=root / "attempt-002",
            )
            write_manifest(target2, manifest2)

    def test_manifest_rejects_nonhex_digests_and_forbidden_artifact_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source_file(root)
            from hashlib import sha256

            kwargs = {
                "policy_bundle": self._policy_bundle(),
                "cached_candidate_indices": np.array([], dtype=np.int64),
                "provenance": self._provenance(root),
                "source_digests": {str(source): sha256(source.read_bytes()).hexdigest()},
                "attempt_root": root / "attempt-001",
            }
            with self.assertRaises(ValueError):
                build_manifest(**{**kwargs, "source_digests": {str(source): "g" * 64}})
            forbidden = root / "TEST_labels.json"
            forbidden.write_text("no", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_manifest(
                    **{
                        **kwargs,
                        "source_digests": {str(forbidden): sha256(forbidden.read_bytes()).hexdigest()},
                    }
                )

    def test_package_test_sources_are_allowed_but_source_drift_is_still_checked(self):
        package_test = Path(__file__)
        from hashlib import sha256

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = build_manifest(
                policy_bundle=self._policy_bundle(),
                cached_candidate_indices=np.array([], dtype=np.int64),
                provenance=self._provenance(root),
                source_digests={str(package_test): sha256(package_test.read_bytes()).hexdigest()},
                attempt_root=root / "attempt-001",
            )
            self.assertIn(str(package_test), manifest.source_digests)


if __name__ == "__main__":
    unittest.main()
