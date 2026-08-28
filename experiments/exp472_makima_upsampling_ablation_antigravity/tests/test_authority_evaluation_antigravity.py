"""Production-Path authority and label-firewall tests for Exp472 Task 5."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import unittest

from experiments.exp472_makima_upsampling_ablation_antigravity import authority_antigravity as authority_module
from experiments.exp472_makima_upsampling_ablation_antigravity import evaluation_antigravity as evaluation_module
from experiments.exp472_makima_upsampling_ablation_antigravity.authority_antigravity import (
    Exp472ReuseAuthorityAntigravity, Exp472VerifiedExp471BundleAntigravity,
    verify_exp471_reuse_exp472_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.evaluation_antigravity import (
    Exp472VerifiedEBundleAntigravity, evaluate_exp472_antigravity,
)

_ROOT = Path("/private/tmp/exp472-task5-fixture-antigravity")
_METHODS = ("A", "B", "C", "D", "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d")

def _digest(value: str) -> str: return sha256(value.encode()).hexdigest()

def _bundle() -> Exp472VerifiedExp471BundleAntigravity:
    datasets = tuple(f"dataset_{index:04d}" for index in range(1117))
    coordinates = {name: {"count": 2, "indices": (2*i, 2*i+1), "blob_sha256": _digest(name+"blob"), "row_sha256": _digest(name+"row")} for i, name in enumerate(datasets)}
    scores, statuses = {}, {}
    abstentions = 0
    for method_index, method in enumerate(_METHODS):
        scores[method], statuses[method] = {}, {}
        for index, name in enumerate(datasets):
            abstained = (index < 195 and method_index < 2) or (index < 5 and method_index == 2)
            statuses[method][name] = "abstained" if abstained else "scored"
            scores[method][name] = None if abstained else (0.1, 0.9)
            abstentions += abstained
    assert abstentions == 395
    coordinate_sha = authority_module._coordinate_digest(datasets, coordinates)
    auth = Exp472ReuseAuthorityAntigravity(_digest("source"), _digest("cache"), _digest("evaluation"), _digest("report"), _digest("database"), coordinate_sha, _METHODS, 11170, 10775, 395, 0, 1117, 922)
    return Exp472VerifiedExp471BundleAntigravity(_ROOT, auth, datasets, coordinates, scores, statuses, _digest("path-bytes"))

def _e_bundle(bundle: Exp472VerifiedExp471BundleAntigravity, *, drift: bool = False, nonfinite: bool = False) -> Exp472VerifiedEBundleAntigravity:
    coordinates = {key: dict(value) for key, value in bundle.coordinates.items()}
    if drift: coordinates[bundle.dataset_order[0]]["indices"] = (99999, 0)
    scores = {dataset: ((float("nan"), 0.9) if nonfinite and i == 0 else (0.2, 0.8)) for i, dataset in enumerate(bundle.dataset_order)}
    return Exp472VerifiedEBundleAntigravity(_ROOT, bundle.authority.database_sha256, bundle.dataset_order, coordinates, scores, {x: 1 for x in bundle.dataset_order}, _digest("e-path-bytes"))

class Exp472AuthorityEvaluationTestsAntigravity(unittest.TestCase):
    def _run(self, exp471, e, labels):
        with patch.object(authority_module, "_read_verified_exp471_path_antigravity", return_value=exp471), patch.object(evaluation_module, "_read_verified_e_path_antigravity", return_value=e):
            return evaluate_exp472_antigravity(_ROOT, _ROOT, labels)

    def test_path_verifier_requires_full_terminal_inventory(self):
        bundle = _bundle()
        with patch.object(authority_module, "_read_verified_exp471_path_antigravity", return_value=bundle):
            verified = verify_exp471_reuse_exp472_antigravity(_ROOT)
        self.assertEqual((verified.authority.total, verified.authority.scored, verified.authority.abstained, verified.authority.failed), (11170, 10775, 395, 0))
        self.assertEqual((len(verified.dataset_order), verified.authority.common_support), (1117, 922))

    def test_failure_before_verified_e_never_calls_labels(self):
        calls = []
        def labels(_): calls.append(True); return {}
        with patch.object(authority_module, "_read_verified_exp471_path_antigravity", side_effect=ValueError("bad Exp471")):
            with self.assertRaisesRegex(ValueError, "bad Exp471"):
                evaluate_exp472_antigravity(_ROOT, _ROOT, labels)
        self.assertEqual(calls, [])

    def test_coordinate_and_nonfinite_e_fail_before_labels(self):
        bundle = _bundle(); calls = []
        def labels(_): calls.append(True); return {}
        with self.assertRaises(ValueError): self._run(bundle, _e_bundle(bundle, drift=True), labels)
        with patch.object(authority_module, "_read_verified_exp471_path_antigravity", return_value=bundle), patch.object(evaluation_module, "_read_verified_e_path_antigravity", side_effect=ValueError("nonfinite E")):
            with self.assertRaisesRegex(ValueError, "nonfinite E"):
                evaluate_exp472_antigravity(_ROOT, _ROOT, labels)
        self.assertEqual(calls, [])

    def test_happy_path_opens_labels_once_after_path_binding(self):
        bundle = _bundle(); calls = []
        def labels(datasets): calls.append(datasets); return {name: (0, 1) for name in datasets}
        result = self._run(bundle, _e_bundle(bundle), labels)
        self.assertEqual(calls, [bundle.dataset_order])
        self.assertEqual(result.method_order, ("A", "B", "C", "D", "E", *_METHODS[4:]))
        self.assertEqual(result.artifact_record()["schema"], "exp472-evaluation-antigravity.v1")
        self.assertEqual(result.denominators["full_population"]["dataset_count"], 1117)
