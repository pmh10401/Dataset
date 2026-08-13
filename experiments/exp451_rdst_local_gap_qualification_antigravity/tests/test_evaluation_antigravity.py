from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import ConfusionAntigravity, DatasetBaselineAntigravity
from experiments.exp451_rdst_local_gap_qualification_antigravity.contracts_antigravity import FrozenPredictionAuthorityAntigravity
from experiments.exp451_rdst_local_gap_qualification_antigravity.predecessor_antigravity import Exp448PredecessorAuthorityAntigravity
from experiments.exp451_rdst_local_gap_qualification_antigravity.sources_antigravity import LabelSnapshotAntigravity


SHA = "a" * 40
LENGTHS = (1800,) + (1500,) * 47
FAMILIES = ("Crop","ElectricDevices","FordA","FordB","Yoga","ECG5000","ChlorineConcentration","StarLightCurves")
IDENTITIES = tuple((i+1, FAMILIES[i % 8], f"{FAMILIES[i % 8]}_dataset{i}") for i in range(48))


def fixture_antigravity(root: Path):
    from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import EvaluationDependenciesAntigravity, EvaluationEvidenceRowAntigravity
    rows = tuple(DatasetBaselineAntigravity(i+1, IDENTITIES[i][1], frozenset(), ConfusionAntigravity(0, 0, 1, LENGTHS[i] - 1)) for i in range(48))
    predecessor = Exp448PredecessorAuthorityAntigravity((1,2,3,4,5,6,7), "ea9856c7de039697ed4f352864def0fde1c0d336714e68c9b0cd586d2c796f05", "a58a74f69dcc13a64efb2a679ee7e93eca40ac8c5db87b29b12ed11ea068b21c", "54ac58d89367ec5396dafdb180777f787384ffbd21ead1a96764bfe627bbddd5", 1117, 48, tuple(range(1,49)), tuple(x[2] for x in IDENTITIES), rows, (1930,314,1988,191668), (6,6,6,6,6,6,6,6), 72300, LENGTHS, 0)
    authority = FrozenPredictionAuthorityAntigravity(SHA, IDENTITIES, LENGTHS, tuple((0,) for _ in range(48)))
    evidence = tuple(EvaluationEvidenceRowAntigravity(i, IDENTITIES[i], LENGTHS[i], (0,)) for i in range(48))
    counters = {"constructed": 0, "called": 0}
    def opener_factory():
        counters["constructed"] += 1
        def opener(value):
            counters["called"] += 1
            labels = tuple(np.r_[np.uint8(1), np.zeros(length - 1, dtype=np.uint8)] for length in LENGTHS)
            return LabelSnapshotAntigravity(tuple(x[2] for x in IDENTITIES), labels, LENGTHS, 72300, 1)
        return opener
    deps = EvaluationDependenciesAntigravity(lambda value: value, lambda: predecessor, evidence, opener_factory, root)
    return authority, deps, counters


class EvaluationAntigravityTest(unittest.TestCase):
    def test_public_evaluate_binds_current_external_identity_baseline_authority_before_labels_antigravity(self):
        from unittest.mock import patch
        from experiments.exp451_rdst_local_gap_qualification_antigravity import evaluation_antigravity as module
        from experiments.exp451_rdst_local_gap_qualification_antigravity.predecessor_antigravity import verify_exp448_predecessor_antigravity
        predecessor=verify_exp448_predecessor_antigravity()
        self.assertEqual(module._baseline_authority_sha256_antigravity(predecessor),"3bb629ca696c6231fb0f6741f5df7e642ac1b0a63e23788436a24bc7eda44232")
        self.assertEqual(predecessor.target_dataset_order[:4],(101,102,103,148)); self.assertEqual(predecessor.label_access_count,0)
        identities=tuple((row.dataset_order,row.family,name) for row,name in zip(predecessor.target_baseline_rows,predecessor.target_dataset_names,strict=True)); authority=FrozenPredictionAuthorityAntigravity(SHA,identities,predecessor.target_test_lengths,tuple(() for _ in identities)); captured=[]
        def private(value,reviewed,deps): captured.append(deps.predecessor_loader()); return "sentinel"
        with patch.object(module,"verify_exp448_predecessor_antigravity",return_value=predecessor), patch.object(module,"_evaluate_with_dependencies_antigravity",side_effect=private):
            self.assertEqual(module.evaluate_antigravity(authority,SHA),"sentinel")
        self.assertEqual(captured,[predecessor])

    def test_qualification_boundaries_and_exact_types_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import qualifies_antigravity
        self.assertTrue(qualifies_antigravity(ConfusionAntigravity(1931,314,1987,191668), -1))
        self.assertFalse(qualifies_antigravity(ConfusionAntigravity(1931,315,1987,191667), -1))
        self.assertFalse(qualifies_antigravity(ConfusionAntigravity(1930,314,1988,191668), 0))
        with self.assertRaises(ValueError): qualifies_antigravity((1931,314,1987,191668), -1)
        with self.assertRaises(ValueError): qualifies_antigravity(ConfusionAntigravity(1931,314,1987,191668), True)

    def test_prelabel_gate_rejects_order_pair_length_index_commit_and_inventory_before_opener_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import EvaluationDependenciesAntigravity, _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            authority, deps, counters = fixture_antigravity(Path(directory) / "evaluation_antigravity")
            attacks = [deps.evidence_rows[:-1], deps.evidence_rows + (deps.evidence_rows[-1],), (deps.evidence_rows[1], deps.evidence_rows[0], *deps.evidence_rows[2:]), tuple(dataclasses.replace(row, identity=(row.identity[0], "wrong", row.identity[2])) if i == 3 else row for i,row in enumerate(deps.evidence_rows)), tuple(dataclasses.replace(row, target_length=row.target_length + 1) if i == 4 else row for i,row in enumerate(deps.evidence_rows)), tuple(dataclasses.replace(row, indices=()) if i == 5 else row for i,row in enumerate(deps.evidence_rows))]
            for evidence in attacks:
                bad = EvaluationDependenciesAntigravity(deps.prediction_validator, deps.predecessor_loader, evidence, deps.label_opener_factory, Path(directory) / f"bad{len(evidence)}_{counters['constructed']}_antigravity")
                with self.assertRaises(ValueError): _evaluate_with_dependencies_antigravity(authority, SHA, bad)
            with self.assertRaises(ValueError): _evaluate_with_dependencies_antigravity(authority, "b"*40, deps)
            self.assertEqual(counters, {"constructed": 0, "called": 0})

    def test_exact_fixture_opens_once_and_builds_counterfactual_bootstrap_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        with tempfile.TemporaryDirectory() as directory:
            authority, deps, counters = fixture_antigravity(Path(directory) / "evaluation_antigravity")
            result = _evaluate_with_dependencies_antigravity(authority, SHA, deps)
            self.assertEqual(counters, {"constructed": 1, "called": 1})
            self.assertEqual((result.replaced_dataset_count,result.reused_dataset_count,result.global_dataset_count), (48,1069,1117))
            self.assertEqual((result.label_session_count,result.label_query_count,result.label_row_count), (1,48,72300))
            self.assertEqual(result.bootstrap_dataset_pairs, 48)
            self.assertEqual(result.target_order, IDENTITIES)
            self.assertEqual(result.candidate_global, ConfusionAntigravity(1978,314,1940,191668))
            self.assertEqual(result.bootstrap["draw_sha256"], "264c780d7e5949994f1de3053158ea20a7cc3d0ae858f29770d9849b18ad4d6a")
            self.assertEqual(len(result.dataset_rows), 48)
            self.assertTrue(all(sum(dataclasses.astuple(row.candidate)) == row.target_length for row in result.dataset_rows))


if __name__ == "__main__": unittest.main()
