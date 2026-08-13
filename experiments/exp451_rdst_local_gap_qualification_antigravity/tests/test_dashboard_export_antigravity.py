from __future__ import annotations
import tempfile, unittest
from pathlib import Path
from .test_evaluation_antigravity import SHA, fixture_antigravity

class DashboardExportAntigravityTest(unittest.TestCase):
    def test_public_result_is_aggregate_only_and_sensitive_input_rejected_antigravity(self):
        from experiments.exp451_rdst_local_gap_qualification_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp451_rdst_local_gap_qualification_antigravity.dashboard_export_antigravity import public_result_antigravity, validate_public_result_antigravity
        with tempfile.TemporaryDirectory() as directory:
            authority,deps,_=fixture_antigravity(Path(directory)/"evaluation_antigravity"); result=_evaluate_with_dependencies_antigravity(authority,SHA,deps)
            public=public_result_antigravity(result)
            flattened=str(public).lower()
            for forbidden in ("dataset", "indices", "label", "sha", "path", "database", "sql", "audit", "checkpoint", "log"):
                self.assertNotIn(forbidden, flattened)
            self.assertEqual(public["runtime"], {"sessions":1,"queries":48,"rows":72300})
            for key in ("raw_labels", "dataset_names", "local_path", "sha256", "sql_log", "candidate_indices"):
                with self.assertRaises(ValueError): validate_public_result_antigravity({**public, key: []})

if __name__ == "__main__": unittest.main()
