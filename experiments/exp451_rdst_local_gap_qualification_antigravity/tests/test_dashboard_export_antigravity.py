from __future__ import annotations
import copy, json
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
            self.assertEqual(public["bootstrap"]["scope"],"target_48_paired")
            self.assertEqual(public["bootstrap"]["f1_delta"],result.bootstrap["f1_delta"])
            self.assertEqual(public["bootstrap"]["f1_interval"],result.bootstrap["f1_interval"])
            flattened=str(public).lower()
            for forbidden in ("dataset", "indices", "label", "sha", "path", "database", "sql", "audit", "checkpoint", "log"):
                self.assertNotIn(forbidden, flattened)
            self.assertEqual(public["runtime"], {"sessions":1,"queries":48,"rows":72300})
            for key in ("raw_labels", "dataset_names", "local_path", "sha256", "sql_log", "candidate_indices"):
                with self.assertRaises(ValueError): validate_public_result_antigravity({**public, key: []},expected_result=result)
            malformed=[]
            wrong=copy.deepcopy(public); wrong["global"]["candidate"]["f1"] += .1; malformed.append(wrong)
            wrong=copy.deepcopy(public); wrong["qualification"]["qualified"] = not wrong["qualification"]["qualified"]; malformed.append(wrong)
            wrong=copy.deepcopy(public); wrong["runtime"]["rows"] = 72299; malformed.append(wrong)
            wrong=copy.deepcopy(public); wrong["family_fn_deltas"].pop(next(iter(wrong["family_fn_deltas"]))); malformed.append(wrong)
            wrong=copy.deepcopy(public); wrong["status"] = "/tmp/private/database.sqlite"; malformed.append(wrong)
            wrong=copy.deepcopy(public); wrong["global"]["candidate"]["fp"]=315; wrong["global"]["candidate"]["tn"]-=1; wrong["global"]["candidate"]["f1"]=2*wrong["global"]["candidate"]["tp"]/(2*wrong["global"]["candidate"]["tp"]+wrong["global"]["candidate"]["fp"]+wrong["global"]["candidate"]["fn"]); malformed.append(wrong)
            wrong=copy.deepcopy(public); wrong["runtime"]["rows"]=True; malformed.append(wrong)
            wrong=copy.deepcopy(public); wrong["family_fn_deltas"]["InventedFamily"]=wrong["family_fn_deltas"].pop(next(iter(wrong["family_fn_deltas"]))); malformed.append(wrong)
            wrong=copy.deepcopy(public); wrong["transitions"]["events"]={key:0 for key in wrong["transitions"]["events"]}; wrong["transitions"]["affected_counts"]={key:48 for key in wrong["transitions"]["affected_counts"]}; wrong["bootstrap"]["fn_delta"]=999; malformed.append(wrong)
            wrong=copy.deepcopy(public); wrong["bootstrap"]["fp_delta"]+=1; malformed.append(wrong)
            wrong=copy.deepcopy(public); wrong["bootstrap"]["f1_delta"]+=.01; malformed.append(wrong)
            wrong=copy.deepcopy(public); wrong["bootstrap"]["scope"]="global"; malformed.append(wrong)
            wrong=copy.deepcopy(public); wrong["transitions"]["events"]["false_alarm_losses"]+=1; malformed.append(wrong)
            for attack in malformed:
                with self.assertRaises(ValueError): validate_public_result_antigravity(attack,expected_result=result)
            self.assertEqual(validate_public_result_antigravity(public,expected_result=result),public)
            reordered=json.loads(json.dumps(public,sort_keys=True))
            self.assertEqual(validate_public_result_antigravity(reordered,expected_result=result),reordered)
            with self.assertRaises(TypeError): validate_public_result_antigravity(public)

if __name__ == "__main__": unittest.main()
