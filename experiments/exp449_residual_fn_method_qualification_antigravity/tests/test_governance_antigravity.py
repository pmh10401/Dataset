import json
import re
import unittest
from pathlib import Path


EXPECTED_PENDING_CANDIDATE = {
    "experiment": None,
    "status": "none_exp446_publication_inspected_complete_not_pending_champion",
    "prediction_sha256": None,
    "evaluation": None,
}


def strict_json_antigravity(path: Path):
    def reject_duplicate_antigravity(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_antigravity)


class Exp449GovernanceTest(unittest.TestCase):
    def test_planned_truth_and_global_constraints_are_exact_antigravity(self):
        prereg = Path("docs/EXP449_PREREGISTRATION_antigravity_KO.html").read_text(encoding="utf-8")
        self.assertIn("global_fp_max = 314", prereg)
        self.assertIn("global_fn_must_be_below = 1988", prereg)
        self.assertIn("official_label_sessions = 1", prereg)
        state = strict_json_antigravity(Path("RESEARCH_STATE.json"))
        self.assertEqual(state["active_experiment"], "Exp449")
        self.assertEqual(state["exp449"]["label_access_count"], 0)
        self.assertFalse(state["exp449"]["official_readiness_started"])
        self.assertEqual(state["pending_candidate"], EXPECTED_PENDING_CANDIDATE)

    def test_global_human_readable_document_policy_is_exact_antigravity(self):
        agents = Path("AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("Every newly created human-readable document", agents)
        self.assertIn("self-contained .html", agents)
        self.assertIn("no Markdown duplicate", agents)
        self.assertIn("JSON, lock, and code files are unaffected", agents)
        self.assertIn("Historical documents are not mass-converted", agents)

    def test_launcher_thread_inventory_requires_aeon_and_rejects_tbb_antigravity(self):
        runbook = Path("docs/EXP449_RUNBOOK_antigravity_KO.html").read_text(encoding="utf-8")
        expected = {
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "VECLIB_MAXIMUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "BLIS_NUM_THREADS=1",
            "AEON_NUM_THREADS=1",
        }
        observed = set(re.findall(r"[A-Z][A-Z0-9_]*=1", runbook))
        self.assertEqual(observed, expected)
        self.assertNotIn("TBB_NUM_THREADS=1", runbook)


if __name__ == "__main__":
    unittest.main()
