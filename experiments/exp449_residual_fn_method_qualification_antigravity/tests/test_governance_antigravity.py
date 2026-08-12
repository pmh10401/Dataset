import json
import re
import subprocess
import unittest
from html.parser import HTMLParser
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


class PlanCheckboxParserAntigravity(HTMLParser):
    """Collect task-heading and step checkbox states from the implementation plan."""

    def __init__(self):
        super().__init__()
        self._heading_depth = 0
        self._heading_tag = None
        self._heading_parts = []
        self._heading_checkboxes = []
        self._current_task = None
        self.tasks = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in ("h2", "h3"):
            self._heading_depth += 1
            self._heading_tag = tag
            self._heading_parts = []
            self._heading_checkboxes = []
            return
        if tag != "input" or attributes.get("type") != "checkbox":
            return
        checkbox = {"disabled": "disabled" in attributes, "checked": "checked" in attributes}
        if self._heading_depth:
            self._heading_checkboxes.append(checkbox)
        elif self._current_task is not None:
            self.tasks[self._current_task]["steps"].append(checkbox)

    def handle_data(self, data):
        if self._heading_depth:
            self._heading_parts.append(data)

    def handle_endtag(self, tag):
        if tag != self._heading_tag or not self._heading_depth:
            return
        self._heading_depth -= 1
        self._heading_tag = None
        match = re.fullmatch(r"\s*Task ([1-8]):.*", "".join(self._heading_parts), flags=re.DOTALL)
        if match:
            task_number = int(match.group(1))
            self.tasks[task_number] = {"heading": self._heading_checkboxes, "steps": []}
            self._current_task = task_number


def parse_plan_checkboxes_antigravity(path: Path):
    parser = PlanCheckboxParserAntigravity()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser.tasks


class Exp449GovernanceTest(unittest.TestCase):
    def test_html_plan_records_only_verified_task_and_step_progress_antigravity(self):
        tasks = parse_plan_checkboxes_antigravity(
            Path("docs/superpowers/plans/2026-08-12-exp449-residual-fn-method-qualification-antigravity.html")
        )
        self.assertEqual(set(tasks), set(range(1, 9)))
        for task_number, state in tasks.items():
            self.assertEqual(len(state["heading"]), 1, f"Task {task_number} requires one task-level checkbox")
            self.assertTrue(state["heading"][0]["disabled"])
            self.assertEqual(state["heading"][0]["checked"], task_number in {1, 2, 3, 4, 5, 6})
        self.assertEqual(len(tasks[1]["steps"]), 8)
        self.assertEqual(len(tasks[2]["steps"]), 7)
        self.assertEqual(len(tasks[3]["steps"]), 8)
        self.assertTrue(
            all(
                box["disabled"] and box["checked"]
                for task_number in (1, 2, 3, 4, 5, 6)
                for box in tasks[task_number]["steps"]
            )
        )
        self.assertEqual(len(tasks[5]["steps"]), 8)
        self.assertEqual(len(tasks[6]["steps"]), 9)
        self.assertEqual(sum(box["checked"] for state in tasks.values() for box in state["steps"]), 48)
        self.assertTrue(
            all(
                not box["checked"]
                for task_number in range(7, 9)
                for box in tasks[task_number]["steps"]
            )
        )

    def test_task5_commit_session_addendum_records_ship_closeout_antigravity(self):
        tasks = parse_plan_checkboxes_antigravity(
            Path("docs/superpowers/plans/2026-08-12-exp449-task5-commit-session-authority-antigravity.html")
        )
        self.assertEqual(set(tasks), {1, 2})
        self.assertEqual(tuple(len(tasks[number]["steps"]) for number in (1, 2)), (6, 9))
        self.assertTrue(
            all(
                box["disabled"] and box["checked"]
                for state in tasks.values()
                for group in (state["heading"], state["steps"])
                for box in group
            )
        )

    def test_codemap_generator_records_task3_implementation_truth_antigravity(self):
        """Break caught: regenerated map leaves the implemented-for-review Task 3 boundary in the future."""
        from scratch.build_codemap_exp449_antigravity import build_map_antigravity

        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        graph, _lock, html = build_map_antigravity(source_commit, "2026-08-12T11:30:00+00:00")
        lifecycle = next(node for node in graph["nodes"] if node["id"] == "receipt_lifecycle")
        self.assertIn("Task-2", lifecycle["role"])
        self.assertIn("implemented", graph["scope"][0])
        self.assertIn("Task 3 readiness/CLI/launcher is implemented-for-review", html)

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
