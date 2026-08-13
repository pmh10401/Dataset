"""Governance regression for the planned, label-free Exp451 sibling."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
PROTECTED_EXP449_KEYS_ANTIGRAVITY = (
    "active_experiment",
    "active_phase",
    "status",
    "active_documents",
    "pending_candidate",
    "exp449",
    "next_action",
    "exact_next_command",
    "official_readiness_command_after_fresh_sol_ship",
    "approval_scope",
)
PROTECTED_EXP449_SHA256_ANTIGRAVITY = (
    "5041bc34c5323658779fbf017925dec80be5e315efd56f5189c94f7c9dcb36b2"
)


def strict_json_antigravity(path: Path) -> object:
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates_antigravity)


def protected_exp449_state_sha256_antigravity(state: dict[str, object]) -> str:
    payload = {key: state[key] for key in PROTECTED_EXP449_KEYS_ANTIGRAVITY}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    return hashlib.sha256(data).hexdigest()


class GovernanceAntigravityTest(unittest.TestCase):
    def test_exp451_planned_truth_antigravity(self) -> None:
        state = strict_json_antigravity(ROOT_ANTIGRAVITY / "RESEARCH_STATE.json")
        self.assertIsInstance(state, dict)
        self.assertEqual(state["active_experiment"], "Exp449")
        self.assertEqual(state["exp451"]["status"], "planned_not_started")
        self.assertEqual(
            protected_exp449_state_sha256_antigravity(state),
            PROTECTED_EXP449_SHA256_ANTIGRAVITY,
        )
        self.assertEqual(state["exp451"]["label_access_count"], 0)
        self.assertFalse(state["exp451"]["official_readiness_started"])
        self.assertEqual(state["exp451"]["target_dataset_count"], 48)
        self.assertEqual(state["exp451"]["target_test_rows"], 72_300)

    def test_exp451_governance_documents_are_self_contained_antigravity(self) -> None:
        for relative_path in (
            "docs/EXP451_PREREGISTRATION_antigravity_KO.html",
            "docs/EXP451_RUNBOOK_antigravity_KO.html",
            "docs/EXP451_IMPLEMENTATION_REPORT_antigravity.html",
        ):
            content = (ROOT_ANTIGRAVITY / relative_path).read_text(encoding="utf-8")
            self.assertIn("<!doctype html>", content.lower())
            self.assertIn("Exp451", content)


if __name__ == "__main__":
    unittest.main()
