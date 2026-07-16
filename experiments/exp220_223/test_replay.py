from __future__ import annotations

import numpy as np

from experiments.exp220_223.replay import build_raw_audit_report, replay_profile


def test_replay_reconstructs_exp190_and_exp195_without_exploratory_lane() -> None:
    result = replay_profile(
        profile_id="profile-a",
        n_query=5,
        exp137_lanes={"hard": {0}, "standard": set(), "priority": {4}},
        conservative={0, 1, 2, 4},
        canonical_knn_p=np.asarray([0.001, 0.004, 0.2, 0.9, 0.003]),
        exp195_zero_train_guard=True,
        expected_exp190={"high": {1}, "standard": {2}},
        expected_exp195={1},
    )

    assert result["passed"] is True
    assert result["actual"]["exp190_high"] == [1]
    assert result["actual"]["exp190_standard"] == [2]
    assert result["actual"]["exp195_shadow"] == [1]
    assert "exploratory" not in result["actual"]


def test_replay_reports_candidate_and_score_mismatch() -> None:
    result = replay_profile(
        profile_id="profile-a",
        n_query=2,
        exp137_lanes={"hard": set(), "standard": set(), "priority": set()},
        conservative={0},
        canonical_knn_p=np.asarray([0.005, 0.9]),
        exp195_zero_train_guard=False,
        expected_exp190={"high": set(), "standard": {0}},
        expected_exp195=set(),
        expected_canonical_knn_p=np.asarray([0.006, 0.9]),
    )

    assert result["passed"] is False
    assert result["candidate_match"] is False
    assert result["score_match"] is False


def test_raw_audit_opens_shadow_gate_only_for_full_exact_coverage() -> None:
    partial = build_raw_audit_report(
        details=[{"profile_id": "a", "passed": True}],
        expected_profiles=639,
        input_hashes={"policy": "abc"},
    )
    assert partial["status"] == "partial"
    assert partial["shadow_execution_gate"] == "blocked_incomplete_raw_replay"

    complete = build_raw_audit_report(
        details=[{"profile_id": str(index), "passed": True} for index in range(639)],
        expected_profiles=639,
        input_hashes={"policy": "abc"},
    )
    assert complete["status"] == "passed"
    assert complete["shadow_execution_gate"] == "passed"
