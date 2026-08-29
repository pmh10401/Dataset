"""Tests for the Exp478 official 1117-dataset census."""

from __future__ import annotations

from html.parser import HTMLParser
import sqlite3

import pytest

from experiments.exp478_full_1117_official_routing_antigravity.run_antigravity import (
    DATABASE_PATH_ANTIGRAVITY,
    OFFICIAL_SOURCE_PATHS_ANTIGRAVITY,
    _audit_nonfinite_normal_train_antigravity,
    _overall_pass_antigravity,
    _resume_rows_antigravity,
    _summarise_routing_antigravity,
    evaluate_dataset_antigravity,
    render_dashboard_antigravity,
    select_population_antigravity,
)


def test_population_is_exactly_1117_with_two_explicit_exclusions() -> None:
    connection = sqlite3.connect(f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True)
    try:
        included, exclusions = select_population_antigravity(connection)
    finally:
        connection.close()

    assert len(included) == 1117
    assert len(set(included)) == 1117
    assert exclusions == [
        {
            "dataset": "CornellWhaleChallenge",
            "reason": "normal_class_not_mapped",
            "normal_train_rows": 18378,
        },
        {
            "dataset": "Phoneme_normal_39",
            "reason": "fewer_than_two_normal_train_rows",
            "normal_train_rows": 1,
        },
    ]


def test_checkpoint_is_not_reused_after_official_source_change() -> None:
    checkpoint = {
        "source_exp477_result_sha256": "source-result",
        "source_hashes": {"official_router": "old"},
        "rows": [{"dataset": "Crop_normal_1", "status": "ok"}],
    }
    assert _resume_rows_antigravity(
        checkpoint,
        population=["Crop_normal_1"],
        source_result_hash="source-result",
        source_hashes={"official_router": "new"},
    ) == {}


def test_checkpoint_sources_include_the_direct_ragged_feature_dependency() -> None:
    assert "ragged_features" in OFFICIAL_SOURCE_PATHS_ANTIGRAVITY
    assert OFFICIAL_SOURCE_PATHS_ANTIGRAVITY["ragged_features"].name == (
        "ragged_features_antigravity.py"
    )


def test_routing_summary_counts_feature_to_direct_disagreement() -> None:
    summary = _summarise_routing_antigravity(
        [
            {
                "dataset": "Synthetic_normal_0",
                "status": "ok",
                "feature_method": "hydra_local_gap",
                "direct_method": "minirocket_local_gap",
                "pipeline_method": "minirocket_local_gap",
                "pipeline_match": True,
                "deterministic": True,
                "candidate_count": 1,
                "elapsed_seconds": 0.0,
            }
        ]
    )
    assert summary["feature_direct_mismatches"] == 1


def test_overall_pass_requires_all_three_nonfinite_audit_counts() -> None:
    routing = {
        "success": 1117,
        "failures": 0,
        "pipeline_mismatches": 0,
        "feature_direct_mismatches": 0,
        "nondeterministic": 0,
        "non_singleton": 0,
    }
    scores = {"scored": 962, "routed_without_scores": 155}
    assert _overall_pass_antigravity(
        routing,
        scores,
        {"affected_datasets": 19, "excluded_rows": 164, "nonfinite_values": 983},
    )
    assert not _overall_pass_antigravity(
        routing,
        scores,
        {"affected_datasets": 19, "excluded_rows": 164, "nonfinite_values": 982},
    )


def test_nonfinite_audit_records_the_exact_rows_excluded_from_features() -> None:
    connection = sqlite3.connect(f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True)
    try:
        included, _ = select_population_antigravity(connection)
        audit = _audit_nonfinite_normal_train_antigravity(connection, included)
    finally:
        connection.close()

    assert audit["affected_datasets"] == 19
    assert audit["excluded_rows"] == 164
    assert audit["nonfinite_values"] == 983
    assert audit["families"] == [
        "DodgerLoopDay",
        "DodgerLoopGame",
        "DodgerLoopWeekend",
        "MelbournePedestrian",
    ]


@pytest.mark.parametrize(
    "dataset",
    ["Crop_normal_1", "ECG5000_normal_2", "AllGestureWiimoteY_normal_3"],
)
def test_real_dataset_route_is_deterministic_singleton_and_pipeline_consistent(
    dataset: str,
) -> None:
    row = evaluate_dataset_antigravity(dataset)
    assert row["status"] == "ok"
    assert row["deterministic"] is True
    assert row["candidate_count"] == 1
    assert row["direct_method"] == row["feature_method"] == row["pipeline_method"]


class _CheckboxParserAntigravity(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.checkboxes: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "input" and values.get("type") == "checkbox":
            self.checkboxes.append(values)


def test_completed_dashboard_separates_1117_routes_from_962_scores() -> None:
    result = {
        "state": "complete",
        "population": {"registered": 1119, "included": 1117, "excluded": 2},
        "routing": {
            "completed": 1117,
            "success": 1117,
            "failures": 0,
            "feature_direct_mismatches": 0,
            "pipeline_mismatches": 0,
            "nondeterministic": 0,
            "non_singleton": 0,
            "method_counts": {"minirocket_local_gap": 1000, "hydra_local_gap": 117},
        },
        "score_replay": {
            "scored": 962,
            "routed_without_scores": 155,
            "average_auc": 0.98,
            "average_zf_tp": 2.04,
            "severe_auc_loss_over_0_05": 45,
            "qualification_average_auc": 0.982340991,
            "qualification_average_zf_tp": 2.040540541,
        },
        "nonfinite_row_handling": {
            "affected_datasets": 19,
            "excluded_rows": 164,
            "nonfinite_values": 983,
            "families": ["DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend", "MelbournePedestrian"],
            "strategy": "exclude_nonfinite_rows_without_interpolation",
        },
        "exclusions": [
            {"dataset": "CornellWhaleChallenge", "reason": "normal_class_not_mapped"},
            {"dataset": "Phoneme_normal_39", "reason": "fewer_than_two_normal_train_rows"},
        ],
        "source_exp477_result_sha256": "e794cc506d7579fc63e2fe96ccfebc794a330975f6a4375889419263bc877949",
        "generated_at_utc": "2026-08-28T00:00:00+00:00",
    }
    html = render_dashboard_antigravity(result)
    parser = _CheckboxParserAntigravity()
    parser.feed(html)

    assert "1117" in html and "962" in html and "155" in html
    assert len(parser.checkboxes) >= 7
    assert all("checked" in checkbox for checkbox in parser.checkboxes)
