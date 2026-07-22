from __future__ import annotations

import sqlite3
from pathlib import Path

from .contracts import EXP272_ROOT, candidate_rows, write_csv


DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")


def run_event_applicability_audit(output_root: Path) -> tuple[Path, Path]:
    rows, datasets = candidate_rows()
    connection = sqlite3.connect(f"file:{DB_PATH.resolve()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        index_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_instances_dataset_split_index'"
        ).fetchone()
        if not index_sql or "instance_index" not in str(index_sql[0]):
            raise ValueError("instance index uniqueness contract missing")
        detail = []
        for row in rows:
            n_test = connection.execute(
                "SELECT test_total_count FROM datasets WHERE name=?", (row["dataset"],)
            ).fetchone()
            if n_test is None or int(n_test[0]) != int(row["n_test"]):
                raise ValueError(f"DB/prediction TEST population mismatch: {row['dataset']}")
            lane_fields = (
                "hard_indices", "shadow_indices", "priority_indices", "standard_indices",
                "recovery_shadow_indices", "recovery_priority_indices", "data_quality_review_indices",
            )
            lane_values = [set(row[field]) for field in lane_fields]
            overlap = sum(len(left & right) for pos, left in enumerate(lane_values) for right in lane_values[pos + 1:])
            union_count = len(set().union(*lane_values))
            raw_count = sum(len(values) for values in lane_values)
            detail.append({
                "dataset": row["dataset"], "n_test_instances": int(n_test[0]),
                "candidate_index_unit": "complete_test_series_instance_virtual_run",
                "within_series_time_index": False,
                "adjacent_index_event_merge_applicable": False,
                "lane_overlap_count": overlap,
                "duplicate_candidate_count": raw_count - union_count,
                "prediction_changed": False,
            })
    finally:
        connection.close()
    summary = [{
        "experiment_id": "experiment_280_event_merge_applicability_audit",
        "datasets": len(datasets),
        "lane_overlap_count": sum(int(row["lane_overlap_count"]) for row in detail),
        "duplicate_candidate_count": sum(int(row["duplicate_candidate_count"]) for row in detail),
        "adjacent_index_event_merge_applicable": False,
        "reason": "candidate index identifies a complete TEST series instance, not an intra-series time point",
        "prediction_changed": False, "hard_policy_changed": False,
        "retrospective": True,
    }]
    root = Path(output_root); detail_path = root / "exp280_event_applicability.csv"; summary_path = root / "exp280_summary.csv"
    write_csv(detail_path, detail); write_csv(summary_path, summary)
    return detail_path, summary_path


__all__ = ["run_event_applicability_audit"]
