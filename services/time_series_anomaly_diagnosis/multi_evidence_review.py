from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

EVIDENCE_SHA256 = "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b"
MANIFEST_SHA256 = "0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0"


@dataclass(frozen=True)
class DatasetReviewEvidence:
    dataset_name: str
    test_count: int
    priority_1_indices: tuple[int, ...]
    priority_2_indices: tuple[int, ...]
    priority_3_indices: tuple[int, ...]
    support_by_index: dict[int, tuple[str, ...]]
    automatic_hard_indices: tuple[int, ...] = ()


def parse_review_evidence_resource(
    payload: dict[str, object],
) -> dict[str, DatasetReviewEvidence]:
    if payload.get("schema") != "review_evidence_v1":
        raise ValueError("review evidence resource schema mismatch")
    rows = payload.get("datasets")
    if not isinstance(rows, list):
        raise ValueError("review evidence resource datasets must be a list")

    result: dict[str, DatasetReviewEvidence] = {}
    expected_keys = {"dataset", "priority_1", "priority_2", "priority_3", "support"}
    for source in rows:
        if not isinstance(source, dict) or set(source) != expected_keys:
            raise ValueError("review evidence resource row has invalid fields")
        dataset_name = source["dataset"]
        support = source["support"]
        if not isinstance(dataset_name, str) or not isinstance(support, dict):
            raise ValueError("review evidence resource row has invalid types")
        if dataset_name in result:
            raise ValueError("review evidence resource contains duplicate datasets")
        try:
            result[dataset_name] = DatasetReviewEvidence(
                dataset_name=dataset_name,
                test_count=0,
                priority_1_indices=tuple(int(value) for value in source["priority_1"]),
                priority_2_indices=tuple(int(value) for value in source["priority_2"]),
                priority_3_indices=tuple(int(value) for value in source["priority_3"]),
                support_by_index={
                    int(index): tuple(str(family) for family in families)
                    for index, families in support.items()
                },
            )
        except (TypeError, ValueError) as error:
            raise ValueError("review evidence resource row has invalid values") from error

    return result


def load_review_evidence(
    evidence_path: Path,
    manifest_path: Path,
) -> dict[str, DatasetReviewEvidence]:
    if not evidence_path.exists():
        raise ValueError(f"review evidence missing: {evidence_path.name}")
    if not manifest_path.exists():
        raise ValueError(f"review evidence manifest missing: {manifest_path.name}")

    evidence_bytes = evidence_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()

    computed_evidence_sha256 = sha256(evidence_bytes).hexdigest()
    computed_manifest_sha256 = sha256(manifest_bytes).hexdigest()

    if computed_evidence_sha256 != EVIDENCE_SHA256:
        raise ValueError("review evidence SHA-256 mismatch")
    if computed_manifest_sha256 != MANIFEST_SHA256:
        raise ValueError("review evidence manifest SHA-256 mismatch")

    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("review evidence manifest must be a JSON object")

    if manifest.get("prediction_sha256") != computed_evidence_sha256:
        raise ValueError("manifest prediction_sha256 does not match evidence file")
    if manifest.get("review_tier_policy_frozen_before_evaluation") is not True:
        raise ValueError("review evidence policy was not frozen before evaluation")

    result: dict[str, DatasetReviewEvidence] = {}
    for line in evidence_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        source = json.loads(line)
        dataset_name = source["dataset"]
        result[dataset_name] = DatasetReviewEvidence(
            dataset_name=dataset_name,
            test_count=int(source["n_test"]),
            priority_1_indices=tuple(int(x) for x in source["review_priority_1_indices"]),
            priority_2_indices=tuple(int(x) for x in source["review_priority_2_indices"]),
            priority_3_indices=tuple(int(x) for x in source["review_priority_3_indices"]),
            support_by_index={
                int(index): tuple(str(family) for family in families)
                for index, families in source.get("review_support_by_index", {}).items()
            },
            automatic_hard_indices=tuple(
                int(x) for x in source.get("automatic_hard_indices", ())
            ),
        )

    if len(result) != 1117:
        raise ValueError(f"review evidence must contain 1117 unique datasets, found {len(result)}")

    return result


def rank_review_candidates(
    dataset_name: str,
    detection_rows,
    evidence_by_dataset: dict[str, DatasetReviewEvidence],
) -> dict[str, object]:
    evidence = evidence_by_dataset.get(dataset_name)
    if evidence is None:
        return {
            "evidence_available": False,
            "automatic_promotion": False,
            "candidates": [],
            "unavailable_candidate_indices": [],
        }

    detection_by_index = {row["series_index"]: row for row in detection_rows}
    automatic_indices = {
        row["series_index"] for row in detection_rows if row.get("automatic_anomaly")
    }

    candidates = []
    unavailable_indices = []

    for tier, indices in (
        (1, evidence.priority_1_indices),
        (2, evidence.priority_2_indices),
        (3, evidence.priority_3_indices),
    ):
        for series_index in indices:
            if series_index in automatic_indices:
                continue
            score_row = detection_by_index.get(series_index)
            if score_row is None:
                unavailable_indices.append(series_index)
                continue
            families = evidence.support_by_index.get(series_index, ())
            candidates.append(
                {
                    "series_index": series_index,
                    "priority": f"P{tier}",
                    "support_count": len(families),
                    "source_families": list(families),
                    "score": float(score_row["score"]),
                    "p_value": float(score_row["p_value"]),
                }
            )

    candidates.sort(
        key=lambda row: (
            int(row["priority"][1]),
            -row["support_count"],
            -row["score"],
            row["p_value"],
            row["series_index"],
        )
    )

    return {
        "evidence_available": True,
        "automatic_promotion": False,
        "candidates": candidates,
        "unavailable_candidate_indices": sorted(set(unavailable_indices)),
    }
