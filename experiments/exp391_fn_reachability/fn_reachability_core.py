from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path


ALPHAS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
VOTE_THRESHOLDS = (1, 2, 3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_jsonl(
    prediction_path: Path,
    manifest_path: Path,
) -> tuple[list[dict], dict]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("contains_test_labels") is not False:
        raise ValueError("frozen prediction manifest contains TEST labels")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("frozen prediction manifest contains evaluation metrics")
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("frozen prediction SHA256 mismatch")
    rows = [
        json.loads(line)
        for line in Path(prediction_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("frozen prediction row count mismatch")
    datasets = [row.get("dataset") for row in rows]
    if any(not isinstance(dataset, str) or not dataset for dataset in datasets):
        raise ValueError("frozen prediction has an invalid dataset")
    if len(set(datasets)) != len(datasets):
        raise ValueError("frozen prediction contains duplicate datasets")
    return rows, manifest


def _validated_indices(
    values: Iterable[int],
    *,
    n_test: int,
    label: str,
) -> set[int]:
    indices = list(values)
    if any(type(index) is not int or index < 0 or index >= n_test for index in indices):
        raise ValueError(f"{label} contains out-of-range indices")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{label} contains duplicate indices")
    return set(indices)


def _true_indices(labels: Sequence[int | str]) -> set[int]:
    return {index for index, label in enumerate(labels) if str(label) == "1"}


def classify_baseline(
    hard_indices: Iterable[int],
    labels: Sequence[int | str],
) -> str:
    hard = _validated_indices(
        hard_indices,
        n_test=len(labels),
        label="Hard lane",
    )
    if not hard:
        return "zero_no_hard"
    if hard & _true_indices(labels):
        return "hard_has_tp"
    return "zero_wrong_hard"


def build_vote_candidates(
    evidence: Mapping[str, Mapping[str, object]],
    *,
    vote_threshold: int,
    n_test: int,
    excluded_indices: Iterable[int],
) -> set[int]:
    if type(vote_threshold) is not int or vote_threshold < 1:
        raise ValueError("vote_threshold must be a positive integer")
    excluded = _validated_indices(
        excluded_indices,
        n_test=n_test,
        label="excluded_indices",
    )
    votes: Counter[int] = Counter()
    for source, payload in evidence.items():
        if payload.get("available") is not True:
            continue
        values = payload.get("indices")
        if not isinstance(values, list):
            raise ValueError(f"{source} indices must be a list")
        source_indices = _validated_indices(
            values,
            n_test=n_test,
            label=f"{source} evidence",
        )
        votes.update(source_indices - excluded)
    return {index for index, count in votes.items() if count >= vote_threshold}


def evaluate_recovery(
    *,
    candidate_indices: Iterable[int],
    hard_indices: Iterable[int],
    review_indices: Iterable[int],
    labels: Sequence[int | str],
) -> dict[str, int | float]:
    n_test = len(labels)
    candidates = _validated_indices(
        candidate_indices,
        n_test=n_test,
        label="candidate_indices",
    )
    hard = _validated_indices(hard_indices, n_test=n_test, label="hard_indices")
    review = _validated_indices(review_indices, n_test=n_test, label="review_indices")
    truth = _true_indices(labels)
    missed = truth - hard
    recovered = candidates & missed
    added_fp = candidates - truth
    review_overlap = candidates & review
    review_recovered = recovered & review
    review_overlap_fp = review_overlap - truth
    return {
        "candidate_count": len(candidates),
        "recovered_fn": len(recovered),
        "added_fp": len(added_fp),
        "candidate_precision": (
            len(recovered) / len(candidates) if candidates else 0.0
        ),
        "review_overlap_count": len(review_overlap),
        "review_recovered_fn": len(review_recovered),
        "review_overlap_fp": len(review_overlap_fp),
        "review_overlap_precision": (
            len(review_recovered) / len(review_overlap) if review_overlap else 0.0
        ),
        "novel_recovered_fn": len(recovered - review),
    }


def aggregate_recovery_rows(rows: Sequence[Mapping[str, object]]) -> dict:
    if not rows:
        raise ValueError("cannot aggregate empty recovery rows")
    fixed_keys = ("alpha", "vote_threshold", "population")
    for key in fixed_keys:
        values = {row.get(key) for row in rows}
        if len(values) != 1:
            raise ValueError(f"recovery rows mix multiple {key} values")
    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "baseline_fn",
            "candidate_count",
            "recovered_fn",
            "added_fp",
            "review_overlap_count",
            "review_recovered_fn",
            "review_overlap_fp",
            "novel_recovered_fn",
        )
    }
    candidate_count = totals["candidate_count"]
    baseline_fn = totals["baseline_fn"]
    return {
        "alpha": rows[0]["alpha"],
        "vote_threshold": rows[0]["vote_threshold"],
        "population": rows[0]["population"],
        "datasets": len({str(row["dataset"]) for row in rows}),
        **totals,
        "candidate_precision": (
            totals["recovered_fn"] / candidate_count if candidate_count else 0.0
        ),
        "fn_recovery_rate": (
            totals["recovered_fn"] / baseline_fn if baseline_fn else 0.0
        ),
        "review_overlap_precision": (
            totals["review_recovered_fn"] / totals["review_overlap_count"]
            if totals["review_overlap_count"]
            else 0.0
        ),
        "selected_for_deployment": False,
        "test_metrics_select_policy": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }
