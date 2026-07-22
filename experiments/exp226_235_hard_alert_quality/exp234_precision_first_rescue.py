from __future__ import annotations

from collections.abc import Iterable, Mapping

from experiments.exp226_235_hard_alert_quality.contracts import PredictionRow


EXPERIMENT_ID = "experiment_234_precision_first_research_rescue"


def research_rescue_candidates(
    *,
    source_hits: Mapping[str, Iterable[int]],
    excluded_indices: Iterable[int],
    minimum_sources: int,
) -> frozenset[int]:
    required = int(minimum_sources)
    if required < 2 or required > len(source_hits):
        raise ValueError("minimum source count outside available source range")
    counts: dict[int, int] = {}
    for source in sorted(source_hits):
        for index in set(int(value) for value in source_hits[source]):
            counts[index] = counts.get(index, 0) + 1
    excluded = {int(value) for value in excluded_indices}
    return frozenset(
        index for index, count in counts.items() if count >= required and index not in excluded
    )


def build_rescue_prediction(
    *,
    dataset: str,
    variant: str,
    candidate_indices: Iterable[int],
    evidence: Mapping[str, object],
) -> PredictionRow:
    research = tuple(sorted(set(int(value) for value in candidate_indices)))
    return PredictionRow(
        experiment_id=EXPERIMENT_ID,
        dataset=str(dataset),
        variant=str(variant),
        hard_indices=(),
        review_indices=(),
        research_indices=research,
        evidence=dict(evidence),
        autonomous_alert=False,
        contains_test_labels=False,
        contains_evaluation_metrics=False,
    )

