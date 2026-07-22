from __future__ import annotations

from collections.abc import Iterable

from experiments.exp226_235_hard_alert_quality.contracts import PredictionRow


EXPERIMENT_ID = "experiment_235_shadow_evidence_export"


def build_shadow_rows(
    predictions: Iterable[PredictionRow],
    *,
    prediction_sha256: str,
    generated_at: str = "not_recorded",
) -> list[dict[str, object]]:
    digest = str(prediction_sha256).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("prediction SHA-256 must be a 64-character hexadecimal digest")
    rows: list[dict[str, object]] = []
    for prediction in sorted(predictions, key=lambda row: (row.dataset, row.variant)):
        candidates = sorted(
            set(prediction.hard_indices)
            | set(prediction.review_indices)
            | set(prediction.research_indices)
        )
        for index in candidates:
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "source_experiment_id": prediction.experiment_id,
                    "dataset": prediction.dataset,
                    "variant": prediction.variant,
                    "candidate_index": int(index),
                    "evidence": dict(prediction.evidence),
                    "prediction_sha256": digest,
                    "generated_at": str(generated_at),
                    "autonomous_alert": False,
                    "contains_test_labels": False,
                    "contains_evaluation_metrics": False,
                    "scope": "prospective_shadow_input_only",
                }
            )
    return rows
