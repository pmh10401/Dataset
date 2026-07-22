from __future__ import annotations

import numpy as np

from experiments.exp226_235_hard_alert_quality.contracts import ScoreBundle
from experiments.exp237_new_score_source_qualification.config import ALPHAS, SOURCE_MAX_LENGTH, SOURCE_NAMES
from experiments.exp237_new_score_source_qualification.score_sources import (
    fit_score_source,
    score_rows,
)


def deterministic_fold_plan(
    n_rows: int, *, seed: int, folds: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_rows < folds:
        raise ValueError("fold plan requires at least one query row per fold")
    rng = np.random.default_rng(int(seed))
    shuffled = np.arange(int(n_rows), dtype=int)
    rng.shuffle(shuffled)
    queries = np.array_split(shuffled, int(folds))
    plan: list[tuple[np.ndarray, np.ndarray]] = []
    for index, query in enumerate(queries):
        fit = np.concatenate([part for part_index, part in enumerate(queries) if part_index != index])
        plan.append((np.sort(fit), np.sort(query)))
    return plan


def build_score_bundle(
    dataset: str,
    train: np.ndarray,
    test: np.ndarray,
    *,
    seed: int,
) -> ScoreBundle:
    train_matrix = np.asarray(train, dtype=np.float64)
    test_matrix = np.asarray(test, dtype=np.float64)
    if train_matrix.ndim != 2 or test_matrix.ndim != 2:
        raise ValueError("aligned TRAIN and TEST values must be two-dimensional")
    if train_matrix.shape[1] != test_matrix.shape[1]:
        raise ValueError("aligned TRAIN and TEST lengths differ")
    original_length = int(train_matrix.shape[1])
    if original_length > SOURCE_MAX_LENGTH:
        old = np.linspace(0.0, 1.0, original_length)
        new = np.linspace(0.0, 1.0, SOURCE_MAX_LENGTH)
        train_matrix = np.vstack([np.interp(new, old, row) for row in train_matrix])
        test_matrix = np.vstack([np.interp(new, old, row) for row in test_matrix])
    if len(train_matrix) < 5:
        empty = {name: np.asarray([], dtype=float) for name in SOURCE_NAMES}
        return ScoreBundle(
            dataset=str(dataset),
            train_oof=empty,
            test_scores={name: values.copy() for name, values in empty.items()},
            provenance={
                "seed": int(seed),
                "n_train": len(train_matrix),
                "n_test": len(test_matrix),
                "calibration_abstained": True,
                "abstention_reason": "n_train_lt_5",
                "contains_test_labels": False,
            },
        )
    plan = deterministic_fold_plan(len(train_matrix), seed=seed)
    train_oof = {name: np.empty(len(train_matrix), dtype=float) for name in SOURCE_NAMES}
    for fit_indices, query_indices in plan:
        for name in SOURCE_NAMES:
            model = fit_score_source(name, train_matrix[fit_indices])
            train_oof[name][query_indices] = score_rows(model, train_matrix[query_indices])
    test_scores: dict[str, np.ndarray] = {}
    for name in SOURCE_NAMES:
        model = fit_score_source(name, train_matrix)
        test_scores[name] = score_rows(model, test_matrix)
    for values in (*train_oof.values(), *test_scores.values()):
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite Exp237 score: dataset={dataset}")
    return ScoreBundle(
        dataset=str(dataset),
        train_oof=train_oof,
        test_scores=test_scores,
        provenance={
            "seed": int(seed),
            "n_train": len(train_matrix),
            "n_test": len(test_matrix),
            "train_derived_target_length": original_length,
            "score_length": train_matrix.shape[1],
            "folds": 5,
            "calibration_abstained": False,
            "contains_test_labels": False,
            "elastic_method": "bounded_shift_alignment_not_exact_dtw",
            "subsequence_method": "train_dictionary_discord_not_exact_matrix_profile",
        },
    )
