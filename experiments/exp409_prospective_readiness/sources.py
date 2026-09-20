"""Fresh, label-blind source recomputation for Exp409 readiness checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes, sha256_bytes
from experiments.exp403_408.consensus import ConsensusSource
from experiments.exp403_408.contracts import (
    ALPHAS,
    FrozenConfig,
    LearnedSourceScores,
    ShadowSourcePrediction,
    train_eligibility,
)
from experiments.exp403_408.feature_registry import (
    FEATURE_SOURCES,
    load_feature_blocks,
)
from experiments.exp403_408.synthetic_anomaly import (
    MODEL_NAMES,
    SyntheticAnomalyConfig,
    _crossfit_normal_scores,
    score_synthetic_models,
    synthetic_shadow_predictions,
)
from experiments.exp403_408.tri_view_ssl import (
    _produce_raw_scores,
    score_tri_view,
    shadow_predictions,
)

from .config import FROZEN_CONFIG, Exp409FrozenConfig
from .contracts import SourceStatus


_FEATURE_ABSTENTION_REASONS = frozenset(
    {
        "source_artifact_missing",
        "source_not_loaded",
        "source_abstained",
        "source_dataset_missing",
        "source_train_count_mismatch",
    }
)
_LEARNED_ABSTENTION_REASONS = frozenset(
    {"n_train_below_50", "calibration_insufficient_oof"}
)


@dataclass(frozen=True)
class SourceBatch:
    sources: tuple[ConsensusSource, ...]
    statuses: tuple[SourceStatus, ...]
    output_sha256: Mapping[str, str]


def build_source_batch(
    dataset: str,
    bundle: object,
    config: Exp409FrozenConfig = FROZEN_CONFIG,
) -> SourceBatch:
    """Build all thirteen registered statuses from current values in memory."""
    if config != FROZEN_CONFIG:
        raise ValueError("Exp409 source configuration is not frozen")
    if str(getattr(bundle, "dataset", dataset)) != str(dataset):
        raise ValueError("Exp409 source dataset identity mismatch")

    train = _matrix(getattr(bundle, "train_normal", None), "TRAIN")
    test = _matrix(getattr(bundle, "test_values", None), "TEST")
    if train.shape[1] != test.shape[1]:
        raise ValueError("Exp409 TRAIN and TEST run lengths differ")
    target_length = int(getattr(bundle, "target_length", train.shape[1]))
    if target_length != train.shape[1]:
        raise ValueError("Exp409 target length mismatch")
    hard = frozenset(int(index) for index in getattr(bundle, "hard_indices", ()))
    if any(index < 0 or index >= len(test) for index in hard):
        raise ValueError("Exp409 Hard index is outside TEST range")

    sources: list[ConsensusSource] = []
    statuses: list[SourceStatus] = []
    output_sha256: dict[str, str] = {}

    _append_existing_sources(
        str(dataset),
        bundle,
        train=train,
        test=test,
        hard=hard,
        sources=sources,
        statuses=statuses,
        output_sha256=output_sha256,
    )
    _append_learned_sources(
        str(dataset),
        train=train,
        test=test,
        target_length=target_length,
        hard=hard,
        seed=config.seed,
        sources=sources,
        statuses=statuses,
        output_sha256=output_sha256,
    )

    expected = {source.name for source in FEATURE_SOURCES}
    expected.add("tri_view")
    expected.update(f"synthetic_{name}" for name in MODEL_NAMES)
    identities = [status.source for status in statuses]
    if (
        len(identities) != 13
        or len(set(identities)) != 13
        or set(identities) != expected
        or set(output_sha256) != expected
    ):
        raise ValueError("Exp409 source status coverage is invalid")
    return SourceBatch(tuple(sources), tuple(statuses), dict(output_sha256))


def _append_existing_sources(
    dataset: str,
    bundle: object,
    *,
    train: np.ndarray,
    test: np.ndarray,
    hard: frozenset[int],
    sources: list[ConsensusSource],
    statuses: list[SourceStatus],
    output_sha256: dict[str, str],
) -> None:
    registered = {source.name: source for source in FEATURE_SOURCES}
    blocks = tuple(load_feature_blocks(dataset, bundle))
    by_name = {str(getattr(block, "name", "")): block for block in blocks}
    if len(by_name) != len(blocks) or set(by_name) != set(registered):
        raise ValueError("Exp409 feature source coverage is invalid")

    for registered_source in FEATURE_SOURCES:
        name = registered_source.name
        block = by_name[name]
        group = str(getattr(block, "group", ""))
        source_hash = str(getattr(block, "source_hash", ""))
        if group != registered_source.group or not _is_sha256(source_hash):
            raise ValueError(f"Exp409 existing source identity is invalid: {name}")

        if not bool(getattr(block, "available", False)):
            reason = str(getattr(block, "reason", ""))
            if reason not in _FEATURE_ABSTENTION_REASONS:
                raise ValueError(
                    f"Exp409 feature abstention reason is invalid: {name}"
                )
            status = SourceStatus(name, "abstained", reason)
            statuses.append(status)
            output_sha256[name] = _output_hash(status=status)
            continue

        train_oof = _vector(
            getattr(block, "train_oof", None), len(train), f"{name} TRAIN scores"
        )
        test_scores = _vector(
            getattr(block, "test_values", None), len(test), f"{name} TEST scores"
        )
        source = ConsensusSource(
            name=name,
            lineage=group,
            kind="existing",
            train_oof_scores=tuple(float(value) for value in train_oof),
            shadow_predictions=_existing_shadow_predictions(
                dataset, train_oof, test_scores, hard, name
            ),
            dataset=dataset,
        )
        status = SourceStatus(name, "available", "")
        sources.append(source)
        statuses.append(status)
        output_sha256[name] = _output_hash(
            status=status, source=source, upstream_sha256=source_hash
        )


def _append_learned_sources(
    dataset: str,
    *,
    train: np.ndarray,
    test: np.ndarray,
    target_length: int,
    hard: frozenset[int],
    seed: int,
    sources: list[ConsensusSource],
    statuses: list[SourceStatus],
    output_sha256: dict[str, str],
) -> None:
    learned_names = ("tri_view",) + tuple(
        f"synthetic_{name}" for name in MODEL_NAMES
    )
    if not train_eligibility(len(train)).learned_methods:
        for name in learned_names:
            status = SourceStatus(name, "abstained", "n_train_below_50")
            statuses.append(status)
            output_sha256[name] = _output_hash(status=status)
        return

    frozen = FrozenConfig(seed=int(seed), target_length=int(target_length))
    tri_score = score_tri_view(train, test, frozen, device="cpu")
    tri_status, tri_source = _tri_view_source(
        dataset, train, test, hard, frozen, tri_score
    )
    statuses.append(tri_status)
    if tri_source is not None:
        sources.append(tri_source)
    output_sha256["tri_view"] = _output_hash(
        status=tri_status, source=tri_source
    )

    synthetic_config = SyntheticAnomalyConfig(frozen=frozen)
    synthetic_scores = tuple(
        score_synthetic_models(train, test, synthetic_config)
    )
    by_model = {
        str(score.model_name): score
        for score in synthetic_scores
        if isinstance(score, LearnedSourceScores)
    }
    if (
        len(by_model) != len(synthetic_scores)
        or set(by_model) != set(MODEL_NAMES)
    ):
        raise ValueError("Exp409 synthetic model coverage is invalid")

    for model_index, model_name in enumerate(MODEL_NAMES):
        source_name = f"synthetic_{model_name}"
        score = by_model[model_name]
        status, source = _synthetic_source(
            dataset,
            train,
            test,
            hard,
            synthetic_config,
            score,
            model_name=model_name,
            seed=int(seed) + model_index * 10_000,
        )
        statuses.append(status)
        if source is not None:
            sources.append(source)
        output_sha256[source_name] = _output_hash(status=status, source=source)


def _tri_view_source(
    dataset: str,
    train: np.ndarray,
    test: np.ndarray,
    hard: frozenset[int],
    frozen: FrozenConfig,
    score: LearnedSourceScores,
) -> tuple[SourceStatus, ConsensusSource | None]:
    status = _learned_status("tri_view", score, expected_model="")
    if status.status == "abstained":
        return status, None
    raw = _produce_raw_scores(train, test, frozen, "cpu")
    oof = _vector(raw.train_oof_scores, len(train), "tri_view TRAIN OOF scores")
    raw_test = _vector(raw.test_scores, len(test), "tri_view raw TEST scores")
    _validate_available_score(score, expected_model="", n_test=len(test))
    if tuple(raw_test) != tuple(float(value) for value in score.raw_test_scores):
        raise ValueError("Exp409 tri_view fresh raw score mismatch")
    source = ConsensusSource(
        name="tri_view",
        lineage="tri_view_encoder",
        kind="tri_view",
        train_oof_scores=tuple(float(value) for value in oof),
        shadow_predictions=shadow_predictions(dataset, score, hard),
        dataset=dataset,
    )
    return status, source


def _synthetic_source(
    dataset: str,
    train: np.ndarray,
    test: np.ndarray,
    hard: frozenset[int],
    config: SyntheticAnomalyConfig,
    score: LearnedSourceScores,
    *,
    model_name: str,
    seed: int,
) -> tuple[SourceStatus, ConsensusSource | None]:
    source_name = f"synthetic_{model_name}"
    status = _learned_status(source_name, score, expected_model=model_name)
    if status.status == "abstained":
        return status, None
    _validate_available_score(score, expected_model=model_name, n_test=len(test))
    oof = _vector(
        _crossfit_normal_scores(
            train, config=config, model_name=model_name, seed=int(seed)
        ),
        len(train),
        f"{source_name} TRAIN OOF scores",
    )
    source = ConsensusSource(
        name=source_name,
        lineage="synthetic_anomaly",
        kind="synthetic",
        train_oof_scores=tuple(float(value) for value in oof),
        shadow_predictions=synthetic_shadow_predictions(dataset, score, hard),
        dataset=dataset,
    )
    return status, source


def _learned_status(
    source_name: str,
    score: LearnedSourceScores,
    *,
    expected_model: str,
) -> SourceStatus:
    if not isinstance(score, LearnedSourceScores):
        raise ValueError(f"Exp409 learned source is malformed: {source_name}")
    if str(score.model_name) != expected_model:
        raise ValueError(f"Exp409 learned source identity is invalid: {source_name}")
    if score.available:
        if score.abstention_reason:
            raise ValueError(
                f"Exp409 learned source availability is inconsistent: {source_name}"
            )
        return SourceStatus(source_name, "available", "")
    reason = str(score.abstention_reason)
    if reason not in _LEARNED_ABSTENTION_REASONS:
        raise ValueError(
            f"Exp409 learned abstention reason is invalid: {source_name}"
        )
    if score.raw_test_scores or score.scores_by_alpha:
        raise ValueError(
            f"Exp409 learned abstention payload is invalid: {source_name}"
        )
    return SourceStatus(source_name, "abstained", reason)


def _validate_available_score(
    score: LearnedSourceScores, *, expected_model: str, n_test: int
) -> None:
    if (
        not score.available
        or str(score.model_name) != expected_model
        or score.abstention_reason
    ):
        raise ValueError("Exp409 learned source score identity is invalid")
    _vector(score.raw_test_scores, n_test, "learned raw TEST scores")
    expected_alpha = {str(alpha) for alpha in ALPHAS}
    if set(score.scores_by_alpha) != expected_alpha:
        raise ValueError("Exp409 learned source alpha coverage is invalid")
    for alpha in ALPHAS:
        values = _vector(
            score.scores_by_alpha[str(alpha)],
            n_test,
            f"learned {alpha} p-values",
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Exp409 learned source p-value is invalid")


def _existing_shadow_predictions(
    dataset: str,
    train: np.ndarray,
    test: np.ndarray,
    hard: frozenset[int],
    source: str,
) -> dict[str, tuple[ShadowSourcePrediction, ...]]:
    p_values = np.asarray(
        [
            (np.count_nonzero(train >= value) + 1.0) / (len(train) + 1.0)
            for value in test
        ],
        dtype=float,
    )
    return {
        str(alpha): tuple(
            ShadowSourcePrediction(
                dataset=dataset,
                candidate_index=int(index),
                scores_by_alpha={
                    str(other): float(p_values[index]) for other in ALPHAS
                },
                source=source,
            )
            for index, value in enumerate(p_values)
            if index not in hard and float(value) <= float(alpha)
        )
        for alpha in ALPHAS
    }


def _output_hash(
    *,
    status: SourceStatus,
    source: ConsensusSource | None = None,
    upstream_sha256: str = "",
) -> str:
    payload: dict[str, object] = {"status": asdict(status)}
    if upstream_sha256:
        payload["upstream_sha256"] = upstream_sha256
    if source is not None:
        payload["source"] = {
            "name": source.name,
            "lineage": source.lineage,
            "kind": source.kind,
            "dataset": source.dataset,
            "train_oof_scores": _json_scores(source.train_oof_scores),
            "shadow_predictions": {
                str(alpha): [
                    {
                        "dataset": row.dataset,
                        "candidate_index": row.candidate_index,
                        "scores_by_alpha": {
                            str(key): float(value)
                            for key, value in sorted(row.scores_by_alpha.items())
                        },
                        "source": row.source,
                    }
                    for row in source.shadow_predictions[str(alpha)]
                ]
                for alpha in ALPHAS
            },
        }
    return sha256_bytes(canonical_json_bytes(payload))


def _json_scores(
    values: Mapping[object, float] | Sequence[float],
) -> object:
    if isinstance(values, Mapping):
        return [
            [str(key), float(value)]
            for key, value in sorted(values.items(), key=lambda item: str(item[0]))
        ]
    return [float(value) for value in values]


def _matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] == 0
        or result.shape[1] == 0
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"Exp409 {name} values are invalid")
    return result


def _vector(value: object, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 1
        or len(result) != int(size)
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"Exp409 {name} are invalid")
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        and math.isfinite(float(int(value, 16)))
    )
