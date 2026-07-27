from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import log

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from experiments.exp393_396.contracts import GrammarConfig


@dataclass(frozen=True)
class ComponentScore:
    transition: float
    dwell: float
    recovery: float


@dataclass(frozen=True)
class DiscreteGrammar:
    transition_probability: np.ndarray
    dwell_lengths: dict[int, np.ndarray]
    high_support_states: frozenset[int]
    transition_surprise_q95: float


@dataclass(frozen=True)
class GrammarModel:
    clusterer: MiniBatchKMeans
    grammar: DiscreteGrammar
    descriptor_center: np.ndarray
    descriptor_scale: np.ndarray


def _top_tail_mean(values: Sequence[float], rate: float = 0.05) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    count = max(1, int(np.ceil(array.size * rate)))
    return float(np.mean(np.partition(array, array.size - count)[-count:]))


def _runs(path: Sequence[int]) -> list[tuple[int, int]]:
    if not path:
        return []
    result: list[tuple[int, int]] = []
    current = int(path[0])
    length = 1
    for value in path[1:]:
        value = int(value)
        if value == current:
            length += 1
        else:
            result.append((current, length))
            current = value
            length = 1
    result.append((current, length))
    return result


def fit_discrete_grammar(
    train_paths: Sequence[Sequence[int]],
    state_count: int,
    high_support_rate: float = 0.05,
    transition_tail_rate: float = 0.95,
) -> DiscreteGrammar:
    if state_count < 1:
        raise ValueError("state_count must be positive")
    transition_counts = np.ones((state_count, state_count), dtype=float)
    occupancy = np.zeros(state_count, dtype=float)
    dwell: dict[int, list[int]] = {state: [] for state in range(state_count)}

    for path in train_paths:
        values = [int(value) for value in path]
        for value in values:
            if value < 0 or value >= state_count:
                raise ValueError("state index is out of range")
            occupancy[value] += 1
        for left, right in zip(values[:-1], values[1:]):
            transition_counts[left, right] += 1
        for state, length in _runs(values):
            dwell[state].append(length)

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    probability = transition_counts / row_sums
    total_occupancy = float(occupancy.sum())
    if total_occupancy == 0:
        raise ValueError("at least one state observation is required")
    high_support = frozenset(
        int(state)
        for state, count in enumerate(occupancy)
        if count / total_occupancy >= high_support_rate
    )

    observed_surprises: list[float] = []
    for path in train_paths:
        values = [int(value) for value in path]
        observed_surprises.extend(
            -log(max(probability[left, right], 1e-300))
            for left, right in zip(values[:-1], values[1:])
        )
    q95 = (
        float(np.quantile(observed_surprises, transition_tail_rate))
        if observed_surprises
        else 0.0
    )
    return DiscreteGrammar(
        transition_probability=probability,
        dwell_lengths={
            state: np.asarray(lengths, dtype=int) for state, lengths in dwell.items()
        },
        high_support_states=high_support,
        transition_surprise_q95=q95,
    )


def score_state_path(
    model: DiscreteGrammar,
    path: Sequence[int],
) -> ComponentScore:
    values = [int(value) for value in path]
    transition_surprises = [
        -log(max(model.transition_probability[left, right], 1e-300))
        for left, right in zip(values[:-1], values[1:])
    ]
    transition = _top_tail_mean(transition_surprises)

    dwell_surprises: list[float] = []
    for state, length in _runs(values):
        reference = model.dwell_lengths.get(state, np.empty(0, dtype=int))
        tail_probability = float(
            (1 + np.count_nonzero(reference >= length)) / (reference.size + 1)
        )
        dwell_surprises.append(-log(max(tail_probability, 1e-300)))
    dwell = _top_tail_mean(dwell_surprises)

    longest_recovery = 0
    for index, surprise in enumerate(transition_surprises):
        if surprise <= model.transition_surprise_q95:
            continue
        recovery = 0
        returned = False
        for state in values[index + 1 :]:
            if state in model.high_support_states:
                returned = True
                break
            recovery += 1
        if not returned:
            recovery += 1
        longest_recovery = max(longest_recovery, recovery)

    return ComponentScore(
        transition=float(transition),
        dwell=float(dwell),
        recovery=float(longest_recovery),
    )


def _descriptor_scaling(
    sequences: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack([np.asarray(sequence, dtype=float) for sequence in sequences])
    center = np.median(combined, axis=0)
    scale = np.median(np.abs(combined - center), axis=0) * 1.4826
    positive = scale[scale > 1e-12]
    fallback = float(np.median(positive)) if positive.size else 1.0
    return center, np.where(scale > 1e-12, scale, fallback)


def fit_grammar_model(
    descriptor_sequences: Sequence[np.ndarray],
    config: GrammarConfig,
) -> GrammarModel:
    if not descriptor_sequences:
        raise ValueError("at least one descriptor sequence is required")
    center, scale = _descriptor_scaling(descriptor_sequences)
    normalized = [
        (np.asarray(sequence, dtype=float) - center) / scale
        for sequence in descriptor_sequences
    ]
    combined = np.vstack(normalized)
    distinct = np.unique(np.round(combined, decimals=12), axis=0).shape[0]
    effective_states = min(config.state_count, distinct)
    if effective_states < 3:
        raise ValueError("effective_state_count_lt_3")
    clusterer = MiniBatchKMeans(
        n_clusters=effective_states,
        random_state=config.random_seed,
        n_init=config.n_init,
        max_iter=config.max_iter,
        batch_size=min(1024, combined.shape[0]),
        reassignment_ratio=0.0,
    )
    clusterer.fit(combined)
    paths = [clusterer.predict(sequence).tolist() for sequence in normalized]
    grammar = fit_discrete_grammar(
        paths,
        state_count=effective_states,
        high_support_rate=config.high_support_rate,
        transition_tail_rate=config.transition_tail_rate,
    )
    return GrammarModel(
        clusterer=clusterer,
        grammar=grammar,
        descriptor_center=center,
        descriptor_scale=scale,
    )


def score_trace(model: GrammarModel, descriptors: np.ndarray) -> ComponentScore:
    normalized = (
        np.asarray(descriptors, dtype=float) - model.descriptor_center
    ) / model.descriptor_scale
    path = model.clusterer.predict(normalized).tolist()
    return score_state_path(model.grammar, path)


def _leave_one_out_percentile(values: np.ndarray, index: int) -> float:
    reference = np.delete(values, index)
    if reference.size == 0:
        return 0.5
    return float(
        (1 + np.count_nonzero(reference <= values[index])) / (reference.size + 1)
    )


def crossfit_train_scores(
    descriptor_sequences: Sequence[np.ndarray],
    config: GrammarConfig,
    fit_observer: Callable[[Sequence[int]], None] | None = None,
) -> dict[str, np.ndarray]:
    count = len(descriptor_sequences)
    if count < 5:
        raise ValueError("n_train_lt_5")
    raw = {
        "transition": np.empty(count, dtype=float),
        "dwell": np.empty(count, dtype=float),
        "recovery": np.empty(count, dtype=float),
    }
    fold_count = min(5, count)
    for fold in range(fold_count):
        held_out_indices = [
            index for index in range(count) if index % fold_count == fold
        ]
        fit_indices = [
            index for index in range(count) if index % fold_count != fold
        ]
        if fit_observer is not None:
            fit_observer(fit_indices)
        model = fit_grammar_model(
            [descriptor_sequences[index] for index in fit_indices],
            config,
        )
        for held_out in held_out_indices:
            score = score_trace(model, descriptor_sequences[held_out])
            raw["transition"][held_out] = score.transition
            raw["dwell"][held_out] = score.dwell
            raw["recovery"][held_out] = score.recovery

    percentiles = np.column_stack(
        [
            np.asarray(
                [
                    _leave_one_out_percentile(raw[name], index)
                    for index in range(count)
                ]
            )
            for name in ("transition", "dwell", "recovery")
        ]
    )
    raw["composite"] = np.mean(percentiles, axis=1)
    return raw


def conformal_p_value(reference: Sequence[float], value: float) -> float:
    array = np.asarray(reference, dtype=float)
    if array.size == 0:
        raise ValueError("conformal reference must not be empty")
    return float((1 + np.count_nonzero(array >= value)) / (array.size + 1))


def candidate_indices(p_values: Sequence[float], alpha: float) -> list[int]:
    return [
        index
        for index, value in enumerate(p_values)
        if float(value) <= float(alpha)
    ]
