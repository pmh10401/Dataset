from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
import math

import numpy as np

from .acceleration import (
    ACCELERATION_POLICY,
    AccelerationPolicy,
    audit_dataset_pairs,
    boundary_safe_minimum,
    boundary_safe_threshold,
    choose_backend,
    full_pairwise_rms,
    minimum_rms,
)
from .backends import pairwise_rms
from .config import FROZEN_CONFIG, Exp411FrozenConfig
from .contracts import DatasetBackendAudit, DatasetQualification, TrainRun
from .distances import (
    VIEW_NAMES,
    calibrate_thresholds,
    cpu_self_excluded_rms,
    tag_queries,
    view_matrices,
)
from .folds import group_crossfit_plan
from .perturbations import (
    generate_train_only_perturbations,
    perturbation_types,
)


def _quantile_key(value: float) -> str:
    return f"{float(value):.2f}"


def passes_gate(
    row: Mapping[str, float],
    config: Exp411FrozenConfig = FROZEN_CONFIG,
) -> bool:
    return (
        float(row["normal_acceptance"]) >= config.minimum_normal_acceptance
        and float(row["pooled_synthetic"])
        <= config.pooled_synthetic_false_normal_max
        and float(row["worst_perturbation"])
        <= config.per_perturbation_false_normal_max
        and float(row["bootstrap_upper"]) <= config.bootstrap_upper_max
    )


def select_quantile(
    rows: Mapping[float, Mapping[str, float]],
    config: Exp411FrozenConfig = FROZEN_CONFIG,
) -> float | None:
    qualified = [
        float(quantile)
        for quantile, row in rows.items()
        if passes_gate(row, config)
    ]
    return max(qualified) if qualified else None


def qualify_dataset(
    *,
    dataset: str,
    family: str,
    runs: Sequence[TrainRun],
    backend: str = "numpy_cpu",
    accelerated: bool = False,
    backend_audit: list[DatasetBackendAudit] | None = None,
    parity_trace: list[dict[str, object]] | None = None,
    config: Exp411FrozenConfig = FROZEN_CONFIG,
) -> DatasetQualification:
    config.validate_research_parameters()
    ordered_runs = tuple(runs)
    if any(run.dataset != dataset for run in ordered_runs):
        raise ValueError("Exp411 dataset qualification received a foreign run")
    if accelerated:
        return _qualify_dataset_accelerated(
            dataset=dataset,
            family=family,
            runs=ordered_runs,
            backend_audit=backend_audit,
            parity_trace=parity_trace,
            config=config,
        )
    keys = tuple(_quantile_key(value) for value in config.quantiles)
    types = perturbation_types()
    normal_near = {key: 0 for key in keys}
    synthetic_total = {key: 0 for key in keys}
    synthetic_near = {key: 0 for key in keys}
    perturbation_total = {
        key: {name: 0 for name in types}
        for key in keys
    }
    perturbation_near = {
        key: {name: 0 for name in types}
        for key in keys
    }
    normal_abstained = 0
    folds = group_crossfit_plan(ordered_runs, seed=config.seed)
    if not folds:
        normal_abstained = len(ordered_runs)
        for run in ordered_runs:
            for quantile in config.quantiles:
                _append_trace(
                    parity_trace,
                    dataset=dataset,
                    query=run,
                    quantile=quantile,
                    kind="normal",
                    subtype=None,
                    candidate_index=run.instance_index,
                    tag="abstain",
                )

    for fold_number, fold in enumerate(folds):
        fit_by_length: dict[int, list[int]] = defaultdict(list)
        for index in fold.fit_indices:
            fit_by_length[len(ordered_runs[index].values)].append(index)
        query_by_length: dict[int, list[int]] = defaultdict(list)
        for index in fold.query_indices:
            query_by_length[len(ordered_runs[index].values)].append(index)

        for length, query_indices in sorted(query_by_length.items()):
            fit_indices = fit_by_length.get(length, [])
            representative_indices = _identity_representatives(
                ordered_runs,
                fit_indices,
            )
            if len(representative_indices) < config.minimum_same_length_fit_groups:
                normal_abstained += len(query_indices)
                for query_index in query_indices:
                    for quantile in config.quantiles:
                        _append_trace(
                            parity_trace,
                            dataset=dataset,
                            query=ordered_runs[query_index],
                            quantile=quantile,
                            kind="normal",
                            subtype=None,
                            candidate_index=ordered_runs[query_index].instance_index,
                            tag="abstain",
                        )
                continue
            fit_values = np.asarray(
                [ordered_runs[index].values for index in representative_indices],
                dtype=np.float64,
            )
            prepared = _prepare_fit_views(
                fit_values,
                backend=backend,
                config=config,
            )
            if len(prepared) < 2 or "raw_rms" not in prepared:
                normal_abstained += len(query_indices)
                for query_index in query_indices:
                    for quantile in config.quantiles:
                        _append_trace(
                            parity_trace,
                            dataset=dataset,
                            query=ordered_runs[query_index],
                            quantile=quantile,
                            kind="normal",
                            subtype=None,
                            candidate_index=ordered_runs[query_index].instance_index,
                            tag="abstain",
                        )
                continue
            fit_hashes = {
                ordered_runs[index].values_sha256
                for index in representative_indices
            }
            for query_index in query_indices:
                query_run = ordered_runs[query_index]
                query = np.asarray([query_run.values], dtype=np.float64)
                normal_distances = _query_distances(
                    prepared,
                    query,
                    backend=backend,
                )
                synthetic = generate_train_only_perturbations(
                    query,
                    target_length=length,
                    seed=_derived_seed(
                        dataset,
                        fold_number,
                        query_run.values_sha256,
                    ),
                )
                synthetic_distances = {
                    name: _query_distances(
                        prepared,
                        values,
                        backend=backend,
                    )
                    for name, values in synthetic.by_type.items()
                }
                for quantile in config.quantiles:
                    key = _quantile_key(quantile)
                    thresholds = {
                        name: values["thresholds"][key]
                        for name, values in prepared.items()
                    }
                    normal_tag = tag_queries(
                        normal_distances,
                        thresholds,
                        exact=(query_run.values_sha256 in fit_hashes,),
                    )[0]
                    normal_near[key] += int(normal_tag == "near")
                    _append_trace(
                        parity_trace,
                        dataset=dataset,
                        query=query_run,
                        quantile=quantile,
                        kind="normal",
                        subtype=None,
                        candidate_index=query_run.instance_index,
                        tag=normal_tag,
                    )
                    for name in types:
                        values = synthetic.by_type[name]
                        tags = tag_queries(
                            synthetic_distances[name],
                            thresholds,
                            exact=(False,) * len(values),
                        )
                        for synthetic_index, tag in enumerate(tags):
                            _append_trace(
                                parity_trace,
                                dataset=dataset,
                                query=query_run,
                                quantile=quantile,
                                kind="synthetic",
                                subtype=name,
                                candidate_index=synthetic_index,
                                tag=tag,
                            )
                        accepted = sum(tag == "near" for tag in tags)
                        perturbation_total[key][name] += len(values)
                        perturbation_near[key][name] += accepted
                        synthetic_total[key] += len(values)
                        synthetic_near[key] += accepted

    return DatasetQualification(
        dataset=str(dataset),
        family=str(family),
        identity_group_count=len({run.values_sha256 for run in ordered_runs}),
        normal_total=len(ordered_runs),
        normal_abstained=normal_abstained,
        normal_near=normal_near,
        synthetic_total=synthetic_total,
        synthetic_near=synthetic_near,
        perturbation_total=perturbation_total,
        perturbation_near=perturbation_near,
    )


def _qualify_dataset_accelerated(
    *,
    dataset: str,
    family: str,
    runs: Sequence[TrainRun],
    backend_audit: list[DatasetBackendAudit] | None,
    parity_trace: list[dict[str, object]] | None,
    config: Exp411FrozenConfig,
    policy: AccelerationPolicy = ACCELERATION_POLICY,
) -> DatasetQualification:
    policy.validate()
    keys = tuple(_quantile_key(value) for value in config.quantiles)
    types = perturbation_types()
    normal_near = {key: 0 for key in keys}
    synthetic_total = {key: 0 for key in keys}
    synthetic_near = {key: 0 for key in keys}
    perturbation_total = {
        key: {name: 0 for name in types}
        for key in keys
    }
    perturbation_near = {
        key: {name: 0 for name in types}
        for key in keys
    }
    normal_abstained = 0
    folds = group_crossfit_plan(runs, seed=config.seed)
    if not folds:
        if backend_audit is not None:
            backend_audit.append(
                DatasetBackendAudit(
                    backend="numpy_cpu",
                    sampled_pairs=(),
                    max_abs_error=0.0,
                    fallback_reason=None,
                )
            )
        for run in runs:
            for quantile in config.quantiles:
                _append_trace(
                    parity_trace,
                    dataset=dataset,
                    query=run,
                    quantile=quantile,
                    kind="normal",
                    subtype=None,
                    candidate_index=run.instance_index,
                    tag="abstain",
                )
        return DatasetQualification(
            dataset=str(dataset),
            family=str(family),
            identity_group_count=len({run.values_sha256 for run in runs}),
            normal_total=len(runs),
            normal_abstained=len(runs),
            normal_near=normal_near,
            synthetic_total=synthetic_total,
            synthetic_near=synthetic_near,
            perturbation_total=perturbation_total,
            perturbation_near=perturbation_near,
        )

    length_contexts, audits = _precompute_fold_view_state(
        runs,
        folds,
        policy=policy,
        config=config,
    )
    if backend_audit is not None:
        backend_audit.append(_combine_backend_audits(audits))

    for fold_number, fold in enumerate(folds):
        query_by_length: dict[int, list[int]] = defaultdict(list)
        for index in fold.query_indices:
            query_by_length[len(runs[index].values)].append(index)
        for length, query_indices in sorted(query_by_length.items()):
            context = length_contexts[length]
            fold_state = context["folds"][fold_number]
            views = fold_state["views"]
            if (
                len(views) < 2
                or "raw_rms" not in views
                or not fold_state["eligible"]
            ):
                normal_abstained += len(query_indices)
                for query_index in query_indices:
                    for quantile in config.quantiles:
                        _append_trace(
                            parity_trace,
                            dataset=dataset,
                            query=runs[query_index],
                            quantile=quantile,
                            kind="normal",
                            subtype=None,
                            candidate_index=runs[query_index].instance_index,
                            tag="abstain",
                        )
                continue

            synthetics = [
                generate_train_only_perturbations(
                    np.asarray([runs[query_index].values], dtype=np.float64),
                    target_length=length,
                    seed=_derived_seed(
                        dataset,
                        fold_number,
                        runs[query_index].values_sha256,
                    ),
                )
                for query_index in query_indices
            ]
            synthetic_distances = _synthetic_distances_for_fold(
                context,
                fold_state,
                synthetics,
                types=types,
                policy=policy,
            )
            fit_hashes = fold_state["fit_hashes"]
            for query_position, query_index in enumerate(query_indices):
                query_run = runs[query_index]
                normal_distances = {
                    name: np.asarray(
                        [values["normal"][query_index]],
                        dtype=np.float64,
                    )
                    for name, values in views.items()
                }
                for quantile in config.quantiles:
                    key = _quantile_key(quantile)
                    thresholds = {
                        name: values["thresholds"][key]
                        for name, values in views.items()
                    }
                    normal_tag = tag_queries(
                        normal_distances,
                        thresholds,
                        exact=(query_run.values_sha256 in fit_hashes,),
                    )[0]
                    normal_near[key] += int(normal_tag == "near")
                    _append_trace(
                        parity_trace,
                        dataset=dataset,
                        query=query_run,
                        quantile=quantile,
                        kind="normal",
                        subtype=None,
                        candidate_index=query_run.instance_index,
                        tag=normal_tag,
                    )
                    for name in types:
                        values = synthetics[query_position].by_type[name]
                        distances = {
                            view_name: synthetic_distances[name][view_name][
                                query_position
                            ]
                            for view_name in views
                        }
                        tags = tag_queries(
                            distances,
                            thresholds,
                            exact=(False,) * len(values),
                        )
                        for synthetic_index, tag in enumerate(tags):
                            _append_trace(
                                parity_trace,
                                dataset=dataset,
                                query=query_run,
                                quantile=quantile,
                                kind="synthetic",
                                subtype=name,
                                candidate_index=synthetic_index,
                                tag=tag,
                            )
                        accepted = sum(tag == "near" for tag in tags)
                        perturbation_total[key][name] += len(values)
                        perturbation_near[key][name] += accepted
                        synthetic_total[key] += len(values)
                        synthetic_near[key] += accepted

    return DatasetQualification(
        dataset=str(dataset),
        family=str(family),
        identity_group_count=len({run.values_sha256 for run in runs}),
        normal_total=len(runs),
        normal_abstained=normal_abstained,
        normal_near=normal_near,
        synthetic_total=synthetic_total,
        synthetic_near=synthetic_near,
        perturbation_total=perturbation_total,
        perturbation_near=perturbation_near,
    )


def _precompute_fold_view_state(
    runs: Sequence[TrainRun],
    folds: Sequence[object],
    *,
    policy: AccelerationPolicy,
    config: Exp411FrozenConfig,
) -> tuple[dict[int, dict[str, object]], tuple[DatasetBackendAudit, ...]]:
    all_by_length: dict[int, list[int]] = defaultdict(list)
    for index, run in enumerate(runs):
        all_by_length[len(run.values)].append(index)
    contexts: dict[int, dict[str, object]] = {}
    audits: list[DatasetBackendAudit] = []
    for length, indices in sorted(all_by_length.items()):
        representative_indices = _identity_representatives(runs, indices)
        raw_values = np.asarray(
            [runs[index].values for index in representative_indices],
            dtype=np.float64,
        )
        digests = tuple(
            runs[index].values_sha256 for index in representative_indices
        )
        digest_to_position = {
            digest: position for position, digest in enumerate(digests)
        }
        context: dict[str, object] = {
            "raw_values": raw_values,
            "digests": digests,
            "digest_to_position": digest_to_position,
            "folds": {},
        }
        for fold_number, fold in enumerate(folds):
            fit_indices = [
                index
                for index in fold.fit_indices
                if len(runs[index].values) == length
            ]
            fit_representatives = _identity_representatives(runs, fit_indices)
            fit_hashes = {
                runs[index].values_sha256 for index in fit_representatives
            }
            query_indices = tuple(
                index
                for index in fold.query_indices
                if len(runs[index].values) == length
            )
            context["folds"][fold_number] = {
                "eligible": (
                    len(fit_representatives)
                    >= config.minimum_same_length_fit_groups
                    and bool(query_indices)
                ),
                "query_indices": query_indices,
                "fit_positions": tuple(
                    digest_to_position[runs[index].values_sha256]
                    for index in fit_representatives
                ),
                "fit_hashes": fit_hashes,
                "views": {},
            }
        if len(raw_values) >= 2:
            for view_name in VIEW_NAMES:
                audits.append(
                    audit_dataset_pairs(
                        _one_view_matrix(raw_values, view_name),
                        digests,
                        policy,
                    )
                )
        contexts[length] = context

    force_numpy = any(
        audit.fallback_reason is not None for audit in audits
    )
    for length, context in sorted(contexts.items()):
        raw_values = np.asarray(context["raw_values"], dtype=np.float64)
        digest_to_position = context["digest_to_position"]
        for view_name in VIEW_NAMES:
            view_values = _one_view_matrix(raw_values, view_name)
            usable_folds = []
            for fold_number, fold_state in context["folds"].items():
                if not fold_state["eligible"]:
                    continue
                fit_positions = np.asarray(
                    fold_state["fit_positions"],
                    dtype=np.int64,
                )
                fit_view = view_values[fit_positions]
                median = float(np.median(fit_view))
                mad = float(np.median(np.abs(fit_view - median)))
                if math.isfinite(mad) and mad > config.mad_floor:
                    usable_folds.append((fold_number, fit_positions, mad))
            if not usable_folds:
                continue
            selected_backend = (
                "numpy_cpu"
                if force_numpy
                else choose_backend(
                    len(view_values) * len(view_values),
                    view_values.shape[1],
                    policy,
                )
            )
            pairwise = full_pairwise_rms(
                view_values,
                backend=selected_backend,
            )
            for fold_number, fit_positions, mad in usable_folds:
                fold_state = context["folds"][fold_number]
                fit_scaled = np.ascontiguousarray(
                    view_values[fit_positions] / mad,
                    dtype=np.float64,
                )
                fit_pairwise = (
                    pairwise[np.ix_(fit_positions, fit_positions)] / mad
                )
                np.fill_diagonal(fit_pairwise, np.inf)
                reference = np.min(fit_pairwise, axis=1)
                thresholds = {
                    _quantile_key(quantile): boundary_safe_threshold(
                        fit_scaled,
                        reference,
                        quantile,
                        tolerance=policy.distance_tolerance,
                    )
                    for quantile in config.quantiles
                }
                query_indices = fold_state["query_indices"]
                query_positions = np.asarray(
                    [
                        digest_to_position[runs[index].values_sha256]
                        for index in query_indices
                    ],
                    dtype=np.int64,
                )
                normal = (
                    np.min(
                        pairwise[np.ix_(query_positions, fit_positions)],
                        axis=1,
                    )
                    / mad
                )
                normal = boundary_safe_minimum(
                    fit_scaled,
                    np.ascontiguousarray(
                        view_values[query_positions] / mad,
                        dtype=np.float64,
                    ),
                    normal,
                    tuple(thresholds.values()),
                    tolerance=policy.distance_tolerance,
                )
                fold_state["views"][view_name] = {
                    "scale": mad,
                    "thresholds": thresholds,
                    "normal": {
                        query_index: float(distance)
                        for query_index, distance in zip(
                            query_indices,
                            normal,
                            strict=True,
                        )
                    },
                    "backend": selected_backend,
                }
            del pairwise
            del view_values
    return contexts, tuple(audits)


def _synthetic_distances_for_fold(
    context: Mapping[str, object],
    fold_state: Mapping[str, object],
    synthetics: Sequence[object],
    *,
    types: Sequence[str],
    policy: AccelerationPolicy,
) -> dict[str, dict[str, list[np.ndarray]]]:
    raw_values = np.asarray(context["raw_values"], dtype=np.float64)
    fit_positions = np.asarray(fold_state["fit_positions"], dtype=np.int64)
    result: dict[str, dict[str, list[np.ndarray]]] = {
        name: {} for name in types
    }
    for view_name, view_state in fold_state["views"].items():
        scale = float(view_state["scale"])
        fit_view = np.ascontiguousarray(
            _one_view_matrix(raw_values[fit_positions], view_name) / scale,
            dtype=np.float64,
        )
        thresholds = tuple(view_state["thresholds"].values())
        for perturbation_name in types:
            sizes = tuple(
                len(synthetic.by_type[perturbation_name])
                for synthetic in synthetics
            )
            joined = np.concatenate(
                [
                    np.asarray(
                        synthetic.by_type[perturbation_name],
                        dtype=np.float64,
                    )
                    for synthetic in synthetics
                ],
                axis=0,
            )
            query_view = np.ascontiguousarray(
                _one_view_matrix(joined, view_name) / scale,
                dtype=np.float64,
            )
            backend = (
                "numpy_cpu"
                if view_state["backend"] == "numpy_cpu"
                else choose_backend(
                    len(fit_view) * len(query_view),
                    fit_view.shape[1],
                    policy,
                )
            )
            observed = _batched_synthetic_distances(
                fit_view,
                query_view,
                backend=backend,
                batch_rows=policy.synthetic_batch_rows,
            )
            observed = boundary_safe_minimum(
                fit_view,
                query_view,
                observed,
                thresholds,
                tolerance=policy.distance_tolerance,
            )
            split: list[np.ndarray] = []
            start = 0
            for size in sizes:
                split.append(observed[start : start + size])
                start += size
            result[perturbation_name][view_name] = split
    return result


def _combine_backend_audits(
    audits: Sequence[DatasetBackendAudit],
) -> DatasetBackendAudit:
    if not audits:
        return DatasetBackendAudit(
            backend="numpy_cpu",
            sampled_pairs=(),
            max_abs_error=0.0,
            fallback_reason=None,
        )
    worst = max(audits, key=lambda value: value.max_abs_error)
    fallback = next(
        (
            value.fallback_reason
            for value in audits
            if value.fallback_reason is not None
        ),
        None,
    )
    return DatasetBackendAudit(
        backend=(
            "numpy_cpu"
            if fallback is not None
            or all(value.backend == "numpy_cpu" for value in audits)
            else "numba_cpu"
        ),
        sampled_pairs=worst.sampled_pairs,
        max_abs_error=max(value.max_abs_error for value in audits),
        fallback_reason=fallback,
    )


def _one_view_matrix(rows: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if name == "raw_rms":
        return values
    if name == "diff_rms":
        return np.diff(values, axis=1)
    if name == "spectrum_rms":
        return np.abs(np.fft.rfft(values, axis=1))[:, 1:]
    raise ValueError(f"unknown Exp411 distance view: {name}")


def _append_trace(
    trace: list[dict[str, object]] | None,
    *,
    dataset: str,
    query: TrainRun,
    quantile: float,
    kind: str,
    subtype: str | None,
    candidate_index: int,
    tag: str,
) -> None:
    if trace is None:
        return
    trace.append(
        {
            "dataset": str(dataset),
            "query_instance_index": int(query.instance_index),
            "query_identity_sha256": str(query.values_sha256),
            "quantile": _quantile_key(quantile),
            "kind": str(kind),
            "subtype": subtype,
            "candidate_index": int(candidate_index),
            "tag": str(tag),
        }
    )


def _identity_representatives(
    runs: Sequence[TrainRun],
    indices: Sequence[int],
) -> tuple[int, ...]:
    representatives: dict[str, int] = {}
    for index in indices:
        digest = runs[index].values_sha256
        representatives.setdefault(digest, index)
    return tuple(representatives[digest] for digest in sorted(representatives))


def _prepare_fit_views(
    fit_values: np.ndarray,
    *,
    backend: str,
    config: Exp411FrozenConfig,
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name, matrix in view_matrices(fit_values).items():
        median = float(np.median(matrix))
        mad = float(np.median(np.abs(matrix - median)))
        if not math.isfinite(mad) or mad <= config.mad_floor:
            continue
        scaled = np.ascontiguousarray(matrix / mad, dtype=np.float64)
        reference = cpu_self_excluded_rms(scaled)
        result[name] = {
            "fit": scaled,
            "scale": mad,
            "thresholds": {
                _quantile_key(quantile): calibrate_thresholds(
                    {name: reference},
                    quantile,
                )[name]
                for quantile in config.quantiles
            },
        }
    return result


def _query_distances(
    prepared: Mapping[str, Mapping[str, object]],
    query_values: object,
    *,
    backend: str,
) -> dict[str, np.ndarray]:
    query_views = view_matrices(query_values)
    return {
        name: pairwise_rms(
            values["fit"],
            np.asarray(query_views[name], dtype=np.float64)
            / float(values["scale"]),
            backend=backend,
        )
        for name, values in prepared.items()
    }


def _batched_synthetic_distances(
    fit: np.ndarray,
    query: np.ndarray,
    *,
    backend: str,
    batch_rows: int,
) -> np.ndarray:
    fit_values = np.ascontiguousarray(fit, dtype=np.float64)
    query_values = np.ascontiguousarray(query, dtype=np.float64)
    if (
        fit_values.ndim != 2
        or query_values.ndim != 2
        or fit_values.shape[0] == 0
        or query_values.shape[0] == 0
        or fit_values.shape[1] != query_values.shape[1]
        or int(batch_rows) < 1
    ):
        raise ValueError("Exp411 synthetic distance batch is invalid")
    output = []
    for start in range(0, len(query_values), int(batch_rows)):
        output.append(
            minimum_rms(
                fit_values,
                query_values[start : start + int(batch_rows)],
                backend=backend,
            )
        )
    return np.concatenate(output)


def _derived_seed(dataset: str, fold: int, identity: str) -> int:
    material = f"Exp411:{dataset}:{int(fold)}:{identity}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "big") % (2**32)


def aggregate_qualification(
    rows: Sequence[DatasetQualification],
    *,
    bootstrap_iterations: int | None = None,
    config: Exp411FrozenConfig = FROZEN_CONFIG,
) -> dict[str, object]:
    values = tuple(rows)
    if not values:
        raise ValueError("Exp411 qualification rows are empty")
    iterations = (
        config.bootstrap_iterations
        if bootstrap_iterations is None
        else int(bootstrap_iterations)
    )
    if iterations < 1:
        raise ValueError("Exp411 bootstrap iterations must be positive")
    quantile_rows: dict[float, dict[str, float]] = {}
    for quantile in config.quantiles:
        key = _quantile_key(quantile)
        normal_total = sum(
            row.normal_total - row.normal_abstained
            for row in values
        )
        normal_near = sum(int(row.normal_near[key]) for row in values)
        synthetic_total = sum(int(row.synthetic_total[key]) for row in values)
        synthetic_near = sum(int(row.synthetic_near[key]) for row in values)
        by_type_total = {
            name: sum(int(row.perturbation_total[key][name]) for row in values)
            for name in perturbation_types()
        }
        by_type_near = {
            name: sum(int(row.perturbation_near[key][name]) for row in values)
            for name in perturbation_types()
        }
        if sum(by_type_total.values()) != synthetic_total:
            raise ValueError("Exp411 synthetic perturbation totals mismatch")
        normal_acceptance = normal_near / normal_total if normal_total else 0.0
        pooled_synthetic = (
            synthetic_near / synthetic_total if synthetic_total else 1.0
        )
        worst_perturbation = max(
            (
                by_type_near[name] / by_type_total[name]
                if by_type_total[name]
                else 1.0
            )
            for name in perturbation_types()
        )
        bootstrap_upper = _family_bootstrap_upper(
            values,
            key=key,
            iterations=iterations,
            seed=config.seed,
        )
        quantile_rows[float(quantile)] = {
            "normal_acceptance": normal_acceptance,
            "pooled_synthetic": pooled_synthetic,
            "worst_perturbation": worst_perturbation,
            "bootstrap_upper": bootstrap_upper,
        }
    selected = select_quantile(quantile_rows, config)
    return {
        "status": "qualified" if selected is not None else "no_qualified_near_rule",
        "selected_quantile": selected,
        "quantiles": quantile_rows,
        "dataset_count": len(values),
        "family_count": len({row.family for row in values}),
    }


def _family_bootstrap_upper(
    rows: Sequence[DatasetQualification],
    *,
    key: str,
    iterations: int,
    seed: int,
) -> float:
    by_family: dict[str, tuple[int, int]] = {}
    grouped_near: dict[str, int] = defaultdict(int)
    grouped_total: dict[str, int] = defaultdict(int)
    for row in rows:
        grouped_near[row.family] += int(row.synthetic_near[key])
        grouped_total[row.family] += int(row.synthetic_total[key])
    for family in grouped_total:
        by_family[family] = (grouped_near[family], grouped_total[family])
    families = tuple(sorted(by_family))
    rng = np.random.default_rng(int(seed))
    rates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = rng.choice(families, size=len(families), replace=True)
        near = sum(by_family[str(family)][0] for family in sampled)
        total = sum(by_family[str(family)][1] for family in sampled)
        rates[index] = near / total if total else 1.0
    return float(np.quantile(rates, 0.95, method="higher"))


__all__ = [
    "aggregate_qualification",
    "passes_gate",
    "qualify_dataset",
    "select_quantile",
]
