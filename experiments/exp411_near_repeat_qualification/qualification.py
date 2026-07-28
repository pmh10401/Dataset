from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from hashlib import sha256
import math

import numpy as np

from .backends import pairwise_rms
from .config import FROZEN_CONFIG, Exp411FrozenConfig
from .contracts import DatasetQualification, TrainRun
from .distances import VIEW_NAMES, calibrate_thresholds, tag_queries, view_matrices
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
    config: Exp411FrozenConfig = FROZEN_CONFIG,
) -> DatasetQualification:
    config.validate()
    ordered_runs = tuple(runs)
    if any(run.dataset != dataset for run in ordered_runs):
        raise ValueError("Exp411 dataset qualification received a foreign run")
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
                    for name in types:
                        values = synthetic.by_type[name]
                        tags = tag_queries(
                            synthetic_distances[name],
                            thresholds,
                            exact=(False,) * len(values),
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
        reference = np.asarray(
            [
                pairwise_rms(
                    np.delete(scaled, index, axis=0),
                    scaled[index : index + 1],
                    backend=backend,
                )[0]
                for index in range(len(scaled))
            ],
            dtype=np.float64,
        )
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
