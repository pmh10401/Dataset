from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .config import (
    ALPHA,
    CACHE_CONTRACTS,
    EXPECTED_DATASETS,
    FROZEN_AUDIT_CONFIGS,
    FROZEN_AUDIT_HASHES,
    FROZEN_CACHE_MANIFEST_HASHES,
    FROZEN_RECON_ARCHIVE_HASH_INVENTORY_SHA256,
    MINIMUM_TRAIN,
    MINI_CACHE_ROOT,
    OUTPUT_ROOT,
    RECON_ARCHIVE_HASH_INVENTORY,
    RECON_CACHE_ROOT,
    RISK_TARGET,
    SEED,
)


CALIBRATION_LIMITATION = (
    "Leave-one-out calibration combines query-excluded OOF scores produced by "
    "different fold models."
)
RECONSTRUCTION_PROVENANCE_BLOCKER = (
    "Five-source combined TRAIN-risk approval is blocked until Exp257 has a "
    "producer-signed/pinned manifest tying the frozen archives to query-excluded "
    "fitting and exact seeds, or a separately preregistered regeneration is performed."
)
CERTIFIED_SOURCE_GROUPS = (
    "kernel",
    "statistical",
    "structural",
    "change_point",
)
_EXPECTED_CALIBRATION = {
    "kernel": "five_fold_transformer_crossfit",
    "statistical": "five_fold_oof",
    "structural": "five_fold_oof",
    "change_point": "five_fold_oof",
}
_TRAIN_KEYS = {
    "kernel_1024": "kernels_1024__train_by_seed",
    "kernel_4096": "kernels_4096__train_by_seed",
    "statistical": "combined_40__train_oof",
    "structural": "structural_combined_440__train_oof",
    "change_point": "change_point_combined_33__train_oof",
    "reconstruction": "mask_025__train_by_seed",
}


class TrainRiskContractError(ValueError):
    """Raised when a frozen TRAIN-risk source violates the audit contract."""


@dataclass(frozen=True)
class PolicyRiskMetrics:
    recovery_shadow_false_reviews: int
    recovery_priority_false_reviews: int
    combined_false_reviews: int
    recovery_shadow_false_review_rate: float
    recovery_priority_false_review_rate: float
    combined_false_review_rate: float
    recovery_shadow_wilson_upper_95: float
    recovery_priority_wilson_upper_95: float
    combined_wilson_upper_95: float
    recovery_shadow_below_risk_target: bool
    recovery_priority_below_risk_target: bool
    combined_below_risk_target: bool


@dataclass(frozen=True)
class DatasetTrainRiskMetrics:
    dataset: str
    n_train: int
    abstained: bool
    abstention_reason: str
    normal_runs: int
    reconstruction_available: bool
    sensitivity_available_source_count: int
    certified: PolicyRiskMetrics
    sensitivity: PolicyRiskMetrics | None


_PROVENANCE_TOKEN_GUARD = object()


class _ValidatedProvenance:
    __slots__ = ("_guard",)

    def __init__(self, guard: object) -> None:
        if guard is not _PROVENANCE_TOKEN_GUARD:
            raise TrainRiskContractError("invalid validated provenance token")
        self._guard = guard


def _is_validated_provenance(value: object) -> bool:
    return (
        isinstance(value, _ValidatedProvenance)
        and getattr(value, "_guard", None) is _PROVENANCE_TOKEN_GUARD
    )


@dataclass(frozen=True)
class TrainRiskRow:
    dataset: str
    n_train: int
    abstained: bool
    abstention_reason: str
    normal_runs: int
    source_count: int
    certified_sources: str
    provenance_verified: bool
    policy_risk_validated: bool
    risk_target_pass: bool
    reconstruction_available_but_provenance_unverified: bool
    reconstruction_provenance_verified: bool
    reconstruction_provenance_blocker: str
    recovery_shadow_false_reviews: int
    recovery_priority_false_reviews: int
    combined_false_reviews: int
    recovery_shadow_false_review_rate: float
    recovery_priority_false_review_rate: float
    combined_false_review_rate: float
    recovery_shadow_wilson_upper_95: float
    recovery_priority_wilson_upper_95: float
    combined_wilson_upper_95: float
    recovery_shadow_meets_risk_target: bool
    recovery_priority_meets_risk_target: bool
    combined_meets_risk_target: bool
    five_source_sensitivity_calculated: bool
    five_source_sensitivity_configured_source_count: int | None
    five_source_sensitivity_available_source_count: int | None
    five_source_sensitivity_recovery_shadow_false_reviews: int | None
    five_source_sensitivity_recovery_priority_false_reviews: int | None
    five_source_sensitivity_combined_false_reviews: int | None
    five_source_sensitivity_recovery_shadow_false_review_rate: float | None
    five_source_sensitivity_recovery_priority_false_review_rate: float | None
    five_source_sensitivity_combined_false_review_rate: float | None
    five_source_sensitivity_recovery_shadow_wilson_upper_95: float | None
    five_source_sensitivity_recovery_priority_wilson_upper_95: float | None
    five_source_sensitivity_combined_wilson_upper_95: float | None
    five_source_sensitivity_provenance_verified: bool
    five_source_sensitivity_policy_risk_validated: bool
    five_source_sensitivity_risk_target_pass: bool | None
    five_source_sensitivity_deployment_eligible: bool
    five_source_sensitivity_approval_blocker: str
    alpha: float = ALPHA
    seed: int = SEED
    risk_target: float = RISK_TARGET
    confidence: float = 0.95
    calibration: str = "query_excluded_loo_empirical_tail_ge"
    calibration_limitation: str = CALIBRATION_LIMITATION
    query_excluded_source_fitting: bool = True
    deployment_eligible: bool = False
    contains_test_values: bool = False
    contains_test_labels: bool = False
    contains_evaluation_metrics: bool = False


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as exc:
        raise TrainRiskContractError(f"cannot read frozen source: {path}") from exc


def loo_tail_pvalues(scores: np.ndarray) -> np.ndarray:
    """Compute empirical upper-tail p-values with each query removed."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise TrainRiskContractError("leave-one-out scores must be one-dimensional")
    result = np.ones(values.shape[0], dtype=float)
    for index, query in enumerate(values):
        reference = np.delete(values, index)
        finite = reference[np.isfinite(reference)]
        if np.isfinite(query) and finite.size:
            result[index] = (1 + np.count_nonzero(finite >= query)) / (
                finite.size + 1
            )
    return result


def seed_intersection_flags(scores_by_seed: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(scores_by_seed, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1:
        raise TrainRiskContractError("seed score arrays must be non-empty and two-dimensional")
    return np.logical_and.reduce(
        [loo_tail_pvalues(seed_scores) <= alpha for seed_scores in values]
    )


def wilson_upper_bound(
    false_reviews: int, normal_runs: int, confidence: float = 0.95
) -> float:
    if (
        type(false_reviews) is not int
        or type(normal_runs) is not int
        or false_reviews < 0
        or normal_runs < 0
        or false_reviews > normal_runs
    ):
        raise TrainRiskContractError(
            "false reviews and normal runs must satisfy 0 <= false_reviews <= normal_runs"
        )
    if not 0.5 < confidence < 1.0:
        raise TrainRiskContractError("confidence must be between 0.5 and 1.0")
    if normal_runs == 0:
        return 1.0
    observed = false_reviews / normal_runs
    z = NormalDist().inv_cdf(confidence)
    denominator = 1.0 + z**2 / normal_runs
    center = observed + z**2 / (2.0 * normal_runs)
    spread = z * np.sqrt(
        observed * (1.0 - observed) / normal_runs
        + z**2 / (4.0 * normal_runs**2)
    )
    return float((center + spread) / denominator)


def validate_query_excluded_provenance(
    calibration_by_source: Mapping[str, str],
) -> None:
    if set(calibration_by_source) != set(CERTIFIED_SOURCE_GROUPS):
        raise TrainRiskContractError(
            "certified source provenance must contain exactly kernel, statistical, "
            "structural, and change_point"
        )
    for source in CERTIFIED_SOURCE_GROUPS:
        if calibration_by_source[source] != _EXPECTED_CALIBRATION[source]:
            raise TrainRiskContractError(
                f"query-excluded source provenance mismatch: {source}"
            )


def _validate_array_lengths(
    dataset: str, n_train: int, source_arrays: Mapping[str, np.ndarray]
) -> None:
    required = {
        "kernel_1024",
        "kernel_4096",
        "statistical",
        "structural",
        "change_point",
    }
    if not required <= set(source_arrays):
        missing = sorted(required - set(source_arrays))
        raise TrainRiskContractError(
            f"missing TRAIN source array: dataset={dataset} source={missing[0]}"
        )
    one_dimensional = ("statistical", "structural", "change_point")
    for source in one_dimensional:
        values = np.asarray(source_arrays[source])
        if values.ndim != 1 or values.shape[0] != n_train:
            raise TrainRiskContractError(
                f"TRAIN array length mismatch: dataset={dataset} source={source}"
            )
    for source in ("kernel_1024", "kernel_4096"):
        values = np.asarray(source_arrays[source])
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != n_train:
            raise TrainRiskContractError(
                f"TRAIN array length mismatch: dataset={dataset} source={source}"
            )
    if "reconstruction" in source_arrays:
        if n_train < 100:
            raise TrainRiskContractError(
                f"unexpected reconstruction TRAIN array: dataset={dataset}"
            )
        values = np.asarray(source_arrays["reconstruction"])
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != n_train:
            raise TrainRiskContractError(
                f"TRAIN array length mismatch: dataset={dataset} source=reconstruction"
            )


def _risk_values(false_reviews: int, normal_runs: int) -> tuple[float, float, bool]:
    rate = false_reviews / normal_runs if normal_runs else 0.0
    upper = wilson_upper_bound(false_reviews, normal_runs)
    return rate, upper, normal_runs > 0 and upper <= RISK_TARGET


def _policy_metrics(flags: Sequence[np.ndarray], normal_runs: int) -> PolicyRiskMetrics:
    support = np.stack(flags, axis=0).sum(axis=0)
    shadow = int(np.count_nonzero(support >= 3))
    priority = int(np.count_nonzero(support == 2))
    combined = shadow + priority
    shadow_rate, shadow_upper, shadow_below = _risk_values(shadow, normal_runs)
    priority_rate, priority_upper, priority_below = _risk_values(priority, normal_runs)
    combined_rate, combined_upper, combined_below = _risk_values(combined, normal_runs)
    return PolicyRiskMetrics(
        recovery_shadow_false_reviews=shadow,
        recovery_priority_false_reviews=priority,
        combined_false_reviews=combined,
        recovery_shadow_false_review_rate=shadow_rate,
        recovery_priority_false_review_rate=priority_rate,
        combined_false_review_rate=combined_rate,
        recovery_shadow_wilson_upper_95=shadow_upper,
        recovery_priority_wilson_upper_95=priority_upper,
        combined_wilson_upper_95=combined_upper,
        recovery_shadow_below_risk_target=shadow_below,
        recovery_priority_below_risk_target=priority_below,
        combined_below_risk_target=combined_below,
    )


def _abstained_policy_metrics() -> PolicyRiskMetrics:
    return PolicyRiskMetrics(
        recovery_shadow_false_reviews=0,
        recovery_priority_false_reviews=0,
        combined_false_reviews=0,
        recovery_shadow_false_review_rate=0.0,
        recovery_priority_false_review_rate=0.0,
        combined_false_review_rate=0.0,
        recovery_shadow_wilson_upper_95=1.0,
        recovery_priority_wilson_upper_95=1.0,
        combined_wilson_upper_95=1.0,
        recovery_shadow_below_risk_target=False,
        recovery_priority_below_risk_target=False,
        combined_below_risk_target=False,
    )


def compute_train_risk_metrics(
    dataset: str, *, n_train: int, source_arrays: Mapping[str, np.ndarray]
) -> DatasetTrainRiskMetrics:
    """Compute TRAIN-risk metrics without making any provenance claim."""

    if type(dataset) is not str or not dataset:
        raise TrainRiskContractError("dataset name must be a non-empty string")
    if type(n_train) is not int or n_train < 0:
        raise TrainRiskContractError("n_train must be a non-negative integer")
    if n_train < MINIMUM_TRAIN:
        return DatasetTrainRiskMetrics(
            dataset=dataset,
            n_train=n_train,
            abstained=True,
            abstention_reason="n_train_below_5",
            normal_runs=0,
            reconstruction_available=False,
            sensitivity_available_source_count=0,
            certified=_abstained_policy_metrics(),
            sensitivity=None,
        )

    _validate_array_lengths(dataset, n_train, source_arrays)
    kernel = seed_intersection_flags(source_arrays["kernel_1024"], ALPHA) | (
        seed_intersection_flags(source_arrays["kernel_4096"], ALPHA)
    )
    certified_flags = [
        kernel,
        loo_tail_pvalues(source_arrays["statistical"]) <= ALPHA,
        loo_tail_pvalues(source_arrays["structural"]) <= ALPHA,
        loo_tail_pvalues(source_arrays["change_point"]) <= ALPHA,
    ]
    reconstruction_available = n_train >= 100 and "reconstruction" in source_arrays
    sensitivity_flags = list(certified_flags)
    if reconstruction_available:
        sensitivity_flags.append(
            seed_intersection_flags(source_arrays["reconstruction"], ALPHA)
        )
    return DatasetTrainRiskMetrics(
        dataset=dataset,
        n_train=n_train,
        abstained=False,
        abstention_reason="",
        normal_runs=n_train,
        reconstruction_available=reconstruction_available,
        sensitivity_available_source_count=len(sensitivity_flags),
        certified=_policy_metrics(certified_flags, n_train),
        sensitivity=_policy_metrics(sensitivity_flags, n_train),
    )


def audit_dataset_train_risk(
    dataset: str,
    *,
    n_train: int,
    source_arrays: Mapping[str, np.ndarray],
    _validated_provenance: object | None = None,
) -> TrainRiskRow:
    if _validated_provenance is not None and not _is_validated_provenance(
        _validated_provenance
    ):
        raise TrainRiskContractError("invalid validated provenance token")
    metrics = compute_train_risk_metrics(
        dataset, n_train=n_train, source_arrays=source_arrays
    )
    provenance_verified = _is_validated_provenance(_validated_provenance)
    policy_risk_validated = provenance_verified and not metrics.abstained
    certified = metrics.certified
    sensitivity = metrics.sensitivity
    return TrainRiskRow(
        dataset=metrics.dataset,
        n_train=metrics.n_train,
        abstained=metrics.abstained,
        abstention_reason=metrics.abstention_reason,
        normal_runs=metrics.normal_runs,
        source_count=len(CERTIFIED_SOURCE_GROUPS),
        certified_sources="|".join(CERTIFIED_SOURCE_GROUPS),
        provenance_verified=provenance_verified,
        policy_risk_validated=policy_risk_validated,
        risk_target_pass=(
            policy_risk_validated and certified.combined_below_risk_target
        ),
        reconstruction_available_but_provenance_unverified=(
            metrics.reconstruction_available
        ),
        reconstruction_provenance_verified=False,
        reconstruction_provenance_blocker=(
            RECONSTRUCTION_PROVENANCE_BLOCKER
            if metrics.reconstruction_available
            else ""
        ),
        recovery_shadow_false_reviews=certified.recovery_shadow_false_reviews,
        recovery_priority_false_reviews=certified.recovery_priority_false_reviews,
        combined_false_reviews=certified.combined_false_reviews,
        recovery_shadow_false_review_rate=certified.recovery_shadow_false_review_rate,
        recovery_priority_false_review_rate=certified.recovery_priority_false_review_rate,
        combined_false_review_rate=certified.combined_false_review_rate,
        recovery_shadow_wilson_upper_95=certified.recovery_shadow_wilson_upper_95,
        recovery_priority_wilson_upper_95=certified.recovery_priority_wilson_upper_95,
        combined_wilson_upper_95=certified.combined_wilson_upper_95,
        recovery_shadow_meets_risk_target=(
            policy_risk_validated and certified.recovery_shadow_below_risk_target
        ),
        recovery_priority_meets_risk_target=(
            policy_risk_validated and certified.recovery_priority_below_risk_target
        ),
        combined_meets_risk_target=(
            policy_risk_validated and certified.combined_below_risk_target
        ),
        five_source_sensitivity_calculated=sensitivity is not None,
        five_source_sensitivity_configured_source_count=(
            5 if sensitivity is not None else None
        ),
        five_source_sensitivity_available_source_count=(
            metrics.sensitivity_available_source_count
            if sensitivity is not None
            else None
        ),
        five_source_sensitivity_recovery_shadow_false_reviews=(
            sensitivity.recovery_shadow_false_reviews if sensitivity else None
        ),
        five_source_sensitivity_recovery_priority_false_reviews=(
            sensitivity.recovery_priority_false_reviews if sensitivity else None
        ),
        five_source_sensitivity_combined_false_reviews=(
            sensitivity.combined_false_reviews if sensitivity else None
        ),
        five_source_sensitivity_recovery_shadow_false_review_rate=(
            sensitivity.recovery_shadow_false_review_rate if sensitivity else None
        ),
        five_source_sensitivity_recovery_priority_false_review_rate=(
            sensitivity.recovery_priority_false_review_rate if sensitivity else None
        ),
        five_source_sensitivity_combined_false_review_rate=(
            sensitivity.combined_false_review_rate if sensitivity else None
        ),
        five_source_sensitivity_recovery_shadow_wilson_upper_95=(
            sensitivity.recovery_shadow_wilson_upper_95 if sensitivity else None
        ),
        five_source_sensitivity_recovery_priority_wilson_upper_95=(
            sensitivity.recovery_priority_wilson_upper_95 if sensitivity else None
        ),
        five_source_sensitivity_combined_wilson_upper_95=(
            sensitivity.combined_wilson_upper_95 if sensitivity else None
        ),
        five_source_sensitivity_provenance_verified=False,
        five_source_sensitivity_policy_risk_validated=False,
        five_source_sensitivity_risk_target_pass=None,
        five_source_sensitivity_deployment_eligible=False,
        five_source_sensitivity_approval_blocker=RECONSTRUCTION_PROVENANCE_BLOCKER,
        query_excluded_source_fitting=provenance_verified,
    )


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainRiskContractError(f"cannot read {description}: {path}") from exc
    if type(value) is not dict:
        raise TrainRiskContractError(f"{description} is not an object: {path}")
    return value


def _load_frozen_manifests() -> tuple[
    tuple[str, ...],
    dict[str, int],
    dict[str, dict[str, dict[str, str]]],
    _ValidatedProvenance,
]:
    manifests: dict[str, dict[str, dict[str, str]]] = {}
    calibration: dict[str, str] = {}
    for name, experiment, source, cache_root, config_hash, mode in CACHE_CONTRACTS:
        root = Path(cache_root)
        audit_path = root.parent / "audit.json"
        manifest_path = root.parent / "cache_manifest.csv"
        if _sha256_file(audit_path) != FROZEN_AUDIT_HASHES[name]:
            raise TrainRiskContractError(f"frozen audit SHA256 mismatch: {name}")
        if _sha256_file(manifest_path) != FROZEN_CACHE_MANIFEST_HASHES[name]:
            raise TrainRiskContractError(f"frozen cache manifest SHA256 mismatch: {name}")
        audit = _read_json(audit_path, f"{name} source audit")
        config = audit.get("config")
        if type(config) is not dict or any(
            config.get(field) != value
            for field, value in FROZEN_AUDIT_CONFIGS[name].items()
        ):
            raise TrainRiskContractError(f"frozen source config mismatch: {name}")
        calibration[name] = str(config.get("train_calibration"))
        try:
            with manifest_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except OSError as exc:
            raise TrainRiskContractError(f"cannot read cache manifest: {name}") from exc
        by_dataset = {row.get("dataset", ""): row for row in rows}
        if len(rows) != EXPECTED_DATASETS or len(by_dataset) != EXPECTED_DATASETS:
            raise TrainRiskContractError(f"cache population mismatch: {name}")
        for dataset, row in by_dataset.items():
            cache_name = hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:20]
            if (
                not dataset
                or row.get("experiment") != str(experiment)
                or row.get("source") != source
                or row.get("config_hash") != config_hash
                or row.get("contains_test_labels") != "False"
                or row.get("cache_file") != f"cache/{cache_name}.npz"
            ):
                raise TrainRiskContractError(
                    f"cache manifest contract mismatch: {name} dataset={dataset}"
                )
            try:
                n_train = int(row["n_train"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TrainRiskContractError(
                    f"invalid n_train in cache manifest: {name} dataset={dataset}"
                ) from exc
            if n_train < 0:
                raise TrainRiskContractError(
                    f"invalid n_train in cache manifest: {name} dataset={dataset}"
                )
        manifests[name] = by_dataset
    validate_query_excluded_provenance(calibration)
    population = tuple(sorted(manifests["kernel"]))
    if any(set(rows) != set(population) for rows in manifests.values()):
        raise TrainRiskContractError("TRAIN cache population mismatch across sources")
    n_train_by_dataset = {
        dataset: int(manifests["kernel"][dataset]["n_train"])
        for dataset in population
    }
    for dataset, expected in n_train_by_dataset.items():
        if any(int(rows[dataset]["n_train"]) != expected for rows in manifests.values()):
            raise TrainRiskContractError(
                f"TRAIN cache length mismatch across manifests: dataset={dataset}"
            )
    return (
        population,
        n_train_by_dataset,
        manifests,
        _ValidatedProvenance(_PROVENANCE_TOKEN_GUARD),
    )


def _load_reconstruction_inventory(
    population: Sequence[str], n_train_by_dataset: Mapping[str, int]
) -> dict[str, str]:
    audit_path = RECON_CACHE_ROOT.parent / "audit.json"
    if _sha256_file(audit_path) != FROZEN_AUDIT_HASHES["reconstruction"]:
        raise TrainRiskContractError("frozen audit SHA256 mismatch: reconstruction")
    audit = _read_json(audit_path, "reconstruction source audit")
    if (
        audit.get("contains_test_labels") is not False
        or audit.get("population_rule") != "train_normal_count >= 100"
    ):
        raise TrainRiskContractError("reconstruction source provenance mismatch")
    if (
        _sha256_file(RECON_ARCHIVE_HASH_INVENTORY)
        != FROZEN_RECON_ARCHIVE_HASH_INVENTORY_SHA256
    ):
        raise TrainRiskContractError("frozen reconstruction inventory SHA256 mismatch")
    inventory = _read_json(RECON_ARCHIVE_HASH_INVENTORY, "reconstruction inventory")
    archives = inventory.get("archives")
    expected = {
        f"{hashlib.sha256(dataset.encode('utf-8')).hexdigest()[:20]}.npz"
        for dataset in population
        if n_train_by_dataset[dataset] >= 100
    }
    if (
        type(archives) is not dict
        or set(archives) != expected
        or inventory.get("cache_root") != str(RECON_CACHE_ROOT)
        or inventory.get("source_audit_sha256")
        != FROZEN_AUDIT_HASHES["reconstruction"]
    ):
        raise TrainRiskContractError("frozen reconstruction inventory contract mismatch")
    return {str(name): str(digest) for name, digest in archives.items()}


def _read_train_key(
    archive: Mapping[str, np.ndarray], key: str, dataset: str, source: str
) -> np.ndarray:
    try:
        return np.asarray(archive[key], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainRiskContractError(
            f"missing or invalid TRAIN cache key: dataset={dataset} source={source} key={key}"
        ) from exc


def _load_dataset_train_arrays(
    dataset: str,
    n_train: int,
    manifests: Mapping[str, Mapping[str, Mapping[str, str]]],
    reconstruction_inventory: Mapping[str, str],
) -> dict[str, np.ndarray]:
    cache_name = hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:20]
    roots = {str(contract[0]): Path(contract[3]) for contract in CACHE_CONTRACTS}
    arrays: dict[str, np.ndarray] = {}
    for source, output_keys in {
        "kernel": ("kernel_1024", "kernel_4096"),
        "statistical": ("statistical",),
        "structural": ("structural",),
        "change_point": ("change_point",),
    }.items():
        path = roots[source] / f"{cache_name}.npz"
        if _sha256_file(path) != manifests[source][dataset].get("cache_sha256"):
            raise TrainRiskContractError(
                f"TRAIN cache SHA256 mismatch: dataset={dataset} source={source}"
            )
        try:
            with np.load(path, allow_pickle=False) as archive:
                for output_key in output_keys:
                    arrays[output_key] = _read_train_key(
                        archive, _TRAIN_KEYS[output_key], dataset, source
                    )
        except TrainRiskContractError:
            raise
        except (OSError, ValueError) as exc:
            raise TrainRiskContractError(
                f"cannot read TRAIN cache: dataset={dataset} source={source}"
            ) from exc
    if n_train >= 100:
        filename = f"{cache_name}.npz"
        path = RECON_CACHE_ROOT / filename
        if _sha256_file(path) != reconstruction_inventory.get(filename):
            raise TrainRiskContractError(
                f"TRAIN cache SHA256 mismatch: dataset={dataset} source=reconstruction"
            )
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays["reconstruction"] = _read_train_key(
                    archive,
                    _TRAIN_KEYS["reconstruction"],
                    dataset,
                    "reconstruction",
                )
        except TrainRiskContractError:
            raise
        except (OSError, ValueError) as exc:
            raise TrainRiskContractError(
                f"cannot read TRAIN cache: dataset={dataset} source=reconstruction"
            ) from exc
    return arrays


def write_train_risk_audit(
    rows: Sequence[TrainRiskRow],
    output_root: Path = OUTPUT_ROOT,
    *,
    _validated_provenance: object | None = None,
) -> Path:
    if not _is_validated_provenance(_validated_provenance):
        raise TrainRiskContractError("invalid validated provenance token")
    if (
        len(rows) != EXPECTED_DATASETS
        or len({row.dataset for row in rows}) != EXPECTED_DATASETS
    ):
        raise TrainRiskContractError("TRAIN-risk audit must contain 1,117 unique datasets")
    for row in rows:
        if (
            row.source_count != len(CERTIFIED_SOURCE_GROUPS)
            or row.certified_sources != "|".join(CERTIFIED_SOURCE_GROUPS)
            or not row.provenance_verified
            or not row.query_excluded_source_fitting
            or row.policy_risk_validated != (not row.abstained)
            or row.five_source_sensitivity_provenance_verified
            or row.five_source_sensitivity_policy_risk_validated
            or row.five_source_sensitivity_risk_target_pass is not None
            or row.five_source_sensitivity_deployment_eligible
        ):
            raise TrainRiskContractError(
                f"uncertified or invalid TRAIN-risk row: dataset={row.dataset}"
            )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "02_train_normal_risk_audit.csv"
    temporary = path.with_suffix(".csv.tmp")
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(TrainRiskRow.__annotations__)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(rows, key=lambda item: item.dataset):
                writer.writerow(asdict(row))
        temporary.replace(path)
    except OSError as exc:
        raise TrainRiskContractError(f"cannot write TRAIN-risk audit: {path}") from exc
    return path


def audit_train_risk(
    *,
    output_root: Path = OUTPUT_ROOT,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[TrainRiskRow, ...]:
    """Audit Exp266's review policy using frozen TRAIN OOF cache arrays only."""

    (
        population,
        n_train_by_dataset,
        manifests,
        validated_provenance,
    ) = _load_frozen_manifests()
    reconstruction_inventory = _load_reconstruction_inventory(
        population, n_train_by_dataset
    )
    rows: list[TrainRiskRow] = []
    for dataset in population:
        n_train = n_train_by_dataset[dataset]
        arrays = (
            {}
            if n_train < MINIMUM_TRAIN
            else _load_dataset_train_arrays(
                dataset, n_train, manifests, reconstruction_inventory
            )
        )
        rows.append(
            audit_dataset_train_risk(
                dataset,
                n_train=n_train,
                source_arrays=arrays,
                _validated_provenance=validated_provenance,
            )
        )
        if progress_callback is not None:
            progress_callback(dataset, len(rows), len(population))
    result = tuple(rows)
    if len(result) != EXPECTED_DATASETS or len({row.dataset for row in result}) != EXPECTED_DATASETS:
        raise TrainRiskContractError("TRAIN-risk audit must contain 1,117 unique datasets")
    write_train_risk_audit(
        result,
        output_root,
        _validated_provenance=validated_provenance,
    )
    return result
