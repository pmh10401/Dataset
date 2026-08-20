"""Label-gated AP/AUROC evaluation for the four frozen Exp450 score roots.

This module deliberately has no module-level TEST-label import.  The frozen
score inventories and all public-baseline identities are checked before the
single evaluation-only label capability is resolved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from experiments.paper_public_baselines.calibration import upper_tail_p_values
from experiments.paper_public_baselines.runner import verify_score_root


CLASSICAL_ROOT = Path("/Users/mac/Documents/Dataset/paper_public_baselines_classical_v1")
DEEP_SVDD_ROOT = Path("/Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_1d_derived_full_antigravity_20260817")
DROCC_ROOT = Path("/Users/mac/Documents/Dataset/paper_public_baselines_drocc_mps_unverified_antigravity_20260817")
METHOD_ORDER = (
    "A", "B", "C", "D", "knn_distance", "lof_novelty", "ocsvm_rbf",
    "isolation_forest", "deep_svdd_1d", "drocc_1d",
)
_CONDITIONS = METHOD_ORDER[:4]
_BASELINE_ROOT_ORDER = ("classical", "deep_svdd", "drocc")
_BASELINE_METHODS = {
    "classical": ("knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest"),
    "deep_svdd": ("deep_svdd_1d",),
    "drocc": ("drocc_1d",),
}
BASELINE_ROOTS = MappingProxyType({
    "classical": CLASSICAL_ROOT,
    "deep_svdd": DEEP_SVDD_ROOT,
    "drocc": DROCC_ROOT,
})
_BOOTSTRAP_SEED = 45020260820
_BOOTSTRAP_SAMPLES = 20_000
_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class MetricTablesExp450Antigravity:
    method_order: tuple[str, ...]
    per_dataset: tuple[Mapping[str, object], ...]
    macro: Mapping[str, Mapping[str, object]]
    micro: Mapping[str, Mapping[str, object]]
    paired_statistics: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class Exp450EvaluationAntigravity:
    manifests: Mapping[str, object]
    metric_tables: MetricTablesExp450Antigravity
    curve_tables: Mapping[str, Mapping[str, object]]


def evaluate_exp450_antigravity(
    root: Path,
    *,
    baseline_roots: Mapping[str, object] | None = None,
    verifier: Callable[[str], Mapping[str, object]] | None = None,
    label_loader: Callable[[tuple[str, ...]], Mapping[str, Sequence[object]]] | None = None,
    baseline_verifier: Callable[[Path], object] = verify_score_root,
) -> Exp450EvaluationAntigravity:
    """Verify every score source and identity before the one TEST-label read.

    ``verifier`` is intentionally injected: Task 6's pure verifier needs the
    already-established run context and catalog.  Task 6 returns a complete,
    self-bound authority, so no caller-provided identity is accepted here.
    """
    if verifier is None:
        raise ValueError("Exp450 manifest verifier is required")
    manifests: dict[str, Mapping[str, object]] = {}
    for condition in _CONDITIONS:
        manifest = verifier(condition)
        if not isinstance(manifest, Mapping):
            raise ValueError("Exp450 manifest verifier returned an invalid manifest")
        manifests[condition] = manifest

    datasets, test_identities, database_sha256 = _exp450_identity(manifests)
    active_baseline_roots = BASELINE_ROOTS if baseline_roots is None else baseline_roots
    baseline_manifests = _verify_baseline_roots(active_baseline_roots, baseline_verifier)
    baseline_scores, baseline_micro_scores = _load_baseline_scores(
        baseline_manifests, datasets, test_identities, database_sha256
    )
    exp_scores = _load_exp450_scores(Path(root), manifests, datasets, test_identities)

    # This is deliberately the first point at which label capability can exist.
    if label_loader is None:
        from experiments.paper_public_baselines.label_access import load_test_labels
        active_label_loader = load_test_labels
    else:
        active_label_loader = label_loader
    labels = _normalize_labels(active_label_loader(datasets), datasets)

    all_scores = {**exp_scores, **baseline_scores}
    all_micro_scores = {**exp_scores, **baseline_micro_scores}
    metrics = metric_tables_exp450_antigravity(labels, all_scores, micro_scores=all_micro_scores)
    curves = curve_tables_exp450_antigravity(labels, all_micro_scores)
    return Exp450EvaluationAntigravity(
        manifests=MappingProxyType(dict(manifests)), metric_tables=metrics,
        curve_tables=MappingProxyType(dict(curves)),
    )


def metric_tables_exp450_antigravity(
    labels: Mapping[str, Sequence[object]],
    scores: Mapping[str, Mapping[str, Sequence[object]]],
    *,
    micro_scores: Mapping[str, Mapping[str, Sequence[object]]] | None = None,
) -> MetricTablesExp450Antigravity:
    """Build denominator-audited per-dataset, macro, micro, and paired tables."""
    datasets = _dataset_order(labels)
    normalized_labels = _normalize_labels(labels, datasets)
    normalized_scores = _normalize_score_maps(scores, datasets)
    normalized_micro = (
        normalized_scores if micro_scores is None
        else _normalize_score_maps(micro_scores, datasets)
    )
    rows: list[Mapping[str, object]] = []
    macro: dict[str, Mapping[str, object]] = {}
    micro: dict[str, Mapping[str, object]] = {}
    values: dict[tuple[str, str, str], float] = {}
    for method in METHOD_ORDER:
        ap_values: list[float] = []
        auroc_values: list[float] = []
        joined_labels: list[np.ndarray] = []
        joined_scores: list[np.ndarray] = []
        for dataset in datasets:
            label_values = np.asarray(normalized_labels[dataset], dtype=np.int8)
            score_values = normalized_scores[method].get(dataset)
            micro_values = normalized_micro[method].get(dataset)
            row: dict[str, object] = {"method": method, "dataset": dataset}
            if score_values is None or micro_values is None:
                row.update({
                    "average_precision": None, "auroc": None,
                    "average_precision_exclusion": "score_unavailable",
                    "auroc_exclusion": "score_unavailable",
                })
            else:
                if score_values.shape != label_values.shape or micro_values.shape != label_values.shape:
                    raise ValueError("score TEST identity differs from labels")
                joined_labels.append(label_values)
                joined_scores.append(micro_values)
                if int(np.count_nonzero(label_values == 1)) == 0:
                    row["average_precision"] = None
                    row["average_precision_exclusion"] = "no_positive_test_labels"
                else:
                    value = float(average_precision_score(label_values, score_values))
                    row["average_precision"] = value
                    row["average_precision_exclusion"] = None
                    ap_values.append(value)
                    values[(method, dataset, "average_precision")] = value
                if len(np.unique(label_values)) != 2:
                    row["auroc"] = None
                    row["auroc_exclusion"] = "single_class_test_labels"
                else:
                    value = float(roc_auc_score(label_values, score_values))
                    row["auroc"] = value
                    row["auroc_exclusion"] = None
                    auroc_values.append(value)
                    values[(method, dataset, "auroc")] = value
            rows.append(MappingProxyType(row))
        macro[method] = MappingProxyType({
            "average_precision": float(np.mean(ap_values)) if ap_values else None,
            "auroc": float(np.mean(auroc_values)) if auroc_values else None,
            "average_precision_denominator": len(ap_values),
            "average_precision_excluded": len(datasets) - len(ap_values),
            "auroc_denominator": len(auroc_values),
            "auroc_excluded": len(datasets) - len(auroc_values),
        })
        micro[method] = MappingProxyType(_micro_summary(joined_labels, joined_scores))
    paired = _paired_statistics(datasets, values)
    return MetricTablesExp450Antigravity(
        METHOD_ORDER, tuple(rows), MappingProxyType(macro), MappingProxyType(micro), paired
    )


def curve_tables_exp450_antigravity(
    labels: Mapping[str, Sequence[object]],
    scores: Mapping[str, Mapping[str, Sequence[object]]],
) -> Mapping[str, Mapping[str, object]]:
    """Return canonical pooled ROC and PR coordinates with sklearn pairings intact."""
    datasets = _dataset_order(labels)
    normalized_labels = _normalize_labels(labels, datasets)
    normalized_scores = _normalize_score_maps(scores, datasets)
    roc_rows: list[Mapping[str, object]] = []
    pr_rows: list[Mapping[str, object]] = []
    for method in METHOD_ORDER:
        present = tuple(dataset for dataset in datasets if dataset in normalized_scores[method])
        if not present:
            continue
        joined_labels = np.concatenate([np.asarray(normalized_labels[dataset], dtype=np.int8) for dataset in present])
        joined_scores = np.concatenate([normalized_scores[method][dataset] for dataset in present])
        if len(np.unique(joined_labels)) != 2:
            continue
        roc_fpr, roc_tpr, roc_threshold = roc_curve(joined_labels, joined_scores)
        pr_precision, pr_recall, _pr_threshold = precision_recall_curve(joined_labels, joined_scores)
        roc_rows.extend(
            MappingProxyType({"method": method, "false_positive_rate": float(x), "true_positive_rate": float(y), "threshold": float(t)})
            for x, y, t in zip(roc_fpr, roc_tpr, roc_threshold, strict=True)
        )
        # sklearn returns precision/recall in paired coordinate order; retain it.
        pr_rows.extend(
            MappingProxyType({"method": method, "recall": float(x), "precision": float(y)})
            for x, y in zip(pr_recall, pr_precision, strict=True)
        )
    return MappingProxyType({
        "roc": MappingProxyType({"x_name": "false_positive_rate", "y_name": "true_positive_rate", "rows": tuple(roc_rows)}),
        "precision_recall": MappingProxyType({"x_name": "recall", "y_name": "precision", "rows": tuple(pr_rows)}),
    })


def _exp450_identity(
    manifests: Mapping[str, Mapping[str, object]],
) -> tuple[tuple[str, ...], Mapping[str, Mapping[str, object]], str]:
    reference_datasets: tuple[str, ...] | None = None
    reference_identities: dict[str, Mapping[str, object]] | None = None
    database_sha256: str | None = None
    reference_context_sha256: str | None = None
    reference_catalog_sha256: str | None = None
    for condition in _CONDITIONS:
        manifest = manifests[condition]
        condition_record = _condition_record(condition, manifest.get("condition"))
        identities = _manifest_test_identities(manifest)
        datasets = tuple(identities)
        db = manifest.get("database_sha256")
        context_sha256 = manifest.get("context_sha256")
        catalog_sha256 = manifest.get("catalog_sha256")
        if (
            not _sha256(db)
            or not _sha256(context_sha256)
            or not _sha256(catalog_sha256)
        ):
            raise ValueError("Exp450 manifest database identity differs")
        if reference_datasets is None:
            reference_datasets, reference_identities = datasets, identities
            database_sha256, reference_context_sha256, reference_catalog_sha256 = db, context_sha256, catalog_sha256
        elif datasets != reference_datasets or identities != reference_identities:
            raise ValueError("Exp450 manifest TEST identity differs")
        elif db != database_sha256 or context_sha256 != reference_context_sha256 or catalog_sha256 != reference_catalog_sha256:
            raise ValueError("Exp450 manifest database identity differs")
    assert reference_datasets is not None and reference_identities is not None and database_sha256 is not None
    return reference_datasets, MappingProxyType(reference_identities), database_sha256


def _verify_baseline_roots(
    roots: Mapping[str, object], baseline_verifier: Callable[[Path], object]
) -> Mapping[str, object]:
    if tuple(roots) != _BASELINE_ROOT_ORDER:
        raise ValueError("baseline roots must include classical, deep_svdd, and drocc in frozen order")
    verified: dict[str, object] = {}
    for name in _BASELINE_ROOT_ORDER:
        candidate = roots[name]
        if not isinstance(candidate, Path):
            raise ValueError("baseline root path differs")
        verified[name] = baseline_verifier(candidate)
    return MappingProxyType(verified)


def _load_baseline_scores(
    manifests: Mapping[str, object], datasets: tuple[str, ...],
    test_identities: Mapping[str, Mapping[str, object]], database_sha256: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    raw_scores: dict[str, dict[str, np.ndarray]] = {}
    micro_scores: dict[str, dict[str, np.ndarray]] = {}
    for root_name in _BASELINE_ROOT_ORDER:
        manifest = manifests[root_name]
        identity, db, shards = _baseline_identity_and_shards(manifest)
        if db != database_sha256:
            raise ValueError("baseline database identity differs")
        if tuple(identity) != datasets:
            raise ValueError("baseline TEST identity differs")
        if isinstance(manifest, Mapping):
            if identity != test_identities:
                raise ValueError("baseline TEST identity differs")
        # FrozenScoreManifest has only verified TEST coordinates.  Equal pinned
        # DB SHA plus exact dataset/count/index order is the authority that its
        # raw values are the same as the Exp450 full row-hash identity.
        elif not _same_test_coordinates(identity, test_identities):
            raise ValueError("baseline TEST identity differs")
        for method in _BASELINE_METHODS[root_name]:
            raw_scores[method] = {}
            micro_scores[method] = {}
            for dataset in datasets:
                shard = shards.get((method, dataset))
                if shard is None:
                    raise ValueError("baseline score coverage differs")
                if shard.get("status") != "complete":
                    continue
                test = _finite_vector(shard.get("test_scores"), "baseline TEST scores")
                train = _finite_vector(shard.get("train_oof_scores"), "baseline TRAIN OOF scores")
                expected = test_identities[dataset]["count"]
                indices = shard.get("test_indices")
                if test.size != expected or indices != list(range(expected)):
                    raise ValueError("baseline TEST identity differs")
                raw_scores[method][dataset] = test
                micro_scores[method][dataset] = 1.0 - upper_tail_p_values(train, test)
    return raw_scores, micro_scores


def _baseline_identity_and_shards(
    manifest: object,
) -> tuple[Mapping[str, Mapping[str, object]], str, Mapping[tuple[str, str], Mapping[str, object]]]:
    if isinstance(manifest, Mapping):
        db = manifest.get("database_sha256")
        identities = _manifest_test_identities(manifest)
        methods = manifest.get("methods")
        if not isinstance(methods, Mapping):
            raise ValueError("baseline score manifest differs")
        shards = {
            (method, dataset): shard
            for method, rows in methods.items() if isinstance(method, str) and isinstance(rows, Mapping)
            for dataset, shard in rows.items() if isinstance(dataset, str) and isinstance(shard, Mapping)
        }
    else:
        root = getattr(manifest, "run_root", None)
        db = getattr(manifest, "database_sha256", None)
        entries = getattr(manifest, "shards", None)
        if not isinstance(root, Path) or not isinstance(entries, tuple):
            raise ValueError("baseline score manifest differs")
        run_config = _strict_json(root / "run_config.json")
        datasets = run_config.get("datasets")
        if not isinstance(datasets, list):
            raise ValueError("baseline score manifest differs")
        loaded: dict[tuple[str, str], Mapping[str, object]] = {}
        identities: dict[str, Mapping[str, object]] = {}
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("baseline score manifest differs")
            shard = _strict_json(root / str(entry.get("path")))
            method, dataset = shard.get("method"), shard.get("dataset")
            if not isinstance(method, str) or not isinstance(dataset, str):
                raise ValueError("baseline score manifest differs")
            count = shard.get("n_test")
            indices = shard.get("test_indices")
            identities[dataset] = _test_coordinate(dataset, {"split": "TEST", "count": count, "indices": indices})
            loaded[(method, dataset)] = shard
        if tuple(identities) != tuple(datasets):
            raise ValueError("baseline dataset identity differs")
        shards = loaded
    if type(db) is not str or len(db) != 64:
        raise ValueError("baseline database identity differs")
    return identities, db, shards


def _load_exp450_scores(
    root: Path, manifests: Mapping[str, Mapping[str, object]], datasets: tuple[str, ...],
    test_identities: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, np.ndarray]]:
    all_scores: dict[str, dict[str, np.ndarray]] = {}
    for condition in _CONDITIONS:
        manifest = manifests[condition]
        direct = manifest.get("score_rows")
        method_scores: dict[str, np.ndarray] = {}
        if isinstance(direct, Mapping):
            for dataset in datasets:
                method_scores[dataset] = _finite_vector(direct.get(dataset), "Exp450 TEST scores")
        else:
            descriptors = manifest.get("shards")
            if not isinstance(descriptors, list):
                raise ValueError("Exp450 manifest score coverage differs")
            descriptor_by_dataset = {item.get("dataset"): item for item in descriptors if isinstance(item, Mapping)}
            for dataset in datasets:
                descriptor = descriptor_by_dataset.get(dataset)
                if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("file"), str):
                    raise ValueError("Exp450 manifest score coverage differs")
                shard = _strict_json(root / str(descriptor["file"]))
                if _normal_test_identity(dataset, shard.get("test_identity")) != test_identities[dataset]:
                    raise ValueError("Exp450 score TEST identity differs")
                policies = shard.get("policies")
                selected = next((row for row in policies if isinstance(row, Mapping) and row.get("policy_id") == "hard_plus_stable"), None) if isinstance(policies, list) else None
                if not isinstance(selected, Mapping):
                    raise ValueError("Exp450 selected policy differs")
                method_scores[dataset] = _finite_vector(selected.get("scores"), "Exp450 TEST scores")
        for dataset, values in method_scores.items():
            if values.size != test_identities[dataset]["count"]:
                raise ValueError("Exp450 score TEST identity differs")
        all_scores[condition] = method_scores
    return all_scores


def _paired_statistics(
    datasets: tuple[str, ...], values: Mapping[tuple[str, str, str], float]
) -> tuple[Mapping[str, object], ...]:
    records: list[Mapping[str, object]] = []
    for metric in ("average_precision", "auroc"):
        common = tuple(
            dataset for dataset in datasets
            if all((method, dataset, metric) in values for method in _CONDITIONS)
        )
        if not common:
            for variant in ("B", "C", "D"):
                records.append(MappingProxyType({"comparison": f"{variant}-A", "metric": metric, "dataset_order": (), "point_estimate": None, "lower": None, "upper": None, "wins": 0, "ties": 0, "losses": 0, "seed": _BOOTSTRAP_SEED, "draw_count": 0, "draw_sha256": None}))
            continue
        draws = np.random.default_rng(_BOOTSTRAP_SEED).integers(0, len(common), size=(_BOOTSTRAP_SAMPLES, len(common)), dtype=np.int64)
        digest = hashlib.sha256(draws.tobytes(order="C")).hexdigest()
        reference = np.asarray([values[("A", dataset, metric)] for dataset in common], dtype=np.float64)
        for variant in ("B", "C", "D"):
            difference = np.asarray([values[(variant, dataset, metric)] for dataset in common], dtype=np.float64) - reference
            estimates = difference[draws].mean(axis=1)
            lower, upper = np.quantile(estimates, (0.025, 0.975))
            records.append(MappingProxyType({
                "comparison": f"{variant}-A", "metric": metric, "dataset_order": common,
                "point_estimate": float(difference.mean()), "lower": float(lower), "upper": float(upper),
                "wins": int(np.count_nonzero(difference > _TIE_TOLERANCE)),
                "ties": int(np.count_nonzero(np.abs(difference) <= _TIE_TOLERANCE)),
                "losses": int(np.count_nonzero(difference < -_TIE_TOLERANCE)),
                "seed": _BOOTSTRAP_SEED, "draw_count": _BOOTSTRAP_SAMPLES, "draw_sha256": digest,
            }))
    return tuple(records)


def _micro_summary(labels: list[np.ndarray], scores: list[np.ndarray]) -> dict[str, object]:
    if not labels:
        return {"average_precision": None, "auroc": None, "row_denominator": 0, "average_precision_exclusion": "no_scored_test_rows", "auroc_exclusion": "no_scored_test_rows"}
    joined_labels = np.concatenate(labels)
    joined_scores = np.concatenate(scores)
    positives = int(np.count_nonzero(joined_labels == 1))
    both = len(np.unique(joined_labels)) == 2
    return {
        "average_precision": float(average_precision_score(joined_labels, joined_scores)) if positives else None,
        "auroc": float(roc_auc_score(joined_labels, joined_scores)) if both else None,
        "row_denominator": int(joined_labels.size),
        "average_precision_exclusion": None if positives else "no_positive_test_rows",
        "auroc_exclusion": None if both else "single_class_pooled_test_rows",
    }


def _dataset_order(labels: Mapping[str, Sequence[object]]) -> tuple[str, ...]:
    if not isinstance(labels, Mapping) or not labels or any(type(dataset) is not str or not dataset for dataset in labels):
        raise ValueError("label datasets are invalid")
    return tuple(labels)


def _normalize_labels(raw: Mapping[str, Sequence[object]], datasets: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    if not isinstance(raw, Mapping) or tuple(raw) != datasets:
        raise ValueError("label dataset order differs")
    normalized: dict[str, tuple[int, ...]] = {}
    for dataset in datasets:
        rows = tuple(raw[dataset])
        if rows and all(isinstance(row, tuple) and len(row) == 2 for row in rows):
            indices, values = zip(*rows, strict=True)
            if indices != tuple(range(len(rows))):
                raise ValueError("TEST label identity differs")
        else:
            values = rows
        if not values or any(type(value) is not int or value not in (0, 1) for value in values):
            raise ValueError("TEST labels must be exact binary integers")
        normalized[dataset] = tuple(values)
    return normalized


def _normalize_score_maps(scores: Mapping[str, Mapping[str, Sequence[object]]], datasets: tuple[str, ...]) -> dict[str, dict[str, np.ndarray]]:
    if not isinstance(scores, Mapping) or tuple(scores) != METHOD_ORDER:
        raise ValueError("score method order differs")
    normalized: dict[str, dict[str, np.ndarray]] = {}
    for method in METHOD_ORDER:
        rows = scores[method]
        if not isinstance(rows, Mapping) or any(dataset not in datasets for dataset in rows):
            raise ValueError("score dataset identity differs")
        normalized[method] = {dataset: _finite_vector(values, "scores") for dataset, values in rows.items()}
    return normalized


def _manifest_test_identities(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    entries = manifest.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest dataset identity differs")
    result: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("dataset"), str):
            raise ValueError("manifest dataset identity differs")
        dataset = str(entry["dataset"])
        if dataset in result:
            raise ValueError("manifest dataset identity differs")
        result[dataset] = _normal_test_identity(dataset, entry.get("test"))
    aggregate = manifest.get("test_identities")
    if not isinstance(aggregate, list) or len(aggregate) != len(result):
        raise ValueError("manifest TEST identity differs")
    normalized_aggregate = [_normal_test_identity(dataset, value) for dataset, value in zip(result, aggregate, strict=True)]
    if normalized_aggregate != list(result.values()):
        raise ValueError("manifest TEST identity differs")
    return result


def _normal_test_identity(dataset: str, value: object) -> Mapping[str, object]:
    required = {"dataset", "dataset_id", "series_length", "split", "indices", "values_blob_sha256", "row_digest", "count"}
    if not isinstance(value, Mapping) or set(value) != required or value.get("dataset") != dataset or value.get("split") != "TEST":
        raise ValueError("TEST identity differs")
    count, indices = value.get("count"), value.get("indices")
    dataset_id, series_length = value.get("dataset_id"), value.get("series_length")
    hashes, digest = value.get("values_blob_sha256"), value.get("row_digest")
    if (
        type(count) is not int or count < 1 or indices != list(range(count))
        or type(dataset_id) is not int or dataset_id < 0
        or type(series_length) is not int or series_length < 1
        or not isinstance(hashes, list) or len(hashes) != count or any(not _sha256(item) for item in hashes)
        or not _sha256(digest)
    ):
        raise ValueError("TEST identity differs")
    bare = {key: value[key] for key in required - {"row_digest", "count"}}
    expected_digest = hashlib.sha256(_canonical_bytes(bare)).hexdigest()
    if digest != expected_digest:
        raise ValueError("TEST identity differs")
    return MappingProxyType({key: value[key] if key != "indices" else list(indices) for key in required})


def _test_coordinate(dataset: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or value.get("split") != "TEST":
        raise ValueError("TEST identity differs")
    count, indices = value.get("count"), value.get("indices")
    if type(count) is not int or count < 1 or indices != list(range(count)):
        raise ValueError("TEST identity differs")
    return MappingProxyType({"dataset": dataset, "split": "TEST", "count": count, "indices": list(indices)})


def _same_test_coordinates(
    baseline: Mapping[str, Mapping[str, object]], exp450: Mapping[str, Mapping[str, object]]
) -> bool:
    return all(
        _test_coordinate(dataset, baseline[dataset]) == _test_coordinate(dataset, exp450[dataset])
        for dataset in exp450
    )


def _condition_record(condition: str, value: object) -> Mapping[str, str]:
    expected_modes = {
        "A": ("linear", "linear"), "B": ("polyphase_pchip", "linear"),
        "C": ("linear", "area"), "D": ("polyphase_pchip", "area"),
    }
    if not isinstance(value, Mapping) or set(value) != {"condition_id", "length_mode", "image_mode"}:
        raise ValueError("Exp450 manifest condition differs")
    length_mode, image_mode = expected_modes[condition]
    if value != {"condition_id": condition, "length_mode": length_mode, "image_mode": image_mode}:
        raise ValueError("Exp450 manifest condition differs")
    return MappingProxyType({"condition_id": condition, "length_mode": length_mode, "image_mode": image_mode})


def _sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("TEST identity differs") from error


def _finite_vector(values: object, name: str) -> np.ndarray:
    if (not isinstance(values, (Sequence, np.ndarray))) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a finite vector")
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite vector") from error
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _strict_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("score artifact is invalid") from error
    if not isinstance(value, Mapping):
        raise ValueError("score artifact is invalid")
    return value


__all__ = [
    "BASELINE_ROOTS", "CLASSICAL_ROOT", "DEEP_SVDD_ROOT", "DROCC_ROOT", "METHOD_ORDER",
    "Exp450EvaluationAntigravity", "MetricTablesExp450Antigravity",
    "curve_tables_exp450_antigravity", "evaluate_exp450_antigravity", "metric_tables_exp450_antigravity",
]
