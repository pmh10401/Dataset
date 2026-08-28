"""One-time, label-gated Exp472 E versus frozen Exp471 D evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .authority_antigravity import (
    Exp472ReuseAuthorityAntigravity,
    bind_e_to_exp471_exp472_antigravity,
    verify_exp471_reuse_exp472_antigravity,
)


METHOD_ORDER_EXP472_ANTIGRAVITY = (
    "A", "B", "C", "D", "E", "knn_distance", "lof_novelty", "ocsvm_rbf",
    "isolation_forest", "deep_svdd_1d", "drocc_1d",
)
_TIE_TOLERANCE_ANTIGRAVITY = 1e-12
_BOOTSTRAP_DRAWS_ANTIGRAVITY = 20_000
_BOOTSTRAP_SEED_ANTIGRAVITY = 47220260828


@dataclass(frozen=True, slots=True)
class Exp472EvaluationAntigravity:
    authority: Exp472ReuseAuthorityAntigravity
    method_order: tuple[str, ...]
    per_dataset: tuple[Mapping[str, object], ...]
    macro: Mapping[str, Mapping[str, object]]
    micro: Mapping[str, Mapping[str, object]]
    paired_statistics: tuple[Mapping[str, object], ...]
    denominators: Mapping[str, Mapping[str, object]]
    curves: Mapping[str, Mapping[str, object]]


def _finite_vector_antigravity(value: object, *, scope: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError(f"Exp472 {scope} differs")
    return values


def _labels_antigravity(value: object, datasets: tuple[str, ...]) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping) or tuple(value) != datasets:
        raise ValueError("Exp472 TEST label coverage differs")
    result: dict[str, np.ndarray] = {}
    for dataset in datasets:
        labels = np.asarray(value[dataset], dtype=np.int8)
        if labels.ndim != 1 or labels.size < 1 or not np.isin(labels, (0, 1)).all():
            raise ValueError("Exp472 TEST labels differ")
        result[dataset] = labels
    return MappingProxyType(result)


def _d_scores_antigravity(d_evidence: Mapping[str, object], datasets: tuple[str, ...]) -> Mapping[str, Mapping[str, np.ndarray | None]]:
    raw = d_evidence.get("scores")
    statuses = d_evidence.get("status", {})
    if not isinstance(raw, Mapping):
        raise ValueError("Exp472 Exp471 score evidence differs")
    result: dict[str, Mapping[str, np.ndarray | None]] = {}
    for method in METHOD_ORDER_EXP472_ANTIGRAVITY:
        if method == "E":
            continue
        by_dataset = raw.get(method)
        if not isinstance(by_dataset, Mapping):
            raise ValueError("Exp472 Exp471 score method coverage differs")
        result[method] = MappingProxyType({
            dataset: None if by_dataset.get(dataset) is None else _finite_vector_antigravity(by_dataset[dataset], scope="Exp471 TEST scores")
            for dataset in datasets
        })
    return MappingProxyType(result)


def _e_scores_antigravity(e_manifest: Mapping[str, object], datasets: tuple[str, ...]) -> tuple[Mapping[str, np.ndarray], Mapping[str, int]]:
    scores = e_manifest.get("scores")
    upsampled = e_manifest.get("upsampled_rows")
    if not isinstance(scores, Mapping) or not isinstance(upsampled, Mapping) or set(scores) != set(datasets) or set(upsampled) != set(datasets):
        raise ValueError("Exp472 E score coverage differs")
    output: dict[str, np.ndarray] = {}
    rows: dict[str, int] = {}
    for dataset in datasets:
        output[dataset] = _finite_vector_antigravity(scores[dataset], scope="E TEST scores")
        value = upsampled[dataset]
        if type(value) is not int or value < 0:
            raise ValueError("Exp472 E upsampled row count differs")
        rows[dataset] = value
    return MappingProxyType(output), MappingProxyType(rows)


def paired_e_minus_d_exp472_antigravity(e_values: Sequence[object], d_values: Sequence[object], *, seed: int = _BOOTSTRAP_SEED_ANTIGRAVITY) -> Mapping[str, object]:
    """Paired percentile bootstrap using bounded-memory batched resampling."""
    e = _finite_vector_antigravity(e_values, scope="paired E values")
    d = _finite_vector_antigravity(d_values, scope="paired D values")
    if e.shape != d.shape or type(seed) is not int:
        raise ValueError("Exp472 paired comparison differs")
    delta = e - d
    generator = np.random.default_rng(seed)
    # Avoid materializing 20,000 x 1,117 indices (~179 MiB) while preserving
    # the same deterministic RNG sequence and percentile result.
    means = np.empty(_BOOTSTRAP_DRAWS_ANTIGRAVITY, dtype=np.float64)
    batch = max(1, min(1024, _BOOTSTRAP_DRAWS_ANTIGRAVITY))
    for start in range(0, _BOOTSTRAP_DRAWS_ANTIGRAVITY, batch):
        stop = min(_BOOTSTRAP_DRAWS_ANTIGRAVITY, start + batch)
        draws = generator.integers(0, delta.size, size=(stop - start, delta.size), dtype=np.intp)
        means[start:stop] = delta[draws].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return MappingProxyType({
        "mean_delta": float(delta.mean()), "ci_lower": float(lower), "ci_upper": float(upper),
        "wins": int(np.count_nonzero(delta > _TIE_TOLERANCE_ANTIGRAVITY)),
        "ties": int(np.count_nonzero(np.abs(delta) <= _TIE_TOLERANCE_ANTIGRAVITY)),
        "losses": int(np.count_nonzero(delta < -_TIE_TOLERANCE_ANTIGRAVITY)),
        "seed": seed, "draw_count": _BOOTSTRAP_DRAWS_ANTIGRAVITY,
    })


def _metric_rows_antigravity(labels: Mapping[str, np.ndarray], scores: Mapping[str, Mapping[str, np.ndarray | None]], upsampled: Mapping[str, int]) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, Mapping[str, object]], Mapping[str, Mapping[str, object]], tuple[Mapping[str, object], ...], Mapping[str, Mapping[str, object]]]:
    datasets = tuple(labels)
    rows: list[Mapping[str, object]] = []
    macro: dict[str, Mapping[str, object]] = {}
    micro: dict[str, Mapping[str, object]] = {}
    values: dict[tuple[str, str, str], float] = {}
    subset = tuple(dataset for dataset in datasets if upsampled[dataset] > 0)
    for method in METHOD_ORDER_EXP472_ANTIGRAVITY:
        ap_values: list[float] = []; auc_values: list[float] = []; joined_y: list[np.ndarray] = []; joined_s: list[np.ndarray] = []
        for dataset in datasets:
            y = labels[dataset]; s = scores[method][dataset]
            row: dict[str, object] = {"method": method, "dataset": dataset, "upsampled_rows": upsampled[dataset]}
            if s is None:
                row.update({"average_precision": None, "average_precision_exclusion": "typed_abstention", "auroc": None, "auroc_exclusion": "typed_abstention"})
            else:
                if s.shape != y.shape:
                    raise ValueError("Exp472 score TEST identity differs from labels")
                joined_y.append(y); joined_s.append(s)
                if np.count_nonzero(y) == 0:
                    row.update({"average_precision": None, "average_precision_exclusion": "no_positive_test_labels"})
                else:
                    ap = float(average_precision_score(y, s)); row.update({"average_precision": ap, "average_precision_exclusion": None}); ap_values.append(ap); values[(method, dataset, "average_precision")] = ap
                if np.unique(y).size != 2:
                    row.update({"auroc": None, "auroc_exclusion": "single_class_test_labels"})
                else:
                    auc = float(roc_auc_score(y, s)); row.update({"auroc": auc, "auroc_exclusion": None}); auc_values.append(auc); values[(method, dataset, "auroc")] = auc
            rows.append(MappingProxyType(row))
        macro[method] = MappingProxyType({"average_precision": float(np.mean(ap_values)) if ap_values else None, "average_precision_denominator": len(ap_values), "average_precision_excluded": len(datasets)-len(ap_values), "auroc": float(np.mean(auc_values)) if auc_values else None, "auroc_denominator": len(auc_values), "auroc_excluded": len(datasets)-len(auc_values)})
        if joined_y and np.unique(np.concatenate(joined_y)).size == 2:
            y = np.concatenate(joined_y); s = np.concatenate(joined_s)
            micro[method] = MappingProxyType({"average_precision": float(average_precision_score(y, s)), "auroc": float(roc_auc_score(y, s)), "observation_denominator": int(y.size), "dataset_denominator": len(joined_y)})
        else:
            micro[method] = MappingProxyType({"average_precision": None, "auroc": None, "observation_denominator": int(sum(x.size for x in joined_y)), "dataset_denominator": len(joined_y)})
    paired: list[Mapping[str, object]] = []
    for metric in ("average_precision", "auroc"):
        common = tuple(dataset for dataset in datasets if ("E", dataset, metric) in values and ("D", dataset, metric) in values)
        statistic = paired_e_minus_d_exp472_antigravity([values[("E", x, metric)] for x in common], [values[("D", x, metric)] for x in common]) if common else MappingProxyType({"mean_delta": None, "ci_lower": None, "ci_upper": None, "wins": 0, "ties": 0, "losses": 0, "seed": _BOOTSTRAP_SEED_ANTIGRAVITY, "draw_count": 0})
        paired.append(MappingProxyType({"comparison": "E-D", "metric": metric, "dataset_order": common, **statistic}))
    denominators = MappingProxyType({
        "full_population": MappingProxyType({"dataset_count": len(datasets), "upsampled_rows_gt_zero": sum(upsampled[x] > 0 for x in datasets)}),
        "upsampled_rows_gt_zero": MappingProxyType({"dataset_count": len(subset), "dataset_order": subset}),
    })
    return tuple(rows), MappingProxyType(macro), MappingProxyType(micro), tuple(paired), denominators


def _curves_antigravity(labels: Mapping[str, np.ndarray], scores: Mapping[str, Mapping[str, np.ndarray | None]]) -> Mapping[str, Mapping[str, object]]:
    roc_rows: list[Mapping[str, object]] = []; pr_rows: list[Mapping[str, object]] = []
    for method in METHOD_ORDER_EXP472_ANTIGRAVITY:
        present = tuple(x for x in labels if scores[method][x] is not None)
        if not present:
            continue
        y = np.concatenate([labels[x] for x in present]); s = np.concatenate([scores[method][x] for x in present])
        if np.unique(y).size != 2:
            continue
        fpr, tpr, threshold = roc_curve(y, s); precision, recall, _ = precision_recall_curve(y, s)
        if not (np.all((0 <= fpr) & (fpr <= 1)) and np.all(np.diff(fpr) >= 0) and (fpr[0], tpr[0]) == (0, 0) and (fpr[-1], tpr[-1]) == (1, 1)):
            raise ValueError("Exp472 ROC coordinates differ")
        if not (np.all((0 <= recall) & (recall <= 1)) and np.all((0 <= precision) & (precision <= 1)) and np.all(np.diff(recall) <= 0)):
            raise ValueError("Exp472 PR recall direction differs")
        roc_rows.extend(MappingProxyType({"method": method, "false_positive_rate": float(x), "true_positive_rate": float(y), "threshold": float(t)}) for x,y,t in zip(fpr,tpr,threshold,strict=True))
        pr_rows.extend(MappingProxyType({"method": method, "recall": float(x), "precision": float(y)}) for x,y in zip(recall,precision,strict=True))
    roc = MappingProxyType({"x_name": "false_positive_rate", "y_name": "true_positive_rate", "rows": tuple(roc_rows)})
    pr = MappingProxyType({"x_name": "recall", "y_name": "precision", "rows": tuple(pr_rows)})
    return MappingProxyType({"roc": roc, "pr": pr, "precision_recall": pr})


def evaluate_exp472_antigravity(e_manifest: Mapping[str, object], exp471_evidence: Mapping[str, object], labels: Mapping[str, Sequence[object]] | Callable[[tuple[str, ...]], Mapping[str, Sequence[object]]]) -> Exp472EvaluationAntigravity:
    """Verify evidence first, then make exactly one caller-owned label access."""
    if not isinstance(exp471_evidence, Mapping):
        raise ValueError("Exp472 Exp471 evidence differs")
    authority = verify_exp471_reuse_exp472_antigravity(exp471_evidence)
    datasets = bind_e_to_exp471_exp472_antigravity(e_manifest, authority)
    e_scores, upsampled = _e_scores_antigravity(e_manifest, datasets)
    d_scores = _d_scores_antigravity(exp471_evidence, datasets)
    # This is the only label boundary; all authority, identity and score checks above are label-free.
    loaded = labels(datasets) if callable(labels) else labels
    normalized_labels = _labels_antigravity(loaded, datasets)
    all_scores = dict(d_scores); all_scores["E"] = e_scores
    rows, macro, micro, paired, denominators = _metric_rows_antigravity(normalized_labels, MappingProxyType(all_scores), upsampled)
    return Exp472EvaluationAntigravity(authority, METHOD_ORDER_EXP472_ANTIGRAVITY, rows, macro, micro, paired, denominators, _curves_antigravity(normalized_labels, MappingProxyType(all_scores)))


__all__ = ["Exp472EvaluationAntigravity", "METHOD_ORDER_EXP472_ANTIGRAVITY", "evaluate_exp472_antigravity", "paired_e_minus_d_exp472_antigravity"]
