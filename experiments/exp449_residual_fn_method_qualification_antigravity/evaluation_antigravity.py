"""One synthetic-session counterfactual evaluator; no public label invocation occurs here."""
from __future__ import annotations

from collections.abc import Mapping
import ctypes
import errno
from hashlib import sha256
import os
from pathlib import Path
import sys

import numpy as np

from .artifacts_antigravity import canonical_json_bytes_antigravity, canonical_jsonl_bytes_antigravity, write_create_only_antigravity
from .bootstrap_antigravity import BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY, paired_delta_antigravity
from .config_antigravity import CANDIDATE_MATRIX_ANTIGRAVITY, FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import CandidateMetricAntigravity, ConfusionAntigravity, DatasetBaselineAntigravity
from .prediction_antigravity import FrozenPredictionAuthorityAntigravity, FrozenPredictionCaptureAntigravity
from .replay_antigravity import FrozenEvaluationAntigravity, inspect_replay_at_fd_antigravity


_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json", "compatibility_antigravity.json", "candidate_registry_antigravity.json",
    "frozen_predictions_antigravity.jsonl", "prediction_manifest_antigravity.json", "evaluation_antigravity.json",
    "candidate_metrics_antigravity.jsonl", "family_metrics_antigravity.jsonl", "bootstrap_antigravity.json",
    "decision_antigravity.json", "dashboard_snapshot_antigravity.json", "artifact_sha256_antigravity.json",
)


def _f1_antigravity(confusion: ConfusionAntigravity) -> float:
    denominator = 2 * confusion.tp + confusion.fp + confusion.fn
    return 0.0 if denominator == 0 else (2.0 * confusion.tp) / denominator


def qualifies_antigravity(confusion: object, *, target_fn_delta: object) -> bool:
    """Apply exactly the four preregistered qualification gates without coercion."""
    if type(confusion) is not ConfusionAntigravity:
        raise ValueError("global confusion must be exact")
    if type(target_fn_delta) is not int:
        raise ValueError("target FN delta must be an exact integer")
    return (confusion.fn < 1988 and confusion.fp <= 314 and _f1_antigravity(confusion) >= FROZEN_CONFIG_ANTIGRAVITY["global_f1_min"] and target_fn_delta < 0)


def _sum_antigravity(items: tuple[ConfusionAntigravity, ...]) -> ConfusionAntigravity:
    return ConfusionAntigravity(sum(item.tp for item in items), sum(item.fp for item in items), sum(item.fn for item in items), sum(item.tn for item in items))


def _from_labels_antigravity(labels: tuple[tuple[int, int], ...], indices: frozenset[int]) -> ConfusionAntigravity:
    tp = fp = fn = tn = 0
    for index, label in labels:
        predicted = index in indices
        if predicted and label == 1: tp += 1
        elif predicted: fp += 1
        elif label == 1: fn += 1
        else: tn += 1
    return ConfusionAntigravity(tp, fp, fn, tn)


def _payload_antigravity(confusion: ConfusionAntigravity) -> dict[str, int]:
    return {"tp": confusion.tp, "fp": confusion.fp, "fn": confusion.fn, "tn": confusion.tn}


def _gates_antigravity(confusion: ConfusionAntigravity, target_fn_delta: int) -> dict[str, bool]:
    return {"global_fn": confusion.fn < 1988, "global_fp": confusion.fp <= 314, "global_f1": _f1_antigravity(confusion) >= FROZEN_CONFIG_ANTIGRAVITY["global_f1_min"], "target_fn_delta": target_fn_delta < 0}


def _exact_labels_antigravity(dataset: str, labels: object) -> tuple[tuple[int, int], ...]:
    if type(labels) is not tuple or not labels:
        raise ValueError("TEST labels are invalid")
    normalized: list[tuple[int, int]] = []
    for expected, row in enumerate(labels):
        if type(row) is not tuple or len(row) != 2 or type(row[0]) is not int or row[0] != expected or type(row[1]) is not int or row[1] not in (0, 1):
            raise ValueError(f"TEST label order differs for {dataset}")
        normalized.append(row)
    return tuple(normalized)


def _prediction_rows_antigravity(capture: object, authority: object) -> tuple[dict[str, object], ...]:
    if type(capture) is not FrozenPredictionCaptureAntigravity:
        raise ValueError("Task5 prediction capture differs")
    try:
        rows = tuple(__import__("json").loads(line) for line in capture.canonical_rows.splitlines())
    except (TypeError, ValueError) as error:
        raise ValueError("Task5 prediction capture differs") from error
    if len(rows) != 96:
        raise ValueError("frozen prediction inventory differs")
    datasets = getattr(authority, "target_dataset_names", None)
    baselines = getattr(authority, "target_baseline_rows", None)
    if type(datasets) is not tuple or type(baselines) is not tuple or len(datasets) != 48 or len(baselines) != 48:
        raise ValueError("Exp448 target authority differs")
    expected: list[tuple[str, str, str]] = []
    for dataset, baseline in zip(datasets, baselines, strict=True):
        if type(dataset) is not str or type(baseline) is not DatasetBaselineAntigravity:
            raise ValueError("Exp448 target identity differs")
        for method in dict(CANDIDATE_MATRIX_ANTIGRAVITY).get(baseline.family, ()):
            expected.append((dataset, baseline.family, method))
    parsed: list[dict[str, object]] = []
    for sequence, (row, (dataset, family, method)) in enumerate(zip(rows, expected, strict=True)):
        required = {"sequence", "candidate_name", "dataset", "family", "method", "indices", "target_length"}
        if not isinstance(row, dict) or set(row) != required or row["sequence"] != sequence or type(row["sequence"]) is not int or row["candidate_name"] != f"{dataset}__{method}" or row["dataset"] != dataset or row["family"] != family or row["method"] != method or type(row["target_length"]) is not int or type(row["indices"]) is not list or any(type(index) is not int or index < 0 for index in row["indices"]) or len(set(row["indices"])) != len(row["indices"]):
            raise ValueError("frozen prediction row differs")
        parsed.append(row)
    return tuple(parsed)


def _audit_antigravity(path: Path, error: BaseException, *, phase: str, label_opened: bool, labels_closed: bool, queries: int, rows: int) -> None:
    payload = {"schema": "exp449.task6.failure", "phase": phase, "label_session_opened": label_opened, "label_session_closed": labels_closed, "label_queries": queries, "label_rows": rows, "error": f"{type(error).__name__}: {error}"}
    try:
        write_create_only_antigravity(path, canonical_json_bytes_antigravity(payload))
    except BaseException as audit_error:
        error.add_note(f"failure_audit_error={type(audit_error).__name__}: {audit_error}")


def _fsync_directory_antigravity(descriptor: int) -> None:
    """Durably flush an already retained directory descriptor."""
    if type(descriptor) is not int:
        raise ValueError("directory descriptor is invalid")
    os.fsync(descriptor)


def _close_descriptor_antigravity(descriptor: int) -> None:
    """Release a retained descriptor; callers decide whether close is reportable."""
    if type(descriptor) is not int:
        raise ValueError("directory descriptor is invalid")
    os.close(descriptor)


def _write_leaf_at_antigravity(directory_fd: int, name: str, payload: bytes) -> None:
    """Create, flush, and retain no pathname dependency for one private leaf."""
    if type(directory_fd) is not int or name not in _LEAVES_ANTIGRAVITY or type(payload) is not bytes:
        raise ValueError("private leaf write is invalid")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    leaf_fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(leaf_fd, payload[offset:])
            if type(written) is not int or written <= 0:
                raise OSError(errno.EIO, "private leaf write made no progress", name)
            offset += written
        os.fsync(leaf_fd)
    finally:
        _close_descriptor_antigravity(leaf_fd)


def _rename_no_replace_at_antigravity(parent_fd: int, source_name: str, destination_name: str) -> None:
    """Use Darwin's descriptor-relative exclusive directory rename."""
    if sys.platform != "darwin":
        raise OSError(errno.ENOTSUP, "exclusive directory rename is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    renamex = getattr(libc, "renameatx_np", None)
    if renamex is None:
        raise OSError(errno.ENOTSUP, "exclusive directory rename is unavailable")
    renamex.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renamex.restype = ctypes.c_int
    # RENAME_EXCL is the Darwin no-replacement flag; its failure leaves both names intact.
    if renamex(parent_fd, os.fsencode(source_name), parent_fd, os.fsencode(destination_name), 0x00000004) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination_name)


def _assert_directory_entry_identity_at_antigravity(parent_fd: int, name: str, retained: os.stat_result) -> None:
    """Prove a hidden namespace entry still names the directory held by its fd."""
    entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (entry.st_dev, entry.st_ino, entry.st_mode, entry.st_nlink) != (retained.st_dev, retained.st_ino, retained.st_mode, retained.st_nlink):
        raise ValueError("sealed publication namespace identity differs")


def _identity_antigravity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink)


def _forensic_name_antigravity(root: Path) -> str:
    return f".{root.name}.forensic_{os.urandom(16).hex()}_antigravity"


def _publish_antigravity(root: Path, payloads: Mapping[str, bytes], expected_prediction_authority: FrozenPredictionCaptureAntigravity) -> None:
    """Publish with post-rename retained-FD verification and bounded identity recovery."""
    if root.exists() or root.is_symlink() or set(payloads) != set(_LEAVES_ANTIGRAVITY[:-1]):
        raise ValueError("publication destination is invalid or already exists")
    parent_fd = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    staging_fd: int | None = None
    staging_name: str | None = None
    primary: BaseException | None = None
    try:
        staging_name = f".{root.name}.staging_{os.urandom(16).hex()}_antigravity"
        os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
        staging_fd = os.open(staging_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        for name in _LEAVES_ANTIGRAVITY[:-1]: _write_leaf_at_antigravity(staging_fd, name, payloads[name])
        hashes = {name: sha256(payloads[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
        _write_leaf_at_antigravity(staging_fd, _LEAVES_ANTIGRAVITY[-1], canonical_json_bytes_antigravity(hashes))
        _fsync_directory_antigravity(staging_fd)
        retained_identity = os.fstat(staging_fd)
        _assert_directory_entry_identity_at_antigravity(parent_fd, staging_name, retained_identity)
        inspect_replay_at_fd_antigravity(staging_fd, expected_prediction_authority=expected_prediction_authority)
        if _identity_antigravity(os.fstat(staging_fd)) != _identity_antigravity(retained_identity):
            raise ValueError("publication directory changed while retained")
        _fsync_directory_antigravity(staging_fd)
        _fsync_directory_antigravity(parent_fd)
        _assert_directory_entry_identity_at_antigravity(parent_fd, staging_name, retained_identity)
        _rename_no_replace_at_antigravity(parent_fd, staging_name, root.name)
        staging_name = None
        # The canonical name is now fallible and must be verified through the retained FD.
        _assert_directory_entry_identity_at_antigravity(parent_fd, root.name, retained_identity)
        inspect_replay_at_fd_antigravity(staging_fd, expected_prediction_authority=expected_prediction_authority)
        if _identity_antigravity(os.fstat(staging_fd)) != _identity_antigravity(retained_identity):
            raise ValueError("publication descriptor identity differs after rename")
        _assert_directory_entry_identity_at_antigravity(parent_fd, root.name, retained_identity)
        _fsync_directory_antigravity(staging_fd)
        _fsync_directory_antigravity(parent_fd)
    except BaseException as error:
        primary = error
        # Never touch a foreign object: only rename canonical when it still is retained identity.
        try:
            retained = os.fstat(staging_fd) if staging_fd is not None else None
            if retained is not None:
                try:
                    current = os.stat(root.name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    current = None
                if current is not None and _identity_antigravity(current) == _identity_antigravity(retained):
                    forensic = _forensic_name_antigravity(root)
                    _rename_no_replace_at_antigravity(parent_fd, root.name, forensic)
                    _assert_directory_entry_identity_at_antigravity(parent_fd, forensic, retained)
                    _fsync_directory_antigravity(parent_fd)
        except BaseException as recovery_error:
            error.add_note(f"recovery_error={type(recovery_error).__name__}: {recovery_error}")
        raise
    finally:
        try:
            if staging_fd is not None:
                try:
                    _close_descriptor_antigravity(staging_fd)
                except BaseException as close_error:
                    if primary is not None:
                        primary.add_note(f"descriptor_close_error={type(close_error).__name__}: {close_error}")
                    else:
                        raise
        finally:
            try:
                _close_descriptor_antigravity(parent_fd)
            except BaseException as close_error:
                if primary is not None:
                    primary.add_note(f"parent_descriptor_close_error={type(close_error).__name__}: {close_error}")
                else:
                    raise


def _evaluate_with_dependencies_antigravity(reviewed_commit: object, dependencies: object) -> FrozenEvaluationAntigravity:
    """Injectable synthetic core; it opens exactly one ordered closure-private session."""
    if type(reviewed_commit) is not str or len(reviewed_commit) != 40 or any(ch not in "0123456789abcdef" for ch in reviewed_commit):
        raise ValueError("reviewed commit is invalid")
    root, audit_path = Path(getattr(dependencies, "root")), Path(getattr(dependencies, "audit_path"))
    phase, opened, closed, queries, label_rows = "preflight", False, False, 0, 0
    session = None
    try:
        if root.exists() or root.is_symlink(): raise ValueError("publication destination already exists")
        authority = getattr(dependencies, "capture_exp448_antigravity")()
        expected_prediction_authority = getattr(dependencies, "capture_task5_prediction_antigravity")()
        if type(expected_prediction_authority) is not FrozenPredictionCaptureAntigravity:
            raise ValueError("Task5 prediction capture differs")
        if expected_prediction_authority.authority.reviewed_commit != reviewed_commit:
            raise ValueError("Task5 reviewed commit differs")
        predictions = _prediction_rows_antigravity(expected_prediction_authority, authority)
        baseline_global = getattr(authority, "baseline_confusion", None)
        baselines = getattr(authority, "target_baseline_rows", None)
        datasets = getattr(authority, "target_dataset_names", None)
        if type(baseline_global) is not ConfusionAntigravity or type(baselines) is not tuple or type(datasets) is not tuple or len(baselines) != 48 or len(datasets) != 48:
            raise ValueError("Exp448 baseline authority differs")
        phase = "label_open"; session = getattr(dependencies, "open_labels_antigravity")(expected_prediction_authority.authority); opened = True
        labels_by_dataset: dict[str, tuple[tuple[int, int], ...]] = {}
        for dataset in datasets:
            phase = "label_query"; labels = _exact_labels_antigravity(dataset, session.labels_for_dataset_antigravity(dataset)); queries += 1; label_rows += len(labels); labels_by_dataset[dataset] = labels
        if (queries, label_rows) != (48, 72300): raise ValueError("TEST label population differs")
        phase = "label_close"; session.close_antigravity(); closed = True
        phase = "post_close"
        if getattr(dependencies, "capture_exp448_antigravity")() != authority: raise ValueError("Exp448 source drift after close")
        by_pair: dict[tuple[str, str], list[tuple[str, frozenset[int]]]] = {(family, method): [] for family, methods in CANDIDATE_MATRIX_ANTIGRAVITY for method in methods}
        for row in predictions: by_pair[(row["family"], row["method"])].append((row["dataset"], frozenset(row["indices"])))
        metrics: list[CandidateMetricAntigravity] = []; bootstrap_rows: list[dict[str, object]] = []
        dataset_evidence = [{"dataset": dataset, "family": baseline.family, "target_length": len(labels_by_dataset[dataset]), "baseline_confusion": _payload_antigravity(baseline.confusion), "candidates": {}} for dataset, baseline in zip(datasets, baselines, strict=True)]
        for family, methods in CANDIDATE_MATRIX_ANTIGRAVITY:
            family_positions = tuple(index for index, baseline in enumerate(baselines) if baseline.family == family)
            base_rows = tuple(baselines[index].confusion for index in family_positions); base_target = _sum_antigravity(base_rows)
            for method in methods:
                current = dict(by_pair[(family, method)])
                if tuple(current) != tuple(datasets[index] for index in family_positions): raise ValueError("candidate dataset membership differs")
                candidate_rows = tuple(_from_labels_antigravity(labels_by_dataset[datasets[index]], current[datasets[index]]) for index in family_positions)
                for position, row in zip(family_positions, candidate_rows, strict=True):
                    dataset_evidence[position]["candidates"][method] = {"indices": list(current[datasets[position]]), "confusion": _payload_antigravity(row)}
                target = _sum_antigravity(candidate_rows)
                if (target.tp + target.fn, target.fp + target.tn) != (base_target.tp + base_target.fn, base_target.fp + base_target.tn): raise ValueError("target confusion does not conserve labels")
                global_ = ConfusionAntigravity(baseline_global.tp - base_target.tp + target.tp, baseline_global.fp - base_target.fp + target.fp, baseline_global.fn - base_target.fn + target.fn, baseline_global.tn - base_target.tn + target.tn)
                delta = target.fn - base_target.fn; metric = CandidateMetricAntigravity(family, method, target, global_, delta, _f1_antigravity(global_), qualifies_antigravity(global_, target_fn_delta=delta)); metrics.append(metric)
                all_base = tuple(item.confusion for item in baselines); all_candidate = list(all_base)
                for position, row in zip(family_positions, candidate_rows, strict=True): all_candidate[position] = row
                base_counts = [[row.tp, row.fp, row.fn, row.tn] for row in all_base]
                candidate_counts = [[row.tp, row.fp, row.fn, row.tn] for row in all_candidate]
                bootstrap = paired_delta_antigravity(np.asarray(base_counts, dtype=np.int64), np.asarray(candidate_counts, dtype=np.int64)); bootstrap_rows.append({"family": family, "method": method, "baseline_dataset_confusions": base_counts, "candidate_dataset_confusions": candidate_counts, **bootstrap})
        phase = "post_compute"
        if getattr(dependencies, "capture_exp448_antigravity")() != authority: raise ValueError("Exp448 source drift after compute")
        metric_payloads = tuple({"family": item.family, "method": item.method, "target_confusion": _payload_antigravity(item.target_confusion), "global_confusion": _payload_antigravity(item.global_confusion), "target_fn_delta": item.target_fn_delta, "global_f1": item.global_f1, "qualified": item.qualified} for item in metrics)
        family_payloads = tuple({"family": family, "candidate_count": 2, "qualified_count": sum(item.qualified for item in metrics if item.family == family), "baseline_target_confusion": _payload_antigravity(_sum_antigravity(tuple(baseline.confusion for baseline in baselines if baseline.family == family)))} for family, _methods in CANDIDATE_MATRIX_ANTIGRAVITY)
        qualified_count = sum(item.qualified for item in metrics); routing_research_eligible = qualified_count > 0
        decision = {"schema": "exp449.decision", "qualified_count": qualified_count, "routing_research_eligible": routing_research_eligible, "automatic_routing": False, "automatic_promotion": False, "gates": {"global_fn_lt": 1988, "global_fp_lte": 314, "global_f1_nonregression": FROZEN_CONFIG_ANTIGRAVITY["global_f1_min"], "target_fn_delta_lt": 0}}
        target_datasets = [{"dataset": item["dataset"], "family": item["family"], "target_length": item["target_length"], "baseline_confusion": item["baseline_confusion"]} for item in dataset_evidence]
        target_authority_sha256 = sha256(canonical_json_bytes_antigravity(target_datasets)).hexdigest()
        evaluation = {"schema": "exp449", "reviewed_commit": reviewed_commit, "candidate_count": 16, "qualified_count": qualified_count, "label_session_count": 1, "label_query_count": 48, "label_row_count": 72300, "routing_research_eligible": routing_research_eligible, "dataset_evidence": dataset_evidence}
        frozen_bytes = expected_prediction_authority.canonical_rows
        dashboard = {"schema": "exp449.public-method-qualification.v1", "experiment": "Exp449", "baseline": {**_payload_antigravity(baseline_global), "f1": _f1_antigravity(baseline_global)}, "candidates": [{"family": item.family, "method": item.method, "target_fn_delta": item.target_fn_delta, "gates": _gates_antigravity(item.global_confusion, item.target_fn_delta), "qualified": item.qualified, "global": {**_payload_antigravity(item.global_confusion), "f1": item.global_f1}} for item in metrics], "families": [family for family, _methods in CANDIDATE_MATRIX_ANTIGRAVITY], "status": "synthetic_complete", "repeatable": True, "qualified_candidate_count": qualified_count, "routing_research_eligible": routing_research_eligible, "runtime": {"label_sessions": 1, "label_queries": 48, "label_rows": 72300}}
        from .dashboard_export_antigravity import public_result_antigravity
        public_result_antigravity(dashboard)
        prediction_manifest = {"schema": "exp449_prediction_manifest", "rows": 96, "candidate_pairs": 16, "frozen_prediction_sha256": sha256(frozen_bytes).hexdigest(), "task5_reviewed_commit": expected_prediction_authority.authority.reviewed_commit, "task5_inventory": list(expected_prediction_authority.authority.inventory), "task5_leaf_sha256": dict(expected_prediction_authority.authority.leaf_sha256), "task5_aggregate_sha256": expected_prediction_authority.authority.aggregate_sha256, "task5_context_sha256": expected_prediction_authority.context_sha256, "task5_source_identity_sha256": expected_prediction_authority.source_identity_sha256, "task5_manifest_sha256": expected_prediction_authority.manifest_sha256, "task5_label_access_count": 0}
        payloads = {"context_antigravity.json": canonical_json_bytes_antigravity({"schema": "exp449", "reviewed_commit": reviewed_commit, "label_access_count": 1, "target_datasets": target_datasets, "target_authority_sha256": target_authority_sha256, "exp448_target_identity_sha256": "0c87af592ea66e2ff479fe8a5fc1ac2222084287d6eae617530f18ec4090bf43"}), "compatibility_antigravity.json": canonical_json_bytes_antigravity({"schema": "exp449.task6", "predecessor": "exp448", "prediction_authority": "task5"}), "candidate_registry_antigravity.json": canonical_json_bytes_antigravity([[family, list(methods)] for family, methods in CANDIDATE_MATRIX_ANTIGRAVITY]), "frozen_predictions_antigravity.jsonl": frozen_bytes, "prediction_manifest_antigravity.json": canonical_json_bytes_antigravity(prediction_manifest), "evaluation_antigravity.json": canonical_json_bytes_antigravity(evaluation), "candidate_metrics_antigravity.jsonl": canonical_jsonl_bytes_antigravity(metric_payloads), "family_metrics_antigravity.jsonl": canonical_jsonl_bytes_antigravity(family_payloads), "bootstrap_antigravity.json": canonical_json_bytes_antigravity({"draw_sha256": BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY, "comparisons": bootstrap_rows}), "decision_antigravity.json": canonical_json_bytes_antigravity(decision), "dashboard_snapshot_antigravity.json": canonical_json_bytes_antigravity(dashboard)}
        phase = "precommit"
        if getattr(dependencies, "capture_exp448_antigravity")() != authority: raise ValueError("Exp448 source drift before commit")
        artifact_map = {name: sha256(payloads[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
        expected = FrozenEvaluationAntigravity(reviewed_commit, 16, qualified_count, 1, 48, 72300, tuple(metrics), sha256(canonical_json_bytes_antigravity(artifact_map)).hexdigest(), sha256(canonical_json_bytes_antigravity(decision)).hexdigest(), routing_research_eligible)
        _publish_antigravity(root, payloads, expected_prediction_authority)
        return expected
    except BaseException as error:
        if session is not None and opened and not closed:
            try: session.close_antigravity(); closed = True
            except BaseException as close_error: error.add_note(f"label_close_error={type(close_error).__name__}: {close_error}")
        _audit_antigravity(audit_path, error, phase=phase, label_opened=opened, labels_closed=closed, queries=queries, rows=label_rows)
        raise


def evaluate_antigravity(reviewed_commit: str) -> FrozenEvaluationAntigravity:
    """Task 7 supplies the official readiness-gated dependency injection; Task 6 never opens labels itself."""
    if type(reviewed_commit) is not str or len(reviewed_commit) != 40:
        raise ValueError("reviewed commit is invalid")
    raise ValueError("Task-6 evaluation requires the closed Task-7 readiness dependency injection")


def inspect_antigravity() -> FrozenEvaluationAntigravity:
    """Compatibility export for the Task-6 source-free inspector."""
    from .replay_antigravity import inspect_antigravity as inspect
    return inspect()


__all__ = ["FrozenEvaluationAntigravity", "evaluate_antigravity", "inspect_antigravity", "qualifies_antigravity"]
