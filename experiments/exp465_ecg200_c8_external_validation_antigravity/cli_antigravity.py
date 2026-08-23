"""Closed private-only command boundary for Exp465 Task 5."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from hashlib import sha256
import json
from math import isfinite, sqrt
import os
from pathlib import Path
import re
import subprocess
import sys

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_COMMIT_ANTIGRAVITY = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[2]
_TRAIN_PATH_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp465_ecg200_train_antigravity/"
    "ECG200_TRAIN_antigravity.ts"
)
OFFICIAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp465_ecg200_c8_external_validation_antigravity_20260822"
)
_PRIVATE_TEST_MODULE_ANTIGRAVITY = "_exp465_cli_private_test_antigravity"
_PRIVATE_TEST_CAPABILITY_ANTIGRAVITY = "exp465_cli_private_test_antigravity_v1"
_LAUNCHER_NAME_ANTIGRAVITY = (
    "run_exp465_ecg200_c8_external_validation_antigravity.py"
)
_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json",
    "method_fold_metrics_antigravity.jsonl",
    "failed_folds_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
    "decision_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "artifact_sha256_antigravity.json",
)


class _ExactSurfaceParserAntigravity(argparse.ArgumentParser):
    """Reject every spelling outside the two preregistered commands."""

    def parse_args(self, args: Sequence[str] | None = None, namespace=None):
        raw = list(sys.argv[1:] if args is None else args)
        if raw == ["inspect"] or (
            len(raw) == 3
            and raw[:2] == ["evaluate", "--reviewed-commit"]
            and _COMMIT_ANTIGRAVITY.fullmatch(raw[2]) is not None
        ):
            return super().parse_args(raw, namespace)
        self.error("only bare inspect or evaluate --reviewed-commit LOWER40 is accepted")


def _reviewed_commit_antigravity(value: str) -> str:
    if _COMMIT_ANTIGRAVITY.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("reviewed commit must be lowercase 40-hex")
    return value


def parser_antigravity() -> argparse.ArgumentParser:
    parser = _ExactSurfaceParserAntigravity(
        prog="run_exp465_ecg200_c8_external_validation_antigravity",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", allow_abbrev=False)
    evaluate.add_argument(
        "--reviewed-commit",
        required=True,
        type=_reviewed_commit_antigravity,
    )
    commands.add_parser("inspect", allow_abbrev=False)
    return parser


def _private_test_root_antigravity() -> Path:
    private = sys.modules.get(_PRIVATE_TEST_MODULE_ANTIGRAVITY)
    if (
        private is None
        or getattr(private, "CAPABILITY_ANTIGRAVITY", None)
        != _PRIVATE_TEST_CAPABILITY_ANTIGRAVITY
    ):
        raise ValueError("private Task 5 root capability is required")
    extra = sorted(
        name
        for name in vars(private)
        if not name.startswith("__")
        and name not in {"CAPABILITY_ANTIGRAVITY", "root_antigravity"}
    )
    if extra:
        raise ValueError("private test seam may inject only root_antigravity")
    root = getattr(private, "root_antigravity", None)
    if (
        not isinstance(root, Path)
        or root.resolve(strict=False) == OFFICIAL_ROOT_ANTIGRAVITY.resolve(strict=False)
    ):
        raise ValueError("private test root must be a non-official Path")
    if root.is_symlink():
        raise ValueError("private test root must be a non-symlink Path")
    return root


def _validate_create_only_root_antigravity(root: Path) -> None:
    if not isinstance(root, Path) or root.is_symlink():
        raise ValueError("result root must be a non-symlink Path")
    if root.resolve(strict=False) == OFFICIAL_ROOT_ANTIGRAVITY.resolve(strict=False):
        raise ValueError("private test root must be a non-official Path")
    if root.exists():
        raise ValueError("publication destination must be a new create-only path")


def _launcher_token_matches_antigravity(command: str) -> bool:
    return any(
        Path(token.strip("\"'")).name == _LAUNCHER_NAME_ANTIGRAVITY
        for token in command.split()
    )


def _validate_no_duplicate_process_antigravity() -> None:
    output = subprocess.check_output(
        ["ps", "-Ao", "pid=,ppid=,command="],
        text=True,
    )
    rows: list[tuple[int, int, str, str]] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), parts[2], line))
        except ValueError:
            continue
    parents = {pid: ppid for pid, ppid, _command, _line in rows}
    ancestors: set[int] = set()
    current = os.getpid()
    while current > 0 and current not in ancestors:
        ancestors.add(current)
        current = parents.get(current, 0)
    active = [
        line
        for pid, _ppid, command, line in rows
        if pid not in ancestors
        and _launcher_token_matches_antigravity(command)
        and "test_cli_antigravity" not in command
        and " -m unittest " not in command
        and "compileall" not in command
        and "build_codemap_exp465_antigravity.py" not in command
    ]
    if active:
        raise ValueError("duplicate Exp465 process is already active")


def _validate_clean_current_commit_antigravity(reviewed_commit: str) -> None:
    head = subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        cwd=_REPOSITORY_ANTIGRAVITY,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ("git", "status", "--porcelain"),
        cwd=_REPOSITORY_ANTIGRAVITY,
        text=True,
    )
    if reviewed_commit != head or dirty:
        raise ValueError("reviewed commit must be the exact clean current commit")
    if OFFICIAL_ROOT_ANTIGRAVITY.exists() or OFFICIAL_ROOT_ANTIGRAVITY.is_symlink():
        raise ValueError("official result root must be absent during Task 5")
    _validate_no_duplicate_process_antigravity()


def _json_leaf_antigravity(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl_leaf_antigravity(path: Path) -> list[object]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _canonical_json_antigravity(payload: bytes) -> object:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("artifact JSON is invalid") from error
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if payload != canonical:
        raise ValueError("artifact JSON is not canonical")
    return value


def _canonical_jsonl_antigravity(payload: bytes) -> tuple[object, ...]:
    if not payload:
        return ()
    if not payload.endswith(b"\n"):
        raise ValueError("artifact JSONL is invalid")
    return tuple(_canonical_json_antigravity(line + b"\n") for line in payload.splitlines())


def _auc_from_scores_antigravity(normal: list[object], changed: list[object]) -> float:
    if not normal or len(normal) != len(changed):
        raise ValueError("score vectors must be equal-length and nonempty")
    if any(type(value) is not float or not isfinite(value) for value in (*normal, *changed)):
        raise ValueError("score vectors must be finite floats")
    wins = sum(change > base for change in changed for base in normal)
    ties = sum(change == base for change in changed for base in normal)
    return float((wins + 0.5 * ties) / (len(normal) * len(changed)))


def _wilson_upper_antigravity(successes: int, trials: int) -> float:
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials or trials < 1:
        raise ValueError("confirmation counts are invalid")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    radius = z * sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * trials)) / trials)
    return float((center + radius) / denominator)


def _persisted_metric_rows_antigravity(rows: tuple[object, ...]) -> tuple[dict[str, object], ...]:
    expected = tuple(
        (method, seed, fold)
        for method in FROZEN_CONFIG_ANTIGRAVITY.method_order
        for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        for fold in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    if len(rows) != len(expected) or any(type(row) is not dict for row in rows):
        raise ValueError("persisted metric rows differ from the frozen inventory")
    typed = tuple(rows)  # type: ignore[assignment]
    actual = tuple((row.get("method"), row.get("seed"), row.get("fold_index")) for row in typed)
    if actual != expected:
        raise ValueError("persisted metric order differs from the frozen contract")
    change_types = ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")
    for row in typed:
        normal = row.get("normal_scores")
        primary = row.get("primary_scores")
        stored_auc = row.get("per_type_auc")
        if type(normal) is not list or type(primary) is not dict or type(stored_auc) is not list or set(primary) != set(change_types) or len(stored_auc) != 4:
            raise ValueError("persisted score fields differ from the frozen contract")
        auc = tuple(_auc_from_scores_antigravity(normal, primary[name]) for name in change_types if type(primary[name]) is list)
        if len(auc) != 4 or tuple(stored_auc) != auc:
            raise ValueError("persisted AUC differs from raw scores")
        mean = float(sum(auc) / len(auc))
        if row.get("mean_auc") != mean or row.get("all_scores_finite") is not True:
            raise ValueError("persisted score summary differs from raw scores")
        counts = row.get("confirmation_counts")
        cosines = row.get("query_cosines")
        if type(counts) is not dict or type(cosines) is not dict:
            raise ValueError("persisted confirmation evidence is invalid")
        if not cosines:
            if any(value != 0 for value in counts.values()):
                raise ValueError("persisted confirmation evidence is invalid")
        elif set(counts) != set(cosines):
            raise ValueError("persisted confirmation evidence is invalid")
        for scope, count in counts.items():
            values = cosines.get(scope, [0.0] * len(normal))
            if type(count) is not int or type(values) is not list or len(values) != len(normal):
                raise ValueError("persisted confirmation evidence is invalid")
            if any(type(value) is not float or not isfinite(value) for value in values):
                raise ValueError("persisted confirmation cosine is invalid")
            if count != sum(value >= FROZEN_CONFIG_ANTIGRAVITY.cosine_min for value in values):
                raise ValueError("persisted confirmation count differs from cosine evidence")
        if row.get("normal_wilson_upper") != _wilson_upper_antigravity(counts["normal"], len(normal)):
            raise ValueError("persisted Wilson value differs from confirmation evidence")
    return typed


def _derived_failure_reasons_antigravity(
    row: dict[str, object],
    control: dict[str, object] | None,
) -> tuple[str, ...]:
    auc = row["per_type_auc"]
    reasons = [
        f"{name}_auc_below_0.85"
        for name, value in zip(
            ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"),
            auc,
            strict=True,
        )
        if value < FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min
    ]
    if row["mean_auc"] < FROZEN_CONFIG_ANTIGRAVITY.mean_auc_min:
        reasons.append("mean_auc_below_0.90")
    if control is not None:
        if auc[1] < control["per_type_auc"][1]:
            reasons.append("gradual_auc_below_c0")
        if auc[3] < control["per_type_auc"][3]:
            reasons.append("period_auc_below_c0")
        return tuple(sorted(set(reasons)))
    return tuple(reasons)


def _pair_diagnostics_antigravity(
    control: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for change_type in ("gradual_drift", "period_change"):
        fixed = 0
        broken = 0
        control_changes = control["primary_scores"][change_type]
        candidate_changes = candidate["primary_scores"][change_type]
        for normal_index, control_normal in enumerate(control["normal_scores"]):
            candidate_normal = candidate["normal_scores"][normal_index]
            for change_index, control_change in enumerate(control_changes):
                candidate_change = candidate_changes[change_index]
                if control_change == control_normal or candidate_change == candidate_normal:
                    continue
                if control_change <= control_normal and candidate_change > candidate_normal:
                    fixed += 1
                elif control_change > control_normal and candidate_change <= candidate_normal:
                    broken += 1
        result[change_type] = {
            "total_pairs": len(control["normal_scores"]) * len(control_changes),
            "fixed_pairs": fixed,
            "broken_pairs": broken,
        }
    return result


def _validate_persisted_derived_fields_antigravity(
    rows: tuple[dict[str, object], ...],
    summaries: tuple[object, ...],
    decision: dict[str, object],
    dashboard: dict[str, object],
) -> None:
    by_identity = {(row["method"], row["seed"], row["fold_index"]): row for row in rows}
    for row in rows:
        control = by_identity[("C0", row["seed"], row["fold_index"])] if row["method"] == "C8" else None
        reasons = _derived_failure_reasons_antigravity(row, control)
        if row.get("failure_reasons") != list(reasons) or row.get("passed") is not (not reasons):
            raise ValueError("persisted pass fields differ from raw scores")
        pair = _pair_diagnostics_antigravity(control, row) if control is not None else {
            name: {"total_pairs": 0, "fixed_pairs": 0, "broken_pairs": 0}
            for name in ("gradual_drift", "period_change")
        }
        if row.get("pair_diagnostics") != pair:
            raise ValueError("persisted pair diagnostics differ from raw scores")
    expected_summaries: list[dict[str, object]] = []
    for method in FROZEN_CONFIG_ANTIGRAVITY.method_order:
        selected = tuple(row for row in rows if row["method"] == method)
        pair = {
            name: {
                key: sum(row["pair_diagnostics"][name][key] for row in selected)
                for key in ("total_pairs", "fixed_pairs", "broken_pairs")
            }
            for name in ("gradual_drift", "period_change")
        }
        expected_summaries.append({
            "method": method,
            "fold_count": len(selected),
            "passed_count": sum(row["passed"] for row in selected),
            "failed_count": sum(not row["passed"] for row in selected),
            "min_per_type_auc": [min(row["per_type_auc"][index] for row in selected) for index in range(4)],
            "min_mean_auc": min(row["mean_auc"] for row in selected),
            "max_normal_wilson_upper": max(row["normal_wilson_upper"] for row in selected),
            "normal_confirmation_count": sum(row["confirmation_counts"]["normal"] for row in selected),
            "all_scores_finite": True,
            "pair_diagnostics": pair,
            "passed": all(row["passed"] for row in selected),
        })
    if list(summaries) != expected_summaries:
        raise ValueError("persisted summaries differ from raw scores")
    c8_rows = tuple(row for row in rows if row["method"] == "C8")
    seed_confirmations = {
        seed: sum(row["confirmation_counts"]["normal"] for row in c8_rows if row["seed"] == seed)
        for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
    }
    decision_reasons: list[str] = []
    if expected_summaries[2]["passed_count"] != FROZEN_CONFIG_ANTIGRAVITY.required_stability_records:
        decision_reasons.append("c8_not_10_of_10")
    for seed, confirmations in seed_confirmations.items():
        if confirmations != 0:
            decision_reasons.append(f"seed_{seed}_normal_confirmations_not_zero")
    for row in c8_rows:
        control = by_identity[("C0", row["seed"], row["fold_index"])]
        if row["per_type_auc"][1] < control["per_type_auc"][1]:
            decision_reasons.append("c8_gradual_auc_below_c0")
        if row["per_type_auc"][3] < control["per_type_auc"][3]:
            decision_reasons.append("c8_period_auc_below_c0")
    failure_reasons = sorted(set(decision_reasons))
    expected_decision = {
        "status": "qualification_passed_ready_for_task5_reproducibility" if not failure_reasons else "qualification_failed_blocked_before_routing_research",
        "primary_candidate": "C8",
        "eligible_for_routing_research": not failure_reasons,
        "primary_passed_count": expected_summaries[2]["passed_count"],
        "primary_required_count": FROZEN_CONFIG_ANTIGRAVITY.required_stability_records,
        "failure_reasons": failure_reasons,
        "routing_applied": False,
        "test_labels_opened": False,
        "label_query_count": 0,
    }
    if decision != expected_decision:
        raise ValueError("persisted decision differs from raw scores")
    expected_dashboard = {
        "후보": "C8",
        "통과_기록": expected_decision["primary_passed_count"],
        "필요_기록": expected_decision["primary_required_count"],
        "다음_재현_가능": expected_decision["eligible_for_routing_research"],
        "실패_이유": failure_reasons,
        "정상_confirmation": {str(seed): seed_confirmations[seed] for seed in seed_confirmations},
        "정상_Wilson_설명값": {str(seed): _wilson_upper_antigravity(seed_confirmations[seed], 69) for seed in seed_confirmations},
        "방법_요약": [{
            "방법": item["method"], "통과": item["passed_count"], "실패": item["failed_count"],
            "최저_AUC": item["min_per_type_auc"], "최저_평균_AUC": item["min_mean_auc"],
            "정상_confirmation": item["normal_confirmation_count"], "점수_유한": item["all_scores_finite"], "통과_여부": item["passed"],
        } for item in expected_summaries],
        "보수성_비교": {
            "gradual_not_below_c0": sum(row["per_type_auc"][1] >= by_identity[("C0", row["seed"], row["fold_index"])]["per_type_auc"][1] for row in c8_rows),
            "period_not_below_c0": sum(row["per_type_auc"][3] >= by_identity[("C0", row["seed"], row["fold_index"])]["per_type_auc"][3] for row in c8_rows),
        },
        "TEST_라벨_조회": 0,
    }
    if dashboard != expected_dashboard:
        raise ValueError("persisted dashboard differs from raw scores")


def _inspect_persisted_root_antigravity(root: Path) -> dict[str, object]:
    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise ValueError("artifact inventory differs")
    if {entry.name for entry in root.iterdir()} != set(_LEAVES_ANTIGRAVITY):
        raise ValueError("artifact inventory differs")
    leaves = {name: (root / name).read_bytes() for name in _LEAVES_ANTIGRAVITY}
    artifact_map = _canonical_json_antigravity(leaves["artifact_sha256_antigravity.json"])
    expected_hashes = {name: sha256(leaves[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
    if artifact_map != expected_hashes:
        raise ValueError("artifact hash map differs")
    context = _canonical_json_antigravity(leaves["context_antigravity.json"])
    decision = _canonical_json_antigravity(leaves["decision_antigravity.json"])
    dashboard = _canonical_json_antigravity(leaves["dashboard_snapshot_antigravity.json"])
    rows = _persisted_metric_rows_antigravity(_canonical_jsonl_antigravity(leaves["method_fold_metrics_antigravity.jsonl"]))
    summaries = _canonical_jsonl_antigravity(leaves["method_summary_antigravity.jsonl"])
    failed = _canonical_jsonl_antigravity(leaves["failed_folds_antigravity.jsonl"])
    if type(context) is not dict or type(decision) is not dict or type(dashboard) is not dict:
        raise ValueError("persisted artifact payload is invalid")
    if context.get("experiment") != FROZEN_CONFIG_ANTIGRAVITY.experiment or context.get("label_query_count") != 0 or context.get("official_evaluate_count") != 0:
        raise ValueError("persisted context violates the Task 5 boundary")
    if decision.get("label_query_count") != 0 or decision.get("test_labels_opened") is not False or decision.get("routing_applied") is not False:
        raise ValueError("inspection opened labels or applied routing")
    if context.get("method_fold_metrics_sha256") != sha256(leaves["method_fold_metrics_antigravity.jsonl"]).hexdigest():
        raise ValueError("persisted metric hash differs")
    if tuple(failed) != tuple(row for row in rows if row.get("passed") is False):
        raise ValueError("persisted failed-fold list differs")
    if len(summaries) != len(FROZEN_CONFIG_ANTIGRAVITY.method_order):
        raise ValueError("persisted summaries differ from the frozen contract")
    _validate_persisted_derived_fields_antigravity(rows, summaries, decision, dashboard)
    return {
        "context": context,
        "method_fold_metrics": list(rows),
        "failed_folds": list(failed),
        "method_summaries": list(summaries),
        "decision": decision,
        "dashboard_snapshot": dashboard,
        "leaf_sha256": {name: sha256(leaves[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY},
        "artifact_map_sha256": sha256(leaves["artifact_sha256_antigravity.json"]).hexdigest(),
        "artifact_sha256": artifact_map,
    }


def _result_payload_from_root_antigravity(root: Path, replayed: dict[str, object]) -> dict[str, object]:
    decision = replayed.get("decision")
    if type(decision) is not dict or decision.get("label_query_count") != 0:
        raise ValueError("inspection opened labels")
    leaf_sha256 = replayed.get("leaf_sha256")
    artifact_map_sha256 = replayed.get("artifact_map_sha256")
    if type(leaf_sha256) is not dict or type(artifact_map_sha256) is not str:
        raise ValueError("inspector must return the Exp465 replay payload")
    return {
        "experiment": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "kind": FROZEN_CONFIG_ANTIGRAVITY.kind,
        "context": replayed["context"],
        "method_fold_metrics": replayed["method_fold_metrics"],
        "failed_folds": replayed["failed_folds"],
        "method_summaries": replayed["method_summaries"],
        "decision": decision,
        "dashboard_snapshot": replayed["dashboard_snapshot"],
        "leaf_sha256": leaf_sha256,
        "artifact_map_sha256": artifact_map_sha256,
        "artifact_sha256": replayed["artifact_sha256"],
    }


def _blocked_fold_indices_antigravity() -> tuple[tuple[object, object], ...]:
    import numpy as np

    all_indices = np.arange(FROZEN_CONFIG_ANTIGRAVITY.normal_rows, dtype=np.int64)
    folds = tuple(np.asarray(values, dtype=np.int64) for values in np.array_split(all_indices, FROZEN_CONFIG_ANTIGRAVITY.folds))
    if tuple(int(values.size) for values in folds) != (14, 14, 14, 14, 13):
        raise ValueError("frozen blocked folds do not cover 69 normal rows")
    result: list[tuple[object, object]] = []
    for query in folds:
        fit = all_indices[~np.isin(all_indices, query)]
        if int(fit.size) + int(query.size) != FROZEN_CONFIG_ANTIGRAVITY.normal_rows:
            raise ValueError("blocked fold indices are invalid")
        result.append((fit, query))
    return tuple(result)


def _evaluate_at_root_antigravity(root: Path) -> dict[str, object]:
    """Run the frozen Task 2→4 flow once into a new private result root."""
    import numpy as np

    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
        capture_exp463_authority_antigravity,
    )

    from .artifacts_antigravity import PublishedQualificationAntigravity, publish_result_antigravity
    from .evaluation_antigravity import QualificationDecisionAntigravity, evaluate_qualification_antigravity
    from .interpolation_antigravity import interpolate_to_140_antigravity
    from .qualification_antigravity import Exp465QualificationResultAntigravity, qualify_methods_antigravity
    from .scoring_antigravity import score_fold_methods_antigravity
    from .sources_antigravity import load_ecg200_normal_train_antigravity

    _validate_create_only_root_antigravity(root)
    normal96 = load_ecg200_normal_train_antigravity(_TRAIN_PATH_ANTIGRAVITY)
    train140 = interpolate_to_140_antigravity(normal96)
    if train140.shape != (
        FROZEN_CONFIG_ANTIGRAVITY.normal_rows,
        FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,
    ) or not np.isfinite(train140).all():
        raise ValueError("frozen ECG200 TRAIN interpolation output is invalid")
    authority = capture_exp463_authority_antigravity()
    records: list[tuple[int, int, object]] = []
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        for fold_index, (fit_indices, query_indices) in enumerate(_blocked_fold_indices_antigravity()):
            records.append((
                seed,
                fold_index,
                score_fold_methods_antigravity(
                    train140,
                    fit_indices,
                    query_indices,
                    seed=seed,
                    fold_index=fold_index,
                    authority=authority,
                ),
            ))
    qualification = qualify_methods_antigravity(tuple(records))
    if type(qualification) is not Exp465QualificationResultAntigravity:
        raise ValueError("qualifier must return the exact Exp465 qualification")
    decision = evaluate_qualification_antigravity(qualification)
    if type(decision) is not QualificationDecisionAntigravity:
        raise ValueError("evaluator must return the exact Exp465 decision")
    publication = publish_result_antigravity(root, qualification, decision)
    if (
        type(publication) is not PublishedQualificationAntigravity
        or publication.root != root
        or len(publication.leaf_sha256) != len(_LEAVES_ANTIGRAVITY)
    ):
        raise ValueError("publisher must create the exact seven-leaf result")
    payload = _result_payload_from_root_antigravity(
        root,
        _inspect_persisted_root_antigravity(root),
    )
    if payload["decision"]["label_query_count"] != 0:  # type: ignore[index]
        raise ValueError("published result opened labels")
    return payload


def _inspect_at_root_antigravity(root: Path) -> dict[str, object]:
    return _result_payload_from_root_antigravity(
        root,
        _inspect_persisted_root_antigravity(root),
    )


def _load_evaluate_antigravity():
    return _evaluate_at_root_antigravity


def _load_inspect_antigravity():
    return _inspect_at_root_antigravity


def _emit_payload_antigravity(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main_antigravity(argv: Sequence[str] | None = None) -> int:
    args = parser_antigravity().parse_args(None if argv is None else list(argv))
    root = _private_test_root_antigravity()
    if args.command == "evaluate":
        _validate_create_only_root_antigravity(root)
        _validate_clean_current_commit_antigravity(args.reviewed_commit)
        payload = _load_evaluate_antigravity()(root)
    else:
        payload = _load_inspect_antigravity()(root)
    _emit_payload_antigravity(payload)
    return 0


__all__ = ("OFFICIAL_ROOT_ANTIGRAVITY", "main_antigravity", "parser_antigravity")
