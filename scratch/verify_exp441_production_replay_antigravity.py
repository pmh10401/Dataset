#!/usr/bin/env python3
"""Final label-free parity audit for the frozen Exp441 source and selector."""
from __future__ import annotations

import ast
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sqlite3
import sys
from time import perf_counter
from typing import Callable, Mapping
from unittest.mock import patch


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(ROOT_ANTIGRAVITY))

from experiments.exp441_hard_empty_single_shot_recovery_antigravity import (  # noqa: E402
    sources_antigravity as sources,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.selector_antigravity import (  # noqa: E402
    select_hard_empty_single_shot_antigravity,
    selected_coordinate_identity_sha256_antigravity,
)


EXPECTED_COUNTS_ANTIGRAVITY = (1117, 840, 394, 73, 49, 30)
EXPECTED_HARD_IDENTITY_ANTIGRAVITY = (
    "858e24e6aa182e78985f571778be02bcc87332639958fa887515c24074654322"
)
EXPECTED_SELECTED_IDENTITY_ANTIGRAVITY = (
    "db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048"
)
REFERENCE_NAME_ANTIGRAVITY = (
    "_verify_frozen_sources_single_worker_reference_antigravity"
)


def _explode_sqlite_antigravity(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("Exp441 label-free replay attempted sqlite3.connect")


def _forbidden_reference_uses_antigravity(
    tree: ast.AST, *, allow_private_definition: bool = False,
) -> tuple[str, ...]:
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == REFERENCE_NAME_ANTIGRAVITY:
            if not allow_private_definition:
                hits.append("definition")
        elif isinstance(node, ast.Name) and node.id == REFERENCE_NAME_ANTIGRAVITY:
            hits.append("name")
        elif isinstance(node, ast.Attribute) and node.attr == REFERENCE_NAME_ANTIGRAVITY:
            hits.append("attribute")
        elif isinstance(node, ast.ImportFrom) and any(
            alias.name == REFERENCE_NAME_ANTIGRAVITY for alias in node.names
        ):
            hits.append("from-import")
        elif isinstance(node, ast.Import) and any(
            alias.name == REFERENCE_NAME_ANTIGRAVITY or alias.asname == REFERENCE_NAME_ANTIGRAVITY
            for alias in node.names
        ):
            hits.append("import")
    return tuple(hits)


def _audit_reference_firewall_antigravity() -> tuple[str, ...]:
    package = Path(sources.__file__).parent
    checked: list[str] = []
    paths = tuple(sorted(package.glob("*.py"))) + (
        ROOT_ANTIGRAVITY / "scratch" / "run_exp441_hard_empty_single_shot_recovery_antigravity.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        hits = _forbidden_reference_uses_antigravity(
            tree, allow_private_definition=path.name == "sources_antigravity.py",
        )
        if hits:
            raise AssertionError(f"sequential reference leaked into {path}: {hits}")
        checked.append(str(path.relative_to(ROOT_ANTIGRAVITY)))
    return tuple(checked)


def _population_counts_antigravity(snapshot: object, selection: object) -> tuple[int, ...]:
    return (
        len(snapshot.context_datasets),
        len(snapshot.provisional_candidates),
        len(selection.outside_hard_candidates),
        len(selection.eligible_candidates),
        len(selection.winners),
        len(selection.selected),
    )


def _require_finite_evidence_antigravity(selection: object) -> None:
    for candidate in selection.provisional_candidates:
        if not (
            math.isfinite(candidate.second_excess_ratio)
            and math.isfinite(candidate.second_empirical_rank)
        ):
            raise AssertionError(f"nonfinite candidate evidence: {candidate.coordinate}")


def main_antigravity() -> int:
    checked_paths = _audit_reference_firewall_antigravity()
    production_parser_calls: list[Mapping[str, int]] = []
    production_events: list[Mapping[str, object]] = []
    original_parser: Callable[..., object] = sources._parse_records_fixed_workers_antigravity
    original_reference: Callable[..., object] = (
        sources._verify_frozen_sources_single_worker_reference_antigravity
    )

    def observed_parser_antigravity(tasks: object, observer: Callable[[Mapping[str, object]], None]) -> object:
        worker_count = sources._fixed_worker_count_antigravity()
        production_parser_calls.append({"tasks": len(tasks), "workers": worker_count})

        def observed_event_antigravity(event: Mapping[str, object]) -> None:
            production_events.append(dict(event))
            observer(event)

        return original_parser(tasks, observed_event_antigravity)

    def forbidden_reference_call_antigravity(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("production route invoked sequential reference")

    production_started = perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(sqlite3, "connect", side_effect=_explode_sqlite_antigravity))
        stack.enter_context(
            patch.object(
                sources,
                "_parse_records_fixed_workers_antigravity",
                new=observed_parser_antigravity,
            )
        )
        reference_spy = stack.enter_context(
            patch.object(
                sources,
                REFERENCE_NAME_ANTIGRAVITY,
                side_effect=forbidden_reference_call_antigravity,
            )
        )
        production_snapshot = sources.verify_frozen_sources_antigravity()
        reference_spy.assert_not_called()
    production_seconds = perf_counter() - production_started

    if production_parser_calls != [{"tasks": 1117, "workers": 11}]:
        raise AssertionError(f"production worker route mismatch: {production_parser_calls!r}")
    if len(production_events) != 1117:
        raise AssertionError(f"production progress event count mismatch: {len(production_events)}")
    if tuple(event["sequence"] for event in production_events) != tuple(range(1117)):
        raise AssertionError("production events are not in frozen context order")

    reference_started = perf_counter()
    with patch.object(sqlite3, "connect", side_effect=_explode_sqlite_antigravity):
        reference_snapshot = original_reference()
    reference_seconds = perf_counter() - reference_started

    if production_snapshot != reference_snapshot:
        raise AssertionError("11-worker and sequential source snapshots differ")
    production_selection = select_hard_empty_single_shot_antigravity(production_snapshot)
    reference_selection = select_hard_empty_single_shot_antigravity(reference_snapshot)
    if production_selection.to_payload() != reference_selection.to_payload():
        raise AssertionError("11-worker and sequential selection payloads differ")

    counts = _population_counts_antigravity(production_snapshot, production_selection)
    if counts != EXPECTED_COUNTS_ANTIGRAVITY:
        raise AssertionError(f"frozen population counts differ: {counts!r}")
    if production_snapshot.hard_coordinate_identity_sha256 != EXPECTED_HARD_IDENTITY_ANTIGRAVITY:
        raise AssertionError("Hard coordinate identity differs")
    selected_identity = selected_coordinate_identity_sha256_antigravity(
        production_selection.selected
    )
    if selected_identity != EXPECTED_SELECTED_IDENTITY_ANTIGRAVITY:
        raise AssertionError("selected coordinate identity differs")
    if production_snapshot.label_access_count != 0 or reference_snapshot.label_access_count != 0:
        raise AssertionError("label-free replay recorded label access")
    _require_finite_evidence_antigravity(production_selection)

    result = {
        "ast_firewall_checked_files": list(checked_paths),
        "ast_firewall_status": "pass",
        "call_spy_status": "pass",
        "counts": {
            "datasets": counts[0],
            "provisional_candidates": counts[1],
            "outside_hard_candidates": counts[2],
            "eligible_candidates": counts[3],
            "winner_datasets": counts[4],
            "selected_candidates": counts[5],
        },
        "finite_evidence": True,
        "hard_coordinate_identity_sha256": production_snapshot.hard_coordinate_identity_sha256,
        "label_access_count": 0,
        "production_committed_events": len(production_events),
        "production_elapsed_seconds": production_seconds,
        "production_workers": production_parser_calls[0]["workers"],
        "reference_elapsed_seconds": reference_seconds,
        "selected_coordinate_identity_sha256": selected_identity,
        "snapshot_payload_parity": True,
        "selection_payload_parity": True,
        "sqlite_connect_guard": "pass",
        "status": "pass",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
