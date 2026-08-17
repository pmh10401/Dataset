#!/usr/bin/env python3
"""Closed lifecycle for the official six-public-baseline paper comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT_ANTIGRAVITY))

CLASSICAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/paper_public_baselines_classical_v1"
)
DEEP_SVDD_SOURCE_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/paper_public_baselines_deep_partial_v1"
)
DEEP_SVDD_VIEW_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "paper_public_baselines_deep_svdd_1d_derived_full_antigravity_20260817"
)
DROCC_MPS_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "paper_public_baselines_drocc_mps_unverified_antigravity_20260817"
)
PROPOSAL_AUTHORITY_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp448_exp447_text_label_boundary_repair_antigravity_20260812"
)
OFFICIAL_COMPARISON_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "paper_public_baselines_official_comparison_antigravity_20260817"
)
EXP449_PROPOSAL_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "paper_proposal_exp449_official_antigravity_20260814"
)
EXP449_CONTINUOUS_COMPARISON_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "paper_public_baselines_official_comparison_exp449_continuous_antigravity_20260817"
)
EXP449_EVALUATION_GROUPS_ANTIGRAVITY = (
    ("classical", CLASSICAL_ROOT_ANTIGRAVITY),
    ("deep_svdd", DEEP_SVDD_VIEW_ROOT_ANTIGRAVITY),
    ("drocc_mps", DROCC_MPS_ROOT_ANTIGRAVITY),
)
COMMANDS_ANTIGRAVITY = (
    "prepare-deep-view",
    "evaluate",
    "combine",
    "verify",
    "evaluate-exp449-continuous",
    "combine-exp449-continuous",
    "verify-exp449-continuous",
)
EVALUATION_ARTIFACT_NAME_ANTIGRAVITY = "artifact_sha256.json"


def build_parser_antigravity() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS_ANTIGRAVITY:
        subparsers.add_parser(command, allow_abbrev=False)
    return parser


def main_antigravity(argv: list[str] | None = None) -> int:
    arguments = build_parser_antigravity().parse_args(argv)
    if arguments.command == "prepare-deep-view":
        result = _prepare_deep_view_antigravity()
    elif arguments.command == "evaluate":
        result = _evaluate_antigravity()
    elif arguments.command == "combine":
        result = _combine_antigravity()
    elif arguments.command == "verify":
        result = _verify_antigravity()
    elif arguments.command == "evaluate-exp449-continuous":
        result = _evaluate_exp449_continuous_antigravity()
    elif arguments.command == "combine-exp449-continuous":
        result = _combine_exp449_continuous_antigravity()
    elif arguments.command == "verify-exp449-continuous":
        result = _verify_exp449_continuous_antigravity()
    else:  # pragma: no cover - argparse closes this branch
        raise AssertionError(arguments.command)
    from experiments.paper_public_baselines.artifacts import canonical_json_bytes

    sys.stdout.write(canonical_json_bytes(result).decode("utf-8"))
    return 0


def _prepare_deep_view_antigravity() -> dict[str, object]:
    from experiments.paper_public_baselines.artifacts import sha256_file
    from experiments.paper_public_baselines.official_view_antigravity import (
        DERIVED_VIEW_PROVENANCE_NAME_ANTIGRAVITY,
        freeze_single_method_view_antigravity,
    )

    manifest = freeze_single_method_view_antigravity(
        DEEP_SVDD_SOURCE_ROOT_ANTIGRAVITY,
        DEEP_SVDD_VIEW_ROOT_ANTIGRAVITY,
        "deep_svdd_1d",
    )
    return {
        "status": "deep_svdd_view_frozen",
        "pair_count": manifest.pair_count,
        "manifest_sha256": sha256_file(
            DEEP_SVDD_VIEW_ROOT_ANTIGRAVITY / "frozen_score_manifest.json"
        ),
        "provenance_sha256": sha256_file(
            DEEP_SVDD_VIEW_ROOT_ANTIGRAVITY
            / DERIVED_VIEW_PROVENANCE_NAME_ANTIGRAVITY
        ),
    }


def _evaluate_antigravity() -> dict[str, object]:
    from experiments.paper_public_baselines.evaluation import (
        evaluate_frozen_scores,
        write_evaluation_bundle,
    )
    from experiments.paper_public_baselines.reporting import load_evaluation_bundle
    from experiments.paper_public_baselines.runner import verify_score_root

    classical = load_evaluation_bundle(CLASSICAL_ROOT_ANTIGRAVITY)
    if classical.metadata.get("method_count") != 4:
        raise ValueError("classical official evaluation is not exact")
    artifacts: dict[str, str] = {}
    for name, root in (
        ("deep_svdd", DEEP_SVDD_VIEW_ROOT_ANTIGRAVITY),
        ("drocc_mps", DROCC_MPS_ROOT_ANTIGRAVITY),
    ):
        verify_score_root(root)
        bundle = evaluate_frozen_scores(root, PROPOSAL_AUTHORITY_ROOT_ANTIGRAVITY)
        artifacts[name] = write_evaluation_bundle(root, bundle)
    return {
        "status": "deep_evaluations_complete",
        "dataset_count": 1117,
        "evaluation_artifact_map_sha256": artifacts,
    }


def _combine_antigravity() -> dict[str, object]:
    from experiments.paper_public_baselines.official_comparison_antigravity import (
        combine_official_evaluations_antigravity,
        write_official_comparison_antigravity,
    )
    from experiments.paper_public_baselines.reporting import load_evaluation_bundle

    comparison = combine_official_evaluations_antigravity(
        load_evaluation_bundle(CLASSICAL_ROOT_ANTIGRAVITY),
        load_evaluation_bundle(DEEP_SVDD_VIEW_ROOT_ANTIGRAVITY),
        load_evaluation_bundle(DROCC_MPS_ROOT_ANTIGRAVITY),
        drocc_parity_status="user_confirmed_no_difference",
    )
    artifact_sha = write_official_comparison_antigravity(
        OFFICIAL_COMPARISON_ROOT_ANTIGRAVITY, comparison
    )
    return {
        "status": "official_comparison_frozen",
        "dataset_count": comparison.dataset_count,
        "public_method_count": len(comparison.public_methods),
        "comparison_artifact_map_sha256": artifact_sha,
    }


def _verify_antigravity() -> dict[str, object]:
    from experiments.paper_public_baselines.artifacts import sha256_file
    from experiments.paper_public_baselines.official_comparison_antigravity import (
        verify_official_comparison_antigravity,
    )
    from experiments.paper_public_baselines.reporting import load_evaluation_bundle
    from experiments.paper_public_baselines.runner import verify_score_root

    score_manifests = {}
    for name, root in (
        ("classical", CLASSICAL_ROOT_ANTIGRAVITY),
        ("deep_svdd", DEEP_SVDD_VIEW_ROOT_ANTIGRAVITY),
        ("drocc_mps", DROCC_MPS_ROOT_ANTIGRAVITY),
    ):
        manifest = verify_score_root(root)
        load_evaluation_bundle(root)
        score_manifests[name] = {
            "pair_count": manifest.pair_count,
            "manifest_sha256": sha256_file(root / "frozen_score_manifest.json"),
            "evaluation_artifact_map_sha256": sha256_file(
                root
                / "evaluation_public_baselines"
                / EVALUATION_ARTIFACT_NAME_ANTIGRAVITY
            ),
        }
    comparison = verify_official_comparison_antigravity(
        OFFICIAL_COMPARISON_ROOT_ANTIGRAVITY
    )
    return {
        "status": "official_comparison_verified",
        "dataset_count": comparison.dataset_count,
        "public_methods": list(comparison.public_methods),
        "score_roots": score_manifests,
        "comparison_artifact_map_sha256": sha256_file(
            OFFICIAL_COMPARISON_ROOT_ANTIGRAVITY
            / "official_comparison_artifact_sha256_antigravity.json"
        ),
    }


def _evaluate_exp449_continuous_antigravity() -> dict[str, object]:
    from experiments.paper_public_baselines.evaluation import write_evaluation_bundle
    from experiments.paper_public_baselines.official_comparison_antigravity import (
        evaluate_frozen_scores_exp449_continuous_antigravity,
    )

    artifacts: dict[str, str] = {}
    for name, score_root in EXP449_EVALUATION_GROUPS_ANTIGRAVITY:
        bundle = evaluate_frozen_scores_exp449_continuous_antigravity(
            score_root, EXP449_PROPOSAL_ROOT_ANTIGRAVITY
        )
        artifacts[name] = write_evaluation_bundle(
            EXP449_CONTINUOUS_COMPARISON_ROOT_ANTIGRAVITY / name, bundle
        )
    return {
        "status": "exp449_continuous_evaluations_complete",
        "dataset_count": 1117,
        "test_row_count": 195900,
        "proposal_alert_count": 2276,
        "evaluation_artifact_map_sha256": artifacts,
    }


def _combine_exp449_continuous_antigravity() -> dict[str, object]:
    from experiments.paper_public_baselines.official_comparison_antigravity import (
        combine_official_evaluations_antigravity,
        write_official_comparison_antigravity,
    )
    from experiments.paper_public_baselines.reporting import load_evaluation_bundle

    bundles = tuple(
        load_evaluation_bundle(EXP449_CONTINUOUS_COMPARISON_ROOT_ANTIGRAVITY / name)
        for name, _score_root in EXP449_EVALUATION_GROUPS_ANTIGRAVITY
    )
    comparison = combine_official_evaluations_antigravity(
        *bundles,
        drocc_parity_status="user_confirmed_no_difference",
        require_proposal_continuous_antigravity=True,
    )
    artifact_sha = write_official_comparison_antigravity(
        EXP449_CONTINUOUS_COMPARISON_ROOT_ANTIGRAVITY, comparison
    )
    return {
        "status": "exp449_continuous_official_comparison_frozen",
        "dataset_count": comparison.dataset_count,
        "public_method_count": len(comparison.public_methods),
        "comparison_artifact_map_sha256": artifact_sha,
    }


def _verify_exp449_continuous_antigravity() -> dict[str, object]:
    from experiments.paper_public_baselines.artifacts import sha256_file
    from experiments.paper_public_baselines.official_comparison_antigravity import (
        OFFICIAL_CONTINUOUS_COMPARISON_SCHEMA_ANTIGRAVITY,
        load_exp449_continuous_proposal_antigravity,
        verify_official_comparison_antigravity,
    )
    from experiments.paper_public_baselines.reporting import load_evaluation_bundle
    from experiments.paper_public_baselines.runner import verify_score_root

    score_manifests: dict[str, str] = {}
    for name, score_root in EXP449_EVALUATION_GROUPS_ANTIGRAVITY:
        verify_score_root(score_root)
        evaluation = load_evaluation_bundle(
            EXP449_CONTINUOUS_COMPARISON_ROOT_ANTIGRAVITY / name
        )
        if evaluation.metadata.get("proposal_continuous_score_available") is not True:
            raise ValueError("Exp449 continuous evaluation authority differs")
        score_manifests[name] = sha256_file(score_root / "frozen_score_manifest.json")
    proposal = load_exp449_continuous_proposal_antigravity(
        EXP449_PROPOSAL_ROOT_ANTIGRAVITY
    )
    comparison = verify_official_comparison_antigravity(
        EXP449_CONTINUOUS_COMPARISON_ROOT_ANTIGRAVITY
    )
    if comparison.metadata.get("schema") != OFFICIAL_CONTINUOUS_COMPARISON_SCHEMA_ANTIGRAVITY:
        raise ValueError("Exp449 continuous comparison schema differs")
    return {
        "status": "exp449_continuous_official_comparison_verified",
        "dataset_count": comparison.dataset_count,
        "test_row_count": proposal.test_row_count,
        "proposal_alert_count": sum(map(len, proposal.alerts_by_dataset.values())),
        "proposal_manifest_sha256": proposal.manifest_sha256,
        "score_manifest_sha256": score_manifests,
        "comparison_artifact_map_sha256": sha256_file(
            EXP449_CONTINUOUS_COMPARISON_ROOT_ANTIGRAVITY
            / "official_comparison_artifact_sha256_antigravity.json"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
