#!/usr/bin/env python3
"""Bounded, values-only source-AST diagnosis for the paper R4 route.

This program deliberately never imports an historical experiment module and
never reads TEST labels.  It SHA-verifies the frozen source text, compiles only
the named pure AST definitions, and compares their values with the paper
implementation on two representative datasets.  Oracle coordinates are loaded
only after all score and selector calculations have finished, for reporting
only; they are never an input to a scoring or selector primitive.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import subprocess
from typing import Iterable, Mapping, Optional, Sequence, Set

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from paper.equipment_anomaly_detection.contracts_paper import HardRouteParityAuthority
from paper.equipment_anomaly_detection.data_paper import load_dataset_values
from paper.equipment_anomaly_detection.hard_route_paper import (
    replay_exp133_134_135_137,
    replay_exp74d_89_90_93,
)
from paper.equipment_anomaly_detection.proposal_score_paper import load_hard_route_oracle_bundle
from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import (
    build_legacy_route_evidence,
    count_cap_threshold_paper,
)
from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import (
    align_series_lengths_paper,
    repair_linear_edge_nearest_paper,
    z_normalize_paper,
)


SOURCE_ROOT = Path("/Users/mac/Documents/Time Series")
ORIGINS = {
    "exp26": ("run_experiment_26_rocket.py", "e11e1e7ed19d4d14112b9af9e72e36f3e1bc746b3bba5a74335bf713773b2eb6"),
    "exp29": ("run_experiment_29_train_normal_threshold_calibration.py", "6f7018ead920254bf00145639f9cdb8ffd39612bb2248980415bf6a6c736c3de"),
    "balanced": ("run_balanced_improvement_experiment.py", "d784f8dcc3a5f05ef7d96829acbe9b9bb206ed5f37db0a5434bd848657f48efd"),
    "exp40": ("run_experiment_40_original_score_normalization_sweep.py", "377109789d95791bdc606c7bed08843ea1a8814cca1710af78a1178926c2c5cb"),
    "exp60": ("run_experiment_60_62_rocket_imaging_selector_variants.py", "4ab38ef8a0ad152ccfc18c1297c40aca31ba79ac29348b9f5da029682507b2ed"),
    "exp89": ("run_experiment_89_74d_with_exp84_candidate.py", "9f591ed8a17c31a15cb6edb2579379d06fa74a49bf4508e488499270ea2c8d08"),
    "exp90": ("run_experiment_90_zero_f1_repair_selector.py", "c65c933587ec68a9c898fb2eb21595ab3d752685d22f69ae8f11413ecc8f3ee0"),
    "exp93": ("run_experiment_93_nonpos_candidate_reranker.py", "5b8b4e047875b430e53a40217d71cad7a1cfdd151903ef8eb63548df882d4b89"),
    "model_hard": ("run_model_hard_research_experiments.py", "f9f06973009259590b4afd165dc275cff9b834a628ccb17e248d015d55680194"),
    "exp131": ("run_experiment_131_rocket_block_b_calibration.py", "32be94dc3fa015df334991a4c301106cb78a2f62cf83b38e505ccfe90544fe5d"),
    "exp132": ("run_experiment_132_block_b_review_integration.py", "358122725c41891464b85310728207bbbaeeabf7dd66a7724ed68e9d4207dbca"),
    "rank": ("run_rank_ensemble_calibration.py", "1191d05fabd85ec174b78eee537710e2748e1c89faca7ec824894831df3d6a9a"),
}

DEFAULT_DATASETS = ("ACSF1_normal_0", "ChlorineConcentration_normal_3")
FROZEN_EXP137_COMMIT = "3b118bf7a65543c7ccc23889d51e5765e5eff645"
FROZEN_EXP137_SOURCE = "run_experiment_137_operational_triage.py"
FROZEN_EXP137_SOURCE_SHA256 = "5e389de7ee26063028b893d4856c9f1e24897f8fae32780ed2437bb90dd3b04c"
SCORE_CACHE_MANIFEST = Path(
    "/Users/mac/Documents/Dataset/experiments/exp234_recovery_antigravity/"
    "score_source_antigravity/01_score_cache_manifest_antigravity.csv"
)
SCORE_CACHE_MANIFEST_SHA256 = "bf73796c046a6102fce53d3b6fc606afa1ed0a5161ebb6e92d94698ab56a68a0"
AUTHORITY = HardRouteParityAuthority(
    exp137_path=Path("/Users/mac/Documents/Dataset/experiments/exp234_recovery_antigravity/experiment_137_operational_triage_results_antigravity.csv"),
    exp137_sha256="942e2f5faac41364a0e91fd9dad7c3e45734cd765a080f31d9d149950826735d",
    exp236_path=Path("/Users/mac/Documents/Dataset/experiments/exp236_direct_run_antigravity/exp236_frozen_predictions.jsonl"),
    exp236_sha256="a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524",
    exp291_path=Path("/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl"),
    exp291_sha256="73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848",
    exp404_path=Path("/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl"),
    exp404_sha256="437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70",
)


def _verified_source(key: str) -> tuple[Path, ast.Module]:
    filename, expected = ORIGINS[key]
    path = SOURCE_ROOT / filename
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise RuntimeError(f"frozen source hash mismatch for {filename}: {actual}")
    return path, ast.parse(payload.decode("utf-8"), filename=str(path))


def _read_verified_file_bytes(path: Path, expected_sha256: str, *, artifact: str) -> bytes:
    """Hash and parse one immutable diagnostic artifact snapshot without reopening its path."""
    path = Path(path)
    try:
        payload = path.read_bytes()
    except FileNotFoundError as error:
        raise RuntimeError(f"{artifact} is unavailable: {path}") from error
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{artifact} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    return payload


def _verified_frozen_exp137_source() -> ast.Module:
    """Read the source blob that created the frozen Exp137 CSV without importing it."""
    result = subprocess.run(
        ["git", "show", f"{FROZEN_EXP137_COMMIT}:{FROZEN_EXP137_SOURCE}"],
        cwd=SOURCE_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(
            "frozen Exp137 source blob is unavailable: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    payload = bytes(result.stdout)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != FROZEN_EXP137_SOURCE_SHA256:
        raise RuntimeError(
            "frozen Exp137 source SHA-256 mismatch: "
            f"expected {FROZEN_EXP137_SOURCE_SHA256}, got {actual}"
        )
    return ast.parse(payload.decode("utf-8"), filename=f"{FROZEN_EXP137_COMMIT}:{FROZEN_EXP137_SOURCE}")


def _frozen_exp137_route_tiers() -> object:
    tree = _verified_frozen_exp137_source()
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "route_tiers"
    ]
    if len(selected) != 1:
        raise RuntimeError("frozen Exp137 route_tiers definition is unavailable")
    run_one = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_one"
    ]
    if len(run_one) != 1 or not any(
        isinstance(node, ast.Compare)
        and ast.unparse(node) == "high | standard != source_hard"
        for node in ast.walk(run_one[0])
    ):
        raise RuntimeError("frozen Exp137 source_hard union invariant is unavailable")
    namespace: dict[str, object] = {}
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            f"{FROZEN_EXP137_COMMIT}:{FROZEN_EXP137_SOURCE}",
            "exec",
        ),
        namespace,
    )
    return namespace["route_tiers"]


def _origin_functions(key: str, names: Sequence[str], namespace: Mapping[str, object]) -> dict[str, object]:
    path, tree = _verified_source(key)
    wanted = set(names)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in selected} != wanted:
        raise RuntimeError(f"source AST definitions unavailable in {path.name}: {sorted(wanted - {node.name for node in selected})}")
    scope = dict(namespace)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return {name: scope[name] for name in names}


def _literal_assignment(key: str, name: str) -> object:
    _, tree = _verified_source(key)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"source constant unavailable: {key}.{name}")


def _base_namespace() -> dict[str, object]:
    return {
        "np": np,
        "math": math,
        "Counter": Counter,
        "PCA": PCA,
        "NearestNeighbors": NearestNeighbors,
        "RobustScaler": RobustScaler,
        "logging": logging,
        "Optional": Optional,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "Set": Set,
        "Iterable": Iterable,
    }


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))) if left.size else 0.0


def _score_comparison(origin: tuple[np.ndarray, np.ndarray], paper: tuple[np.ndarray, np.ndarray], *, rate: float) -> dict[str, object]:
    origin_train, origin_test = (np.asarray(values, dtype=np.float64) for values in origin)
    paper_train, paper_test = (np.asarray(values, dtype=np.float64) for values in paper)
    threshold, _, _ = count_cap_threshold_paper(origin_train, rate)
    origin_candidate = frozenset(np.flatnonzero(origin_test > threshold).astype(int).tolist())
    paper_threshold, _, _ = count_cap_threshold_paper(paper_train, rate)
    paper_candidate = frozenset(np.flatnonzero(paper_test > paper_threshold).astype(int).tolist())
    return {
        "train_exact": bool(np.array_equal(origin_train, paper_train)),
        "test_exact": bool(np.array_equal(origin_test, paper_test)),
        "train_max_abs": _max_abs(origin_train, paper_train),
        "test_max_abs": _max_abs(origin_test, paper_test),
        "origin_candidate": sorted(origin_candidate),
        "paper_candidate": sorted(paper_candidate),
        "candidate_equal": origin_candidate == paper_candidate,
    }


def _source_primitives() -> dict[str, object]:
    common = _base_namespace()
    exp26 = _origin_functions(
        "exp26",
        ("make_kernels", "dilated_convolution", "rocket_transform"),
        {**common, "NUM_KERNELS": 256, "RNG_SEED": 20260706},
    )
    exp29 = _origin_functions(
        "exp29",
        ("reference_train", "knn_score_pair"),
        {**common, "MAX_TRAIN_REFERENCE": 1000, "RNG_SEED": 20260706},
    )
    balanced = _origin_functions("balanced", ("density_knn_score_pair",), common)
    exp40 = _origin_functions("exp40", ("clean_scores", "count_cap_threshold"), common)
    rank = _origin_functions("rank", ("sanitize_series", "resample_series", "align_series_lengths", "z_normalize"), common)

    for collection in (exp29,):
        collection["reference_train"].__globals__.update({"MAX_TRAIN_REFERENCE": 1000, "RNG_SEED": 20260706})
    model = _origin_functions(
        "model_hard",
        (
            "resample_rows", "fixed_width_features", "pca_pair", "minmax_scale_pair",
            "prepare_series_pair_for_scale", "scale_feature_pair", "rp_images",
            "spectrogram_images", "glcm_image_features", "image_stack_for_kind",
            "imaging_features", "score_pair_for_config", "threshold_selection_diagnostics",
            "aeon_collection", "numpy_feature_matrix", "aeon_fit_collection",
            "aeon_multirocket_features", "aeon_hydra_features", "aeon_feature_pair",
            "stable_tail_feature_prune", "maybe_prune_features",
        ),
        {**common, "RNG_SEED": 20260707},
    )
    for function in model.values():
        function.__globals__.update(
            {
                "RNG_SEED": 20260707,
                "knn_score_pair": exp29["knn_score_pair"],
                "scale_feature_pair": model["scale_feature_pair"],
                "imaging_features": model["imaging_features"],
                "pca_pair": model["pca_pair"],
                "image_stack_for_kind": model["image_stack_for_kind"],
                "rp_images": model["rp_images"],
                "spectrogram_images": model["spectrogram_images"],
                "glcm_image_features": model["glcm_image_features"],
                "fixed_width_features": model["fixed_width_features"],
                "minmax_scale_pair": model["minmax_scale_pair"],
                "aeon_collection": model["aeon_collection"],
                "numpy_feature_matrix": model["numpy_feature_matrix"],
                "aeon_fit_collection": model["aeon_fit_collection"],
                "aeon_multirocket_features": model["aeon_multirocket_features"],
                "aeon_hydra_features": model["aeon_hydra_features"],
                "aeon_feature_pair": model["aeon_feature_pair"],
                "stable_tail_feature_prune": model["stable_tail_feature_prune"],
                "maybe_prune_features": model["maybe_prune_features"],
                "density_knn_score_pair": balanced["density_knn_score_pair"],
                "AEON_N_JOBS": 1,
            }
        )
    exp131 = _origin_functions("exp131", ("fixed_reference",), common)
    exp132 = _origin_functions("exp132", ("fixed_reference", "block_scores"), common)
    for collection in (exp131, exp132):
        for function in collection.values():
            function.__globals__.update(
                {
                    "MAX_TRAIN_REFERENCE": 1000,
                    "RNG_SEED": 20260706,
                    "REFERENCE_SEED_OFFSET": 130,
                    "make_kernels": exp26["make_kernels"],
                    "rocket_transform": exp26["rocket_transform"],
                    "density_knn_score_pair": balanced["density_knn_score_pair"],
                }
            )
    selector = _origin_functions(
        "exp60",
        (
            "selection", "indices_at_least", "cap_indices", "cap_indices_count",
            "exp69_base_selection", "rank_ensemble_indices", "top_score_indices",
            "rank_ensemble_guarded_indices", "large_data_budget", "top_score_order",
        ),
        common,
    )
    for function in selector.values():
        function.__globals__.update(selector)
    exp89 = _origin_functions(
        "exp89", ("parse_indices", "as_float", "exp84_bundle_from_row", "choose_rows_for_dataset"), common,
    )
    exp90 = _origin_functions(
        "exp90",
        ("first_top_index_from_bundle", "top_indices_from_bundle", "top_indices_from_exp84", "rank_union", "choose_rows_for_dataset"),
        common,
    )
    exp93 = _origin_functions(
        "exp93",
        ("parse_order", "bundle_order", "exp84_order", "rank_map", "candidate_pool", "score_candidates", "sorted_candidates", "top_non_base", "weak_base_indices", "choose_rows_for_dataset"),
        {**common, "MAX_TOP": 12},
    )
    return {
        **exp26, **exp29, **balanced, **exp40, **rank, **model, **exp131, **exp132, **selector,
        "exp89_parse_indices": exp89["parse_indices"],
        "exp89_as_float": exp89["as_float"],
        "exp89_exp84_bundle": exp89["exp84_bundle_from_row"],
        "exp89_choose": exp89["choose_rows_for_dataset"],
        "exp90_choose": exp90["choose_rows_for_dataset"],
        "exp93_choose": exp93["choose_rows_for_dataset"],
    }


def _source_raw_arrays(values: object, primitives: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sanitize = primitives["sanitize_series"]
    align = primitives["align_series_lengths"]
    normalize = primitives["z_normalize"]
    train_rows = [sanitize(row) for row in values.train_rows]
    test_rows = [sanitize(row) for row in values.test_rows]
    exp40_target = max(8, int(values.metadata_length))
    route_target = min(max(8, int(round(float(np.median([len(row) for row in (*train_rows, *test_rows)]))))), 2048)
    exp40_train = align(train_rows, exp40_target)
    exp40_test = align(test_rows, exp40_target)
    route_train = align(train_rows, route_target)
    route_test = align(test_rows, route_target)
    return normalize(exp40_train).astype(np.float32), normalize(exp40_test).astype(np.float32), route_train, route_test


def _paper_exp40_arrays(values: object) -> tuple[np.ndarray, np.ndarray]:
    train = tuple(repair_linear_edge_nearest_paper(row) for row in values.train_rows)
    test = tuple(repair_linear_edge_nearest_paper(row) for row in values.test_rows)
    target = max(8, int(values.metadata_length))
    return z_normalize_paper(align_series_lengths_paper(train, target)), z_normalize_paper(align_series_lengths_paper(test, target))


def _origin_exp84_config() -> dict[str, object]:
    path, tree = _verified_source("model_hard")
    assignment = next(
        (
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "EXPERIMENT_SPECS" for target in node.targets)
        ),
        None,
    )
    if assignment is None:
        raise RuntimeError(f"source configuration unavailable in {path.name}")
    scope: dict[str, object] = {
        name.id: []
        for name in ast.walk(assignment.value)
        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Load)
    }
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), scope)
    config = scope["EXPERIMENT_SPECS"]["experiment_87_exp84_index_diagnostics"]["configs"][0]
    if not isinstance(config, dict):
        raise RuntimeError("source Exp84 configuration is invalid")
    return dict(config)


def _source_score_rows(values: object, primitives: Mapping[str, object]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, object]]:
    exp40_train, exp40_test, route_train, route_test = _source_raw_arrays(values, primitives)
    reference = primitives["reference_train"](exp40_train, exp40_train.shape[1], 256)
    kernels = primitives["make_kernels"](exp40_train.shape[1], num_kernels=256)
    rocket_features = primitives["rocket_transform"](reference, kernels)
    rocket_test_features = primitives["rocket_transform"](exp40_test, kernels)
    rocket = primitives["density_knn_score_pair"](rocket_features, rocket_test_features, 3, "local_gap")

    configs = _literal_assignment("exp60", "IMAGING_CONFIGS")
    source_model = primitives["score_pair_for_config"]
    source_prepare = primitives["prepare_series_pair_for_scale"]
    record = {"dataset_name": values.dataset}
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {"rocket_exp40": rocket}
    for name in ("exp55_best", "exp56_best"):
        config = dict(configs[name])
        train, test = source_prepare(config.get("series_scale", "per_series_z"), route_train, route_test, primitives["z_normalize"](route_train), primitives["z_normalize"](route_test))
        result[name] = source_model(train, test, route_train.shape[1], config, record)

    exp84_config = _origin_exp84_config()
    route_train_z = primitives["z_normalize"](route_train).astype(np.float32)
    route_test_z = primitives["z_normalize"](route_test).astype(np.float32)
    result["exp84"] = source_model(route_train_z, route_test_z, route_train.shape[1], exp84_config, record)

    a_train, a_test, b_train, b_test, reference_count, seed_offset = primitives["block_scores"](exp40_train, exp40_test)
    result["block_a"] = (a_train, a_test)
    result["block_b"] = (b_train, b_test)
    return result, {
        "source_exp40_train": exp40_train,
        "source_exp40_test": exp40_test,
        "source_route_train": route_train,
        "source_route_test": route_test,
        "reference_count": int(reference_count),
        "reference_seed_offset": int(seed_offset),
    }


def _source_exp74(values: object, source_scores: Mapping[str, tuple[np.ndarray, np.ndarray]], primitives: Mapping[str, object]) -> dict[str, list[int]]:
    bundles = {}
    for name in ("rocket_exp40", "exp55_best", "exp56_best"):
        train, test = source_scores[name]
        threshold, _, _ = primitives["count_cap_threshold"](train, .015)
        bundles[name] = {
            "indices": set(np.flatnonzero(np.asarray(test) > threshold).astype(int).tolist()),
            "test_scores": np.asarray(test, dtype=np.float64),
        }
    base = set(primitives["exp69_base_selection"](bundles)["indices"])
    budget_one = primitives["large_data_budget"](np.empty(len(values.test_rows)), rate=.01, minimum=2, maximum=5)
    budget_two = primitives["large_data_budget"](np.empty(len(values.test_rows)), rate=.02, minimum=3, maximum=8)
    primary = set(primitives["rank_ensemble_guarded_indices"](bundles, budget_two, "rocket_top", max(budget_two, budget_one * 2)))
    review = set(primitives["cap_indices_count"](set(primitives["rank_ensemble_indices"](bundles, budget_two)) - primary, bundles["rocket_exp40"], max(1, budget_one)))
    large = len(values.train_rows) >= 100 and len(values.test_rows) >= 100
    return {
        "hard": sorted(base | primary if large else base),
        "standard_review": sorted(review if large else set()),
        "base": sorted(base),
        "primary": sorted(primary),
        "review_precedence_unresolved": sorted(review & base),
    }


def _indices_text(indices: Iterable[int]) -> str:
    return " ".join(str(int(index)) for index in sorted(indices))


def _frozen_indices(value: object, *, field: str) -> frozenset[int]:
    text = str(value).strip()
    if not text:
        return frozenset()
    values = tuple(int(item) for item in text.split())
    if len(values) != len(set(values)) or any(index < 0 for index in values):
        raise RuntimeError(f"frozen Exp137 {field} is not a duplicate-free non-negative coordinate list")
    return frozenset(values)


def _frozen_exp93_to_exp133_boundary(
    dataset: str,
    oracle_exp137_hard: frozenset[int],
    route_tiers: object,
) -> dict[str, object]:
    """Derive Exp93's stored input from the frozen Exp137 source/result seam.

    This runs only after every raw-score and selector computation.  The frozen
    coordinates are never supplied to a scorer or selector; they prove the
    historical storage boundary through Exp137's SHA-pinned source invariant:
    ``high | standard == source_hard`` before ``route_tiers`` builds lanes.
    """
    exp137_bytes = _read_verified_file_bytes(
        AUTHORITY.exp137_path,
        AUTHORITY.exp137_sha256,
        artifact="frozen Exp137 CSV",
    )
    row: dict[str, str] | None = None
    with io.StringIO(exp137_bytes.decode("utf-8"), newline="") as handle:
        for candidate in csv.DictReader(handle):
            if candidate["dataset_name"] == dataset:
                if row is not None:
                    raise RuntimeError(f"frozen Exp137 has duplicate dataset row: {dataset}")
                row = candidate
    if row is None:
        raise RuntimeError(f"frozen Exp137 dataset row is unavailable: {dataset}")
    high = _frozen_indices(row["hard_alert_indices"], field="hard_alert_indices")
    standard = _frozen_indices(row["standard_review_indices"], field="standard_review_indices")
    priority = _frozen_indices(row["priority_review_indices"], field="priority_review_indices")
    source_exp93 = high | standard
    source_count = int(row["source_exp93_hard_count"])
    if len(source_exp93) != source_count:
        raise RuntimeError(
            "frozen Exp137 source_exp93_hard_count does not match the route_tiers input "
            f"for {dataset}: {len(source_exp93)} != {source_count}"
        )
    tiers = route_tiers(set(high), set(standard), set(priority), int(row["test_size"]))
    if frozenset(tiers["hard"]) != high or frozenset(tiers["standard_review"]) != standard:
        raise RuntimeError(f"frozen Exp137 route_tiers lane invariant failed for {dataset}")
    if high != oracle_exp137_hard:
        raise RuntimeError(f"frozen Exp137 direct oracle projection differs from CSV row for {dataset}")
    return {
        "frozen_source_commit": FROZEN_EXP137_COMMIT,
        "frozen_source_sha256": FROZEN_EXP137_SOURCE_SHA256,
        "frozen_exp93_input": sorted(source_exp93),
        "frozen_exp133_high": sorted(high),
        "frozen_exp133_standard": sorted(standard),
        "frozen_exp135_priority": sorted(priority),
        "frozen_source_exp93_hard_count": source_count,
        "route_tiers_invariant_verified": True,
        "used_as_scoring_input": False,
    }


def _score_cache_report_only(dataset: str, compared_coordinates: Iterable[int]) -> dict[str, object]:
    """Read the SHA-bound later score cache only as post-score supporting evidence."""
    manifest_bytes = _read_verified_file_bytes(
        SCORE_CACHE_MANIFEST,
        SCORE_CACHE_MANIFEST_SHA256,
        artifact="score-cache manifest",
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    rows = list(csv.DictReader(io.StringIO(manifest_bytes.decode("utf-8"), newline="")))
    matches = [row for row in rows if row["dataset_name"] == dataset]
    if len(matches) != 1:
        raise RuntimeError(f"score-cache manifest requires one row for {dataset}, got {len(matches)}")
    row = matches[0]
    cache_path = SCORE_CACHE_MANIFEST.parent / "cache" / f"antigravity_{row['cache_file']}"
    cache_bytes = _read_verified_file_bytes(
        cache_path,
        row["cache_sha256"],
        artifact=f"score-cache for {dataset}",
    )
    cache_sha256 = hashlib.sha256(cache_bytes).hexdigest()
    with np.load(io.BytesIO(cache_bytes), allow_pickle=False) as archive:
        required = {"legacy_block_b_indices", "legacy_block_c_indices"}
        if not required <= set(archive.files):
            raise RuntimeError(f"score-cache legacy block evidence unavailable for {dataset}")
        block_b = frozenset(int(value) for value in archive["legacy_block_b_indices"].tolist())
        block_c = frozenset(int(value) for value in archive["legacy_block_c_indices"].tolist())
    compared = tuple(sorted({int(index) for index in compared_coordinates}))
    return {
        "manifest_sha256": manifest_sha256,
        "cache_sha256": cache_sha256,
        "legacy_block_b_contains_compared": {str(index): index in block_b for index in compared},
        "legacy_block_c_contains_compared": {str(index): index in block_c for index in compared},
        "used_as_scoring_input": False,
    }


def _source_exp84_rows(
    evidence: object,
    source_exp84: tuple[np.ndarray, np.ndarray],
    primitives: Mapping[str, object],
) -> dict[tuple[str, str], dict[str, object]]:
    diagnostic = primitives["threshold_selection_diagnostics"]
    diagnostic.__globals__["format_indices"] = lambda values, sort=True: _indices_text(sorted(values) if sort else values)
    diagnostic.__globals__["format_float_list"] = lambda values, ndigits=6: " ".join(f"{float(value):.{ndigits}g}" for value in values)
    rows: dict[tuple[str, str], dict[str, object]] = {}
    train, test = source_exp84
    for method, rate in (
        ("family_guard_v1", evidence.exp84_family_guard.rate),
        ("count_cap_3pct", evidence.exp84_count_cap_3pct.rate),
    ):
        threshold, _, _ = primitives["count_cap_threshold"](train, rate)
        row = dict(diagnostic(test, threshold, top_n=12))
        row["train_exceed_rate"] = float(np.mean(np.asarray(train) > threshold))
        rows[(evidence.dataset, method)] = row
    return rows


def _source_selector_rows(
    values: object,
    evidence: object,
    source_scores: Mapping[str, tuple[np.ndarray, np.ndarray]],
    source_exp74: Mapping[str, list[int]],
    primitives: Mapping[str, object],
) -> dict[str, list[int]]:
    """Execute Exp89/90/93 AST selectors with values-only score bundles and row stubs."""
    bundles: dict[str, dict[str, object]] = {}
    for name in ("rocket_exp40", "exp55_best", "exp56_best"):
        train, test = source_scores[name]
        threshold, _, _ = primitives["count_cap_threshold"](train, .015)
        bundles[name] = {
            "indices": set(np.flatnonzero(np.asarray(test) > threshold).astype(int).tolist()),
            "test_scores": np.asarray(test, dtype=np.float64),
            "train_exceed_rate": float(np.mean(np.asarray(train) > threshold)),
        }
    record = {"train_series": tuple(range(len(values.train_rows)))}
    # Only the size is read by the AST selectors.  It is a synthetic shape carrier,
    # not a TEST-label vector.
    shape_only_test = np.empty(len(values.test_rows), dtype=np.float32)
    source_rows = _source_exp84_rows(evidence, source_scores["exp84"], primitives)
    source_loader = lambda *_args, **_kwargs: (record, shape_only_test, bundles)
    source_parse = primitives["exp89_parse_indices"]
    source_float = primitives["exp89_as_float"]
    source_exp84 = primitives["exp89_exp84_bundle"]
    source_passthrough = lambda row, selector_name, *_args, **_kwargs: {"selector_name": selector_name, "selected_indices": row["selected_indices"]}
    source_computed = lambda _dataset, _record, _test, _bundles, _base, selector_name, indices, *_args, **_kwargs: {"selector_name": selector_name, "selected_indices": _indices_text(indices)}
    selector_globals = {
        "CALIBRATION_PROFILES": {"relaxed_15pct": {"rocket_exp40": .015, "exp55_best": .015, "exp56_best": .015}},
        "load_candidate_predictions": source_loader,
        "is_large_data_case": lambda source_record, source_test: len(source_record["train_series"]) >= 100 and len(source_test) >= 100,
        "large_data_budget": primitives["large_data_budget"],
        "top_score_indices": primitives["top_score_indices"],
        "indices_at_least": primitives["indices_at_least"],
        "cap_indices_count": primitives["cap_indices_count"],
        "parse_indices": source_parse,
        "as_float": source_float,
        "exp84_bundle_from_row": source_exp84,
        "passthrough_row": source_passthrough,
        "computed_row": source_computed,
        "row_with_metrics": source_computed,
        "MAX_TOP": 12,
        "EXPERIMENT_ID": "diagnostic",
        "top_score_order": primitives["top_score_order"],
    }
    exp74_row = {"selected_indices": _indices_text(source_exp74["hard"]), "train_normal_count": str(len(values.train_rows))}
    review_row = {"selected_indices": _indices_text(source_exp74["standard_review"]), "train_normal_count": str(len(values.train_rows))}

    exp89 = primitives["exp89_choose"]
    exp89.__globals__.update(selector_globals)
    rows89 = exp89((values.dataset, exp74_row, review_row, source_rows))
    selected89 = next(row["selected_indices"] for row in rows89 if row["selector_name"] == "exp84_four_model_fg_plus_noalert_repair")
    exp89_row = {"selected_indices": selected89, "train_normal_count": str(len(values.train_rows))}

    exp90 = primitives["exp90_choose"]
    exp90.__globals__.update(selector_globals)
    rows90 = exp90((values.dataset, exp89_row, exp74_row, source_rows))
    selected90 = next(row["selected_indices"] for row in rows90 if row["selector_name"] == "noalert_top1_train_safe_repair")
    exp90_row = {"selected_indices": selected90, "train_normal_count": str(len(values.train_rows))}

    exp93 = primitives["exp93_choose"]
    exp93.__globals__.update(selector_globals)
    rows93 = exp93((values.dataset, exp90_row, exp90_row, source_rows))
    selected93 = next(row["selected_indices"] for row in rows93 if row["selector_name"] == "nonpos_weak_alert_replace")
    return {
        "exp89": sorted(source_parse(selected89)),
        "exp90": sorted(source_parse(selected90)),
        "exp93": sorted(source_parse(selected93)),
    }


def _diagnose_dataset(dataset: str, primitives: Mapping[str, object]) -> dict[str, object]:
    values = load_dataset_values("/Users/mac/Documents/Dataset/univariate_ts.db", dataset)
    source_scores, source_context = _source_score_rows(values, primitives)
    evidence = build_legacy_route_evidence(values)
    paper_exp40_train, paper_exp40_test = _paper_exp40_arrays(values)
    paper_scores = {
        "rocket_exp40": (evidence.primary["rocket_exp40"].train_score, evidence.primary["rocket_exp40"].test_score),
        "exp55_best": (evidence.primary["exp55_best"].train_score, evidence.primary["exp55_best"].test_score),
        "exp56_best": (evidence.primary["exp56_best"].train_score, evidence.primary["exp56_best"].test_score),
        "exp84": (evidence.exp84.train_score, evidence.exp84.test_score),
        "block_a": (evidence.confirmation_blocks.block_a.train_score, evidence.confirmation_blocks.block_a.test_score),
        "block_b": (evidence.confirmation_blocks.block_b_015.train_score, evidence.confirmation_blocks.block_b_015.test_score),
    }
    rates = {
        "rocket_exp40": .015,
        "exp55_best": .015,
        "exp56_best": .015,
        "exp84": evidence.exp84_family_guard.rate,
        "block_a": .015,
        "block_b": .015,
    }
    score_stages = {name: _score_comparison(source_scores[name], paper_scores[name], rate=rates[name]) for name in rates}
    source_exp74 = _source_exp74(values, source_scores, primitives)
    paper_exp74, paper_exp89, paper_exp90, paper_exp93 = replay_exp74d_89_90_93(evidence)
    paper_exp137 = replay_exp133_134_135_137(paper_exp93, evidence.confirmation_blocks)
    source_selectors = _source_selector_rows(values, evidence, source_scores, source_exp74, primitives)
    source_exp137_hard = sorted(set(source_selectors["exp93"]) & set(score_stages["block_b"]["origin_candidate"]))
    return {
        "dataset": dataset,
        "n_train": len(values.train_rows),
        "n_test": len(values.test_rows),
        "labels_read": False,
        "preprocess_exp40_train_exact": bool(np.array_equal(source_context["source_exp40_train"], paper_exp40_train)),
        "preprocess_exp40_test_exact": bool(np.array_equal(source_context["source_exp40_test"], paper_exp40_test)),
        "preprocess_exp40_train_max_abs": _max_abs(source_context["source_exp40_train"], paper_exp40_train),
        "preprocess_exp40_test_max_abs": _max_abs(source_context["source_exp40_test"], paper_exp40_test),
        "reference": {"count": source_context["reference_count"], "seed_offset": source_context["reference_seed_offset"]},
        "score_stages": score_stages,
        "source_exp74d": source_exp74,
        "source_exp89": source_selectors["exp89"],
        "source_exp90": source_selectors["exp90"],
        "source_exp93": source_selectors["exp93"],
        "source_exp137_hard": source_exp137_hard,
        "paper_exp74d": {"hard": sorted(paper_exp74.hard), "standard_review": sorted(paper_exp74.standard)},
        "paper_exp89": sorted(paper_exp89),
        "paper_exp90": sorted(paper_exp90),
        "paper_exp93": sorted(paper_exp93),
        "paper_exp137": {"hard": sorted(paper_exp137.hard), "standard_review": sorted(paper_exp137.standard), "priority_review": sorted(paper_exp137.priority)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", dest="datasets", choices=DEFAULT_DATASETS, help="bounded representative dataset; repeat at most twice")
    args = parser.parse_args()
    datasets = tuple(args.datasets or DEFAULT_DATASETS)
    if len(datasets) > len(DEFAULT_DATASETS) or len(set(datasets)) != len(datasets):
        raise SystemExit("diagnostic accepts each bounded representative dataset at most once")
    primitives = _source_primitives()
    results = [_diagnose_dataset(dataset, primitives) for dataset in datasets]

    # Authority is deliberately loaded only after every values-only computation above.
    oracle = load_hard_route_oracle_bundle(
        AUTHORITY,
        selected_population=datasets,
        selected_n_test={result["dataset"]: int(result["n_test"]) for result in results},
    )
    frozen_route_tiers = _frozen_exp137_route_tiers()
    for result in results:
        dataset = result["dataset"]
        result["oracle_exp137_report_only"] = sorted(oracle.exp137[dataset])
        result["paper_exp137_matches_oracle"] = result["paper_exp137"]["hard"] == result["oracle_exp137_report_only"]
        frozen_boundary = _frozen_exp93_to_exp133_boundary(
            dataset,
            oracle.exp137[dataset],
            frozen_route_tiers,
        )
        result["frozen_exp93_to_exp133_boundary_report_only"] = frozen_boundary
        current_exp93 = frozenset(int(index) for index in result["paper_exp93"])
        frozen_exp93 = frozenset(int(index) for index in frozen_boundary["frozen_exp93_input"])
        result["paper_exp93_matches_frozen_exp93_input"] = current_exp93 == frozen_exp93
        result["paper_exp93_only"] = sorted(current_exp93 - frozen_exp93)
        result["frozen_exp93_only"] = sorted(frozen_exp93 - current_exp93)
        result["score_cache_report_only"] = _score_cache_report_only(
            dataset,
            current_exp93 ^ frozen_exp93,
        )
    print(json.dumps({"schema": "paper-r4-mechanism1-parity-diagnosis.v1", "datasets": results, "origin_hashes": {key: digest for key, (_, digest) in ORIGINS.items()}, "oracle_used_after_scoring_only": True}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
