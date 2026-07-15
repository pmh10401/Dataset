from __future__ import annotations

import csv
from pathlib import Path

# User-facing score source names
SOURCE_DISPLAY_NAMES = {
    "spectral_derivative": "미분·스펙트럼 통계 점수",
    "shapelet_normal_prototype": "부분 형태 일치 점수",
    "interval_quantile": "구간 분포 점수",
}

# The 13 datasets computed in Exp46
EXP46_DATASETS = {
    "Computers_normal_1",
    "Computers_normal_2",
    "ElectricDevices_normal_4",
    "ElectricDevices_normal_5",
    "LargeKitchenAppliances_normal_1",
    "LargeKitchenAppliances_normal_2",
    "LargeKitchenAppliances_normal_3",
    "RefrigerationDevices_normal_2",
    "RefrigerationDevices_normal_3",
    "ScreenType_normal_1",
    "ScreenType_normal_2",
    "ScreenType_normal_3",
    "SmallKitchenAppliances_normal_2",
}


def is_target(row: dict) -> bool:
    """
    Exp160 target selection criteria:
    - n_train >= 5
    - candidate_count == 0
    """
    try:
        n_train = int(row.get("n_train", 0))
    except (ValueError, TypeError):
        n_train = 0

    try:
        candidate_count = int(row.get("candidate_count", 0))
    except (ValueError, TypeError):
        candidate_count = 0

    return n_train >= 5 and candidate_count == 0


def detect_computed_exp99(rows: list[dict]) -> set[str]:
    """
    Detects which datasets in Exp99 results were actually computed
    (i.e., not a passthrough row).
    A dataset is computed only if a row explicitly marks spectral_target == 1
    or research_target == 1. Selector prose is not accepted as evidence.
    """
    computed = set()
    for row in rows:
        dataset_name = row.get("dataset_name")
        if not dataset_name:
            continue

        spectral_target = 0
        try:
            spectral_target = float(row.get("spectral_target", 0) or 0)
        except ValueError:
            pass

        research_target = 0
        try:
            research_target = float(row.get("research_target", 0) or 0)
        except ValueError:
            pass

        if spectral_target > 0 or research_target > 0:
            computed.add(dataset_name)
    return computed


def detect_computed_exp101(rows: list[dict]) -> set[str]:
    """
    Detects which datasets in Exp101 results were actually computed
    (i.e., not a passthrough row).
    A dataset is computed only if a row explicitly marks shapelet_target == 1
    or research_target == 1. Selector prose is not accepted as evidence.
    """
    computed = set()
    for row in rows:
        dataset_name = row.get("dataset_name")
        if not dataset_name:
            continue

        shapelet_target = 0
        try:
            shapelet_target = float(row.get("shapelet_target", 0) or 0)
        except ValueError:
            pass

        research_target = 0
        try:
            research_target = float(row.get("research_target", 0) or 0)
        except ValueError:
            pass

        if shapelet_target > 0 or research_target > 0:
            computed.add(dataset_name)
    return computed


def detect_computed_exp46(rows: list[dict]) -> set[str]:
    """
    Detects which datasets in Exp46 results were computed.
    For Exp46, all rows in the results file correspond to actual computed datasets (13 in total).
    """
    computed = set()
    for row in rows:
        dataset_name = row.get("dataset_name")
        if dataset_name:
            computed.add(dataset_name)
    # Double check that it matches or is subset of EXP46_DATASETS
    return computed.intersection(EXP46_DATASETS)


def is_policy_eligible(source_key: str, dataset_name: str, exp99_computed: set[str], exp101_computed: set[str], exp46_computed: set[str]) -> bool:
    """
    Determines if the alternative source is policy-eligible.
    - spectral_derivative: Always False (provenance from research/family target is ineligible)
    - shapelet_normal_prototype: Always False (provenance from research/shapelet target is ineligible)
    - interval_quantile: Always False for this historical artifact because its
      13-dataset coverage was selected by family/difficulty diagnostics.
    """
    if source_key == "spectral_derivative":
        return False
    elif source_key == "shapelet_normal_prototype":
        return False
    elif source_key == "interval_quantile":
        return False
    return False


def parse_indices(indices_str: str) -> set[int]:
    """
    Parses a string representing index values into a set of integers.
    Supports delimiters like space, comma, semicolon.
    """
    if not indices_str or not isinstance(indices_str, str):
        return set()
    normalized = indices_str.replace(",", " ").replace(";", " ")
    parts = normalized.split()
    out = set()
    for p in parts:
        try:
            out.add(int(p))
        except ValueError:
            pass
    return out


def calculate_legacy_index_overlap(
    source_key: str,
    dataset_name: str,
    exp_rows: list[dict],
    is_computed: bool
) -> dict | None:
    """
    Calculates index overlap for diagnostic purposes on research_target rows.
    Must only be calculated if the source was actually computed (is_computed is True)
    and if it is marked as research_target == 1 in the source CSV.
    Otherwise, returns None to avoid using passthrough rows as evidence.
    """
    if not is_computed:
        return None

    # Filter rows for this dataset
    ds_rows = [r for r in exp_rows if r.get("dataset_name") == dataset_name]
    if not ds_rows:
        return None

    # Check if research_target == 1
    research_target = 0
    for r in ds_rows:
        try:
            rt = float(r.get("research_target", 0) or 0)
            if rt > 0:
                research_target = 1
                break
        except ValueError:
            pass

    if research_target == 0:
        return None

    # Find the target configuration row
    # Exp99: research_spectral_q98_cap2
    # Exp101: research_shapelet_q98_cap2
    target_selector = ""
    if source_key == "spectral_derivative":
        target_selector = "research_spectral_q98_cap2"
    elif source_key == "shapelet_normal_prototype":
        target_selector = "research_shapelet_q98_cap2"
    else:
        return None

    source_row = next((r for r in ds_rows if r.get("selector_name") == target_selector), None)
    baseline_row = next((r for r in ds_rows if r.get("selector_name") == "baseline_exp93_operating"), None)

    if not source_row or not baseline_row:
        return None

    source_indices = parse_indices(source_row.get("selected_indices", ""))
    baseline_indices = parse_indices(baseline_row.get("selected_indices", ""))

    overlap = source_indices.intersection(baseline_indices)

    return {
        "dataset_name": dataset_name,
        "source_name": source_key,
        "base_indices": sorted(list(baseline_indices)),
        "source_indices": sorted(list(source_indices)),
        "overlap_indices": sorted(list(overlap)),
        "overlap_count": len(overlap),
    }
