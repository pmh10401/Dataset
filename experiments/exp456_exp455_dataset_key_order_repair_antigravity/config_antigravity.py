"""Frozen, planned-only configuration for Exp456."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Exp456ConfigAntigravity:
    experiment: str
    kind: str
    run_identity: str
    run_root: str
    package: str
    approved_design_path: str
    approved_design_sha256: str
    candidates: tuple[str, ...]
    target_family_counts: tuple[tuple[str, int], ...]
    target_dataset_count: int
    baseline_dataset_count: int
    global_fn_must_be_below: int
    global_fp_max: int
    global_f1_min: float
    target_family_fn_delta_max: int
    exp454_sha256: tuple[tuple[str, str], ...]
    official_python: str
    router_inputs_label_free: bool
    mlx_assessment_required: bool


FROZEN_CONFIG_ANTIGRAVITY: Exp456ConfigAntigravity = Exp456ConfigAntigravity(
    experiment="Exp456",
    kind="exp455_dataset_key_order_repair",
    run_identity="exp456-exp455-dataset-key-order-repair-antigravity-20260817",
    run_root="/Users/mac/Documents/Dataset/exp456_exp455_dataset_key_order_repair_antigravity_20260817",
    package="experiments/exp456_exp455_dataset_key_order_repair_antigravity",
    approved_design_path="docs/superpowers/specs/2026-08-17-exp456-exp455-dataset-key-order-repair-antigravity-design.html",
    approved_design_sha256="79c6ad3e8bfad6f7f9b397fc1d20b448ebc656f198de5d449ff897a92de98247",
    candidates=("ECG5000:hydra_local_gap", "ChlorineConcentration:multirocket_local_gap"),
    target_family_counts=(("ECG5000", 5), ("ChlorineConcentration", 3)),
    target_dataset_count=8,
    baseline_dataset_count=40,
    global_fn_must_be_below=1988,
    global_fp_max=314,
    global_f1_min=0.6264199935,
    target_family_fn_delta_max=0,
    exp454_sha256=(
        ("artifact_map", "748fdc62de555279640b4ce588b3155670ebe33e13e5997d40061d96011bff65"),
        ("evaluation", "c2b6321d18ba4c98e426be96ea3d3fb0824e75926fa54ba4600634dded68e1ac"),
        ("decision", "3510a426d3ee083ccb3f8599c64d0c91cc56d0b902780c8b6a2136098cd9a595"),
    ),
    official_python="/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
    router_inputs_label_free=True,
    mlx_assessment_required=False,
)
