"""Frozen, planned-only configuration for Exp455."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Exp455ConfigAntigravity:
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


FROZEN_CONFIG_ANTIGRAVITY: Exp455ConfigAntigravity = Exp455ConfigAntigravity(
    experiment="Exp455",
    kind="conservative_label_free_routing",
    run_identity="exp455-exp454-conservative-label-free-routing-antigravity-20260817",
    run_root="/Users/mac/Documents/Dataset/exp455_exp454_conservative_label_free_routing_antigravity_20260817",
    package="experiments/exp455_exp454_conservative_label_free_routing_antigravity",
    approved_design_path="docs/superpowers/specs/2026-08-17-exp455-conservative-label-free-routing-antigravity-design.html",
    approved_design_sha256="ae391bdaa9b34b8d603722262887143e0bdf8284bdcbf0fa1b059d4841c0369a",
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
