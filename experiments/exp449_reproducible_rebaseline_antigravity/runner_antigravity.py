"""Exp449 values-only Task4--7 composition with an exact local-coverage gate."""

from __future__ import annotations

from collections.abc import Mapping
from time import monotonic
from typing import Protocol

from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision, RawDataset
from paper.equipment_anomaly_detection.hard_route_paper import HardRouteResult
from paper.equipment_anomaly_detection.proposal_score_paper import (
    ProposalScore,
    _collect_task4_task5,
    _label_firewall,
)
from paper.equipment_anomaly_detection.routing_paper import build_hard_evidence_surface, build_policies
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import HardSourceBundle
from paper.equipment_anomaly_detection.sources_paper.low_frequency_paper import score_low_frequency
from paper.equipment_anomaly_detection.sources_paper.stability_paper import classify_stability


class Task45Collector(Protocol):
    """The paper collector boundary, including verified checkpoint inputs."""

    def __call__(
        self,
        raw: Mapping[str, RawDataset],
        backend: FrozenTask8BackendDecision,
        *,
        root: object,
        context: object,
        catalog: dict[str, object],
        acceleration_manifest: object,
    ) -> tuple[Mapping[str, HardSourceBundle], Mapping[str, HardRouteResult]]: ...


def verify_dataset_local_bijection_antigravity(
    catalog: dict[str, object],
    raw: Mapping[str, RawDataset],
    sources: Mapping[str, HardSourceBundle],
    routes: Mapping[str, HardRouteResult],
    *,
    mode: str,
) -> tuple[str, ...]:
    """Require each in-memory Task4/5 mapping to equal the ordered catalog."""
    expected = tuple(entry["dataset"] for entry in catalog["datasets"])
    if expected != tuple(sorted(set(expected))):
        raise ValueError("Exp449 catalog bijection differs")
    if tuple(raw) != expected or tuple(sources) != expected or tuple(routes) != expected:
        raise ValueError("Exp449 dataset-local catalog bijection differs")
    if mode == "full" and len(expected) != 1117:
        raise ValueError("Exp449 full catalog count differs")
    return expected


def _in_catalog_order_antigravity(
    values: Mapping[str, object], expected: tuple[str, ...]
) -> dict[str, object]:
    """Retain every returned key while placing catalog entries in canonical order."""
    rank = {dataset: index for index, dataset in enumerate(expected)}
    return dict(sorted(dict(values).items(), key=lambda item: (rank.get(item[0], len(rank)), item[0])))


def _gate_mode_antigravity(context: object) -> str:
    """Map the frozen context's official mode onto the 1,117-entry gate mode."""
    return "full" if getattr(context, "mode", None) == "official" else "smoke"


def compose_exp449_proposal_score_antigravity(
    raw_datasets: Mapping[str, RawDataset],
    acceleration_manifest: object,
    *,
    root=None,
    context: object = None,
    catalog: dict[str, object],
    collector: Task45Collector = _collect_task4_task5,
) -> ProposalScore:
    """Run Task6/7 only after exact in-memory Task4/5 catalog closure.

    The collector receives the caller-bound recovery inputs unchanged.  This
    module has no writer dependency; tests use an in-memory collector and Task
    4 owns recovery persistence through the paper collector's authority path.
    """
    _label_firewall()
    raw = dict(raw_datasets)
    if (
        not raw
        or tuple(raw) != tuple(sorted(raw))
        or any(not isinstance(values, RawDataset) or values.dataset != dataset for dataset, values in raw.items())
    ):
        raise ValueError("proposal raw dataset mapping is invalid")
    if not isinstance(catalog, dict):
        raise ValueError("Exp449 catalog is invalid")
    backend = FrozenTask8BackendDecision.from_acceleration_manifest(acceleration_manifest)
    expected = tuple(entry["dataset"] for entry in catalog.get("datasets", ()))
    started = monotonic()
    returned_sources, returned_routes = collector(
        raw,
        backend,
        root=root,
        context=context,
        catalog=catalog,
        acceleration_manifest=acceleration_manifest,
    )
    sources = _in_catalog_order_antigravity(returned_sources, expected)
    routes = _in_catalog_order_antigravity(returned_routes, expected)
    verify_dataset_local_bijection_antigravity(catalog, raw, sources, routes, mode=_gate_mode_antigravity(context))
    hard_seconds = float(monotonic() - started)
    _label_firewall()
    low_started = monotonic()
    hard_lanes = {dataset: routes[dataset].exp253.hard for dataset in raw}
    low = score_low_frequency(raw, hard_lanes)
    low_seconds = float(monotonic() - low_started)
    _label_firewall()
    stability_started = monotonic()
    stability = classify_stability(low, raw)
    stability_seconds = float(monotonic() - stability_started)
    surface = build_hard_evidence_surface(sources, routes, raw)
    policies = build_policies(surface, low, stability, raw, routes)
    _label_firewall()
    return ProposalScore(
        raw,
        sources,
        routes,
        low,
        stability,
        policies,
        backend,
        {
            "task4_task5": hard_seconds,
            "task6": low_seconds,
            "task7": stability_seconds,
            "total": float(monotonic() - started),
        },
    )


__all__ = [
    "compose_exp449_proposal_score_antigravity",
    "verify_dataset_local_bijection_antigravity",
]
