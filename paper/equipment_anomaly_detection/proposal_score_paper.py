"""Narrow Task4--7 score composition for the frozen proposal population."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import os
import sys
from time import monotonic
from types import MappingProxyType
from typing import Mapping

from .artifacts_paper import freeze_score_root, write_typed_proposal_shard
from .contracts_paper import FrozenTask8BackendDecision, LowFrequencyResult, RawDataset, Task8FrozenManifest
from .hard_route_paper import HardRouteResult, replay_hard_route
from .routing_paper import PolicyBundle, build_hard_evidence_surface, build_policies
from .sources_paper.hard_sources_paper import HardSourceBundle, score_hard_sources
from .sources_paper.low_frequency_paper import score_low_frequency
from .sources_paper.minirocket_paper import MiniRocketResult, score_minirocket_confirmations
from .sources_paper.stability_paper import StabilityPartition, classify_stability


@dataclass(frozen=True, slots=True)
class ProposalScore:
    raw: Mapping[str, RawDataset]
    sources: Mapping[str, HardSourceBundle]
    routes: Mapping[str, HardRouteResult]
    low_frequency: LowFrequencyResult
    stability: StabilityPartition
    policies: PolicyBundle
    backend: FrozenTask8BackendDecision
    stage_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        raw, sources, routes, timing = dict(self.raw), dict(self.sources), dict(self.routes), dict(self.stage_seconds)
        if not raw or tuple(raw) != tuple(sorted(raw)) or set(raw) != set(sources) or set(raw) != set(routes) or set(raw) != set(self.low_frequency.datasets):
            raise ValueError("proposal score population identities differ")
        if any(not isinstance(value, RawDataset) or value.dataset != dataset for dataset, value in raw.items()) or any(not isinstance(value, HardSourceBundle) for value in sources.values()) or any(not isinstance(value, HardRouteResult) for value in routes.values()):
            raise ValueError("proposal score typed intermediates differ")
        if not isinstance(self.stability, StabilityPartition) or not isinstance(self.policies, PolicyBundle) or not isinstance(self.backend, FrozenTask8BackendDecision) or any(type(value) is not float or value < 0.0 for value in timing.values()):
            raise ValueError("proposal score result is invalid")
        object.__setattr__(self, "raw", MappingProxyType(raw))
        object.__setattr__(self, "sources", MappingProxyType(sources))
        object.__setattr__(self, "routes", MappingProxyType(routes))
        object.__setattr__(self, "stage_seconds", MappingProxyType(timing))


def _label_firewall() -> None:
    if "paper.equipment_anomaly_detection.label_access_paper" in sys.modules:
        raise RuntimeError("label_access_paper must remain unloaded during proposal scoring")


_NUMERICAL_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextmanager
def _pin_inner_numerical_threads():
    previous = {name: os.environ.get(name) for name in _NUMERICAL_THREAD_ENV}
    try:
        for name in _NUMERICAL_THREAD_ENV:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _score_dataset_task4_task5(raw: RawDataset) -> tuple[HardSourceBundle, HardRouteResult]:
    sources = score_hard_sources(raw)
    mini = score_minirocket_confirmations(raw)
    return sources, replay_hard_route(raw, sources, mini)


def _collect_task4_task5(raw: Mapping[str, RawDataset], backend: FrozenTask8BackendDecision) -> tuple[dict[str, HardSourceBundle], dict[str, HardRouteResult]]:
    sources: dict[str, HardSourceBundle] = {}
    routes: dict[str, HardRouteResult] = {}
    if backend.dataset_workers == 1:
        for dataset in raw:
            try:
                sources[dataset], routes[dataset] = _score_dataset_task4_task5(raw[dataset])
            except Exception as error:
                raise RuntimeError(f"Task4+Task5 worker failed for dataset {dataset}: {error}") from error
        return sources, routes
    with _pin_inner_numerical_threads():
        with ProcessPoolExecutor(max_workers=backend.dataset_workers) as executor:
            futures = {dataset: executor.submit(_score_dataset_task4_task5, raw[dataset]) for dataset in raw}
            for dataset in raw:
                try:
                    sources[dataset], routes[dataset] = futures[dataset].result()
                except Exception as error:
                    raise RuntimeError(f"Task4+Task5 worker failed for dataset {dataset}: {error}") from error
    return sources, routes


def compose_proposal_score(raw_datasets: Mapping[str, RawDataset], acceleration_manifest: object) -> ProposalScore:
    """Execute Task4+5 independently, then exactly one population-global Task6 and selected-only Task7."""
    _label_firewall()
    raw = dict(raw_datasets)
    if not raw or tuple(raw) != tuple(sorted(raw)) or any(not isinstance(values, RawDataset) or values.dataset != dataset for dataset, values in raw.items()):
        raise ValueError("proposal raw dataset mapping is invalid")
    backend = FrozenTask8BackendDecision.from_acceleration_manifest(acceleration_manifest)
    started = monotonic()
    sources, routes = _collect_task4_task5(raw, backend)
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
    policies = build_policies(surface, low, stability, raw)
    _label_firewall()
    return ProposalScore(raw, sources, routes, low, stability, policies, backend, {"task4_task5": hard_seconds, "task6": low_seconds, "task7": stability_seconds, "total": float(monotonic() - started)})


def write_and_freeze_proposal_score(root, catalog: dict[str, object], proposal: ProposalScore) -> Task8FrozenManifest:
    """Pass only Task4--7 typed values to Task8's create-only shard writer."""
    if not isinstance(proposal, ProposalScore) or tuple(entry["dataset"] for entry in catalog.get("datasets", ())) != tuple(proposal.raw):
        raise ValueError("proposal score catalog identity differs")
    for dataset in proposal.raw:
        write_typed_proposal_shard(root, catalog, proposal.raw[dataset], proposal.sources[dataset], proposal.routes[dataset], proposal.low_frequency, proposal.stability, proposal.policies, proposal.backend)
    return freeze_score_root(root, catalog)


__all__ = ["ProposalScore", "compose_proposal_score", "write_and_freeze_proposal_score"]
