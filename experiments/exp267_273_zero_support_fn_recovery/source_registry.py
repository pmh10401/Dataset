"""Frozen identities and factories for Exp268-270 score sources.

Variants identify calibration views.  They are deliberately not evidence
families: the two TCN views originate from the same fitted normal forecaster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import SOURCE_SPECS
from .forecast_rff_source import RFFNormalForecastResidual
from .forecast_tcn_source import CausalTCNNormalForecast
from .scattering_source import MultiscaleScatteringDeviation
from .signature_source import OrderedPathSignatureDeviation


@dataclass(frozen=True)
class SourceRegistration:
    """One concrete calibration view and its non-duplicable evidence identity."""

    key: str
    source: str
    variant: str
    algorithm: str
    family: str
    evidence_identity: str
    minimum_train: int
    model_based: bool
    source_version: str = "v1"


_SOURCE_BY_NAME = {spec.name: spec for spec in SOURCE_SPECS}

SOURCE_REGISTRY: dict[str, SourceRegistration] = {
    "scattering": SourceRegistration(
        key="scattering",
        source="scattering",
        variant="primary",
        algorithm="MultiscaleScatteringDeviation",
        family="scattering",
        evidence_identity="scattering",
        minimum_train=_SOURCE_BY_NAME["scattering"].minimum_train,
        model_based=False,
    ),
    "signature": SourceRegistration(
        key="signature",
        source="signature",
        variant="primary",
        algorithm="OrderedPathSignatureDeviation",
        family="signature",
        evidence_identity="signature",
        minimum_train=_SOURCE_BY_NAME["signature"].minimum_train,
        model_based=False,
    ),
    "rff_forecast": SourceRegistration(
        key="rff_forecast",
        source="rff_forecast",
        variant="primary",
        algorithm="RFFNormalForecastResidual",
        family="normal_forecast",
        evidence_identity="rff_forecast",
        minimum_train=_SOURCE_BY_NAME["rff_forecast"].minimum_train,
        model_based=False,
    ),
    "tcn_seed_median": SourceRegistration(
        key="tcn_seed_median",
        source="tcn_forecast",
        variant="seed_median",
        algorithm="CausalTCNNormalForecast",
        family="normal_forecast",
        evidence_identity="tcn_forecast",
        minimum_train=_SOURCE_BY_NAME["tcn_forecast"].minimum_train,
        model_based=True,
    ),
    "tcn_seed_intersection": SourceRegistration(
        key="tcn_seed_intersection",
        source="tcn_forecast",
        variant="seed_intersection",
        algorithm="CausalTCNNormalForecast",
        family="normal_forecast",
        evidence_identity="tcn_forecast",
        minimum_train=_SOURCE_BY_NAME["tcn_forecast"].minimum_train,
        model_based=True,
    ),
}

_ALIASES = {
    "tcn_forecast": "tcn_seed_median",
    "tcn": "tcn_seed_median",
    "rff": "rff_forecast",
}


def registration(value: str) -> SourceRegistration:
    """Return a frozen concrete variant registration."""
    key = _ALIASES.get(str(value), str(value))
    try:
        return SOURCE_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"unknown Exp267-270 source variant: {value}") from exc


def source_variants(source: str) -> tuple[SourceRegistration, ...]:
    """Return deterministic calibration variants emitted by one physical source."""
    canonical = registration(source).source
    return tuple(item for item in SOURCE_REGISTRY.values() if item.source == canonical)


def source_names() -> tuple[str, ...]:
    return ("scattering", "signature", "rff_forecast", "tcn_forecast")


def build_source(
    source: str,
    *,
    device: str = "cpu",
    benchmark_record: Mapping[str, object] | None = None,
):
    """Build a source once; no TEST or family policy enters this factory."""
    canonical = registration(source).source
    factories = {
        "scattering": MultiscaleScatteringDeviation,
        "signature": OrderedPathSignatureDeviation,
        "rff_forecast": RFFNormalForecastResidual,
        "tcn_forecast": lambda: CausalTCNNormalForecast(
            device=device, benchmark_record=benchmark_record
        ),
    }
    return factories[canonical]()
