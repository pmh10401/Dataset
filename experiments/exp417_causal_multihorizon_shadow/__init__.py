"""Frozen, label-free contracts for Exp417 causal Shadow research."""

from .config import Exp417FrozenConfig, FROZEN_CONFIG
from .contracts import (
    BackendAudit,
    DatasetResult,
    FrozenLaneSnapshot,
    FrozenRunContext,
    FrozenTrainContract,
    GroupQualification,
    HorizonScore,
    LengthGroup,
    ProcessedGroup,
    RawSeries,
    RawTestDataset,
    RawTrainDataset,
    RunSummary,
    SeedDecision,
    ShadowCandidate,
)

__all__ = [
    "BackendAudit",
    "DatasetResult",
    "Exp417FrozenConfig",
    "FROZEN_CONFIG",
    "FrozenLaneSnapshot",
    "FrozenRunContext",
    "FrozenTrainContract",
    "GroupQualification",
    "HorizonScore",
    "LengthGroup",
    "ProcessedGroup",
    "RawSeries",
    "RawTestDataset",
    "RawTrainDataset",
    "RunSummary",
    "SeedDecision",
    "ShadowCandidate",
]
