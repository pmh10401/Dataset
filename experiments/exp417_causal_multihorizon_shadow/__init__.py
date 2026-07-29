"""Frozen, label-free contracts for Exp417 causal Shadow research."""

from .config import Exp417FrozenConfig, FROZEN_CONFIG
from .contracts import (
    BackendAudit,
    DatasetDecisionEvidence,
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
    TestInstanceDecision,
)

__all__ = [
    "BackendAudit",
    "DatasetDecisionEvidence",
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
    "TestInstanceDecision",
]
