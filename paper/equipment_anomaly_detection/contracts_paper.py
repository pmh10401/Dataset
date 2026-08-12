"""Immutable values shared by the paper rerun provenance boundary."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LineageNode:
    node_id: str
    stage: str
    input_kind: str
    module: str
    symbols: tuple[str, ...]
    source_sha256: str | None
    paper_module: str | None
    label_access: bool
    parity_oracle: str | None = None


@dataclass(frozen=True, slots=True)
class FrozenPaperConfig:
    run_identity: str = "paper-proposal-clean-rerun-20260812"
    database_path: str = "/Users/mac/Documents/Dataset/univariate_ts.db"
    database_sha256: str = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
    excluded_datasets: tuple[str, str] = ("CornellWhaleChallenge", "Wafer_normal_1")
    dataset_count: int = 1117
    test_row_count: int = 195900
    seed: int = 20260812
    ablation_alert_counts: tuple[int, int, int, int] = (2218, 2248, 2244, 2222)


@dataclass(frozen=True, slots=True)
class PaperCodemap:
    required_fields: tuple[str, ...]
    modules: dict[str, dict[str, object]]
