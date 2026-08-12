"""Frozen public identity and local paths for the paper rerun."""
from __future__ import annotations

from pathlib import Path

from .contracts_paper import FrozenPaperConfig


PAPER_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PAPER_ROOT.parents[1]
LINEAGE_LOCK_PATH = PAPER_ROOT / "lineage_lock_paper.json"
SOURCE_PROVENANCE_PATH = PAPER_ROOT / "sources_paper" / "source_provenance_paper.json"
ARTIFACT_PARITY_CLAIM_PATH = PAPER_ROOT / "artifact_parity_claim_paper.json"
FROZEN_CONFIG = FrozenPaperConfig()
