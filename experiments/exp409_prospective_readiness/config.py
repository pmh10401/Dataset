"""Immutable preregistered configuration for Exp409."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Exp409FrozenConfig:
    experiment_id: str = "Exp409"
    variant: str = "two_new_plus_existing_independent"
    alpha: str = "0.005"
    review_budget: int = 100
    seed: int = 20260727
    independence_threshold: float = 0.90
    minimum_oof_rows: int = 20
    disposition: str = "review_only"
    performance_evaluation: str = "not_evaluated"
    prospective_validated: bool = False
    deployment_eligible: bool = False
    operational_policy_changed: bool = False

    def __post_init__(self) -> None:
        expected = {
            "experiment_id": "Exp409",
            "variant": "two_new_plus_existing_independent",
            "alpha": "0.005",
            "review_budget": 100,
            "seed": 20260727,
            "independence_threshold": 0.90,
            "minimum_oof_rows": 20,
            "disposition": "review_only",
            "performance_evaluation": "not_evaluated",
            "prospective_validated": False,
            "deployment_eligible": False,
            "operational_policy_changed": False,
        }
        if asdict(self) != expected:
            raise ValueError("Exp409 research configuration is frozen")

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


FROZEN_CONFIG = Exp409FrozenConfig()
