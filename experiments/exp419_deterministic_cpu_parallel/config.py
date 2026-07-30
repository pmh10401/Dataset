"""Exact immutable runtime configuration preregistered for Exp419."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Exp419FrozenConfig:
    experiment_id: str = "Exp419"
    expected_datasets: int = 1117
    worker_candidates: tuple[int, ...] = (1, 2, 4, 6)
    start_method: str = "spawn"
    speedup_gate: float = 2.0
    memory_limit_bytes: int = 27_487_790_694
    torch_threads: int = 1
    benchmark_quantiles: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)

    def validate(self) -> None:
        expected = type(self)()
        for field in fields(self):
            actual = getattr(self, field.name)
            wanted = getattr(expected, field.name)
            if type(actual) is not type(wanted) or actual != wanted:
                raise ValueError(f"Exp419 frozen config mismatch: {field.name}")


FROZEN_CONFIG = Exp419FrozenConfig()
