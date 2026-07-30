"""Exact immutable runtime configuration preregistered for Exp419."""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

from experiments.exp417_causal_multihorizon_shadow.config import (
    Exp417FrozenConfig,
    FROZEN_CONFIG as EXP417_FROZEN_CONFIG,
)


@dataclass(frozen=True)
class Exp419FrozenConfig:
    experiment_id: str = "Exp419"
    expected_datasets: int = 1117
    exp417_config: Exp417FrozenConfig = EXP417_FROZEN_CONFIG
    database_sha256: str = EXP417_FROZEN_CONFIG.database_sha256
    expected_hard: int = EXP417_FROZEN_CONFIG.expected_hard
    expected_standard: int = EXP417_FROZEN_CONFIG.expected_standard
    expected_priority: int = EXP417_FROZEN_CONFIG.expected_priority
    expected_automatic_normal: int = EXP417_FROZEN_CONFIG.expected_automatic_normal
    exp291_prediction_sha256: str = EXP417_FROZEN_CONFIG.exp291_prediction_sha256
    exp291_manifest_sha256: str = EXP417_FROZEN_CONFIG.exp291_manifest_sha256
    exp292_prediction_sha256: str = EXP417_FROZEN_CONFIG.exp292_prediction_sha256
    exp292_manifest_sha256: str = EXP417_FROZEN_CONFIG.exp292_manifest_sha256
    worker_candidates: tuple[int, ...] = (1, 2, 4, 6)
    start_method: str = "spawn"
    speedup_gate: float = 2.0
    memory_limit_bytes: int = 27_487_790_694
    torch_threads: int = 1
    benchmark_quantiles: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    benchmark_proxy_headroom_gate: float = 2.5
    benchmark_proxy_headroom_workers: tuple[int, ...] = (4, 6)
    corrected_benchmark_root: str = (
        "/Users/mac/Documents/Dataset/"
        "exp419_deterministic_cpu_parallel_benchmark_balanced"
    )
    benchmark_evidence_filename: str = (
        "EXP419_NON_OFFICIAL_RUNTIME_BENCHMARK.json"
    )

    def validate(self) -> None:
        if type(self.exp417_config) is not Exp417FrozenConfig:
            raise ValueError("Exp419 frozen config mismatch: exp417_config")
        self.exp417_config.validate()
        expected = type(self)()
        for field in fields(self):
            actual = getattr(self, field.name)
            wanted = getattr(expected, field.name)
            if type(actual) is not type(wanted) or actual != wanted:
                raise ValueError(f"Exp419 frozen config mismatch: {field.name}")
        for field_name in (
            "database_sha256",
            "expected_hard",
            "expected_standard",
            "expected_priority",
            "expected_automatic_normal",
            "exp291_prediction_sha256",
            "exp291_manifest_sha256",
            "exp292_prediction_sha256",
            "exp292_manifest_sha256",
        ):
            if getattr(self, field_name) != getattr(self.exp417_config, field_name):
                raise ValueError(f"Exp419 Exp417 binding mismatch: {field_name}")


FROZEN_CONFIG = Exp419FrozenConfig()


def _require_exact_path(path: Path, expected: Path, name: str) -> Path:
    if not isinstance(path, Path) or path != expected:
        raise ValueError(f"{name} must use the exact lexical path")
    try:
        observed = path.resolve(strict=False)
        canonical = expected.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{name} canonical path cannot be resolved") from error
    if observed != canonical or canonical != expected:
        raise ValueError(f"{name} must use the exact canonical path")
    return path


def corrected_benchmark_root() -> Path:
    return Path(FROZEN_CONFIG.corrected_benchmark_root)


def corrected_benchmark_evidence_path() -> Path:
    return (
        corrected_benchmark_root()
        / FROZEN_CONFIG.benchmark_evidence_filename
    )


def corrected_trial_roots() -> tuple[Path, ...]:
    root = corrected_benchmark_root()
    return tuple(
        root / f"workers-{count}"
        for count in FROZEN_CONFIG.worker_candidates
    )


def require_exact_benchmark_root(path: Path) -> Path:
    return _require_exact_path(
        path, corrected_benchmark_root(), "benchmark root"
    )


def require_exact_benchmark_evidence_path(path: Path) -> Path:
    return _require_exact_path(
        path,
        corrected_benchmark_evidence_path(),
        "benchmark evidence",
    )


def require_exact_trial_roots(
    roots: Sequence[Path],
) -> tuple[Path, ...]:
    if not isinstance(roots, Sequence) or isinstance(
        roots, (str, bytes, bytearray)
    ):
        raise ValueError("trial roots must be an exact ordered sequence")
    frozen = tuple(roots)
    expected = corrected_trial_roots()
    if len(frozen) != len(expected):
        raise ValueError("trial roots must be the exact ordered roots")
    return tuple(
        _require_exact_path(root, wanted, "trial root")
        for root, wanted in zip(frozen, expected, strict=True)
    )
