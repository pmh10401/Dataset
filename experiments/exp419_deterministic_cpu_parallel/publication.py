"""No-clobber atomic publication wrapper for exact Exp419 evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from experiments.exp417_causal_multihorizon_shadow.checkpoints import (
    Exp417CheckpointStore,
)
from experiments.exp417_causal_multihorizon_shadow.publication import (
    _require_disjoint_checkpoint_output,
    publish_exact_bundle,
)
from experiments.exp417_causal_multihorizon_shadow.reporting import (
    TrustedCheckpointProvenance,
)

from .contracts import (
    BenchmarkSelectionProof,
    ParallelRun,
    RuntimeBenchmark,
    ShardPlan,
)
from .reporting import (
    OFFICIAL_FILES,
    render_exp419_payloads,
    verify_exp419_payloads,
)


def publish_exp419(
    payloads: Mapping[str, bytes],
    output_dir: Path,
    *,
    checkpoints: Exp417CheckpointStore,
    run: ParallelRun,
    benchmark: RuntimeBenchmark,
    expected_selection_proof: BenchmarkSelectionProof,
    plan: ShardPlan,
    pre_publish_verify: Callable[[], None],
) -> Mapping[str, str]:
    """Publish only a fresh, complete, canonical Exp419 ten-file bundle."""
    if not callable(pre_publish_verify):
        raise ValueError("Exp419 pre-publish verifier must be callable")
    if (
        type(expected_selection_proof) is not BenchmarkSelectionProof
        or benchmark.selection_proof != expected_selection_proof
    ):
        raise ValueError(
            "Exp419 publication benchmark selection proof mismatch"
        )
    if (
        not isinstance(payloads, Mapping)
        or set(payloads) != set(OFFICIAL_FILES)
        or len(payloads) != len(OFFICIAL_FILES)
        or any(
            type(name) is not str or type(payload) is not bytes
            for name, payload in payloads.items()
        )
    ):
        raise ValueError("Exp419 publication requires the exact ordered ten-file set")
    output = Path(output_dir)
    checkpoint_root = getattr(checkpoints, "root", None)
    if not isinstance(checkpoint_root, Path):
        raise ValueError("Exp419 publication requires a canonical checkpoint store")
    _require_disjoint_checkpoint_output(checkpoint_root, output)

    trusted = TrustedCheckpointProvenance.from_store(checkpoints)
    expected = dict(
        render_exp419_payloads(run, benchmark, plan, trusted)
    )
    if dict(payloads) != expected:
        raise ValueError("Exp419 payloads differ from verified run evidence")
    verify_exp419_payloads(
        payloads,
        trusted_checkpoint=trusted,
        expected_selection_proof=expected_selection_proof,
    )

    def fresh_revalidation() -> None:
        fresh = TrustedCheckpointProvenance.from_store(checkpoints)
        if fresh != trusted:
            raise ValueError("Exp419 trusted canonical checkpoint changed")
        rerendered = dict(
            render_exp419_payloads(run, benchmark, plan, fresh)
        )
        if rerendered != dict(payloads):
            raise ValueError("Exp419 inputs or Git-bound sources changed")
        verify_exp419_payloads(
            payloads,
            trusted_checkpoint=fresh,
            expected_selection_proof=expected_selection_proof,
        )
        pre_publish_verify()

    return publish_exact_bundle(
        payloads,
        output,
        exact_names=OFFICIAL_FILES,
        pre_publish_verify=fresh_revalidation,
    )
