"""Optional MLX scoring behind the frozen Exp435 parity and speed gate.

The production path in this module is deliberately small.  NumPy remains the
reference implementation; MLX is only used after a readiness smoke proves
that the exact float32 score values and candidate coordinates agree.  MLX is
imported inside :class:`MlxEngine` so importing the Exp435 prediction package
never makes an optional dependency a readiness requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import json
from numbers import Integral
from time import perf_counter
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import BackendBatchEvidence, BackendDecision
from .features_antigravity import score_channel_cpu


_BATCH_SIZES = (16, 32, 64, 128, 256)
_WARMUPS = 2
_TIMED_RUNS = 5


class MlxUnavailable(RuntimeError):
    """Raised when MLX discovery fails, retaining partial provenance."""

    def __init__(
        self,
        reason: str,
        *,
        mlx_version: str | None = None,
        mlx_metal_version: str | None = None,
        device: str | None = None,
        device_info: str | None = None,
        metal_available: bool | None = None,
    ) -> None:
        self.reason = str(reason) or "mlx_unavailable"
        self.mlx_version = mlx_version
        self.mlx_metal_version = mlx_metal_version
        self.device = device
        self.device_info = device_info
        self.metal_available = metal_available
        super().__init__(self.reason)


class MlxEngineProtocol(Protocol):
    """Minimal engine protocol accepted by the benchmark and test doubles."""

    def score(
        self,
        query: object,
        references: object,
        k: int,
        *,
        batch_size: int = 16,
    ) -> object:
        ...


@dataclass(frozen=True)
class MlxFixture:
    """A label-free query/reference batch used by the readiness smoke.

    ``thresholds`` are optional because a synthetic smoke may derive one from
    its CPU scores.  They are score thresholds, never TEST labels.
    """

    queries: NDArray[np.float32]
    references: NDArray[np.float32]
    thresholds: NDArray[np.float32] | None = None
    k: int = FROZEN_CONFIG.neighbor_count

    def __post_init__(self) -> None:
        queries = _query_matrix(self.queries)
        references = _reference_matrix(self.references)
        if references.shape[1] != queries.shape[1]:
            raise ValueError("fixture query/reference lengths differ")
        if not isinstance(self.k, Integral) or isinstance(self.k, bool) or int(self.k) < 1:
            raise ValueError("fixture k must be a positive integer")
        if int(self.k) > references.shape[0]:
            raise ValueError("fixture k cannot exceed reference count")
        thresholds: NDArray[np.float32] | None
        if self.thresholds is None:
            thresholds = None
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                thresholds = np.asarray(self.thresholds, dtype=np.float32)
            if thresholds.ndim == 0:
                thresholds = np.full(queries.shape[0], thresholds.item(), dtype=np.float32)
            if thresholds.ndim != 1 or thresholds.shape[0] != queries.shape[0]:
                raise ValueError("fixture thresholds must have one value per query")
            if not np.isfinite(thresholds).all():
                raise ValueError("fixture thresholds must be finite")
            thresholds = np.ascontiguousarray(thresholds, dtype=np.float32)
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "k", int(self.k))


def _query_matrix(value: object) -> NDArray[np.float32]:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("queries must be a non-empty vector or matrix")
    if not np.isfinite(array).all():
        raise ValueError("queries must be finite float32 values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _reference_matrix(value: object) -> NDArray[np.float32]:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("references must be a non-empty matrix")
    if not np.isfinite(array).all():
        raise ValueError("references must be finite float32 values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _version(distribution: str, fallback: object = None) -> str:
    try:
        value = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        value = fallback
    if value is None:
        return "unknown"
    return str(value)


def _canonical_info(value: object) -> str | None:
    """Serialize Metal device information without depending on its shape."""

    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return str(value)


class MlxEngine:
    """Lazy MLX implementation of the exact CPU RMS/nearest-k median score."""

    def __init__(self) -> None:
        try:
            import mlx.core as mx  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - platform dependent
            raise MlxUnavailable("mlx_unavailable") from exc
        self.mx = mx
        self.mlx_version = _version("mlx", getattr(mx, "__version__", None))
        # Apple distributes Metal support inside the ``mlx`` wheel.  There is
        # no separate ``mlx-metal`` distribution to require; this marker is a
        # truthful provenance label for the bundled backend version.
        if self.mlx_version == "unknown":
            raise MlxUnavailable(
                "mlx_version_unavailable",
                mlx_version=self.mlx_version,
            )
        try:
            self.device = str(mx.default_device())
        except Exception as exc:  # pragma: no cover - implementation dependent
            raise MlxUnavailable(
                f"mlx_device_unavailable:{type(exc).__name__}",
                mlx_version=self.mlx_version,
            ) from exc
        metal = getattr(mx, "metal", None)
        metal_is_available = getattr(metal, "is_available", None)
        try:
            self.metal_available = bool(metal_is_available()) if callable(metal_is_available) else False
        except Exception as exc:  # pragma: no cover - implementation dependent
            raise MlxUnavailable(
                f"mlx_metal_probe_failed:{type(exc).__name__}",
                mlx_version=self.mlx_version,
                device=self.device,
                metal_available=None,
            ) from exc
        device_info_call = getattr(metal, "device_info", None)
        try:
            raw_device_info = device_info_call() if callable(device_info_call) else None
        except Exception as exc:  # pragma: no cover - implementation dependent
            raise MlxUnavailable(
                f"mlx_metal_device_info_failed:{type(exc).__name__}",
                mlx_version=self.mlx_version,
                device=self.device,
                metal_available=self.metal_available,
            ) from exc
        self.device_info = _canonical_info(raw_device_info)
        self.mlx_metal_version = f"bundled-with-mlx-{self.mlx_version}"
        if not self.metal_available:
            raise MlxUnavailable(
                "mlx_metal_unavailable",
                mlx_version=self.mlx_version,
                mlx_metal_version=self.mlx_metal_version,
                device=self.device,
                device_info=self.device_info,
                metal_available=False,
            )

    def score(
        self,
        query: object,
        references: object,
        k: int,
        *,
        batch_size: int = 16,
    ) -> NDArray[np.float32]:
        queries = _query_matrix(query)
        refs = _reference_matrix(references)
        if refs.shape[1] != queries.shape[1]:
            raise ValueError("query and references have incompatible lengths")
        if not isinstance(k, Integral) or isinstance(k, bool) or int(k) < 1:
            raise ValueError("k must be a positive integer")
        if int(k) > refs.shape[0]:
            raise ValueError("k cannot exceed reference count")
        if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or int(batch_size) < 1:
            raise ValueError("batch_size must be a positive integer")

        mx = self.mx
        # Transfers are intentionally inside the timed stage.  Reference
        # chunks bound the largest dense allocation to Q x batch_size while
        # retaining the exact global nearest-k set across every chunk.
        query_mx = mx.array(queries, dtype=mx.float32)
        nearest = None
        for start in range(0, refs.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), refs.shape[0])
            refs_mx = mx.array(refs[start:stop], dtype=mx.float32)
            delta = refs_mx[None, :, :] - query_mx[:, None, :]
            distances = mx.sqrt(mx.mean(mx.square(delta), axis=2))
            chunk_nearest = mx.sort(distances, axis=1)[:, : min(int(k), stop - start)]
            if nearest is None:
                nearest = chunk_nearest
            else:
                nearest = mx.sort(mx.concatenate((nearest, chunk_nearest), axis=1), axis=1)[
                    :, : int(k)
                ]
        if nearest is None:  # pragma: no cover - refs is validated non-empty
            raise ValueError("references must be a non-empty matrix")
        if int(k) % 2:
            scores = nearest[:, int(k) // 2]
        else:
            left = nearest[:, int(k) // 2 - 1]
            right = nearest[:, int(k) // 2]
            scores = (left + right) / mx.array(np.float32(2.0), dtype=mx.float32)
        # ``eval`` materializes the graph; ``synchronize`` is included where
        # supported so benchmark wall time includes device completion.
        mx.eval(scores)
        synchronize = getattr(mx, "synchronize", None)
        if callable(synchronize):
            synchronize()
        output = np.asarray(scores, dtype=np.float32)
        if output.ndim != 1 or output.shape[0] != queries.shape[0]:
            raise ValueError("MLX score output has the wrong shape")
        if not np.isfinite(output).all():
            raise ValueError("MLX score output is nonfinite")
        return np.ascontiguousarray(output, dtype=np.float32)


def _engine_call(
    engine: object,
    queries: NDArray[np.float32],
    references: NDArray[np.float32],
    k: int,
    batch_size: int,
) -> NDArray[np.float32]:
    method = getattr(engine, "score", None)
    if not callable(method):
        method = getattr(engine, "score_channel", None)
    if not callable(method):
        method = getattr(engine, "score_channel_mlx", None)
    if not callable(method):
        raise TypeError("MLX engine must expose score(...) or score_channel(...)")
    try:
        raw = method(queries, references, int(k), batch_size=int(batch_size))
    except TypeError as first:
        # Permit small test doubles that implement only the positional core.
        try:
            raw = method(queries, references, int(k))
        except TypeError:
            try:
                raw = method(queries, references)
            except TypeError:
                raise first
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.asarray(raw, dtype=np.float32)
    if output.ndim == 0:
        output = output.reshape(1)
    if output.ndim != 1 or output.shape[0] != queries.shape[0]:
        raise ValueError("MLX score output has the wrong shape")
    return np.ascontiguousarray(output, dtype=np.float32)


def _score_channel_with_engine(
    queries: object,
    references: object,
    k: int,
    *,
    engine: object,
    batch_size: int,
) -> NDArray[np.float32]:
    query_array = _query_matrix(queries)
    reference_array = _reference_matrix(references)
    if reference_array.shape[1] != query_array.shape[1]:
        raise ValueError("query and references have incompatible lengths")
    if not isinstance(k, Integral) or isinstance(k, bool) or int(k) < 1:
        raise ValueError("k must be a positive integer")
    if int(k) > reference_array.shape[0]:
        raise ValueError("k cannot exceed reference count")
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or int(batch_size) < 1:
        raise ValueError("batch_size must be a positive integer")
    output = _engine_call(
        engine,
        query_array,
        reference_array,
        int(k),
        int(batch_size),
    )
    if not np.isfinite(output).all():
        raise ValueError("MLX score output is nonfinite")
    return np.ascontiguousarray(output, dtype=np.float32)


def score_channel_mlx(
    query: object,
    references: object,
    k: int = FROZEN_CONFIG.neighbor_count,
    *,
    batch_size: int = 256,
    engine: object | None = None,
) -> NDArray[np.float32]:
    """Score one or more queries through the optional MLX dense path.

    A direct call raises :class:`MlxUnavailable` when MLX cannot be imported;
    readiness callers should use :func:`benchmark_mlx_adoption`, which turns
    that condition into a CPU decision.
    """

    FROZEN_CONFIG.validate()
    active_engine = engine if engine is not None else MlxEngine()
    return _score_channel_with_engine(
        query,
        references,
        k,
        engine=active_engine,
        batch_size=batch_size,
    )


def _cpu_batch_scores(fixture: MlxFixture) -> NDArray[np.float32]:
    return np.asarray(
        [score_channel_cpu(row, fixture.references, fixture.k) for row in fixture.queries],
        dtype=np.float32,
    )


def _normalise_fixture(item: object) -> MlxFixture:
    if isinstance(item, MlxFixture):
        return item
    if isinstance(item, Mapping):
        query = item.get("queries", item.get("query"))
        refs = item.get("references", item.get("reference"))
        thresholds = item.get("thresholds", item.get("threshold"))
        k = item.get("k", FROZEN_CONFIG.neighbor_count)
        if query is None or refs is None:
            raise ValueError("fixture mapping needs queries and references")
        return MlxFixture(query, refs, thresholds, k)  # type: ignore[arg-type]
    query = getattr(item, "queries", getattr(item, "query", None))
    refs = getattr(item, "references", getattr(item, "reference", None))
    if query is not None and refs is not None:
        return MlxFixture(
            query,
            refs,
            getattr(item, "thresholds", getattr(item, "threshold", None)),
            getattr(item, "k", FROZEN_CONFIG.neighbor_count),
        )
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        query, refs = item[0], item[1]
        # A tuple of numeric arrays is one fixture; a tuple of mappings is a
        # collection and is handled by _normalise_fixtures instead.
        try:
            np.asarray(query, dtype=np.float32)
            np.asarray(refs, dtype=np.float32)
        except (TypeError, ValueError):
            pass
        else:
            return MlxFixture(
                query,
                refs,
                item[2] if len(item) >= 3 else None,
                item[3] if len(item) >= 4 else FROZEN_CONFIG.neighbor_count,
            )
    raise ValueError("fixture must provide queries and references")


def _normalise_fixtures(fixtures: object, smoke_seed: int) -> tuple[MlxFixture, ...]:
    if fixtures is None:
        fixtures = ()
    if isinstance(fixtures, (MlxFixture, Mapping)):
        raw_items: tuple[object, ...] = (fixtures,)
    elif isinstance(fixtures, (tuple, list)):
        # ``(queries, references[, thresholds[, k]])`` is a single fixture.
        if len(fixtures) >= 2:
            try:
                first = np.asarray(fixtures[0], dtype=np.float32)
                second = np.asarray(fixtures[1], dtype=np.float32)
            except (TypeError, ValueError):
                first = second = np.asarray((), dtype=np.float32)
            if first.ndim in (1, 2) and second.ndim == 2 and first.size and second.size:
                raw_items = (fixtures,)
            else:
                raw_items = tuple(fixtures)
        else:
            raw_items = tuple(fixtures)
    else:
        raw_items = tuple(fixtures) if isinstance(fixtures, Sequence) else (fixtures,)
    if not raw_items:
        rng = np.random.default_rng(int(smoke_seed))
        references = rng.normal(size=(8, 16)).astype(np.float32)
        queries = rng.normal(size=(32, 16)).astype(np.float32)
        return (MlxFixture(queries, references),)
    return tuple(_normalise_fixture(item) for item in raw_items)


def _thresholds(fixture: MlxFixture, cpu_scores: NDArray[np.float32]) -> NDArray[np.float32]:
    if fixture.thresholds is not None:
        return fixture.thresholds
    threshold = np.float32(
        np.quantile(
            cpu_scores,
            1.0 - FROZEN_CONFIG.candidate_alpha,
            method=FROZEN_CONFIG.quantile_method,
        )
    )
    return np.full(cpu_scores.shape, threshold, dtype=np.float32)


def _coordinates(scores: NDArray[np.float32], thresholds: NDArray[np.float32], fixture_index: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(fixture_index), int(index))
        for index in np.flatnonzero(scores > thresholds).tolist()
    )


def _measure(
    engine: object,
    kind: str,
    batch_size: int,
    call: Callable[[], object],
    clock: Callable[[], float],
) -> tuple[float, object]:
    holder: dict[str, object] = {}

    def invoke() -> None:
        holder["value"] = call()

    timed_call = getattr(engine, "timed_call", None)
    if callable(timed_call):
        elapsed = timed_call(kind, int(batch_size), invoke, clock)
    else:
        started = float(clock())
        invoke()
        elapsed = float(clock()) - started
    elapsed = float(elapsed)
    if not np.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("nonpositive benchmark timing")
    return elapsed, holder["value"]


def _metadata(engine: object, name: str, *aliases: str) -> str | None:
    for candidate in (name,) + aliases:
        value = getattr(engine, candidate, None)
        if value is not None:
            return str(value)
    return None


def _engine_provenance(engine: object | None) -> dict[str, object]:
    if engine is None:
        return {
            "mlx_version": None,
            "mlx_metal_version": None,
            "device": None,
            "device_info": None,
            "metal_available": None,
        }
    metal_available = getattr(engine, "metal_available", None)
    if metal_available is not None:
        metal_available = bool(metal_available)
    return {
        "mlx_version": _metadata(engine, "mlx_version", "version"),
        "mlx_metal_version": _metadata(engine, "mlx_metal_version", "metal_version"),
        "device": _metadata(engine, "device", "device_name"),
        "device_info": _metadata(engine, "device_info", "metal_device_info"),
        "metal_available": metal_available,
    }


def _runtime_failure_reason(exc: BaseException) -> str:
    """Return a stable, precise reason without catching process interrupts."""

    message = str(exc).strip().replace("\n", " ")
    if len(message) > 120:
        message = message[:120]
    return f"mlx_runtime_error:{type(exc).__name__}" + (f":{message}" if message else "")


def _make_decision(
    *,
    engine: object | None,
    adopted: bool,
    reason: str,
    smoke_seed: int,
    finite_outputs: bool,
    candidate_identity_match: bool,
    parity_passed: bool,
    normalized_error_max: float | None,
    absolute_error_max: float | None,
    timings: tuple[tuple[int, float, float], ...],
    selected_batch_size: int | None,
    batch_evidence: tuple[BackendBatchEvidence, ...] = (),
    provenance: Mapping[str, object] | None = None,
) -> BackendDecision:
    evidence = next(
        (item for item in batch_evidence if item.batch_size == selected_batch_size),
        None,
    )
    if evidence is not None and evidence.cpu_median_seconds is not None and evidence.mlx_median_seconds is not None:
        cpu_seconds = evidence.cpu_median_seconds
        mlx_seconds = evidence.mlx_median_seconds
        speedup = evidence.speedup
    elif timings and selected_batch_size is not None:
        selected = next(item for item in timings if item[0] == selected_batch_size)
        cpu_seconds, mlx_seconds = selected[1], selected[2]
        speedup = cpu_seconds / mlx_seconds
    else:
        cpu_seconds = mlx_seconds = speedup = None
    fields = dict(_engine_provenance(engine))
    if provenance is not None:
        fields.update(provenance)
    if cpu_seconds is None or mlx_seconds is None or selected_batch_size is None:
        return BackendDecision(
            backend="cpu",
            adopted=False,
            smoke_seed=smoke_seed,
            finite_outputs=finite_outputs,
            candidate_identity_match=candidate_identity_match,
            parity_passed=False,
            normalized_error_max=normalized_error_max,
            absolute_error_max=absolute_error_max,
            mlx_version=fields.get("mlx_version"),
            mlx_metal_version=fields.get("mlx_metal_version"),
            device=fields.get("device"),
            device_info=fields.get("device_info"),
            metal_available=fields.get("metal_available"),
            batch_evidence=batch_evidence,
            reason=reason,
        )
    return BackendDecision(
        backend="mlx" if adopted else "cpu",
        adopted=adopted,
        smoke_seed=smoke_seed,
        finite_outputs=finite_outputs,
        candidate_identity_match=candidate_identity_match,
        parity_passed=parity_passed,
        normalized_error_max=normalized_error_max,
        absolute_error_max=absolute_error_max,
        speedup=speedup,
        selected_batch_size=selected_batch_size,
        cpu_median_seconds=cpu_seconds,
        mlx_median_seconds=mlx_seconds,
        mlx_version=fields.get("mlx_version") or ("unknown" if adopted else None),
        mlx_metal_version=fields.get("mlx_metal_version") or ("unknown" if adopted else None),
        device=fields.get("device") or ("unknown" if adopted else None),
        device_info=fields.get("device_info"),
        metal_available=fields.get("metal_available"),
        timings=timings,
        batch_evidence=batch_evidence,
        reason=reason,
    )


def benchmark_mlx_adoption(
    fixtures: object,
    smoke_seed: int = FROZEN_CONFIG.mlx_smoke_seed,
    clock: Callable[[], float] = perf_counter,
    *,
    engine: object | None = None,
) -> BackendDecision:
    """Run the frozen MLX readiness gate and return a CPU/MLX decision."""

    FROZEN_CONFIG.validate()
    if smoke_seed != FROZEN_CONFIG.mlx_smoke_seed:
        raise ValueError("MLX smoke seed must equal frozen seed 43501")
    try:
        prepared = _normalise_fixtures(fixtures, smoke_seed)
    except (TypeError, ValueError):
        return BackendDecision.cpu_fallback("invalid_fixture")
    active_engine = engine
    provenance: dict[str, object] | None = None
    if active_engine is None:
        try:
            active_engine = MlxEngine()
        except MlxUnavailable as exc:
            provenance = {
                "mlx_version": exc.mlx_version,
                "mlx_metal_version": exc.mlx_metal_version,
                "device": exc.device,
                "device_info": exc.device_info,
                "metal_available": exc.metal_available,
            }
            return BackendDecision.cpu_fallback(
                exc.reason,
                mlx_version=exc.mlx_version,
                mlx_metal_version=exc.mlx_metal_version,
                device=exc.device,
                device_info=exc.device_info,
                metal_available=exc.metal_available,
            )
        except RuntimeError as exc:
            return BackendDecision.cpu_fallback(_runtime_failure_reason(exc))

    # CPU scores are computed once from the exact input and reused for every
    # batch timing.  This is not a TEST evaluation; thresholds are fixed
    # fixture values or a deterministic smoke quantile.
    cpu_reference = tuple(_cpu_batch_scores(item) for item in prepared)
    thresholds = tuple(_thresholds(item, scores) for item, scores in zip(prepared, cpu_reference))
    cpu_coordinates = tuple(
        _coordinates(scores, threshold, fixture_index)
        for fixture_index, (scores, threshold) in enumerate(zip(cpu_reference, thresholds))
    )
    cpu_flat = np.concatenate(cpu_reference).astype(np.float32, copy=False)

    timings_list: list[tuple[int, float, float]] = []
    batch_records: list[BackendBatchEvidence] = []

    def cpu_call() -> NDArray[np.float32]:
        return np.concatenate(
            tuple(_cpu_batch_scores(item) for item in prepared)
        ).astype(np.float32, copy=False)

    def mlx_call(batch_size: int) -> NDArray[np.float32]:
        return np.concatenate(
            tuple(
                _score_channel_with_engine(
                    item.queries,
                    item.references,
                    item.k,
                    engine=active_engine,
                    batch_size=batch_size,
                )
                for item in prepared
            )
        ).astype(np.float32, copy=False)

    for batch_size in _BATCH_SIZES:
        batch_finite = True
        batch_identity = True
        batch_absolute = 0.0
        batch_normalized = 0.0
        batch_pairs = 0
        batch_error: str | None = None
        cpu_times: list[float] = []
        mlx_times: list[float] = []
        try:
            for _ in range(_WARMUPS):
                _measure(active_engine, "cpu", batch_size, cpu_call, clock)
                _measure(active_engine, "mlx", batch_size, lambda b=batch_size: mlx_call(b), clock)
            for _ in range(_TIMED_RUNS):
                try:
                    cpu_seconds, cpu_output_raw = _measure(active_engine, "cpu", batch_size, cpu_call, clock)
                    mlx_seconds, mlx_output_raw = _measure(
                        active_engine,
                        "mlx",
                        batch_size,
                        lambda b=batch_size: mlx_call(b),
                        clock,
                    )
                except (TypeError, ValueError, FloatingPointError) as exc:
                    message = str(exc).lower()
                    batch_error = batch_error or (
                        "score_shape_mismatch" if "shape" in message else "nonfinite_output"
                    )
                    batch_finite = False
                    batch_identity = False
                    break
                except RuntimeError as exc:
                    batch_error = batch_error or _runtime_failure_reason(exc)
                    batch_finite = False
                    batch_identity = False
                    break
                cpu_times.append(cpu_seconds)
                mlx_times.append(mlx_seconds)
                cpu_output = np.asarray(cpu_output_raw, dtype=np.float32)
                mlx_output = np.asarray(mlx_output_raw, dtype=np.float32)
                if cpu_output.shape != mlx_output.shape or cpu_output.shape != cpu_flat.shape:
                    batch_error = batch_error or "score_shape_mismatch"
                    batch_finite = False
                    continue
                if not np.isfinite(cpu_output).all() or not np.isfinite(mlx_output).all():
                    batch_finite = False
                    batch_error = batch_error or "nonfinite_output"
                    continue
                batch_pairs += int(cpu_output.size)
                # Error arithmetic follows the preregistered scalar formula;
                # score values remain the exact float32 values returned by the
                # two backends, while the comparison itself avoids an
                # accidental float32 rounding of the 1e-12 floor.
                cpu_error_values = np.asarray(cpu_output, dtype=np.float64)
                mlx_error_values = np.asarray(mlx_output, dtype=np.float64)
                absolute = np.abs(cpu_error_values - mlx_error_values)
                denominators = np.maximum(
                    np.maximum(np.abs(cpu_error_values), np.abs(mlx_error_values)),
                    float(FROZEN_CONFIG.numeric_floor),
                )
                normalized = absolute / denominators
                batch_absolute = max(batch_absolute, float(np.max(absolute)))
                batch_normalized = max(batch_normalized, float(np.max(normalized)))
                offset = 0
                mlx_coordinates: list[tuple[int, int]] = []
                for fixture_index, item_scores in enumerate(cpu_reference):
                    count = item_scores.shape[0]
                    mlx_coordinates.extend(
                        _coordinates(
                            mlx_output[offset : offset + count],
                            thresholds[fixture_index],
                            fixture_index,
                        )
                    )
                    offset += count
                if tuple(mlx_coordinates) != tuple(item for group in cpu_coordinates for item in group):
                    batch_identity = False
                    batch_error = batch_error or "candidate_identity_mismatch"
        except (TypeError, ValueError, FloatingPointError) as exc:
            message = str(exc).lower()
            batch_error = batch_error or (
                "score_shape_mismatch" if "shape" in message else "nonfinite_output"
            )
            batch_finite = False
            batch_identity = False
        except RuntimeError as exc:
            batch_error = batch_error or _runtime_failure_reason(exc)
            batch_finite = False
            batch_identity = False

        complete_timing = len(cpu_times) == _TIMED_RUNS and len(mlx_times) == _TIMED_RUNS
        if complete_timing:
            cpu_median = float(np.median(np.asarray(cpu_times, dtype=np.float64)))
            mlx_median = float(np.median(np.asarray(mlx_times, dtype=np.float64)))
            speedup = cpu_median / mlx_median
            timings_list.append((int(batch_size), cpu_median, mlx_median))
        else:
            cpu_median = mlx_median = speedup = None
        parity_passed = bool(
            batch_finite
            and batch_identity
            and batch_pairs > 0
            and batch_absolute <= FROZEN_CONFIG.score_tolerance
            and batch_normalized <= FROZEN_CONFIG.score_tolerance
        )
        speed_passed = bool(parity_passed and speedup is not None and speedup >= FROZEN_CONFIG.mlx_min_speedup)
        if batch_error is None:
            if not parity_passed:
                if not batch_identity:
                    batch_error = "candidate_identity_mismatch"
                elif not batch_finite:
                    batch_error = "nonfinite_output"
                elif batch_absolute > FROZEN_CONFIG.score_tolerance:
                    batch_error = "absolute_error_exceeded"
                elif batch_normalized > FROZEN_CONFIG.score_tolerance:
                    batch_error = "normalized_error_exceeded"
            elif not speed_passed:
                batch_error = "speedup_below_threshold"
            else:
                batch_error = "passed"
        batch_records.append(
            BackendBatchEvidence(
                batch_size=int(batch_size),
                finite_outputs=batch_finite,
                candidate_identity_match=batch_identity,
                parity_passed=parity_passed,
                normalized_error_max=batch_normalized if batch_pairs else None,
                absolute_error_max=batch_absolute if batch_pairs else None,
                speed_passed=speed_passed,
                cpu_median_seconds=cpu_median,
                mlx_median_seconds=mlx_median,
                speedup=speedup,
                reason=batch_error,
            )
        )

    timings = tuple(timings_list)
    evidence = tuple(batch_records)
    passing = [item for item in evidence if item.adopted]
    if passing:
        selected = min(passing, key=lambda item: (item.mlx_median_seconds or float("inf"), item.batch_size))
        reason = "adopted"
        return _make_decision(
            engine=active_engine,
            adopted=True,
            reason=reason,
            smoke_seed=smoke_seed,
            finite_outputs=selected.finite_outputs,
            candidate_identity_match=selected.candidate_identity_match,
            parity_passed=True,
            normalized_error_max=selected.normalized_error_max,
            absolute_error_max=selected.absolute_error_max,
            timings=timings,
            selected_batch_size=selected.batch_size,
            batch_evidence=evidence,
            provenance=provenance,
        )

    parity_evidence = [item for item in evidence if item.parity_passed]
    representative = (
        min(
            parity_evidence,
            key=lambda item: (item.mlx_median_seconds or float("inf"), item.batch_size),
        )
        if parity_evidence
        else min(
            (item for item in evidence if item.mlx_median_seconds is not None),
            key=lambda item: (item.mlx_median_seconds or float("inf"), item.batch_size),
            default=None,
        )
    )
    if any(item.reason == "candidate_identity_mismatch" for item in evidence):
        reason = "candidate_identity_mismatch"
    elif any(item.reason == "nonfinite_output" for item in evidence):
        reason = "nonfinite_output"
    elif any(item.reason == "absolute_error_exceeded" for item in evidence):
        reason = "absolute_error_exceeded"
    elif any(item.reason == "normalized_error_exceeded" for item in evidence):
        reason = "normalized_error_exceeded"
    elif any(item.reason == "score_shape_mismatch" for item in evidence):
        reason = "score_shape_mismatch"
    elif any(item.reason.startswith("mlx_runtime_error:") for item in evidence):
        reason = next(item.reason for item in evidence if item.reason.startswith("mlx_runtime_error:"))
    else:
        reason = "speedup_below_threshold"
    selected_fallback = representative.batch_size if representative is not None and representative.mlx_median_seconds is not None else None
    return _make_decision(
        engine=active_engine,
        adopted=False,
        reason=reason,
        smoke_seed=smoke_seed,
        finite_outputs=representative.finite_outputs if representative is not None else False,
        candidate_identity_match=representative.candidate_identity_match if representative is not None else False,
        parity_passed=representative.parity_passed if representative is not None else False,
        normalized_error_max=representative.normalized_error_max if representative is not None else None,
        absolute_error_max=representative.absolute_error_max if representative is not None else None,
        timings=timings,
        selected_batch_size=selected_fallback,
        batch_evidence=evidence,
        provenance=provenance,
    )


def score_with_backend(
    decision: BackendDecision,
    query: object,
    references: object,
    k: int = FROZEN_CONFIG.neighbor_count,
    *,
    engine: object | None = None,
) -> float:
    """Score one query using only the already-approved backend decision.

    A runtime MLX error is fail-closed.  The selected backend is frozen by
    readiness, so an adopted MLX run cannot silently mix in CPU scores.
    """

    if not isinstance(decision, BackendDecision):
        raise ValueError("decision must be a BackendDecision")
    if decision.adopted and decision.backend == "mlx":
        try:
            active_engine = engine if engine is not None else MlxEngine()
            if decision.mlx_version is not None:
                actual_version = _metadata(active_engine, "mlx_version", "version")
                if actual_version != decision.mlx_version:
                    raise ValueError("MLX version is not the approved backend")
            if decision.mlx_metal_version is not None:
                actual_metal = _metadata(active_engine, "mlx_metal_version", "metal_version")
                if actual_metal != decision.mlx_metal_version:
                    raise ValueError("mlx-metal version is not the approved backend")
            if decision.device is not None:
                actual_device = _metadata(active_engine, "device", "device_name")
                if actual_device != decision.device:
                    raise ValueError("MLX device is not the approved backend")
            if decision.device_info is None:
                raise ValueError("approved MLX device_info is missing")
            actual_device_info = _metadata(active_engine, "device_info", "metal_device_info")
            if actual_device_info != decision.device_info:
                raise ValueError("MLX device_info is not the approved backend")
            if decision.metal_available is not True:
                raise ValueError("approved MLX Metal availability is false")
            if getattr(active_engine, "metal_available", None) is not True:
                raise ValueError("MLX Metal availability is not the approved backend")
            output = score_channel_mlx(
                query,
                references,
                k,
                batch_size=decision.selected_batch_size or 256,
                engine=active_engine,
            )
            if output.shape != (1,) or not np.isfinite(output).all():
                raise ValueError("MLX runtime output is invalid")
            return float(output[0])
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise MlxUnavailable(_runtime_failure_reason(exc)) from exc
    return float(score_channel_cpu(query, references, k))


__all__ = [
    "MlxEngine",
    "MlxEngineProtocol",
    "MlxFixture",
    "MlxUnavailable",
    "benchmark_mlx_adoption",
    "score_channel_mlx",
    "score_with_backend",
]
