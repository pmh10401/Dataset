from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, TensorDataset

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp239_248_diverse_score_sources.foundation import crossfit_plan
from experiments.exp267_273_zero_support_fn_recovery.config import (
    SECOND_SEED,
    SEED,
    TCN_MAX_TRAIN_ELEMENTS,
    TCN_MIN_TRAIN,
)
from experiments.exp267_273_zero_support_fn_recovery.contracts import SourceScore


TCN_CHANNELS: Final = 32
TCN_KERNEL_SIZE: Final = 3
TCN_DILATIONS: Final = (1, 2, 4, 8)
TCN_DROPOUT: Final = 0.1
TCN_PATCH_SIZE: Final = 8
TCN_BATCH_SIZE: Final = 32
TCN_MAX_EPOCHS: Final = 20
TCN_PATIENCE: Final = 3
TCN_LEARNING_RATE: Final = 1e-3
TCN_WEIGHT_DECAY: Final = 1e-4
TCN_EVIDENCE_FAMILY: Final = "normal_forecast"
TCN_SOURCE_NAME: Final = "causal_tcn_normal_forecast"
_BENCHMARK_ROWS: Final = 100
_BENCHMARK_TARGET_LENGTH: Final = 128
_BENCHMARK_WARM_ITERATIONS: Final = 3
_BENCHMARK_TIMED_ITERATIONS: Final = 5
_BENCHMARK_RTOL: Final = 1e-4
_BENCHMARK_ATOL: Final = 1e-5
_MINIMUM_TARGET_LENGTH: Final = TCN_PATCH_SIZE + 1
_BENCHMARK_RECORD_HASH_FIELD: Final = "benchmark_record_sha256"


class _CausalConv1d(nn.Module):
    """A convolution with left-only padding so every output uses its prefix."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = int(dilation) * (TCN_KERNEL_SIZE - 1)
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=TCN_KERNEL_SIZE,
            dilation=int(dilation),
            padding=0,
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.convolution(functional.pad(values, (self.left_padding, 0)))


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.causal = _CausalConv1d(channels, dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(TCN_DROPOUT)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.dropout(self.activation(self.causal(values)))


class CausalTCN(nn.Module):
    """Frozen four-block causal residual TCN with eight future-patch outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(1, TCN_CHANNELS, kernel_size=1)
        self.blocks = nn.ModuleList(
            _CausalResidualBlock(TCN_CHANNELS, dilation)
            for dilation in TCN_DILATIONS
        )
        self.patch_head = nn.Conv1d(TCN_CHANNELS, TCN_PATCH_SIZE, kernel_size=1)

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.input_projection(values)
        for block in self.blocks:
            hidden = block(hidden)
        return self.patch_head(hidden)


def set_deterministic_seed(seed: int) -> None:
    """Set the two framework RNGs used by the frozen TRAIN-only fitting path."""
    numeric_seed = int(seed)
    np.random.seed(numeric_seed)
    torch.manual_seed(numeric_seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(numeric_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(numeric_seed)
    torch.backends.cudnn.benchmark = False


def build_tcn_model() -> CausalTCN:
    return CausalTCN()


def deterministic_tcn_fold_plan(
    n_train: int, seed: int
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return the frozen five-fold TRAIN-only outer cross-fit partition."""
    return crossfit_plan(int(n_train), seed=int(seed), folds=5)


def _validate_rows(rows: np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} rows must have shape (batch, time)")
    if values.shape[1] != int(expected_length):
        raise ValueError(f"{name} rows must match the train target length")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} rows must be finite")
    return values


def _future_patch_targets(rows: Tensor) -> tuple[Tensor, Tensor]:
    """Return causal inputs and all eight-step TRAIN reconstruction targets."""
    if rows.ndim != 2 or rows.shape[1] < _MINIMUM_TARGET_LENGTH:
        raise ValueError("TCN forecasting requires at least nine time points")
    positions = rows.shape[1] - TCN_PATCH_SIZE
    inputs = rows[:, :positions].unsqueeze(1)
    targets = torch.stack(
        tuple(rows[:, offset : offset + positions] for offset in range(1, TCN_PATCH_SIZE + 1)),
        dim=1,
    )
    return inputs, targets


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _release_device_memory(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def _device_available(device: str) -> bool:
    return (device == "cpu") or (
        device == "mps" and torch.backends.mps.is_available()
    ) or (device == "cuda" and torch.cuda.is_available())


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def benchmark_record_hash(record: Mapping[str, object]) -> str:
    """Hash a benchmark record while excluding its self-referential hash field."""
    return _canonical_hash(
        {key: value for key, value in record.items() if key != _BENCHMARK_RECORD_HASH_FIELD}
    )


def _frozen_benchmark_record(
    record: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if record is None:
        return None
    # A JSON round-trip detaches the source from a caller-owned mutable record.
    return json.loads(json.dumps(record, sort_keys=True, allow_nan=False))


def _is_passing_mps_benchmark(record: Mapping[str, object] | None) -> bool:
    valid, _ = _benchmark_record_validation(record)
    return bool(valid and record is not None and record.get("selected_device") == "mps")


def choose_device(requested: str, benchmark: Mapping[str, object] | None = None) -> str:
    """Choose only a frozen, benchmark-approved execution device.

    ``auto`` deliberately falls back to CPU until the pipeline supplies a
    recorded benchmark decision. This prevents a per-dataset MPS decision.
    """
    choice = str(requested).lower()
    if choice not in {"auto", "cpu", "mps"}:
        raise ValueError("requested device must be one of: auto, cpu, mps")
    if choice == "cpu":
        return "cpu"
    if _is_passing_mps_benchmark(benchmark) and _device_available("mps"):
        return "mps"
    return "cpu"


def _benchmark_payload() -> dict[str, object]:
    return {
        "source": TCN_SOURCE_NAME,
        "architecture": {
            "channels": TCN_CHANNELS,
            "kernel_size": TCN_KERNEL_SIZE,
            "dilations": TCN_DILATIONS,
            "dropout": TCN_DROPOUT,
            "future_patch": TCN_PATCH_SIZE,
        },
        "synthetic_rows": _BENCHMARK_ROWS,
        "synthetic_target_length": _BENCHMARK_TARGET_LENGTH,
        "warm_iterations": _BENCHMARK_WARM_ITERATIONS,
        "timed_iterations": _BENCHMARK_TIMED_ITERATIONS,
        "parity_rtol": _BENCHMARK_RTOL,
        "parity_atol": _BENCHMARK_ATOL,
        "seed": SEED,
        "torch_version": torch.__version__,
    }


def _payload_hash(payload: Mapping[str, object]) -> str:
    return _canonical_hash(payload)


def _benchmark_record_validation(
    record: Mapping[str, object] | None,
) -> tuple[bool, str]:
    """Verify current config, required fields, and canonical record integrity."""
    if record is None:
        return False, "missing_benchmark_record"
    try:
        normalized = json.loads(json.dumps(record, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError):
        return False, "benchmark_record_not_canonical_json"
    if not isinstance(normalized, dict):
        return False, "benchmark_record_not_mapping"

    expected_payload = json.loads(json.dumps(_benchmark_payload(), sort_keys=True))
    for field, expected in expected_payload.items():
        if normalized.get(field) != expected:
            return False, f"benchmark_payload_mismatch:{field}"
    if normalized.get("benchmark_config_hash") != _payload_hash(_benchmark_payload()):
        return False, "stale_or_tampered_benchmark_config_hash"

    required = (
        "benchmark_config_hash",
        _BENCHMARK_RECORD_HASH_FIELD,
        "requested_device",
        "cpu_median_seconds",
        "mps_available",
        "mps_finite",
        "mps_close_to_cpu",
        "mps_faster_than_cpu",
        "selected_device",
        "selection_reason",
    )
    missing = tuple(field for field in required if field not in normalized)
    if missing:
        return False, f"missing_benchmark_fields:{','.join(missing)}"
    if normalized["requested_device"] not in {"auto", "cpu", "mps"}:
        return False, "invalid_requested_device"
    if normalized["selected_device"] not in {"cpu", "mps"}:
        return False, "invalid_selected_device"
    if not isinstance(normalized["selection_reason"], str) or not normalized["selection_reason"]:
        return False, "invalid_selection_reason"
    if not isinstance(normalized["cpu_median_seconds"], (int, float)) or not np.isfinite(
        float(normalized["cpu_median_seconds"])
    ) or float(normalized["cpu_median_seconds"]) <= 0.0:
        return False, "invalid_cpu_benchmark_timing"
    for field in ("mps_available", "mps_finite", "mps_close_to_cpu", "mps_faster_than_cpu"):
        if type(normalized[field]) is not bool:
            return False, f"invalid_boolean:{field}"
    stored_hash = normalized[_BENCHMARK_RECORD_HASH_FIELD]
    if not isinstance(stored_hash, str) or stored_hash != benchmark_record_hash(normalized):
        return False, "tampered_benchmark_record_hash"
    if normalized["selected_device"] == "mps":
        if not all(
            normalized[field]
            for field in ("mps_available", "mps_finite", "mps_close_to_cpu", "mps_faster_than_cpu")
        ):
            return False, "mps_selected_without_passing_checks"
        mps_seconds = normalized.get("mps_median_seconds")
        if not isinstance(mps_seconds, (int, float)) or not np.isfinite(float(mps_seconds)) or float(mps_seconds) <= 0.0:
            return False, "invalid_mps_benchmark_timing"
    return True, "verified"


def _synthetic_benchmark_batch() -> np.ndarray:
    generator = np.random.default_rng(SEED)
    time_axis = np.linspace(0.0, 8.0 * np.pi, _BENCHMARK_TARGET_LENGTH, dtype=np.float32)
    offsets = np.linspace(0.0, 0.5, _BENCHMARK_ROWS, dtype=np.float32)[:, None]
    return (
        np.sin(time_axis[None, :] + offsets)
        + 0.05 * np.sin(3.0 * time_axis[None, :] + offsets)
        + generator.normal(0.0, 0.01, size=(_BENCHMARK_ROWS, _BENCHMARK_TARGET_LENGTH))
    ).astype(np.float32)


def _benchmark_device_once(device: str) -> dict[str, object]:
    if not _device_available(device):
        raise RuntimeError(f"device is unavailable: {device}")
    set_deterministic_seed(SEED)
    model = build_tcn_model().to(device).eval()
    values = torch.as_tensor(_synthetic_benchmark_batch(), device=device)
    inputs, _ = _future_patch_targets(values)
    try:
        with torch.no_grad():
            for _ in range(_BENCHMARK_WARM_ITERATIONS):
                output = model(inputs)
            _synchronize(device)
            timings = []
            for _ in range(_BENCHMARK_TIMED_ITERATIONS):
                start = time.perf_counter()
                output = model(inputs)
                _synchronize(device)
                timings.append(time.perf_counter() - start)
            result = output.detach().cpu().numpy().copy()
        if not np.isfinite(result).all():
            raise ValueError(f"non-finite {device} benchmark output")
        return {
            "device": device,
            "median_seconds": float(np.median(timings)),
            "timings_seconds": tuple(float(item) for item in timings),
            "output": result,
            "finite": True,
        }
    finally:
        del model, values, inputs
        _release_device_memory(device)


def benchmark_device(
    artifact_dir: Path, *, requested: str = "auto", force: bool = False
) -> dict[str, object]:
    """Run the fixed one-time CPU/MPS benchmark and freeze its decision on disk."""
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tcn_device_benchmark.json"
    payload = _benchmark_payload()
    payload_hash = _payload_hash(payload)
    if path.exists() and not force:
        cached = json.loads(path.read_text(encoding="utf-8"))
        valid, _ = _benchmark_record_validation(cached)
        if valid:
            return cached

    cpu = _benchmark_device_once("cpu")
    record: dict[str, object] = {
        **payload,
        "benchmark_config_hash": payload_hash,
        "requested_device": str(requested).lower(),
        "cpu_median_seconds": cpu["median_seconds"],
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_finite": False,
        "mps_close_to_cpu": False,
        "mps_faster_than_cpu": False,
        "selected_device": "cpu",
        "selection_reason": "CPU requested or MPS not benchmark-qualified",
    }
    if str(requested).lower() not in {"auto", "cpu", "mps"}:
        raise ValueError("requested device must be one of: auto, cpu, mps")
    if str(requested).lower() != "cpu" and torch.backends.mps.is_available():
        try:
            mps = _benchmark_device_once("mps")
            close = bool(
                np.allclose(
                    np.asarray(cpu["output"]),
                    np.asarray(mps["output"]),
                    rtol=_BENCHMARK_RTOL,
                    atol=_BENCHMARK_ATOL,
                )
            )
            faster = bool(float(mps["median_seconds"]) < float(cpu["median_seconds"]))
            record.update(
                {
                    "mps_median_seconds": mps["median_seconds"],
                    "mps_finite": bool(mps["finite"]),
                    "mps_close_to_cpu": close,
                    "mps_faster_than_cpu": faster,
                }
            )
            if bool(mps["finite"]) and close and faster:
                record.update(
                    {
                        "selected_device": "mps",
                        "selection_reason": "MPS is finite, CPU-parity verified, and faster",
                    }
                )
        except Exception as exc:  # benchmark failures are frozen as CPU decisions.
            record["mps_error"] = f"{type(exc).__name__}: {exc}"
    normalized_record = json.loads(json.dumps(record, sort_keys=True))
    normalized_record[_BENCHMARK_RECORD_HASH_FIELD] = benchmark_record_hash(
        normalized_record
    )
    valid, reason = _benchmark_record_validation(normalized_record)
    if not valid:
        raise RuntimeError(f"failed to create a valid TCN benchmark record: {reason}")
    path.write_text(
        json.dumps(normalized_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return normalized_record


def estimate_tcn_batch_bytes(batch_rows: int, target_length: int) -> int:
    """Conservative activation estimate for one bounded batch, not a policy cap."""
    batch = int(batch_rows)
    length = int(target_length)
    if batch < 1 or length < _MINIMUM_TARGET_LENGTH:
        raise ValueError("batch rows and target length must be positive TCN inputs")
    positions = length - TCN_PATCH_SIZE
    float_bytes = np.dtype(np.float32).itemsize
    activations = batch * positions * (
        1 + TCN_PATCH_SIZE + TCN_CHANNELS * (1 + len(TCN_DILATIONS))
    )
    return int(activations * float_bytes)


def tcn_compute_provenance(n_train: int, target_length: int) -> dict[str, object]:
    """Record the fixed batch-memory contract before execution eligibility is known."""
    return {
        "n_train": int(n_train),
        "target_length": int(target_length),
        "train_elements": int(n_train) * int(target_length),
        "tcn_max_train_elements": TCN_MAX_TRAIN_ELEMENTS,
        "maximum_training_batch_rows": TCN_BATCH_SIZE,
        "estimated_batch_activation_bytes": estimate_tcn_batch_bytes(
            TCN_BATCH_SIZE, int(target_length)
        ),
        "train_rows_sampled": False,
        "target_length_resampled": False,
    }


def tcn_runtime_estimate(
    benchmark_record: Mapping[str, object] | None,
    *,
    selected_device: str,
    n_train: int,
    target_length: int,
) -> dict[str, object]:
    """Return a deterministic inference-only lower bound from frozen timing data."""
    valid, reason = _benchmark_record_validation(benchmark_record)
    if not valid:
        return {"available": False, "reason": reason}
    if benchmark_record is None:  # Defensive narrowing for static readers.
        return {"available": False, "reason": "missing_benchmark_record"}
    time_key = f"{selected_device}_median_seconds"
    benchmark_seconds = benchmark_record.get(time_key)
    rows = benchmark_record.get("synthetic_rows")
    length = benchmark_record.get("synthetic_target_length")
    if (
        not isinstance(benchmark_seconds, (int, float))
        or not np.isfinite(float(benchmark_seconds))
        or float(benchmark_seconds) <= 0.0
        or not isinstance(rows, int)
        or not isinstance(length, int)
        or rows < 1
        or length <= TCN_PATCH_SIZE
    ):
        return {
            "available": False,
            "reason": "benchmark_record_missing_selected_device_timing",
        }
    positions = int(target_length) - TCN_PATCH_SIZE
    benchmark_positions = int(length) - TCN_PATCH_SIZE
    batch_rows = min(TCN_BATCH_SIZE, int(n_train))
    per_batch = (
        float(benchmark_seconds)
        * (float(batch_rows) / float(rows))
        * (float(positions) / float(benchmark_positions))
    )
    updates_per_fit = math.ceil(int(n_train) / TCN_BATCH_SIZE)
    max_fit_count = 2 * (5 + 1)
    max_update_count = max_fit_count * TCN_MAX_EPOCHS * updates_per_fit
    return {
        "available": True,
        "selected_device": selected_device,
        "benchmark_record_hash": benchmark_record[_BENCHMARK_RECORD_HASH_FIELD],
        "benchmark_median_seconds": float(benchmark_seconds),
        "estimated_forward_seconds_per_batch": per_batch,
        "max_fit_count": max_fit_count,
        "max_epochs": TCN_MAX_EPOCHS,
        "updates_per_fit": updates_per_fit,
        "max_training_update_count": max_update_count,
        "estimated_training_forward_lower_bound_seconds": per_batch * max_update_count,
        "assumption": "linear batch/time scaling; inference-only lower bound; backward and IO excluded",
    }


def _make_loader(rows: np.ndarray, *, seed: int) -> DataLoader[tuple[Tensor]]:
    values = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float32))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(values),
        batch_size=TCN_BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def _fit_model(train_rows: np.ndarray, *, seed: int, device: str) -> tuple[CausalTCN, int]:
    """Fit one frozen seed/model using only the supplied TRAIN-normal rows."""
    set_deterministic_seed(seed)
    model = build_tcn_model().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TCN_LEARNING_RATE, weight_decay=TCN_WEIGHT_DECAY
    )
    loss_fn = nn.HuberLoss()
    loader = _make_loader(train_rows, seed=seed)
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    epochs_completed = 0
    for epoch in range(TCN_MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        examples = 0
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            inputs, targets = _future_patch_targets(batch)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = loss_fn(prediction, targets)
            if not torch.isfinite(loss):
                raise ValueError("non-finite TCN TRAIN loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_cpu)
            examples += len(batch_cpu)
        current_loss = total_loss / max(1, examples)
        epochs_completed = epoch + 1
        if current_loss < best_loss - 1e-12:
            best_loss = current_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= TCN_PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("TCN fit did not produce a finite model state")
    model.load_state_dict(best_state)
    model.eval()
    return model, epochs_completed


def _residual_scores(model: CausalTCN, rows: np.ndarray, *, device: str) -> np.ndarray:
    """Return one mean absolute eight-patch residual score per trace."""
    dataset = TensorDataset(torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float32)))
    loader = DataLoader(dataset, batch_size=TCN_BATCH_SIZE, shuffle=False, num_workers=0)
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            inputs, targets = _future_patch_targets(batch)
            residual = torch.abs(model(inputs) - targets).mean(dim=(1, 2))
            values.append(residual.detach().cpu().numpy())
    output = np.concatenate(values).astype(float, copy=False)
    if not np.isfinite(output).all():
        raise ValueError("non-finite TCN residual scores")
    return output


def _score_one_seed(
    *,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Build deterministic OOF and final TEST residual scores for one seed."""
    folds = deterministic_tcn_fold_plan(len(train_rows), seed)
    train_oof = np.empty(len(train_rows), dtype=float)
    epochs: list[int] = []
    for fit_indices, query_indices in folds:
        model, completed = _fit_model(train_rows[fit_indices], seed=seed, device=device)
        try:
            train_oof[query_indices] = _residual_scores(
                model, train_rows[query_indices], device=device
            )
        finally:
            del model
            _release_device_memory(device)
        epochs.append(completed)
    final_model, completed = _fit_model(train_rows, seed=seed, device=device)
    try:
        test_scores = _residual_scores(final_model, test_rows, device=device)
    finally:
        del final_model
        _release_device_memory(device)
    epochs.append(completed)
    if not np.isfinite(train_oof).all() or not np.isfinite(test_scores).all():
        raise ValueError("non-finite TCN seed scores")
    return train_oof, test_scores, tuple(epochs)


def score_variants(first: np.ndarray, second: np.ndarray) -> dict[str, np.ndarray]:
    """Produce frozen scalar median and conservative conjunction score views.

    The candidate-level ``seed_intersection`` remains the conjunction of the
    two independently calibrated seed p-values in Task 7. Its scalar view is
    the pointwise minimum only for cache inspection and never replaces that
    conjunction rule.
    """
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("TCN seed score variants require matching finite arrays")
    return {
        "seed_median": np.median(np.stack((left, right), axis=0), axis=0),
        "seed_intersection": np.minimum(left, right),
    }


class CausalTCNNormalForecast:
    """Exp270-B TRAIN-normal causal TCN residual source.

    ``score`` emits the median variant directly and retains both per-seed score
    vectors in label-free provenance for the future cache writer. The two
    variants remain one ``normal_forecast`` evidence family.
    """

    name = TCN_SOURCE_NAME
    minimum_train = TCN_MIN_TRAIN

    def __init__(
        self,
        *,
        device: str = "cpu",
        benchmark_record: Mapping[str, object] | None = None,
    ) -> None:
        self.benchmark_record = _frozen_benchmark_record(benchmark_record)
        self.benchmark_record_verified, self.benchmark_verification_reason = (
            _benchmark_record_validation(self.benchmark_record)
        )
        self.benchmark_record_hash = (
            self.benchmark_record.get(_BENCHMARK_RECORD_HASH_FIELD)
            if self.benchmark_record is not None
            else None
        )
        self.benchmark_record_computed_hash = (
            benchmark_record_hash(self.benchmark_record)
            if self.benchmark_record is not None
            else None
        )
        self.device = choose_device(device, self.benchmark_record)

    def score(self, bundle: DatasetBundle, seed: int, artifact_dir: Path) -> SourceScore:
        del artifact_dir
        raw_train = np.asarray(bundle.train_normal, dtype=np.float32)
        if raw_train.ndim != 2:
            raise ValueError("TRAIN rows must have shape (batch, time)")
        target_length = int(raw_train.shape[1])
        train_rows = _validate_rows(
            raw_train, expected_length=target_length, name="TRAIN"
        )
        common_provenance: dict[str, object] = {
            "seed": int(seed),
            "seed_order": (SEED, SECOND_SEED),
            "n_train": int(len(train_rows)),
            "minimum_train": self.minimum_train,
            "target_length": target_length,
            "target_length_from_train_only": True,
            "architecture": {
                "causal_residual_blocks": len(TCN_DILATIONS),
                "channels": TCN_CHANNELS,
                "kernel_size": TCN_KERNEL_SIZE,
                "dilations": TCN_DILATIONS,
                "activation": "GELU",
                "dropout": TCN_DROPOUT,
                "future_patch": TCN_PATCH_SIZE,
            },
            "training": {
                "optimizer": "AdamW",
                "learning_rate": TCN_LEARNING_RATE,
                "weight_decay": TCN_WEIGHT_DECAY,
                "batch_size": TCN_BATCH_SIZE,
                "maximum_epochs": TCN_MAX_EPOCHS,
                "patience": TCN_PATIENCE,
                "loss": "Huber",
                "early_stopping_metric": "fit_train_huber_loss",
                "held_out_validation": False,
            },
            "device": self.device,
            "selected_device": self.device,
            "benchmark_record": self.benchmark_record,
            "benchmark_record_hash": self.benchmark_record_hash,
            "benchmark_record_computed_hash": self.benchmark_record_computed_hash,
            "benchmark_record_verified": self.benchmark_record_verified,
            "benchmark_verification_reason": self.benchmark_verification_reason,
            "evidence_family": TCN_EVIDENCE_FAMILY,
            "variants": ("seed_median", "seed_intersection"),
            "seed_intersection_rule": "both independently calibrated seed p-values",
            "contains_test_labels": False,
            **(
                tcn_compute_provenance(len(train_rows), target_length)
                if target_length >= _MINIMUM_TARGET_LENGTH
                else {}
            ),
        }
        if target_length < _MINIMUM_TARGET_LENGTH:
            return SourceScore(
                source=self.name,
                variant="seed_median",
                train_oof=np.asarray([], dtype=float),
                test_scores=np.asarray([], dtype=float),
                eligible=False,
                abstained=True,
                abstention_reason="target_length_below_9",
                provenance={
                    **common_provenance,
                    "training_row_coverage": "none: target-length eligibility abstention",
                    "training_executed": False,
                    "benchmark_executed_by_source": False,
                    "cross_fitted": False,
                    "self_reference_violations": 0,
                },
            )
        if len(train_rows) < self.minimum_train:
            return SourceScore(
                source=self.name,
                variant="seed_median",
                train_oof=np.asarray([], dtype=float),
                test_scores=np.asarray([], dtype=float),
                eligible=False,
                abstained=True,
                abstention_reason=f"n_train_below_{self.minimum_train}",
                provenance={
                    **common_provenance,
                    "training_row_coverage": "none: n_train eligibility abstention",
                    "training_executed": False,
                    "benchmark_executed_by_source": False,
                    "cross_fitted": False,
                    "self_reference_violations": 0,
                },
            )
        train_elements = int(len(train_rows)) * target_length
        if train_elements > TCN_MAX_TRAIN_ELEMENTS:
            return SourceScore(
                source=self.name,
                variant="seed_median",
                train_oof=np.asarray([], dtype=float),
                test_scores=np.asarray([], dtype=float),
                eligible=False,
                abstained=True,
                abstention_reason="train_work_exceeds_5000000",
                provenance={
                    **common_provenance,
                    "train_elements": train_elements,
                    "tcn_max_train_elements": TCN_MAX_TRAIN_ELEMENTS,
                    "compute_safety_boundary": "TRAIN normal row count times TRAIN target length",
                    "training_row_coverage": "none: compute safety abstention",
                    "training_executed": False,
                    "benchmark_executed_by_source": False,
                    "cross_fitted": False,
                    "self_reference_violations": 0,
                },
            )

        test_rows = _validate_rows(
            bundle.test_values, expected_length=target_length, name="TEST"
        )
        common_provenance["n_test"] = int(len(test_rows))
        common_provenance["runtime_estimate"] = tcn_runtime_estimate(
            self.benchmark_record,
            selected_device=self.device,
            n_train=len(train_rows),
            target_length=target_length,
        )
        seeds = (SEED, SECOND_SEED)
        per_seed = {
            current_seed: _score_one_seed(
                train_rows=train_rows,
                test_rows=test_rows,
                seed=current_seed,
                device=self.device,
            )
            for current_seed in seeds
        }
        first_train, first_test, first_epochs = per_seed[SEED]
        second_train, second_test, second_epochs = per_seed[SECOND_SEED]
        train_variants = score_variants(first_train, second_train)
        test_variants = score_variants(first_test, second_test)
        folds = deterministic_tcn_fold_plan(len(train_rows), int(seed))
        overlap_counts = tuple(
            int(len(np.intersect1d(fit, query))) for fit, query in folds
        )
        return SourceScore(
            source=self.name,
            variant="seed_median",
            train_oof=train_variants["seed_median"],
            test_scores=test_variants["seed_median"],
            eligible=True,
            abstained=False,
            abstention_reason=None,
            provenance={
                **common_provenance,
                "cross_fitted": True,
                "training_executed": True,
                "benchmark_executed_by_source": False,
                "training_row_coverage": "all TRAIN-normal rows in every fold and final fit",
                "global_fold_count": len(folds),
                "self_reference_violations": int(sum(overlap_counts)),
                "outer_fit_query_overlap_counts": overlap_counts,
                "seed_epochs_completed": {
                    str(SEED): first_epochs,
                    str(SECOND_SEED): second_epochs,
                },
                "per_seed_train_oof": {
                    str(SEED): tuple(float(value) for value in first_train),
                    str(SECOND_SEED): tuple(float(value) for value in second_train),
                },
                "per_seed_test_scores": {
                    str(SEED): tuple(float(value) for value in first_test),
                    str(SECOND_SEED): tuple(float(value) for value in second_test),
                },
                "seed_intersection_scalar_view": tuple(
                    float(value) for value in test_variants["seed_intersection"]
                ),
            },
        )
