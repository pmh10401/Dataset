"""Deterministic, hash-bound Exp472 paper reporting.

The report is deliberately a post-evaluation publisher: it accepts only the
opaque capability issued after persisted evaluation replay, and never imports a
scorer or a label reader.  Frozen terminal scores plus private-label evidence
therefore flow through full replay before the capability reaches the fifteen
scientific leaves; the completion marker only binds their manifest, avoiding a
recursive self-hash.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import csv
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import zlib

from .artifacts_antigravity import validate_diagnostic_exp472_antigravity
from .contracts_antigravity import canonical_json_bytes_exp472_antigravity
from .evaluation_antigravity import _require_verified_persisted_evaluation_antigravity


REPORT_LEAVES_ANTIGRAVITY = (
    "exp472_paper_report_antigravity.html",
    "exp472_paper_summary_ko_antigravity.md",
    "exp472_methods_appendix_antigravity.md",
    "exp472_aggregate_metrics_antigravity.csv",
    "exp472_per_dataset_metrics_antigravity.csv",
    "exp472_paired_e_minus_d_antigravity.csv",
    "exp472_upsampling_subset_antigravity.csv",
    "exp472_shape_diagnostics_antigravity.csv",
    "exp472_denominator_audit_antigravity.csv",
    "exp472_roc_coordinates_antigravity.csv",
    "exp472_pr_coordinates_antigravity.csv",
    "exp472_roc_curve_antigravity.svg",
    "exp472_pr_curve_antigravity.svg",
    "exp472_curves_antigravity.png",
    "exp472_backend_provenance_antigravity.json",
)
REPORT_MANIFEST_NAME_ANTIGRAVITY = "exp472_report_manifest_antigravity.json"
REPORT_COMPLETE_NAME_ANTIGRAVITY = ".exp472_report_complete_antigravity.json"
_MANIFEST_SCHEMA_ANTIGRAVITY = "exp472-report-manifest-antigravity.v2"
_COMPLETE_SCHEMA_ANTIGRAVITY = "exp472-report-complete-antigravity.v2"
_SHAPE_SCHEMA_ANTIGRAVITY = "exp472-shape-diagnostic-antigravity.v3"
_SHAPE_CSV_FIELDS_ANTIGRAVITY = (
    "dataset", "schema", "trace_state", "alignment_status",
    "invocation_count", "row_invocation_count", "makima_row_invocation_count",
    "upsampled_rows", "input_identity_sha256", "result_sha256",
    "invocation_order_sha256", "invocation_aggregate_sha256", "ordinal",
    "phase_tag", "source_tag", "target_len", "row_count", "singleton_rows",
    "equal_rows", "polyphase_downsample_rows", "makima_rows", "makima_row_indices",
    "overshoot", "undershoot", "source_extrema", "output_extrema", "added_extrema",
    "input_sha256", "output_sha256", "input_evidence", "output_evidence",
    "invocation_sha256", "diagnostic_sha256",
)


def _digest_antigravity(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"Exp472 {name} must be a lowercase SHA-256")
    return value


def _plain_antigravity(value: object) -> object:
    """Freeze a finite, JSON-only public record before it enters a leaf."""
    return json.loads(canonical_json_bytes_exp472_antigravity(value).decode("utf-8"))


def _safe_root_antigravity(root: Path, *, create: bool) -> Path:
    target = Path(root)
    if not target.is_absolute() or "antigravity" not in target.name:
        raise ValueError("Exp472 report root must be an absolute antigravity path")
    if create:
        target.mkdir(parents=True, exist_ok=True)
    try:
        details = target.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp472 report root is missing") from error
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("Exp472 report root is unsafe")
    return target


def _leaf_path_antigravity(root: Path, name: str) -> Path:
    if name not in REPORT_LEAVES_ANTIGRAVITY and name not in {REPORT_MANIFEST_NAME_ANTIGRAVITY, REPORT_COMPLETE_NAME_ANTIGRAVITY}:
        raise ValueError("Exp472 report leaf is not authorized")
    if PurePosixPath(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Exp472 report leaf path is unsafe")
    return root / name


def _fsync_directory_antigravity(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_antigravity(path: Path) -> bytes:
    try:
        linked = path.lstat()
    except FileNotFoundError as error:
        raise ValueError("Exp472 report leaf is missing") from error
    if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
        raise ValueError("Exp472 report leaf is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("Exp472 report leaf is unsafe") from error
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_dev != linked.st_dev
                or opened.st_ino != linked.st_ino or opened.st_nlink != 1):
            raise ValueError("Exp472 report leaf is unsafe")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        os.close(descriptor)


def _write_create_or_identical_antigravity(path: Path, payload: bytes) -> None:
    """Durably claim one leaf; retries may only reuse byte-identical output."""
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if _read_regular_antigravity(path) != payload:
            raise ValueError("Exp472 report leaf already exists with different bytes")
        return
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Exp472 report write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory_antigravity(path.parent)
    if _read_regular_antigravity(path) != payload:
        raise ValueError("Exp472 report leaf changed after publication")


def _csv_antigravity(fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=tuple(fieldnames), lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        plain = _plain_antigravity(dict(row))
        if not isinstance(plain, dict):
            raise ValueError("Exp472 CSV row differs")
        writer.writerow({name: "" if plain.get(name) is None else plain.get(name) for name in fieldnames})
    return stream.getvalue().encode("utf-8")


def _evaluation_record_antigravity(evaluation: object) -> Mapping[str, object]:
    if not callable(getattr(evaluation, "artifact_record", None)):
        raise ValueError("Exp472 reporting requires a verified evaluation artifact")
    record = _plain_antigravity(evaluation.artifact_record())
    if not isinstance(record, dict) or record.get("schema") != "exp472-evaluation-antigravity.v3" or not isinstance(record.get("private_label_evidence_sha256"),str):
        raise ValueError("Exp472 evaluation artifact differs")
    _digest_antigravity(record["private_label_evidence_sha256"], "private label evidence digest")
    digest = record.get("evaluation_sha256")
    bare = {key: value for key, value in record.items() if key != "evaluation_sha256"}
    if _digest_antigravity(digest, "evaluation digest") != sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest():
        raise ValueError("Exp472 evaluation digest differs")
    return record


def _evaluation_digest_antigravity(evaluation: object) -> str:
    return _digest_antigravity(_evaluation_record_antigravity(evaluation).get("evaluation_sha256"), "evaluation digest")


def _reuse_authority_digest_antigravity(evaluation: object) -> str:
    """Bind the complete immutable Exp471 reuse authority, not one leaf of it."""
    authority = getattr(evaluation, "authority", None)
    fields = getattr(authority, "__dataclass_fields__", None)
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("Exp472 Exp471 reuse authority differs")
    record: dict[str, object] = {}
    for name in fields:
        value = getattr(authority, name, None)
        record[name] = list(value) if isinstance(value, tuple) else value
    for name in ("exp471_source_authority_sha256", "exp471_cache_manifest_sha256", "exp471_evaluation_sha256", "exp471_report_manifest_sha256", "database_sha256"):
        _digest_antigravity(record.get(name), f"Exp471 {name}")
    return sha256(canonical_json_bytes_exp472_antigravity(record)).hexdigest()


def _methods_antigravity(evaluation: object) -> tuple[str, ...]:
    methods = tuple(getattr(evaluation, "method_order", ()))
    expected = ("A", "B", "C", "D", "E", "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d")
    if methods != expected:
        raise ValueError("Exp472 report requires all 11 frozen methods")
    return methods


def _trace_by_dataset_antigravity(
    value: Sequence[Mapping[str, object]], expected_datasets: tuple[str, ...],
) -> Mapping[str, Mapping[str, object]]:
    """Authenticate exactly one sealed v3 trace for every E dataset."""
    if not expected_datasets or len(set(expected_datasets)) != len(expected_datasets):
        raise ValueError("Exp472 shape diagnostics differ")
    traces: dict[str, Mapping[str, object]] = {}
    for supplied in value:
        if not isinstance(supplied, Mapping) or type(supplied.get("dataset")) is not str:
            raise ValueError("Exp472 shape diagnostics differ")
        dataset = supplied["dataset"]
        if dataset not in expected_datasets or dataset in traces:
            raise ValueError("Exp472 shape diagnostics differ")
        row = dict(supplied)
        try:
            validate_diagnostic_exp472_antigravity(row, dataset=dataset)
        except (TypeError, ValueError) as error:
            raise ValueError("Exp472 shape diagnostics differ") from error
        if row.get("schema") != _SHAPE_SCHEMA_ANTIGRAVITY or row.get("dataset") != dataset:
            raise ValueError("Exp472 shape diagnostics differ")
        traces[dataset] = row
    if set(traces) != set(expected_datasets):
        raise ValueError("Exp472 shape diagnostics differ")
    return traces


def _shape_csv_rows_antigravity(
    value: Sequence[Mapping[str, object]], expected_datasets: tuple[str, ...],
) -> tuple[Mapping[str, object], ...]:
    """Flatten only strict, sealed v3 actual-call traces.

    The output intentionally contains one record per actual Exp450 aligner
    invocation.  A dataset may therefore expose several target lengths and a
    repeated Makima row exposure.  The trace aggregate is repeated per record
    so a CSV row remains independently attributable to its sealed diagnostic.
    """
    traces = _trace_by_dataset_antigravity(value, expected_datasets)
    flattened: list[Mapping[str, object]] = []
    for dataset in expected_datasets:
        row = traces[dataset]
        aggregate = {
            name: row[name]
            for name in (
                "dataset", "schema", "trace_state", "alignment_status",
                "invocation_count", "row_invocation_count",
                "makima_row_invocation_count", "upsampled_rows",
                "input_identity_sha256", "result_sha256",
                "invocation_order_sha256", "invocation_aggregate_sha256",
                "diagnostic_sha256",
            )
        }
        for invocation in row["invocations"]:
            flattened.append({
                **aggregate,
                **{name: invocation[name] for name in (
                    "ordinal", "phase_tag", "source_tag", "target_len", "row_count",
                    "singleton_rows", "equal_rows", "polyphase_downsample_rows",
                    "makima_rows", "overshoot", "undershoot", "source_extrema",
                    "output_extrema", "added_extrema", "input_sha256", "output_sha256",
                    "invocation_sha256",
                )},
                "makima_row_indices": canonical_json_bytes_exp472_antigravity(invocation["makima_row_indices"]).decode("utf-8").strip(),
                "input_evidence": canonical_json_bytes_exp472_antigravity(invocation["input_evidence"]).decode("utf-8").strip(),
                "output_evidence": canonical_json_bytes_exp472_antigravity(invocation["output_evidence"]).decode("utf-8").strip(),
            })
    return tuple(flattened)


def _bind_trace_to_evaluation_antigravity(
    evaluation: object,
    shape_diagnostics: Sequence[Mapping[str, object]],
    expected_datasets: tuple[str, ...],
) -> None:
    """Require a real evaluation's digest-bound trace summary to match leaves."""
    record = _evaluation_record_antigravity(evaluation)
    summary = record.get("diagnostic_summary")
    if not isinstance(summary, Mapping) or set(summary) != set(expected_datasets):
        raise ValueError("Exp472 evaluation diagnostic summary differs")
    traces = _trace_by_dataset_antigravity(shape_diagnostics, expected_datasets)
    for dataset in expected_datasets:
        item = summary[dataset]
        trace = traces[dataset]
        if not isinstance(item, Mapping) or trace.get("dataset") != dataset:
            raise ValueError("Exp472 evaluation diagnostic summary differs")
        if (
            type(item.get("upsampled_rows")) is not int
            or type(item.get("makima_row_invocation_count")) is not int
            or _digest_antigravity(item.get("diagnostic_sha256"), "evaluation diagnostic digest") != trace["diagnostic_sha256"]
            or trace["upsampled_rows"] != item["upsampled_rows"]
            or trace["makima_row_invocation_count"] != item["makima_row_invocation_count"]
        ):
            raise ValueError("Exp472 evaluation diagnostic summary differs")


def _mapping_antigravity(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Exp472 {name} differs")
    return value


def _paired_interpretation_antigravity(evaluation: object) -> tuple[str, str]:
    """Return English and Korean wording without treating zero-crossing as equivalence."""
    pairs = tuple(getattr(evaluation, "paired_statistics", ()))
    for row in pairs:
        if isinstance(row, Mapping) and row.get("comparison") == "E-D" and row.get("population") == "full" and row.get("metric") == "average_precision":
            low, high = row.get("ci_lower"), row.get("ci_upper")
            if type(low) in (int, float) and type(high) in (int, float) and math.isfinite(low) and math.isfinite(high):
                if low > 0:
                    return ("Makima improves AP over PCHIP on the frozen paired interval.", "고정된 짝지은 구간에서 Makima는 PCHIP보다 AP가 개선되었습니다.")
                if high < 0:
                    return ("Makima degrades AP relative to PCHIP on the frozen paired interval.", "고정된 짝지은 구간에서 Makima는 PCHIP보다 AP가 저하되었습니다.")
    return ("No decisive paired AP difference is established; this is not evidence of equivalence.", "짝지은 AP 차이에 대한 결정적 근거는 없습니다. 이는 동등성의 증거가 아닙니다.")


def _curve_rows_antigravity(evaluation: object, key: str, x_name: str, y_name: str) -> tuple[Mapping[str, object], ...]:
    curves = _mapping_antigravity(getattr(evaluation, "curves", None), "curve table")
    table = _mapping_antigravity(curves.get(key), f"{key} curve")
    if table.get("x_name") != x_name or table.get("y_name") != y_name:
        raise ValueError("Exp472 curve axes differ")
    rows = tuple(table.get("rows", ()))
    methods = set()
    result: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("method") not in _methods_antigravity(evaluation):
            raise ValueError("Exp472 curve coordinate differs")
        x, y = row.get(x_name), row.get(y_name)
        if type(x) not in (int, float) or type(y) not in (int, float) or not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("Exp472 curve coordinate differs")
        methods.add(str(row["method"])); result.append(dict(row))
    if methods != set(_methods_antigravity(evaluation)):
        raise ValueError("Exp472 curve table lacks a frozen method")
    return tuple(result)


def _svg_antigravity(rows: Sequence[Mapping[str, object]], x_name: str, y_name: str, title: str) -> bytes:
    ordered = sorted(rows, key=lambda row: (str(row["method"]), float(row[x_name]), float(row[y_name])))
    polylines: list[str] = []
    for method in sorted({str(row["method"]) for row in ordered}):
        points = " ".join(f"{40 + 240 * float(row[x_name]):.6f},{260 - 220 * float(row[y_name]):.6f}" for row in ordered if row["method"] == method)
        polylines.append(f'<polyline data-method="{method}" points="{points}" fill="none" stroke="#1f4e79"/>')
    payload = (f'<svg xmlns="http://www.w3.org/2000/svg" width="320" height="300" viewBox="0 0 320 300"><title>{title}</title><text x="40" y="20">{title}</text><text x="40" y="292">{x_name}</text><text x="2" y="40">{y_name}</text><line x1="40" y1="260" x2="280" y2="260" stroke="#000"/><line x1="40" y1="40" x2="40" y2="260" stroke="#000"/>{''.join(polylines)}</svg>\n')
    return payload.encode("utf-8")


def _png_antigravity(roc_rows: Sequence[Mapping[str, object]], pr_rows: Sequence[Mapping[str, object]]) -> bytes:
    """Render a small deterministic ROC/PR raster from the canonical rows."""
    def chunk(name: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + name + payload + struct.pack(">I", zlib.crc32(name + payload) & 0xffffffff)
    width, height = 400, 240
    pixels = bytearray([255, 255, 255, 255]) * (width * height)
    def point(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            position = 4 * (y * width + x); pixels[position:position + 4] = bytes((*color, 255))
    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0); sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1); error = dx + dy
        while True:
            point(x0, y0, color)
            if (x0, y0) == (x1, y1): break
            twice = 2 * error
            if twice >= dy: error += dy; x0 += sx
            if twice <= dx: error += dx; y0 += sy
    for offset in (0, 190):
        line(offset + 20, 210, offset + 170, 210, (0, 0, 0)); line(offset + 20, 30, offset + 20, 210, (0, 0, 0))
    for rows, x_name, y_name, offset, color in ((roc_rows, "false_positive_rate", "true_positive_rate", 0, (31, 78, 121)), (pr_rows, "recall", "precision", 190, (180, 70, 30))):
        for method in sorted({str(row["method"]) for row in rows}):
            series = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: (float(row[x_name]), float(row[y_name])),
            )
            previous: tuple[int, int] | None = None
            for row in series:
                coordinate = (offset + 20 + round(150 * float(row[x_name])), 210 - round(180 * float(row[y_name])))
                point(*coordinate, color)
                if previous is not None: line(*previous, *coordinate, color)
                previous = coordinate
    raw = b"".join(b"\x00" + bytes(pixels[row * width * 4:(row + 1) * width * 4]) for row in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, level=9)) + chunk(b"IEND", b"")


@dataclass(frozen=True, slots=True)
class Exp472ReportAuthorityAntigravity:
    """The non-recursive report authority bound by a separate marker."""

    leaf_sha256: Mapping[str, str]
    evaluation_sha256: str
    e_manifest_sha256: str
    exp471_reuse_authority_sha256: str
    code_map_lock_sha256: str
    report_manifest_sha256: str

    def __post_init__(self) -> None:
        leaves = dict(self.leaf_sha256)
        if tuple(leaves) != REPORT_LEAVES_ANTIGRAVITY:
            raise ValueError("Exp472 report leaf inventory differs")
        for name, digest in leaves.items():
            _digest_antigravity(digest, name)
        for name in ("evaluation_sha256", "e_manifest_sha256", "exp471_reuse_authority_sha256", "code_map_lock_sha256", "report_manifest_sha256"):
            _digest_antigravity(getattr(self, name), name)
        object.__setattr__(self, "leaf_sha256", dict(leaves))


def _report_leaf_payloads_antigravity(evaluation: object, worker_record: Mapping[str, object], shape_diagnostics: Sequence[Mapping[str, object]]) -> Mapping[str, bytes]:
    methods = _methods_antigravity(evaluation)
    roc_rows = _curve_rows_antigravity(evaluation, "roc", "false_positive_rate", "true_positive_rate")
    pr_rows = _curve_rows_antigravity(evaluation, "pr", "recall", "precision")
    english, korean = _paired_interpretation_antigravity(evaluation)
    macro = _mapping_antigravity(getattr(evaluation, "macro", None), "macro metrics")
    subset_macro = _mapping_antigravity(getattr(evaluation, "subset_macro", None), "subset macro metrics")
    for collection in (macro, subset_macro):
        if set(collection) != set(methods):
            raise ValueError("Exp472 report metrics lack a frozen method")
    aggregate_rows = []
    for scope, metrics in (("full", macro), ("upsampled_rows_gt_zero", subset_macro)):
        for method in methods:
            row = _mapping_antigravity(metrics[method], "metric row")
            aggregate_rows.append({"scope": scope, "method": method, "average_precision": row.get("average_precision", row.get("average_precision_macro")), "auroc": row.get("auroc", row.get("auroc_macro")), "average_precision_denominator": row.get("average_precision_denominator"), "auroc_denominator": row.get("auroc_denominator")})
    per_dataset = tuple(getattr(evaluation, "per_dataset", ()))
    subset_dataset = tuple(getattr(evaluation, "subset_per_dataset", ()))
    paired = tuple(getattr(evaluation, "paired_statistics", ()))
    denominators = _mapping_antigravity(getattr(evaluation, "denominators", None), "denominator audit")
    denominator_rows = []
    for scope in sorted(denominators):
        row = _mapping_antigravity(denominators[scope], "denominator row")
        denominator_rows.append({"scope": scope, "method": "all", "dataset_count": row.get("dataset_count"), "observation_count": row.get("observation_count"), "average_precision_denominator": None, "auroc_denominator": None, "average_precision_excluded": None, "auroc_excluded": None})
    for method in methods:
        row = _mapping_antigravity(macro[method], "macro denominator")
        denominator_rows.append({"scope": "full_population", "method": method, "dataset_count": None, "observation_count": None, "average_precision_denominator": row.get("average_precision_denominator"), "auroc_denominator": row.get("auroc_denominator"), "average_precision_excluded": row.get("average_precision_excluded"), "auroc_excluded": row.get("auroc_excluded")})
    for method in methods:
        row = _mapping_antigravity(subset_macro[method], "subset macro denominator")
        denominator_rows.append({"scope": "upsampled_rows_gt_zero", "method": method, "dataset_count": None, "observation_count": None, "average_precision_denominator": row.get("average_precision_denominator"), "auroc_denominator": row.get("auroc_denominator"), "average_precision_excluded": row.get("average_precision_excluded"), "auroc_excluded": row.get("auroc_excluded")})
    expected_shape_datasets = tuple(
        str(row.get("dataset")) for row in per_dataset
        if isinstance(row, Mapping) and row.get("method") == "E"
    )
    _bind_trace_to_evaluation_antigravity(evaluation, shape_diagnostics, expected_shape_datasets)
    shape_rows = _shape_csv_rows_antigravity(shape_diagnostics, expected_shape_datasets)
    trace_aggregates = _trace_by_dataset_antigravity(shape_diagnostics, expected_shape_datasets)
    denominator_rows.append({
        "scope": "verified_trace", "method": "E", "dataset_count": len(trace_aggregates),
        "observation_count": None, "average_precision_denominator": None,
        "auroc_denominator": None, "average_precision_excluded": None,
        "auroc_excluded": None,
        "unique_upsampled_slots": sum(int(row["upsampled_rows"]) for row in trace_aggregates.values()),
        "makima_row_invocation_count": sum(int(row["makima_row_invocation_count"]) for row in trace_aggregates.values()),
    })
    html = "".join(("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Exp472 Makima versus PCHIP</title></head><body>", "<h1>Exp472: Makima versus PCHIP</h1>", f"<p>{english}</p>", "<p>ROC: x-axis False-positive rate; y-axis True-positive rate.</p>", "<p>Precision–recall: x-axis Recall; y-axis Precision.</p>", "<p>Context includes A–E and six frozen public baselines (11 methods).</p>", "</body></html>\n"))
    summary = "\n".join(("# Exp472 Makima 업샘플링 보조 실험", "", korean, "", "ROC의 x축은 False-positive rate이고 y축은 True-positive rate입니다.", "PR의 x축은 Recall이고 y축은 Precision입니다.", "`upsampled_rows`는 검증된 실제 (TRAIN/TEST, 행 번호) 고유 슬롯 수이며, 이 값이 0보다 큰 데이터셋만 부분집합에 포함합니다.", "`makima_row_invocation_count`는 같은 슬롯이 여러 경로에서 다시 처리된 경우까지 센 호출 노출 횟수이므로 고유 슬롯 수와 혼용하지 않습니다.", "구간이 0을 포함하면 동등하다고 결론내리지 않습니다.", ""))
    appendix = "\n".join(("# Methods appendix", "", "E = polyphase downsampling + Makima upsampling + area/local-mean imaging.", "D = frozen Exp471 polyphase + PCHIP + area/local-mean control.", "Only E was newly scored; A–D and public baselines are reused context.", "The diagnostic table has one row per actual Exp450 aligner invocation and retains phase/source tags, call-specific target length, input/output evidence, route counts, and trace aggregates.", "Subset membership uses only verified unique upsampled slots, not repeated Makima invocation exposure.", ""))
    return {
        "exp472_paper_report_antigravity.html": html.encode("utf-8"),
        "exp472_paper_summary_ko_antigravity.md": summary.encode("utf-8"),
        "exp472_methods_appendix_antigravity.md": appendix.encode("utf-8"),
        "exp472_aggregate_metrics_antigravity.csv": _csv_antigravity(("scope", "method", "average_precision", "auroc", "average_precision_denominator", "auroc_denominator"), aggregate_rows),
        "exp472_per_dataset_metrics_antigravity.csv": _csv_antigravity(("method", "dataset", "average_precision", "average_precision_exclusion", "auroc", "auroc_exclusion"), [dict(row) for row in per_dataset]),
        "exp472_paired_e_minus_d_antigravity.csv": _csv_antigravity(("comparison", "population", "metric", "mean_delta", "ci_lower", "ci_upper", "wins", "ties", "losses", "seed", "draw_count", "percentile", "tie_tolerance"), [dict(row) for row in paired]),
        "exp472_upsampling_subset_antigravity.csv": _csv_antigravity(("method", "dataset", "average_precision", "average_precision_exclusion", "auroc", "auroc_exclusion"), [dict(row) for row in subset_dataset]),
        "exp472_shape_diagnostics_antigravity.csv": _csv_antigravity(_SHAPE_CSV_FIELDS_ANTIGRAVITY, shape_rows),
        "exp472_denominator_audit_antigravity.csv": _csv_antigravity(("scope", "method", "dataset_count", "observation_count", "average_precision_denominator", "auroc_denominator", "average_precision_excluded", "auroc_excluded", "unique_upsampled_slots", "makima_row_invocation_count"), denominator_rows),
        "exp472_roc_coordinates_antigravity.csv": _csv_antigravity(("method", "false_positive_rate", "true_positive_rate", "threshold"), roc_rows),
        "exp472_pr_coordinates_antigravity.csv": _csv_antigravity(("method", "recall", "precision"), pr_rows),
        "exp472_roc_curve_antigravity.svg": _svg_antigravity(roc_rows, "false_positive_rate", "true_positive_rate", "ROC: False-positive rate / True-positive rate"),
        "exp472_pr_curve_antigravity.svg": _svg_antigravity(pr_rows, "recall", "precision", "PR: Recall / Precision"),
        "exp472_curves_antigravity.png": _png_antigravity(roc_rows, pr_rows),
        "exp472_backend_provenance_antigravity.json": canonical_json_bytes_exp472_antigravity(_plain_antigravity(worker_record)),
    }


def _manifest_record_antigravity(authority: Exp472ReportAuthorityAntigravity, worker_record: Mapping[str, object]) -> dict[str, object]:
    bare = {"schema": _MANIFEST_SCHEMA_ANTIGRAVITY, "leaves": dict(authority.leaf_sha256), "evaluation_sha256": authority.evaluation_sha256, "e_manifest_sha256": authority.e_manifest_sha256, "exp471_reuse_authority_sha256": authority.exp471_reuse_authority_sha256, "code_map_lock_sha256": authority.code_map_lock_sha256, "backend_worker": _plain_antigravity(worker_record)}
    return {**bare, "report_manifest_sha256": sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()}


def write_exp472_report_antigravity(evaluation: object, root: Path, worker_record: Mapping[str, object], *, e_manifest_sha256: str, code_map_lock_sha256: str, shape_diagnostics: Sequence[Mapping[str, object]] = ()) -> Exp472ReportAuthorityAntigravity:
    """Publish exactly 15 leaves from a trusted-reader capability only."""
    evaluation = _require_verified_persisted_evaluation_antigravity(evaluation)
    target = _safe_root_antigravity(Path(root), create=True)
    worker = _mapping_antigravity(worker_record, "backend worker provenance")
    if not {"logical_cpu_count", "cpu_workers", "tensor_backend", "mlx_fallback_reason", "pytorch_available", "pytorch_version", "scipy_available", "scipy_version", "mlx_available", "mlx_version", "mlx_metal_available", "mlx_metal_version", "mps_available", "inner_numerical_threads"}.issubset(worker):
        raise ValueError("Exp472 backend worker provenance is incomplete")
    if type(worker["logical_cpu_count"]) is not int or type(worker["cpu_workers"]) is not int or worker["cpu_workers"] != max(1, worker["logical_cpu_count"] - 1):
        raise ValueError("Exp472 CPU worker provenance differs")
    payloads = _report_leaf_payloads_antigravity(evaluation, worker, tuple(shape_diagnostics))
    if tuple(payloads) != REPORT_LEAVES_ANTIGRAVITY:
        raise ValueError("Exp472 report leaf inventory differs")
    for name in REPORT_LEAVES_ANTIGRAVITY:
        _write_create_or_identical_antigravity(_leaf_path_antigravity(target, name), payloads[name])
    reuse_digest = _reuse_authority_digest_antigravity(evaluation)
    provisional = Exp472ReportAuthorityAntigravity(
        {name: sha256(payloads[name]).hexdigest() for name in REPORT_LEAVES_ANTIGRAVITY}, _evaluation_digest_antigravity(evaluation), _digest_antigravity(e_manifest_sha256, "E manifest digest"), reuse_digest, _digest_antigravity(code_map_lock_sha256, "code-map lock digest"), "0" * 64,
    )
    manifest = _manifest_record_antigravity(provisional, worker)
    final = Exp472ReportAuthorityAntigravity(provisional.leaf_sha256, provisional.evaluation_sha256, provisional.e_manifest_sha256, provisional.exp471_reuse_authority_sha256, provisional.code_map_lock_sha256, str(manifest["report_manifest_sha256"]))
    _write_create_or_identical_antigravity(_leaf_path_antigravity(target, REPORT_MANIFEST_NAME_ANTIGRAVITY), canonical_json_bytes_exp472_antigravity(manifest))
    marker = {"schema": _COMPLETE_SCHEMA_ANTIGRAVITY, "report_manifest": REPORT_MANIFEST_NAME_ANTIGRAVITY, "report_manifest_sha256": final.report_manifest_sha256}
    _write_create_or_identical_antigravity(_leaf_path_antigravity(target, REPORT_COMPLETE_NAME_ANTIGRAVITY), canonical_json_bytes_exp472_antigravity(marker))
    if not verify_exp472_report_antigravity(target, final):
        raise ValueError("Exp472 report verification failed after publication")
    return final


def verify_exp472_report_antigravity(root: Path, authority: Exp472ReportAuthorityAntigravity) -> bool:
    """Reopen all published leaves and the marker chain without running science."""
    if not isinstance(authority, Exp472ReportAuthorityAntigravity):
        raise ValueError("Exp472 report authority differs")
    target = _safe_root_antigravity(Path(root), create=False)
    leaves = {name: sha256(_read_regular_antigravity(_leaf_path_antigravity(target, name))).hexdigest() for name in REPORT_LEAVES_ANTIGRAVITY}
    if leaves != dict(authority.leaf_sha256):
        raise ValueError("Exp472 report leaf digest differs")
    raw_manifest = _read_regular_antigravity(_leaf_path_antigravity(target, REPORT_MANIFEST_NAME_ANTIGRAVITY))
    raw_marker = _read_regular_antigravity(_leaf_path_antigravity(target, REPORT_COMPLETE_NAME_ANTIGRAVITY))
    try:
        manifest = json.loads(raw_manifest.decode("utf-8")); marker = json.loads(raw_marker.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exp472 report authority JSON is invalid") from error
    if raw_manifest != canonical_json_bytes_exp472_antigravity(manifest) or raw_marker != canonical_json_bytes_exp472_antigravity(marker):
        raise ValueError("Exp472 report authority JSON is noncanonical")
    bare = {key: value for key, value in manifest.items() if key != "report_manifest_sha256"}
    if (set(manifest) != {"schema", "leaves", "evaluation_sha256", "e_manifest_sha256", "exp471_reuse_authority_sha256", "code_map_lock_sha256", "backend_worker", "report_manifest_sha256"} or manifest.get("schema") != _MANIFEST_SCHEMA_ANTIGRAVITY or manifest.get("leaves") != leaves or manifest.get("evaluation_sha256") != authority.evaluation_sha256 or manifest.get("e_manifest_sha256") != authority.e_manifest_sha256 or manifest.get("exp471_reuse_authority_sha256") != authority.exp471_reuse_authority_sha256 or manifest.get("code_map_lock_sha256") != authority.code_map_lock_sha256 or manifest.get("report_manifest_sha256") != sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest() or manifest.get("report_manifest_sha256") != authority.report_manifest_sha256):
        raise ValueError("Exp472 report manifest differs")
    if marker != {"schema": _COMPLETE_SCHEMA_ANTIGRAVITY, "report_manifest": REPORT_MANIFEST_NAME_ANTIGRAVITY, "report_manifest_sha256": authority.report_manifest_sha256}:
        raise ValueError("Exp472 report completion marker differs")
    return True


__all__ = ["Exp472ReportAuthorityAntigravity", "REPORT_COMPLETE_NAME_ANTIGRAVITY", "REPORT_LEAVES_ANTIGRAVITY", "REPORT_MANIFEST_NAME_ANTIGRAVITY", "verify_exp472_report_antigravity", "write_exp472_report_antigravity"]
