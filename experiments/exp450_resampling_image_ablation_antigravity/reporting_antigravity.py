"""Immutable paper artifacts for the frozen Exp450 continuous-score ablation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import html
import io
import json
import math
import os
from pathlib import Path
import struct
from typing import Mapping, Sequence
import zlib

import numpy as np

from .contracts_antigravity import canonical_json_bytes
from .evaluation_antigravity import Exp450EvaluationAntigravity, METHOD_ORDER


@dataclass(frozen=True, slots=True)
class Exp450ReportArtifactAntigravity:
    root: Path
    metric_csv_path: Path
    metric_json_path: Path
    curve_csv_path: Path
    curve_json_path: Path
    paired_csv_path: Path
    paired_json_path: Path
    png_path: Path
    svg_path: Path
    html_path: Path
    manifest_path: Path


def write_exp450_report_antigravity(
    evaluation: Exp450EvaluationAntigravity, root: Path
) -> Exp450ReportArtifactAntigravity:
    """Write exact evaluation leaves and standalone figures below one supplied root.

    The caller supplies an evaluation that has already crossed the label boundary;
    this module never loads score roots or labels itself.
    """
    if not isinstance(evaluation, Exp450EvaluationAntigravity):
        raise ValueError("Exp450 report requires a verified evaluation")
    destination = Path(root) / "exp450_report_artifacts_antigravity"
    metrics = evaluation.metric_tables
    curves = canonicalize_exp450_json_antigravity(evaluation.curve_tables)
    metric_payload = {
        "method_order": list(metrics.method_order),
        "per_dataset": canonicalize_exp450_json_antigravity(metrics.per_dataset),
        "macro": canonicalize_exp450_json_antigravity(metrics.macro),
        "micro": canonicalize_exp450_json_antigravity(metrics.micro),
    }
    paired_payload = {"paired_statistics": canonicalize_exp450_json_antigravity(metrics.paired_statistics)}
    leaves = {
        "exp450_metric_tables_antigravity.csv": _csv_bytes(
            metrics.per_dataset,
            ("method", "dataset", "average_precision", "auroc", "average_precision_exclusion", "auroc_exclusion"),
        ),
        "exp450_metric_tables_antigravity.json": canonical_json_bytes(metric_payload),
        "exp450_curve_tables_antigravity.csv": _curve_csv_bytes(curves),
        "exp450_curve_tables_antigravity.json": canonical_json_bytes(curves),
        "exp450_paired_differences_antigravity.csv": _paired_csv_bytes(metrics.paired_statistics),
        "exp450_paired_differences_antigravity.json": canonical_json_bytes(paired_payload),
    }
    svg = _curve_svg(curves)
    png = _curve_png(curves)
    document = _html_document(metric_payload, paired_payload, curves, svg)
    leaves.update({
        "exp450_curves_antigravity.svg": svg.encode("utf-8"),
        "exp450_curves_antigravity.png": png,
        "exp450_paper_report_antigravity.html": document.encode("utf-8"),
    })
    manifest = {name: sha256(payload).hexdigest() for name, payload in sorted(leaves.items())}
    for name, payload in leaves.items():
        _write_create_only(destination / name, payload)
    manifest_path = destination / "exp450_report_manifest_antigravity.json"
    _write_create_only(manifest_path, canonical_json_bytes(manifest))
    return Exp450ReportArtifactAntigravity(
        root=destination,
        metric_csv_path=destination / "exp450_metric_tables_antigravity.csv",
        metric_json_path=destination / "exp450_metric_tables_antigravity.json",
        curve_csv_path=destination / "exp450_curve_tables_antigravity.csv",
        curve_json_path=destination / "exp450_curve_tables_antigravity.json",
        paired_csv_path=destination / "exp450_paired_differences_antigravity.csv",
        paired_json_path=destination / "exp450_paired_differences_antigravity.json",
        png_path=destination / "exp450_curves_antigravity.png",
        svg_path=destination / "exp450_curves_antigravity.svg",
        html_path=destination / "exp450_paper_report_antigravity.html",
        manifest_path=manifest_path,
    )


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("Exp450 report artifact already exists with different bytes") from None
        return
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Exp450 report artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def canonicalize_exp450_json_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): canonicalize_exp450_json_antigravity(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonicalize_exp450_json_antigravity(item) for item in value]
    if isinstance(value, np.ndarray):
        return canonicalize_exp450_json_antigravity(value.tolist())
    if isinstance(value, np.generic):
        return canonicalize_exp450_json_antigravity(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for raw_row in rows:
        row = canonicalize_exp450_json_antigravity(raw_row)
        assert isinstance(row, dict)
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    return stream.getvalue().encode("utf-8")


def _curve_csv_bytes(curves: Mapping[str, object]) -> bytes:
    rows: list[dict[str, object]] = []
    for curve_name in ("roc", "precision_recall"):
        curve = curves[curve_name]
        assert isinstance(curve, dict)
        for row in curve["rows"]:
            assert isinstance(row, dict)
            rows.append({"curve": curve_name, **row})
    fields = ("curve", "method", "false_positive_rate", "true_positive_rate", "threshold", "recall", "precision")
    return _csv_bytes(rows, fields)


def _paired_csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    plain_rows = canonicalize_exp450_json_antigravity(rows)
    assert isinstance(plain_rows, list)
    fields = tuple(sorted({field for row in plain_rows for field in row}))
    return _csv_bytes(plain_rows, fields)


def _curve_svg(curves: Mapping[str, object]) -> str:
    panels = (("roc", "False-positive rate", "True-positive rate"), ("precision_recall", "Recall", "Precision"))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="580" viewBox="0 0 1200 580">', '<rect width="1200" height="580" fill="white"/>']
    for panel, (name, x_label, y_label) in enumerate(panels):
        left = 70 + panel * 590
        top, width, height = 70, 480, 400
        parts.extend((f'<text x="{left}" y="35" font-size="18">{html.escape(name)}</text>', f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="black"/>', f'<text x="{left + 160}" y="510" font-size="16">{x_label}</text>', f'<text x="{left - 42}" y="{top + 230}" font-size="16" transform="rotate(-90 {left - 42} {top + 230})">{y_label}</text>'))
        curve = curves[name]
        assert isinstance(curve, dict)
        rows = curve["rows"]
        assert isinstance(rows, list)
        for index, method in enumerate(METHOD_ORDER):
            points = [row for row in rows if isinstance(row, dict) and row["method"] == method]
            hue = (index * 47) % 360
            legend_y = top + 14 + index * 15
            parts.append(f'<line data-legend-method="{method}" x1="{left + 8}" y1="{legend_y}" x2="{left + 18}" y2="{legend_y}" stroke="hsl({hue},65%,42%)" stroke-width="2"/>')
            parts.append(f'<text x="{left + 22}" y="{top + 18 + index * 15}" font-size="11" fill="hsl({hue},65%,32%)">{method}</text>')
            if points:
                x_key, y_key = curve["x_name"], curve["y_name"]
                encoded = " ".join(f"{left + float(row[x_key]) * width:.12g},{top + height - float(row[y_key]) * height:.12g}" for row in points)
                parts.append(f'<polyline fill="none" stroke="hsl({hue},65%,42%)" stroke-width="1.5" points="{encoded}"><title>{method}</title></polyline>')
    parts.append('</svg>')
    return "".join(parts)


def _curve_png(curves: Mapping[str, object]) -> bytes:
    """Draw a dependency-free PNG counterpart to the exact standalone SVG."""
    width, height = 1200, 580
    pixels = bytearray(b"\xff" * (width * height * 3))
    glyphs = {
        "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
        "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
        "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
        "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
        "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
        "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
        "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
        "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
        "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
        "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
        "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
        "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
        "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
        "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
        "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
        "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
        "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
        "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
        "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
        "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
        "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
        "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
        "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
        "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
        "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
        " ": ("000", "000", "000", "000", "000", "000", "000"),
    }

    def point(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset:offset + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
        dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            point(x0, y0, color)
            if (x0, y0) == (x1, y1):
                return
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy

    def text(x: int, y: int, label: str, color: tuple[int, int, int], scale: int = 1, vertical: bool = False) -> None:
        """Render an intentionally small, dependency-free 5×7 bitmap label."""
        for letter_index, character in enumerate(label.upper()):
            glyph = glyphs.get(character, glyphs[" "])
            for row, bits in enumerate(glyph):
                for column, bit in enumerate(bits):
                    if bit != "1":
                        continue
                    for dy in range(scale):
                        for dx in range(scale):
                            source_x = (letter_index * 6 + column) * scale + dx
                            source_y = row * scale + dy
                            if vertical:
                                point(x - source_y, y + source_x, color)
                            else:
                                point(x + source_x, y + source_y, color)

    panels = (
        ("roc", "ROC", "FALSE POSITIVE RATE", "TRUE POSITIVE RATE"),
        ("precision_recall", "PRECISION RECALL", "RECALL", "PRECISION"),
    )
    for panel, (name, title, x_label, y_label) in enumerate(panels):
        left, top, panel_width, panel_height = 70 + panel * 590, 70, 480, 400
        line(left, top + panel_height, left + panel_width, top + panel_height, (0, 0, 0))
        line(left, top, left, top + panel_height, (0, 0, 0))
        text(left, 20, title, (0, 0, 0), scale=2)
        text(left + 145, 525, x_label, (0, 0, 0), scale=1)
        text(left - 12, top + 130, y_label, (0, 0, 0), scale=1, vertical=True)
        curve = curves[name]
        assert isinstance(curve, dict)
        x_key, y_key = curve["x_name"], curve["y_name"]
        for index, method in enumerate(METHOD_ORDER):
            rows = [row for row in curve["rows"] if row["method"] == method]
            color = ((41 + index * 53) % 220, (79 + index * 97) % 220, (131 + index * 37) % 220)
            text(left + 8, top + 12 + index * 14, method, color)
            previous: tuple[int, int] | None = None
            for row in rows:
                current = (round(left + float(row[x_key]) * panel_width), round(top + panel_height - float(row[y_key]) * panel_height))
                if previous is not None:
                    line(*previous, *current, color)
                previous = current
    raw = b"".join(b"\x00" + bytes(pixels[row * width * 3:(row + 1) * width * 3]) for row in range(height))
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, level=9)) + chunk(b"IEND", b"")


def _html_document(metrics: Mapping[str, object], paired: Mapping[str, object], curves: Mapping[str, object], svg: str) -> str:
    def metric_table(name: str, values: object) -> str:
        assert isinstance(values, dict)
        rows = "".join(f"<tr><td>{html.escape(method)}</td><td>{_number(item.get('average_precision'))}</td><td>{_number(item.get('auroc'))}</td></tr>" for method, item in values.items() if isinstance(item, dict))
        return f"<h2>{name}</h2><table><thead><tr><th>Method</th><th>Average precision</th><th>AUROC</th></tr></thead><tbody>{rows}</tbody></table>"
    paired_rows = paired["paired_statistics"]
    assert isinstance(paired_rows, list)
    paired_table = "".join(f"<tr><td>{html.escape(str(row.get('comparison')))}</td><td>{html.escape(str(row.get('metric')))}</td><td>{_number(row.get('point_estimate'))}</td><td>{_number(row.get('lower'))}</td><td>{_number(row.get('upper'))}</td></tr>" for row in paired_rows if isinstance(row, dict))
    curve_json = json.dumps(curves, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    paired_json = json.dumps(paired, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Exp450 resampling/image ablation</title><body>
<h1>Exp450 continuous resampling/image ablation</h1>
<p>This is an ablation report only; it makes no automatic promotion claim.</p>
{metric_table("Macro AP / AUROC", metrics["macro"])}{metric_table("Micro AP / AUROC", metrics["micro"])}
<h2>Paired differences</h2><table><thead><tr><th>Comparison</th><th>Metric</th><th>Difference</th><th>Lower</th><th>Upper</th></tr></thead><tbody>{paired_table}</tbody></table>
<h2>ROC: False-positive rate × True-positive rate</h2><h2>Precision–recall: Recall × Precision</h2><figure>{svg}</figure>
<script id=\"exp450-curves\" type=\"application/json\">{curve_json}</script><script id=\"exp450-paired\" type=\"application/json\">{paired_json}</script>
</body></html>"""


def _number(value: object) -> str:
    return "" if value is None else f"{float(value):.12g}"


__all__ = ["Exp450ReportArtifactAntigravity", "canonicalize_exp450_json_antigravity", "write_exp450_report_antigravity"]
