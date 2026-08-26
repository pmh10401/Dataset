"""Deterministic paper leaves for the validated Exp471 typed-abstain artifact."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import html
import io
import json
import math
import os
from pathlib import Path
import struct
from typing import Any
import zlib

from experiments.exp450_resampling_image_ablation_antigravity import (
    reporting_antigravity as _curve_raster_antigravity,
)

from .config_antigravity import METHOD_ORDER_ANTIGRAVITY
from . import cache_artifacts_antigravity
from .contracts_antigravity import (
    Exp471EvaluationArtifactAntigravity,
    canonical_json_bytes_exp471_antigravity,
)
from .evaluation_antigravity import _validate_artifact_integrity_antigravity


_ROOT_NAME_ANTIGRAVITY = "exp471_typed_report_artifacts_antigravity"
_MANIFEST_NAME_ANTIGRAVITY = "exp471_typed_report_manifest_antigravity.json"
_LEAF_NAMES_ANTIGRAVITY = (
    "exp471_coverage_antigravity.csv",
    "exp471_coverage_antigravity.json",
    "exp471_full_population_decisions_antigravity.csv",
    "exp471_full_population_decisions_antigravity.json",
    "exp471_scored_support_metrics_antigravity.csv",
    "exp471_scored_support_metrics_antigravity.json",
    "exp471_common_support_metrics_antigravity.csv",
    "exp471_common_support_metrics_antigravity.json",
    "exp471_scored_support_curves_antigravity.svg",
    "exp471_scored_support_curves_antigravity.png",
    "exp471_common_support_curves_antigravity.svg",
    "exp471_common_support_curves_antigravity.png",
    "exp471_paper_report_antigravity.html",
    _MANIFEST_NAME_ANTIGRAVITY,
)


@dataclass(frozen=True, slots=True)
class Exp471TypedReportArtifactAntigravity:
    """Digest-bound report leaves rooted below a caller-owned staging directory."""

    root: Path
    leaves: Mapping[str, str]


def _plain_antigravity(value: object) -> object:
    return json.loads(canonical_json_bytes_exp471_antigravity(value).decode("utf-8"))


def _csv_bytes_antigravity(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(fields), extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {field: "" if row.get(field) is None else row.get(field) for field in fields}
        )
    return stream.getvalue().encode("utf-8")


def _metric_csv_rows_antigravity(table: Mapping[str, object]) -> tuple[
    tuple[Mapping[str, object], ...], tuple[str, ...]
]:
    rows: list[dict[str, object]] = []
    for row in table["per_dataset"]:
        assert isinstance(row, Mapping)
        rows.append({"record_type": "dataset", **_plain_antigravity(row)})
    for method in METHOD_ORDER_ANTIGRAVITY:
        macro = table["macro"][method]
        micro = table["micro"][method]
        assert isinstance(macro, Mapping) and isinstance(micro, Mapping)
        rows.append({"record_type": "macro", "method": method, **_plain_antigravity(macro)})
        rows.append({"record_type": "micro", "method": method, **_plain_antigravity(micro)})
    fields = tuple(sorted({key for row in rows for key in row}))
    return tuple(rows), fields


def _decision_csv_rows_antigravity(table: Mapping[str, object]) -> tuple[
    tuple[Mapping[str, object], ...], tuple[str, ...]
]:
    full = table["full_population"]
    assert isinstance(full, Mapping)
    rows = [
        {"record_type": "dataset", **_plain_antigravity(row)}
        for row in full["dataset_rows"]
    ]
    rows.extend(
        {"record_type": "summary", **_plain_antigravity(row)}
        for row in full["summary_rows"]
    )
    fields = tuple(sorted({key for row in rows for key in row}))
    return tuple(rows), fields


def _curve_svg_antigravity(curves: Mapping[str, object], support: str) -> str:
    """Render both exact axis conventions with all ten validated method blocks."""

    panels = (
        ("roc", "false_positive_rate", "true_positive_rate"),
        ("precision_recall", "recall", "precision"),
    )
    metadata = html.escape(
        canonical_json_bytes_exp471_antigravity(
            {
                "schema": "exp471-typed-curve-figure-antigravity.v1",
                "support": support,
                "roc": {"x": "false_positive_rate", "y": "true_positive_rate"},
                "precision_recall": {"x": "recall", "y": "precision"},
                "method_order": list(METHOD_ORDER_ANTIGRAVITY),
            }
        ).decode("utf-8").rstrip(),
        quote=False,
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="580" viewBox="0 0 1200 580"',
        ' data-roc-x="false_positive_rate" data-roc-y="true_positive_rate"',
        ' data-pr-x="recall" data-pr-y="precision">',
        f"<metadata>{metadata}</metadata><rect width=\"1200\" height=\"580\" fill=\"white\"/>",
    ]
    for panel, (name, x_label, y_label) in enumerate(panels):
        left, top, width, height = 70 + panel * 590, 70, 480, 400
        curve = curves[name]
        assert isinstance(curve, Mapping)
        x_key, y_key = curve["x_name"], curve["y_name"]
        parts.extend(
            (
                f'<text x="{left}" y="35" font-size="18">{name}</text>',
                f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="black"/>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="black"/>',
                f'<text x="{left + 120}" y="510" font-size="16">{x_label}</text>',
                f'<text x="{left - 42}" y="{top + 230}" font-size="16" transform="rotate(-90 {left - 42} {top + 230})">{y_label}</text>',
            )
        )
        rows = curve["rows"]
        assert isinstance(rows, Sequence)
        for index, method in enumerate(METHOD_ORDER_ANTIGRAVITY):
            points = [row for row in rows if isinstance(row, Mapping) and row["method"] == method]
            if not points:
                raise ValueError("Exp471 typed report curve method block is absent")
            hue = (index * 47) % 360
            legend_y = top + 14 + index * 15
            encoded = " ".join(
                f"{left + float(row[x_key]) * width:.12g},{top + height - float(row[y_key]) * height:.12g}"
                for row in points
            )
            parts.extend(
                (
                    f'<line data-legend-method="{method}" x1="{left + 8}" y1="{legend_y}" x2="{left + 18}" y2="{legend_y}" stroke="hsl({hue},65%,42%)" stroke-width="2"/>',
                    f'<text x="{left + 22}" y="{top + 18 + index * 15}" font-size="11" fill="hsl({hue},65%,32%)">{html.escape(method)}</text>',
                    f'<polyline data-method="{method}" fill="none" stroke="hsl({hue},65%,42%)" stroke-width="1.5" points="{encoded}"/>',
                )
            )
    parts.append("</svg>")
    return "".join(parts)


def _png_chunk_antigravity(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload)) + kind + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _curve_png_antigravity(curves: Mapping[str, object], support: str) -> bytes:
    """Attach deterministic semantic metadata to the dependency-free raster."""

    raster = _curve_raster_antigravity._curve_png(curves)
    iend = _png_chunk_antigravity(b"IEND", b"")
    if not raster.endswith(iend):
        raise ValueError("Exp471 typed curve PNG structure differs")
    metadata = canonical_json_bytes_exp471_antigravity(
        {
            "schema": "exp471-typed-curve-png-antigravity.v1",
            "support": support,
            "roc": {"x": "false_positive_rate", "y": "true_positive_rate"},
            "precision_recall": {"x": "recall", "y": "precision"},
            "method_order": list(METHOD_ORDER_ANTIGRAVITY),
            "curves": curves,
        }
    )
    return raster[: -len(iend)] + _png_chunk_antigravity(
        b"tEXt", b"exp471_typed_curve\x00" + metadata
    ) + iend


def _report_html_antigravity(payload: Mapping[str, object], figures: Mapping[str, str]) -> str:
    coverage = _plain_antigravity(payload["coverage"])
    support = _plain_antigravity(payload["support_authority"])
    scored = _plain_antigravity(payload["continuous_tables"]["scored_support"])
    common = _plain_antigravity(payload["continuous_tables"]["all_ten_common_support"])
    decisions = _plain_antigravity(payload["decision_tables"])
    embedded = json.dumps(
        {"coverage": coverage, "support_authority": support, "scored": scored, "common": common, "decisions": decisions},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).replace("</", "<\\/")
    coverage_rows = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(row[field]))}</td>"
            for field in ("method", "population", "scored", "abstained", "failed", "coverage", "abstention_reasons")
        ) + "</tr>"
        for row in coverage["rows"]
    )
    common_count = support["all_ten_common_support_count"]
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Exp471 typed abstain report</title><body>
<h1>Exp471 typed-abstain paper report</h1>
<p>Abstention is a preregistered no-alert outcome, not a runtime failure and not a continuous score.</p>
<p>This report prohibits score completion, filling absent values, automatic promotion, and historical-parity claims.</p>
<p>Continuous AP and AUROC summaries use scored support with explicit per-method denominators. The all-ten-method common support has count {common_count}.</p>
<h2>Coverage</h2><table><thead><tr><th>method</th><th>population</th><th>scored</th><th>abstained</th><th>failed</th><th>coverage</th><th>abstention_reasons</th></tr></thead><tbody>{coverage_rows}</tbody></table>
<h2>Scored support curves</h2><figure>{figures["scored_support"]}</figure>
<h2>All-ten common support curves</h2><figure>{figures["all_ten_common_support"]}</figure>
<script id="exp471-typed-artifact" type="application/json">{embedded}</script>
</body></html>'''


def _validate_typed_destination_antigravity(
    anchor: object, parent: int, expected: os.stat_result
) -> None:
    """Rebind the caller root and typed destination after each durable operation."""

    anchor.validate()
    cache_artifacts_antigravity._validate_child_directory_antigravity(
        parent, _ROOT_NAME_ANTIGRAVITY, expected
    )


def _typed_leaf_names_antigravity(directory: int) -> tuple[str, ...]:
    try:
        names = tuple(sorted(os.listdir(directory)))
    except OSError as error:
        raise ValueError("Exp471 typed report leaf inventory is unavailable") from error
    if any(name.startswith(".") for name in names):
        raise ValueError("Exp471 typed report leaf inventory contains a temporary leaf")
    return names


def expected_typed_exp471_report_leaves_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
) -> dict[str, bytes]:
    """Materialize every immutable leaf solely from the validated v2 artifact."""

    _validate_artifact_integrity_antigravity(artifact)
    payload = artifact.payload
    coverage = _plain_antigravity(payload["coverage"])
    decisions = _plain_antigravity(payload["decision_tables"])
    supports = {
        "scored_support": _plain_antigravity(payload["continuous_tables"]["scored_support"]),
        "all_ten_common_support": _plain_antigravity(payload["continuous_tables"]["all_ten_common_support"]),
    }
    curve_supports = {
        "scored_support": _plain_antigravity(payload["curve_tables"]["scored_support"]["curves"]),
        "all_ten_common_support": _plain_antigravity(payload["curve_tables"]["all_ten_common_support"]["curves"]),
    }
    decision_rows, decision_fields = _decision_csv_rows_antigravity(decisions)
    leaves: dict[str, bytes] = {
        "exp471_coverage_antigravity.csv": _csv_bytes_antigravity(
            coverage["rows"],
            ("method", "population", "scored", "abstained", "failed", "coverage", "abstention_reasons"),
        ),
        "exp471_coverage_antigravity.json": canonical_json_bytes_exp471_antigravity(coverage),
        "exp471_full_population_decisions_antigravity.csv": _csv_bytes_antigravity(decision_rows, decision_fields),
        "exp471_full_population_decisions_antigravity.json": canonical_json_bytes_exp471_antigravity(decisions),
    }
    figures: dict[str, str] = {}
    for support, stem in (
        ("scored_support", "exp471_scored_support"),
        ("all_ten_common_support", "exp471_common_support"),
    ):
        metric_rows, metric_fields = _metric_csv_rows_antigravity(supports[support])
        leaves[f"{stem}_metrics_antigravity.csv"] = _csv_bytes_antigravity(metric_rows, metric_fields)
        leaves[f"{stem}_metrics_antigravity.json"] = canonical_json_bytes_exp471_antigravity(supports[support])
        svg = _curve_svg_antigravity(curve_supports[support], support)
        figures[support] = svg
        leaves[f"{stem}_curves_antigravity.svg"] = svg.encode("utf-8")
        leaves[f"{stem}_curves_antigravity.png"] = _curve_png_antigravity(
            curve_supports[support], support
        )
    leaves["exp471_paper_report_antigravity.html"] = _report_html_antigravity(payload, figures).encode("utf-8")
    manifest = {name: sha256(content).hexdigest() for name, content in sorted(leaves.items())}
    leaves[_MANIFEST_NAME_ANTIGRAVITY] = canonical_json_bytes_exp471_antigravity(
        {"schema": "exp471-typed-report-manifest-antigravity.v1", "files": manifest}
    )
    if tuple(sorted(leaves)) != tuple(sorted(_LEAF_NAMES_ANTIGRAVITY)):
        raise ValueError("Exp471 typed report leaf inventory differs")
    return {f"{_ROOT_NAME_ANTIGRAVITY}/{name}": content for name, content in leaves.items()}


def write_typed_exp471_report_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity, root: Path
) -> Exp471TypedReportArtifactAntigravity:
    """Create or exact-byte reuse every typed report leaf below ``root``."""

    leaves = expected_typed_exp471_report_leaves_antigravity(artifact)
    expected_names = tuple(
        sorted(relative.removeprefix(f"{_ROOT_NAME_ANTIGRAVITY}/") for relative in leaves)
    )
    anchor = cache_artifacts_antigravity._open_absolute_directory_antigravity(
        Path(root), create=True
    )
    destination_descriptor: int | None = None
    destination = Path(root) / _ROOT_NAME_ANTIGRAVITY
    try:
        destination_descriptor, destination_identity = (
            cache_artifacts_antigravity._open_child_directory_antigravity(
                anchor.descriptor, _ROOT_NAME_ANTIGRAVITY, create=True
            )
        )
        _validate_typed_destination_antigravity(
            anchor, anchor.descriptor, destination_identity
        )
        if _typed_leaf_names_antigravity(destination_descriptor):
            existing = _typed_leaf_names_antigravity(destination_descriptor)
            if existing != expected_names:
                raise ValueError("Exp471 typed report leaf inventory differs")
        for relative, content in leaves.items():
            name = relative.removeprefix(f"{_ROOT_NAME_ANTIGRAVITY}/")
            cache_artifacts_antigravity._publish_bytes_at_antigravity(
                destination_descriptor,
                name,
                content,
                temporary_scope="typed_report_leaf",
                post_link_validate_antigravity=lambda: _validate_typed_destination_antigravity(
                    anchor, anchor.descriptor, destination_identity
                ),
            )
        if _typed_leaf_names_antigravity(destination_descriptor) != expected_names:
            raise ValueError("Exp471 typed report leaf inventory differs")
        _validate_typed_destination_antigravity(
            anchor, anchor.descriptor, destination_identity
        )
    finally:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        anchor.close()
    digests = {relative: sha256(content).hexdigest() for relative, content in leaves.items()}
    return Exp471TypedReportArtifactAntigravity(destination, digests)


__all__ = [
    "Exp471TypedReportArtifactAntigravity",
    "expected_typed_exp471_report_leaves_antigravity",
    "write_typed_exp471_report_antigravity",
]
