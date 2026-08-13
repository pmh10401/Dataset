"""Fail-closed subprocess audit of the installed official aeon RDST."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Final

from .config_antigravity import FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY


_EXPECTED_CONSTRUCTOR_SIGNATURE_ANTIGRAVITY: Final = "(max_shapelets: int = 10000, shapelet_lengths: List[int] | numpy.ndarray | None = None, proba_normalization: float = 0.8, threshold_percentiles: List[float] | numpy.ndarray | None = None, alpha_similarity: float = 0.5, use_prime_dilations: bool = False, random_state: int | None = None, n_jobs: int = 1)"
_KNOWN_AEON_WARNING_MESSAGE_ANTIGRAVITY: Final = "Some invalid values (inf or nan) where converted from to 0 during the shapelet transformation."


def _audit_request_antigravity() -> dict[str, object]:
    authority = FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY
    return {"schema": "exp451.rdst_audit.request.v1", "seed": authority.seeds[0], "max_shapelets": authority.max_shapelets, "n_jobs": authority.rdst_n_jobs, "y_policy": authority.fit_label_policy}


def _canonical_json_text_antigravity(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def _warning_is_allowed_antigravity(message: object, category: object) -> bool:
    return type(message) is str and message == _KNOWN_AEON_WARNING_MESSAGE_ANTIGRAVITY and category is UserWarning

_AUDIT_PROGRAM_ANTIGRAVITY: Final = r'''
import hashlib
import importlib.metadata
import inspect
import json
import sys
import warnings

import numpy as np
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform

def strict_load(text):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=no_duplicates)
    if type(value) is not dict:
        raise ValueError("request must be an object")
    return value

request = strict_load(sys.stdin.read())
if set(request) != {"schema", "seed", "max_shapelets", "n_jobs", "y_policy"}:
    raise ValueError("request keys differ")
if request["schema"] != "exp451.rdst_audit.request.v1" or type(request["seed"]) is not int or type(request["max_shapelets"]) is not int or type(request["n_jobs"]) is not int or request["y_policy"] != "y_none_only":
    raise ValueError("request types differ")
source = inspect.getsourcefile(RandomDilatedShapeletTransform)
if source is None:
    raise RuntimeError("RDST source is unavailable")
with open(source, "rb") as handle:
    source_sha256 = hashlib.sha256(handle.read()).hexdigest()
fixture = np.array([
    [[0.0, 0.3, 0.1, 0.5, 0.2, 0.7, 0.4, 0.8, 0.3, 0.6, 0.2, 0.9, 0.1, 0.5, 0.4, 0.7]],
    [[0.1, 0.4, 0.2, 0.6, 0.3, 0.8, 0.5, 0.9, 0.4, 0.7, 0.3, 1.0, 0.2, 0.6, 0.5, 0.8]],
    [[0.9, 0.6, 0.8, 0.5, 0.7, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9, 0.4, 0.8, 0.2]],
    [[0.8, 0.5, 0.7, 0.4, 0.6, 0.3, 0.5, 0.2, 0.6, 0.1, 0.7, 0.0, 0.8, 0.3, 0.7, 0.1]],
], dtype=np.float64)
outputs = []
retained = []
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for _ in range(2):
        transformer = RandomDilatedShapeletTransform(max_shapelets=request["max_shapelets"], random_state=request["seed"], n_jobs=request["n_jobs"])
        features = transformer.fit_transform(fixture, y=None)
        if features.ndim != 2 or not np.isfinite(features).all():
            raise RuntimeError("RDST fixture output is not finite 2D")
        count = getattr(transformer, "n_shapelets_", None)
        if type(count) is not int or count < 1 or features.shape[1] != 3 * count:
            raise RuntimeError("RDST retained-shapelet width differs")
        outputs.append(np.ascontiguousarray(features).tobytes())
        retained.append(count)
if any(str(item.message) != "Some invalid values (inf or nan) where converted from to 0 during the shapelet transformation." or item.category is not UserWarning for item in caught):
    raise RuntimeError("unexpected RDST warning")
if retained[0] != retained[1] or outputs[0] != outputs[1]:
    raise RuntimeError("RDST same-seed fixture is not byte deterministic")
response = {
    "schema": "exp451.rdst_audit.response.v1",
    "aeon_version": importlib.metadata.version("aeon"),
    "public_class_path": "aeon.transformations.collection.shapelet_based.RandomDilatedShapeletTransform",
    "implementation_class_path": RandomDilatedShapeletTransform.__module__ + "." + RandomDilatedShapeletTransform.__qualname__,
    "constructor_signature": str(inspect.signature(RandomDilatedShapeletTransform)),
    "fit_signature": str(inspect.signature(RandomDilatedShapeletTransform.fit)),
    "fit_transform_signature": str(inspect.signature(RandomDilatedShapeletTransform.fit_transform)),
    "source_sha256": source_sha256,
    "seed": request["seed"],
    "max_shapelets": request["max_shapelets"],
    "n_jobs": request["n_jobs"],
    "y_is_none": True,
    "two_dimensional": True,
    "finite": True,
    "deterministic": True,
    "retained_shapelets": retained[0],
    "feature_dimension": 3 * retained[0],
    "first_output_sha256": hashlib.sha256(outputs[0]).hexdigest(),
    "second_output_sha256": hashlib.sha256(outputs[1]).hexdigest(),
}
sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")) + "\n")
'''


def _strict_json_antigravity(payload: str) -> object:
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate audit JSON key")
            result[key] = value
        return result
    return json.loads(payload, object_pairs_hook=reject_duplicates_antigravity)


@dataclass(frozen=True, slots=True)
class RdstAuditResultAntigravity:
    aeon_version: str
    public_class_path: str
    implementation_class_path: str
    constructor_signature: str
    fit_signature: str
    fit_transform_signature: str
    source_sha256: str
    seed: int
    max_shapelets: int
    n_jobs: int
    y_is_none: bool
    two_dimensional: bool
    finite: bool
    deterministic: bool
    retained_shapelets: int
    feature_dimension: int
    first_output_sha256: str
    second_output_sha256: str

    def __post_init__(self) -> None:
        for field in ("aeon_version", "public_class_path", "implementation_class_path", "constructor_signature", "fit_signature", "fit_transform_signature"):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise ValueError(f"{field} must be an exact non-empty string")
        for field in ("source_sha256", "first_output_sha256", "second_output_sha256"):
            if type(getattr(self, field)) is not str or not re.fullmatch(r"[0-9a-f]{64}", getattr(self, field)):
                raise ValueError(f"{field} must be a lowercase SHA256")
        for field in ("seed", "max_shapelets", "n_jobs", "retained_shapelets", "feature_dimension"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"{field} must be an exact positive integer")
        for field in ("y_is_none", "two_dimensional", "finite", "deterministic"):
            if type(getattr(self, field)) is not bool or not getattr(self, field):
                raise ValueError(f"{field} must be exact true")
        if self.feature_dimension != 3 * self.retained_shapelets or self.first_output_sha256 != self.second_output_sha256:
            raise ValueError("RDST fixture evidence differs")


def audit_rdst_antigravity() -> RdstAuditResultAntigravity:
    """Audit only the frozen interpreter, class, source, and label-free fixture."""
    request = _audit_request_antigravity()
    try:
        completed = subprocess.run(
            [FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.fixed_interpreter, "-c", _AUDIT_PROGRAM_ANTIGRAVITY],
            input=_canonical_json_text_antigravity(request), text=True, capture_output=True, check=False, timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("fixed RDST audit subprocess timed out") from error
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError(f"fixed RDST audit subprocess failed: {completed.stderr.strip()}")
    decoded = _strict_json_antigravity(completed.stdout)
    expected_keys = {
        "schema", "aeon_version", "public_class_path", "implementation_class_path", "constructor_signature", "fit_signature", "fit_transform_signature",
        "source_sha256", "seed", "max_shapelets", "n_jobs", "y_is_none", "two_dimensional", "finite",
        "deterministic", "retained_shapelets", "feature_dimension", "first_output_sha256", "second_output_sha256",
    }
    if type(decoded) is not dict or set(decoded) != expected_keys or decoded["schema"] != "exp451.rdst_audit.response.v1":
        raise ValueError("fixed RDST audit response schema differs")
    if completed.stdout != _canonical_json_text_antigravity(decoded):
        raise ValueError("fixed RDST audit response is not canonical")
    values = {key: value for key, value in decoded.items() if key != "schema"}
    result = RdstAuditResultAntigravity(**values)
    authority = FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY
    if result.aeon_version != authority.aeon_version or result.public_class_path != authority.rdst_public_class_path or result.implementation_class_path != authority.rdst_implementation_class_path or result.source_sha256 != authority.source_sha256:
        raise ValueError("fixed RDST binding differs")
    if result.constructor_signature != _EXPECTED_CONSTRUCTOR_SIGNATURE_ANTIGRAVITY or result.fit_signature != "(self, X, y=None)" or result.fit_transform_signature != "(self, X, y=None)":
        raise ValueError("fixed RDST signature differs")
    if (result.seed, result.max_shapelets, result.n_jobs) != (authority.seeds[0], authority.max_shapelets, authority.rdst_n_jobs):
        raise ValueError("fixed RDST fixture parameters differ")
    return result
