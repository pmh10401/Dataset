"""Subprocess-only audit of the fixed aeon method surface.

This module deliberately imports no audited aeon class.  That keeps readiness and
inspect callers able to consume the immutable returned records without loading a
transformer or classifier into their interpreter.
"""

from __future__ import annotations

import json
import subprocess
from typing import Final

from experiments.exp449_residual_fn_method_qualification_antigravity.config_antigravity import (
    FIXED_INTERPRETER_ANTIGRAVITY,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import (
    MethodAuditAntigravity,
)


AUDIT_SPECS_ANTIGRAVITY: Final = (
    ("MiniRocket", "aeon.transformations.collection.convolution_based", "score_compatible"),
    ("MultiRocket", "aeon.transformations.collection.convolution_based", "score_compatible"),
    ("HydraTransformer", "aeon.transformations.collection.convolution_based", "score_compatible"),
    ("WEASEL_V2", "aeon.classification.dictionary_based", "classifier_only"),
    ("DrCIFClassifier", "aeon.classification.interval_based", "classifier_only"),
    ("HIVECOTEV2", "aeon.classification.hybrid", "classifier_only"),
    ("InceptionTimeClassifier", "aeon.classification.deep_learning", "classifier_only"),
    ("TSChiefClassifier", "aeon.classification.hybrid", "unavailable"),
)

_EXPECTED_VERSIONS_ANTIGRAVITY: Final = {
    "python": "3.14.6",
    "aeon": "1.4.0",
    "numpy": "2.3.5",
    "scikit-learn": "1.7.2",
}

_TRANSFORM_METHODS_ANTIGRAVITY: Final = (
    "clone", "fit", "fit_transform", "get_class_tag", "get_class_tags",
    "get_fitted_params", "get_metadata_routing", "get_params", "get_tag",
    "get_tags", "reset", "set_output", "set_params", "set_tags", "transform",
)
_CLASSIFIER_METHODS_ANTIGRAVITY: Final = (
    "clone", "fit", "fit_predict", "fit_predict_proba", "get_class_tag",
    "get_class_tags", "get_fitted_params", "get_metadata_routing", "get_params",
    "get_tag", "get_tags", "predict", "predict_proba", "reset", "score",
    "set_params", "set_tags",
)
_FROZEN_BINDINGS_ANTIGRAVITY: Final = {
    "MiniRocket": ("a0050df93fcbf093a540ebdc8e0a5fef528740adbbd0ad6c6540e31e229c58da", _TRANSFORM_METHODS_ANTIGRAVITY),
    "MultiRocket": ("4fed8bff32f369601f7487990e91c47d69c2ff57eb956f6c8e50fbe6931c3564", _TRANSFORM_METHODS_ANTIGRAVITY),
    "HydraTransformer": ("eb86f11e9fbc7eb6fe34d5d5d6799cec5a47ab02f8347c9c8d6fd303b6fb9799", _TRANSFORM_METHODS_ANTIGRAVITY),
    "WEASEL_V2": ("0e53de67bb675e2670376c1db976d03b032641ec1b82a21f38dbc34907992817", _CLASSIFIER_METHODS_ANTIGRAVITY),
    "DrCIFClassifier": ("a1d8cf8bf8a18e9a5ef22f87e3bdcb2bf81182a2b0bd4549511f16d9872282f7", _CLASSIFIER_METHODS_ANTIGRAVITY + ("temporal_importance_curves",)),
    "HIVECOTEV2": ("899223ec5e6f018f757a1f7a2858a8a9653a0dc5367f55985efb18baa58a2d02", _CLASSIFIER_METHODS_ANTIGRAVITY),
    "InceptionTimeClassifier": ("66c158378c16ffa162c4b87f32d4689aa3a4734539da43229331d0c6eec16581", _CLASSIFIER_METHODS_ANTIGRAVITY[:11] + ("load_model",) + _CLASSIFIER_METHODS_ANTIGRAVITY[11:]),
}

_AUDIT_PROGRAM_ANTIGRAVITY: Final = r'''
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import platform
import sys

specs = json.loads(sys.stdin.read())
records = []
for name, module_name, status in specs:
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, name)
    except AttributeError:
        records.append({"name": name, "module": module_name, "available": False})
        continue
    source = inspect.getsourcefile(cls)
    if source is None:
        raise RuntimeError("class has no source file: " + name)
    methods = sorted(name for name, member in inspect.getmembers(cls, predicate=callable) if not name.startswith("_"))
    with open(source, "rb") as handle:
        source_sha256 = hashlib.sha256(handle.read()).hexdigest()
    records.append({"name": name, "module": module_name, "available": True, "source_sha256": source_sha256, "methods": methods})
print(json.dumps({"versions": {"python": platform.python_version(), "aeon": importlib.metadata.version("aeon"), "numpy": importlib.metadata.version("numpy"), "scikit-learn": importlib.metadata.version("scikit-learn")}, "records": records}, sort_keys=True, separators=(",", ":")))
'''


def _strict_payload_antigravity(payload: str) -> dict[str, object]:
    def reject_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate audit JSON key: {key}")
            result[key] = value
        return result

    decoded = json.loads(payload, object_pairs_hook=reject_duplicates_antigravity)
    if type(decoded) is not dict:
        raise ValueError("audit response must be an object")
    return decoded


def audit_methods_antigravity() -> tuple[MethodAuditAntigravity, ...]:
    """Audit the one frozen interpreter and return deterministic immutable records."""
    completed = subprocess.run(
        [FIXED_INTERPRETER_ANTIGRAVITY, "-c", _AUDIT_PROGRAM_ANTIGRAVITY],
        input=json.dumps(AUDIT_SPECS_ANTIGRAVITY, separators=(",", ":")),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fixed method audit failed: {completed.stderr.strip()}")
    result = _strict_payload_antigravity(completed.stdout)
    versions = result.get("versions")
    records = result.get("records")
    if versions != _EXPECTED_VERSIONS_ANTIGRAVITY:
        raise ValueError(f"fixed interpreter versions differ: {versions!r}")
    if type(records) is not list or len(records) != len(AUDIT_SPECS_ANTIGRAVITY):
        raise ValueError("fixed method audit has an invalid record count")
    audits: list[MethodAuditAntigravity] = []
    for raw, spec in zip(records, AUDIT_SPECS_ANTIGRAVITY, strict=True):
        name, module, status = spec
        if type(raw) is not dict or raw.get("name") != name or raw.get("module") != module or type(raw.get("available")) is not bool:
            raise ValueError("fixed method audit record does not match its specification")
        if status == "unavailable":
            if raw["available"]:
                raise ValueError(f"frozen unavailable class is now available: {name}")
            audits.append(MethodAuditAntigravity(name, status, module, name, _EXPECTED_VERSIONS_ANTIGRAVITY["aeon"], None, ()))
            continue
        if not raw["available"]:
            raise ValueError(f"required class is unavailable: {name}")
        source_sha256 = raw.get("source_sha256")
        methods = raw.get("methods")
        if type(source_sha256) is not str or type(methods) is not list or any(type(method) is not str for method in methods):
            raise ValueError(f"invalid source or method binding: {name}")
        expected_sha256, expected_methods = _FROZEN_BINDINGS_ANTIGRAVITY[name]
        if source_sha256 != expected_sha256 or tuple(methods) != expected_methods:
            raise ValueError(f"frozen public method binding differs: {name}")
        audits.append(MethodAuditAntigravity(name, status, module, name, _EXPECTED_VERSIONS_ANTIGRAVITY["aeon"], source_sha256, tuple(methods)))
    return tuple(audits)
