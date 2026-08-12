"""Strict, side-effect-free provenance validation for frozen paper origins."""
from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path
import re

from .config_paper import REPOSITORY_ROOT
from .contracts_paper import LineageNode, PaperCodemap


_DOTTED_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_NODE_FIELDS = frozenset({
    "node_id", "stage", "input_kind", "module", "symbols", "source_sha256",
    "paper_module", "label_access", "parity_oracle",
})
_CODEMAP_FIELDS = ("callers", "effects", "tests", "source_origins")
_LOCK_FIELDS = frozenset({"schema", "hashes"})


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_strict_json(path: Path) -> object:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def resolve_repo_module_source(module: str) -> Path:
    """Resolve an origin module to one exact repo-local Python source, unexecuted."""
    if type(module) is not str or not _DOTTED_IDENTIFIER.fullmatch(module):
        raise ValueError("origin module must be a dotted identifier")
    root = REPOSITORY_ROOT.resolve(strict=True)
    source = (root / (module.replace(".", "/") + ".py")).resolve(strict=False)
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError("origin module escapes repository root") from error
    if not source.is_file() or source.suffix != ".py":
        raise ValueError(f"origin module source is unavailable: {module}")
    return source


def static_top_level_names(tree: ast.AST) -> frozenset[str]:
    """Return only names genuinely defined or exported at module top level."""
    if not isinstance(tree, ast.Module):
        raise ValueError("origin source AST must be a module")
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.Import):
            names.extend(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.extend(alias.asname or alias.name for alias in node.names)
    if len(names) != len(set(names)):
        raise ValueError("origin source has duplicate top-level names")
    return frozenset(names)


def parse_nodes(payload: object) -> tuple[LineageNode, ...]:
    if not isinstance(payload, list):
        raise ValueError("lineage lock must be a JSON list")
    if any(not isinstance(record, dict) or set(record) != _NODE_FIELDS for record in payload):
        raise ValueError("lineage node schema is invalid")
    nodes: list[LineageNode] = []
    for record in payload:
        symbols = record["symbols"]
        fields = (
            record["node_id"], record["stage"], record["input_kind"], record["module"],
            record["source_sha256"], record["paper_module"], record["parity_oracle"],
        )
        if (
            any(value is not None and type(value) is not str for value in fields)
            or not all(type(value) is str and value for value in fields[:4])
            or not isinstance(symbols, list)
            or not symbols
            or any(type(symbol) is not str or not symbol for symbol in symbols)
            or type(record["label_access"]) is not bool
            or type(record["source_sha256"]) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", record["source_sha256"])
        ):
            raise ValueError("lineage node types are invalid")
        if record["stage"] not in {"raw-computable", "frozen-rule", "parity-oracle"}:
            raise ValueError("lineage node stage is invalid")
        if record["input_kind"] not in {"raw_train_test_values", "frozen_rule", "parity_oracle"}:
            raise ValueError("lineage node input kind is invalid")
        nodes.append(LineageNode(
            node_id=record["node_id"], stage=record["stage"], input_kind=record["input_kind"],
            module=record["module"], symbols=tuple(symbols),
            source_sha256=record["source_sha256"], paper_module=record["paper_module"],
            label_access=record["label_access"], parity_oracle=record["parity_oracle"],
        ))
    result = tuple(nodes)
    if not result or len({node.node_id for node in result}) != len(result):
        raise ValueError("lineage node ids must be non-empty and unique")
    return result


def validate_lineage_lock(path: Path) -> tuple[LineageNode, ...]:
    """Validate exact unexecuted origin source, AST symbols, and SHA256 lineage."""
    nodes = parse_nodes(read_strict_json(path))
    if any(node.input_kind == "prediction_coordinates" for node in nodes):
        raise ValueError("old prediction coordinates are forbidden inputs")
    if any(node.label_access for node in nodes):
        raise ValueError("lineage nodes may not access TEST labels")
    for node in nodes:
        source = resolve_repo_module_source(node.module)
        try:
            exported = static_top_level_names(ast.parse(source.read_text(encoding="utf-8"), filename=str(source)))
        except SyntaxError as error:
            raise ValueError(f"origin source syntax is invalid: {node.node_id}") from error
        if any(symbol not in exported for symbol in node.symbols):
            raise ValueError(f"raw scorer symbol is unavailable: {node.node_id}")
        if sha256_file(source) != node.source_sha256:
            raise ValueError(f"raw scorer source SHA differs: {node.node_id}")
    return nodes


def _parse_codemap(payload: object) -> PaperCodemap:
    if not isinstance(payload, dict) or set(payload) != {"schema", "required_fields", "modules"}:
        raise ValueError("paper codemap schema is invalid")
    if payload["schema"] != "paper-codemap.v1" or not isinstance(payload["required_fields"], list):
        raise ValueError("paper codemap schema is invalid")
    required = tuple(payload["required_fields"])
    modules = payload["modules"]
    if set(required) != set(_CODEMAP_FIELDS) or len(required) != len(_CODEMAP_FIELDS) or not isinstance(modules, dict) or not modules:
        raise ValueError("paper codemap required fields are invalid")
    if any(type(name) is not str or not name.endswith("_paper.py") or not isinstance(detail, dict) or set(detail) != set(_CODEMAP_FIELDS) for name, detail in modules.items()):
        raise ValueError("paper codemap module schema is invalid")
    if any(any(not isinstance(detail[field], list) or any(type(item) is not str for item in detail[field]) for field in _CODEMAP_FIELDS) for detail in modules.values()):
        raise ValueError("paper codemap module values are invalid")
    return PaperCodemap(required, modules)


def validate_codemap_triplet(paper_root: Path) -> PaperCodemap:
    root = Path(paper_root)
    html, codemap, lock = (root / "codemap_paper.html", root / "codemap_paper.json", root / "codemap_paper.lock")
    config, lineage = root / "config_paper.py", root / "lineage_lock_paper.json"
    required_paths = (html, codemap, lock, config, lineage)
    if any(not item.is_file() for item in required_paths):
        raise ValueError("paper codemap triplet is incomplete")
    parsed = _parse_codemap(read_strict_json(codemap))
    lock_payload = read_strict_json(lock)
    if not isinstance(lock_payload, dict) or set(lock_payload) != _LOCK_FIELDS or lock_payload.get("schema") != "paper-codemap-lock.v1":
        raise ValueError("paper codemap lock schema is invalid")
    hashes = lock_payload.get("hashes")
    expected_names = {"config_paper.py", "lineage_lock_paper.json", "codemap_paper.html", "codemap_paper.json"}
    if not isinstance(hashes, dict) or set(hashes) != expected_names or any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()):
        raise ValueError("paper codemap lock hashes are invalid")
    for name, expected in hashes.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"paper codemap hash differs: {name}")
    return parsed


__all__ = [
    "parse_nodes", "read_strict_json", "resolve_repo_module_source", "sha256_file",
    "static_top_level_names", "validate_codemap_triplet", "validate_lineage_lock",
]
