"""Strict, side-effect-free provenance validation for frozen paper origins."""
from __future__ import annotations

import ast
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re

from .config_paper import ARTIFACT_PARITY_CLAIM_PATH, LINEAGE_LOCK_PATH, REPOSITORY_ROOT, SOURCE_PROVENANCE_PATH
from .contracts_paper import LineageNode, PaperCodemap


_DOTTED_IDENTIFIER = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_NODE_FIELDS = frozenset({
    "node_id", "stage", "input_kind", "module", "symbols", "source_sha256",
    "paper_module", "label_access", "parity_oracle",
})
_CODEMAP_FIELDS = ("callers", "effects", "tests", "source_origins")
_REQUIRED_CODEMAP_CALLERS = {
    "calibration_paper.py": frozenset({"hard_route_paper.py", "sources_paper/minirocket_paper.py"}),
    "contracts_paper.py": frozenset({"hard_route_paper.py", "sources_paper/legacy_route_sources_paper.py", "sources_paper/minirocket_paper.py"}),
    "sources_paper/hard_sources_paper.py": frozenset({"hard_route_paper.py"}),
    "sources_paper/legacy_route_sources_paper.py": frozenset({"hard_route_paper.py", "tests_paper/test_hard_route_paper.py"}),
    "sources_paper/minirocket_paper.py": frozenset({"hard_route_paper.py", "tests_paper/test_minirocket_paper.py"}),
    "sources_paper/imaging_paper.py": frozenset({"sources_paper/legacy_route_sources_paper.py"}),
    "sources_paper/knn_paper.py": frozenset({"sources_paper/legacy_route_sources_paper.py"}),
    "sources_paper/multirocket_hydra_paper.py": frozenset({"sources_paper/legacy_route_sources_paper.py"}),
    "sources_paper/preprocessing_paper.py": frozenset({"sources_paper/legacy_route_sources_paper.py", "sources_paper/minirocket_paper.py"}),
    "sources_paper/rocket_paper.py": frozenset({"sources_paper/legacy_route_sources_paper.py"}),
}
_LOCK_FIELDS = frozenset({"schema", "hashes"})
_SOURCE_PROVENANCE_FIELDS = frozenset({"schema", "helpers"})
_SOURCE_PROVENANCE_HELPER_FIELDS = frozenset({"paper_sha256", "origins"})
_SOURCE_PROVENANCE_ORIGIN_FIELDS = frozenset({"module", "symbols", "source_sha256"})
_ARTIFACT_PARITY_CLAIM_FIELDS = frozenset({"schema", "parity_kind", "frozen_datasets", "freeze_before_artifact_read", "available_stages", "unavailable_stages", "artifacts"})
_ARTIFACT_PARITY_AVAILABLE = ("exp137", "exp236")
_ARTIFACT_PARITY_UNAVAILABLE = ("exp74d", "exp89", "exp90", "exp93", "exp253")
_ARTIFACT_PARITY_DATASETS = tuple(f"ACSF1_normal_{index}" for index in range(10))
_ARTIFACT_PARITY_PATHS = {
    "exp137": ("results/exp137/experiment_137_operational_triage_results.csv",),
    "exp236": (
        "outputs/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl",
        "/Users/mac/Documents/Dataset/experiments/exp236_direct_run_antigravity/exp236_frozen_predictions.jsonl",
    ),
}


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_strict_json(path: Path) -> object:
    return _strict_json_loads(path.read_text(encoding="utf-8"))


def reject_duplicate_object_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(payload: str) -> object:
    return json.loads(
        payload,
        object_pairs_hook=reject_duplicate_object_keys,
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


def _paper_source_helper_inventory(root: Path) -> frozenset[str]:
    source_root = root / "sources_paper"
    helpers = {
        path.relative_to(root).as_posix()
        for path in source_root.glob("*_paper.py")
        if "__pycache__" not in path.parts and path.name != "package_paper.py"
    }
    # The route is scientific copied logic too; provenance must bind its bytes
    # and every frozen selector origin, not only scorer helpers under sources/.
    helpers.add("hard_route_paper.py")
    return frozenset(helpers)


def validate_source_provenance_paper(paper_root: Path) -> dict[str, object]:
    """Hash-bind every copied scientific helper and every unexecuted origin."""
    root = Path(paper_root)
    provenance = root / "sources_paper" / "source_provenance_paper.json"
    if not provenance.is_file():
        raise ValueError("paper source provenance is unavailable")
    payload = read_strict_json(provenance)
    if not isinstance(payload, dict) or set(payload) != _SOURCE_PROVENANCE_FIELDS or payload.get("schema") != "paper-source-provenance.v1":
        raise ValueError("paper source provenance schema is invalid")
    helpers = payload.get("helpers")
    expected_helpers = _paper_source_helper_inventory(root)
    if not isinstance(helpers, dict) or set(helpers) != expected_helpers:
        raise ValueError("paper source provenance helper inventory differs")
    for helper, record in helpers.items():
        if not isinstance(record, dict) or set(record) != _SOURCE_PROVENANCE_HELPER_FIELDS:
            raise ValueError("paper source provenance helper schema is invalid")
        if type(record["paper_sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", record["paper_sha256"]):
            raise ValueError("paper source provenance helper hash is invalid")
        if sha256_file(root / helper) != record["paper_sha256"]:
            raise ValueError(f"paper source helper SHA differs: {helper}")
        origins = record["origins"]
        if not isinstance(origins, list) or not origins:
            raise ValueError("paper source provenance origins are invalid")
        for origin in origins:
            if not isinstance(origin, dict) or set(origin) != _SOURCE_PROVENANCE_ORIGIN_FIELDS:
                raise ValueError("paper source provenance origin schema is invalid")
            module, symbols, expected = origin["module"], origin["symbols"], origin["source_sha256"]
            if type(module) is not str or not _DOTTED_IDENTIFIER.fullmatch(module) or not isinstance(symbols, list) or not symbols or any(type(symbol) is not str or not symbol for symbol in symbols) or type(expected) is not str or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("paper source provenance origin values are invalid")
            source = resolve_repo_module_source(module)
            if sha256_file(source) != expected:
                raise ValueError(f"paper source origin SHA differs: {module}")
            exported = static_top_level_names(ast.parse(source.read_text(encoding="utf-8"), filename=str(source)))
            if any(symbol not in exported for symbol in symbols):
                raise ValueError(f"paper source origin symbol is unavailable: {module}")
    return helpers


def validate_artifact_parity_claim(paper_root: Path) -> dict[str, object]:
    """Validate the narrow, post-freeze historical-artifact parity boundary."""
    root = Path(paper_root)
    path = root / ARTIFACT_PARITY_CLAIM_PATH.name
    payload = read_strict_json(path)
    if not isinstance(payload, dict) or set(payload) != _ARTIFACT_PARITY_CLAIM_FIELDS:
        raise ValueError("paper artifact parity claim schema is invalid")
    if payload["schema"] != "paper-artifact-parity-claim.v1" or payload["parity_kind"] != "historical_artifact_lanes":
        raise ValueError("paper artifact parity claim kind is invalid")
    datasets, available, unavailable, artifacts = payload["frozen_datasets"], payload["available_stages"], payload["unavailable_stages"], payload["artifacts"]
    if tuple(datasets) != _ARTIFACT_PARITY_DATASETS:
        raise ValueError("paper artifact parity claim datasets are invalid")
    if type(payload["freeze_before_artifact_read"]) is not bool or payload["freeze_before_artifact_read"] is not True:
        raise ValueError("paper artifact parity claim must require create-only freeze")
    if tuple(available) != _ARTIFACT_PARITY_AVAILABLE or tuple(unavailable) != _ARTIFACT_PARITY_UNAVAILABLE:
        raise ValueError("paper artifact parity stage boundary differs")
    if not isinstance(artifacts, dict) or {stage: tuple(paths) for stage, paths in artifacts.items()} != _ARTIFACT_PARITY_PATHS:
        raise ValueError("paper artifact parity artifact paths are invalid")
    return {"available_stages": tuple(available), "unavailable_stages": tuple(unavailable), "freeze_before_artifact_read": payload["freeze_before_artifact_read"]}


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


def _ast_import_callers(paper_root: Path) -> dict[str, frozenset[str]]:
    """Return the direct paper-module caller graph from unexecuted import ASTs."""
    root = Path(paper_root).resolve(strict=True)
    modules = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*_paper.py")
        if "__pycache__" not in path.parts
    }
    callers: dict[str, set[str]] = {module: set() for module in modules}
    package_prefix = "paper.equipment_anomaly_detection."
    for path in root.rglob("*_paper.py"):
        if "__pycache__" in path.parts:
            continue
        caller = path.relative_to(root).as_posix()
        package = path.relative_to(root).parts[:-1]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                candidates = [
                    alias.name.removeprefix(package_prefix).replace(".", "/") + ".py"
                    for alias in node.names if alias.name.startswith(package_prefix)
                ]
            elif not isinstance(node, ast.ImportFrom):
                continue
            elif node.level:
                base = package[:len(package) - (node.level - 1)]
                candidates = ["/".join(base + tuple(node.module.split("."))) + ".py"] if node.module else ["/".join(base + (alias.name,)) + ".py" for alias in node.names]
            elif node.module == package_prefix.removesuffix("."):
                candidates = [alias.name + ".py" for alias in node.names]
            elif node.module is not None and node.module.startswith(package_prefix):
                base = node.module.removeprefix(package_prefix).replace(".", "/")
                candidates = [base + ".py"]
                candidates.extend(base + "/" + alias.name + ".py" for alias in node.names)
            else:
                continue
            for imported in candidates:
                if imported in callers:
                    callers[imported].add(caller)
    return {module: frozenset(value) for module, value in callers.items()}


def _render_codemap_html(codemap: PaperCodemap) -> str:
    graph = json.dumps(
        {"schema": "paper-codemap.v1", "required_fields": list(codemap.required_fields), "modules": codemap.modules},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    rows = "\n".join(
        "<tr>" + "".join((
            f"<td>{escape(module)}</td>",
            f"<td>{escape('; '.join(detail['callers']))}</td>",
            f"<td>{escape('; '.join(detail['effects']))}</td>",
            f"<td>{escape('; '.join(detail['tests']))}</td>",
            f"<td>{escape('; '.join(detail['source_origins']))}</td>",
        )) + "</tr>"
        for module, detail in codemap.modules.items()
    )
    return "\n".join((
        "<!doctype html>",
        "<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Equipment Anomaly Detection paper codemap</title></head>",
        "<body>",
        "<h1>Equipment Anomaly Detection paper dependency map</h1>",
        "<p>Validation mode: <code>static_ast_sha256</code>. Frozen legacy origins are resolved, parsed, and SHA256-bound without import execution because some have import-time file effects or optional environment dependencies. Tasks 4–7 establish executable parity after scientific copies enter this namespace.</p>",
        "<table><thead><tr><th>Module</th><th>Callers</th><th>Effects</th><th>Tests</th><th>Source origins</th></tr></thead><tbody>",
        rows,
        "</tbody></table>",
        "<script id=\"codemap-graph-paper\" type=\"application/json\">",
        graph,
        "</script>",
        "</body></html>",
        "",
    ))


def _paper_module_inventory(root: Path) -> frozenset[str]:
    return frozenset(
        path.relative_to(root).as_posix()
        for path in root.rglob("*_paper.py")
        if "__pycache__" not in path.parts
    )


def validate_codemap_triplet(paper_root: Path) -> PaperCodemap:
    root = Path(paper_root)
    html, codemap, lock = (root / "codemap_paper.html", root / "codemap_paper.json", root / "codemap_paper.lock")
    config, contracts, lineage = root / "config_paper.py", root / "contracts_paper.py", root / "lineage_lock_paper.json"
    provenance, claim = root / "sources_paper" / "source_provenance_paper.json", root / ARTIFACT_PARITY_CLAIM_PATH.name
    required_paths = (html, codemap, lock, config, contracts, lineage, provenance, claim)
    if any(not item.is_file() for item in required_paths):
        raise ValueError("paper codemap triplet is incomplete")
    lock_payload = read_strict_json(lock)
    if not isinstance(lock_payload, dict) or set(lock_payload) != _LOCK_FIELDS or lock_payload.get("schema") != "paper-codemap-lock.v1":
        raise ValueError("paper codemap lock schema is invalid")
    hashes = lock_payload.get("hashes")
    expected_names = {
        "config_paper.py", "contracts_paper.py", "lineage_lock_paper.json",
        "codemap_paper.html", "codemap_paper.json", "hard_route_paper.py",
        "sources_paper/source_provenance_paper.json", ARTIFACT_PARITY_CLAIM_PATH.name,
    } | _paper_source_helper_inventory(root)
    if not isinstance(hashes, dict) or set(hashes) != expected_names or any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()):
        raise ValueError("paper codemap lock hashes are invalid")
    for name, expected in hashes.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"paper codemap hash differs: {name}")
    parsed = _parse_codemap(read_strict_json(codemap))
    actual_callers = _ast_import_callers(root)
    if any(not callers <= actual_callers.get(module, frozenset()) for module, callers in _REQUIRED_CODEMAP_CALLERS.items()):
        raise ValueError("paper codemap caller graph omits required scientific AST edges")
    if {module: frozenset(detail["callers"]) for module, detail in parsed.modules.items()} != actual_callers:
        raise ValueError("paper codemap caller graph differs from required AST edges")
    validate_source_provenance_paper(root)
    validate_artifact_parity_claim(root)
    if set(parsed.modules) != _paper_module_inventory(root):
        raise ValueError("paper codemap module inventory differs")
    if html.read_text(encoding="utf-8") != _render_codemap_html(parsed):
        raise ValueError("paper codemap HTML graph differs")
    return parsed


__all__ = [
    "LINEAGE_NODES", "parse_nodes", "read_strict_json", "reject_duplicate_object_keys",
    "resolve_repo_module_source", "sha256_file", "static_top_level_names",
    "validate_artifact_parity_claim", "validate_codemap_triplet", "validate_lineage_lock", "validate_source_provenance_paper",
]


LINEAGE_NODES = validate_lineage_lock(LINEAGE_LOCK_PATH)
