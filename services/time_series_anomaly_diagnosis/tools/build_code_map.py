from argparse import ArgumentParser
from hashlib import sha256
from html import escape
import json
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
DOCS = SERVICE_ROOT / "docs"
JSON_PATH = DOCS / "code_map_v1.json"
HTML_PATH = DOCS / "code_map_v1.html"
LOCK_PATH = DOCS / "code_map_v1.lock"
RELATIONSHIPS = (
    {
        "id": "release_bundle",
        "path": "packaging/release_bundle.py",
        "caller": "Windows release preparation",
        "effect": "Snapshots the full read-only DB and splits a history-free Windows distribution with checksums",
        "tests": ["tests/test_release_bundle.py"],
    },
    {
        "id": "packaged_smoke",
        "path": "packaging/smoke_executable.py",
        "caller": "Windows GitHub Actions build",
        "effect": "Exercises offline database registration, analysis, and restart through the packaged executable",
        "tests": ["packaging/smoke_executable.py"],
    },
    {
        "id": "code_map",
        "path": "tools/build_code_map.py",
        "caller": "Task completion workflow",
        "effect": "Builds and validates the service relationship map",
        "tests": ["tests/test_code_map.py"],
    },
    {
        "id": "model_resources",
        "path": "model_resources.py",
        "caller": "dataset_repository.py, analysis workflows",
        "effect": "Loads and verifies service-owned routing, recovery, and label-free review metadata",
        "tests": [
            "tests/test_data_access.py",
            "tests/test_model_resources.py",
            "tests/test_multi_evidence_review.py",
            "tests/test_real_data_smoke.py",
            "tests/test_cross_platform_service.py",
        ],
    },
    {
        "id": "service_resource_builder",
        "path": "tools/build_service_resources.py",
        "caller": "Release resource preparation workflow",
        "effect": "Copies model inputs and sanitizes trusted review evidence into deterministic service resources",
        "tests": ["tests/test_model_resources.py"],
    },
    {
        "id": "dataset_repository",
        "path": "dataset_repository.py",
        "caller": "Analysis API and service workflows",
        "effect": "Opens an explicit read-only source database and provides catalog, chart, and preparation access",
        "tests": [
            "tests/test_data_access.py",
            "tests/test_real_data_smoke.py",
            "tests/test_cross_platform_service.py",
        ],
    },
    {
        "id": "dataset_importer",
        "path": "dataset_importer.py",
        "caller": "Offline administrator module command and portable launcher import mode",
        "effect": "Validates and atomically installs a locally derived database for dataset_repository.py",
        "tests": [
            "tests/test_dataset_importer.py",
            "tests/test_service_launcher.py",
            "tests/test_cross_platform_service.py",
        ],
    },
    {
        "id": "score_sources",
        "path": "score_sources.py",
        "caller": "anomaly_detection.py and policy qualification workflows",
        "effect": "Computes the three approved deterministic aeon LocalGap score sources",
        "tests": [
            "tests/test_score_sources.py",
            "tests/test_real_data_smoke.py",
            "tests/test_cross_platform_service.py",
        ],
    },
    {
        "id": "anomaly_detection",
        "path": "anomaly_detection.py",
        "caller": "Analysis API and service workflows",
        "effect": "Combines two score seeds into A/B/C policy decisions, current review evidence, and evaluation metrics",
        "tests": [
            "tests/test_anomaly_detection.py",
            "tests/test_real_data_smoke.py",
            "tests/test_cross_platform_service.py",
        ],
    },
    {
        "id": "routing_policy",
        "path": "routing_policy.py",
        "caller": "anomaly_detection.py and policy qualification workflows",
        "effect": "Extracts seven original-TRAIN features and selects one score method from anonymous group evidence",
        "tests": [
            "tests/test_routing_policy.py",
            "tests/test_anomaly_detection.py",
        ],
    },
    {
        "id": "policy_qualification",
        "path": "policy_qualification.py",
        "caller": "Task 4/5 qualification CLI",
        "effect": "Builds label-free manifests, reusable score checkpoints, frozen predictions, policy metrics, and acceleration decisions",
        "tests": [
            "tests/test_policy_qualification.py",
            "tests/test_anomaly_detection.py",
        ],
    },
    {
        "id": "residual_fn_routing_qualification",
        "path": "residual_fn_routing_qualification.py",
        "caller": "Cached follow-up qualification workflow",
        "effect": "Replays frozen scores and tests anonymous routing against residual FN with FP guardrails",
        "tests": ["tests/test_residual_fn_routing_qualification.py"],
    },
    {
        "id": "multi_evidence_review",
        "path": "multi_evidence_review.py",
        "caller": "Model resource loader, application server, research reproduction, tests",
        "effect": "Parses label-free review evidence and ranks review-only candidates without changing automatic decisions",
        "tests": ["tests/test_multi_evidence_review.py"],
    },
    {
        "id": "analysis_history",
        "path": "analysis_history.py",
        "caller": "Analysis API, application server, and service workflows",
        "effect": "Persists analysis run status, timing, errors, and completed results in SQLite",
        "tests": ["tests/test_analysis_history.py", "tests/test_cross_platform_service.py"],
    },
    {
        "id": "application_server",
        "path": "application_server.py",
        "caller": "Portable launcher, direct CLI, HTTP clients, tests",
        "effect": "Assembles runtime dependencies, ranks review candidates before labels, and queues current database analyses",
        "tests": [
            "tests/test_application_server.py",
            "tests/test_web_interface.py",
            "tests/test_cross_platform_service.py",
        ],
    },
    {
        "id": "service_launcher",
        "path": "service_launcher.py",
        "caller": "Packaged Windows executable and macOS app",
        "effect": "Resolves portable paths, imports a local database on request, or prepares the Numba cache and starts the server and browser",
        "tests": ["tests/test_service_launcher.py"],
    },
    {
        "id": "desktop_package_builder",
        "path": "packaging/build_package.py",
        "caller": "Release packaging workflow",
        "effect": "Builds and finalizes the current operating system's database-free desktop distribution",
        "tests": ["tests/test_packaging.py"],
    },
    {
        "id": "web_application",
        "path": "web/application.js",
        "caller": "Browser users through web/index.html",
        "effect": "Checks service status and calls application server HTTP APIs to manage datasets, analyses, and results",
        "tests": ["tests/test_web_interface.py"],
    },
)
EDGES = (
    {
        "from": "service_launcher",
        "to": "dataset_importer",
        "type": "installs local database with",
    },
    {
        "from": "dataset_importer",
        "to": "dataset_repository",
        "type": "produces database for",
    },
    {
        "from": "dataset_repository",
        "to": "model_resources",
        "type": "imports",
    },
    {
        "from": "service_resource_builder",
        "to": "model_resources",
        "type": "produces resources for",
    },
    {
        "from": "anomaly_detection",
        "to": "score_sources",
        "type": "imports",
    },
    {
        "from": "anomaly_detection",
        "to": "dataset_repository",
        "type": "imports",
    },
    {
        "from": "anomaly_detection",
        "to": "routing_policy",
        "type": "imports",
    },
    {
        "from": "routing_policy",
        "to": "score_sources",
        "type": "imports",
    },
    {
        "from": "policy_qualification",
        "to": "dataset_repository",
        "type": "imports",
    },
    {
        "from": "policy_qualification",
        "to": "score_sources",
        "type": "imports",
    },
    {
        "from": "policy_qualification",
        "to": "anomaly_detection",
        "type": "imports",
    },
    {
        "from": "policy_qualification",
        "to": "routing_policy",
        "type": "imports",
    },
    {
        "from": "residual_fn_routing_qualification",
        "to": "policy_qualification",
        "type": "imports",
    },
    {
        "from": "residual_fn_routing_qualification",
        "to": "anomaly_detection",
        "type": "imports",
    },
    {
        "from": "residual_fn_routing_qualification",
        "to": "routing_policy",
        "type": "imports",
    },
    {
        "from": "residual_fn_routing_qualification",
        "to": "score_sources",
        "type": "imports",
    },
    {
        "from": "application_server",
        "to": "model_resources",
        "type": "consumes",
    },
    {
        "from": "application_server",
        "to": "multi_evidence_review",
        "type": "ranks review candidates with",
    },
    {
        "from": "model_resources",
        "to": "multi_evidence_review",
        "type": "parses review evidence with",
    },
    {
        "from": "application_server",
        "to": "dataset_repository",
        "type": "consumes",
    },
    {
        "from": "application_server",
        "to": "anomaly_detection",
        "type": "consumes",
    },
    {
        "from": "application_server",
        "to": "analysis_history",
        "type": "consumes",
    },
    {
        "from": "web_application",
        "to": "application_server",
        "type": "calls",
    },
    {
        "from": "service_launcher",
        "to": "application_server",
        "type": "starts",
    },
    {
        "from": "desktop_package_builder",
        "to": "service_launcher",
        "type": "packages",
    },
    {
        "from": "desktop_package_builder",
        "to": "model_resources",
        "type": "packages resources for",
    },
    {
        "from": "desktop_package_builder",
        "to": "web_application",
        "type": "packages assets for",
    },
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_code_map() -> dict[str, object]:
    nodes = []
    for relationship in RELATIONSHIPS:
        path = SERVICE_ROOT / relationship["path"]
        nodes.append({**relationship, "sha256": file_sha256(path)})
    return {
        "schema": "time_series_anomaly_diagnosis_code_map_v1",
        "nodes": nodes,
        "edges": [dict(edge) for edge in EDGES],
        "unknown_relationships": [],
    }


def render_html(payload: dict[str, object]) -> str:
    rows = "".join(
        f"<tr><td>{escape(node['id'])}</td><td>{escape(node['path'])}</td>"
        f"<td>{escape(node['caller'])}</td><td>{escape(node['effect'])}</td>"
        f"<td>{escape(', '.join(node['tests']))}</td></tr>"
        for node in payload["nodes"]
    )
    edge_rows = "".join(
        f"<tr><td>{escape(edge['from'])}</td><td>{escape(edge.get('type', '->'))}</td><td>{escape(edge['to'])}</td></tr>"
        for edge in payload.get("edges", [])
    )
    edge_section = f"<h2>간선 (Edges)</h2><table>{edge_rows}</table>" if edge_rows else ""
    return (
        f"<!doctype html><html lang='ko'><meta charset='utf-8'>"
        f"<title>서비스 코드맵 v1</title><h2>모듈</h2><table>{rows}</table>{edge_section}</html>"
    )


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def write_code_map() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    payload = build_code_map()
    json_bytes = canonical_json(payload)
    JSON_PATH.write_bytes(json_bytes)
    HTML_PATH.write_text(render_html(payload), encoding="utf-8")
    LOCK_PATH.write_bytes(canonical_json({"map_sha256": sha256(json_bytes).hexdigest()}))


def validate_code_map() -> None:
    payload = build_code_map()
    expected_json = canonical_json(payload)
    if not JSON_PATH.exists() or JSON_PATH.read_bytes() != expected_json:
        raise SystemExit("code map JSON is stale")
    expected_lock = canonical_json({"map_sha256": sha256(expected_json).hexdigest()})
    if not LOCK_PATH.exists() or LOCK_PATH.read_bytes() != expected_lock:
        raise SystemExit("code map lock is stale")
    if not HTML_PATH.exists() or HTML_PATH.read_text(encoding="utf-8") != render_html(payload):
        raise SystemExit("code map HTML is stale")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args()
    validate_code_map() if arguments.validate else write_code_map()
