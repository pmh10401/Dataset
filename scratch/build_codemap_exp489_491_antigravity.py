"""Generate the three additive experiment maps, including their source dependencies."""
import argparse
import ast
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    "exp489": "experiments/exp489_lof_score_head_antigravity",
    "exp490": "experiments/exp490_behavior_descriptors_antigravity",
    "exp491": "experiments/exp491_gain_prediction_antigravity",
}


def generate(version, package):
    files = set((ROOT / package).glob("*.py"))
    pending = list(files)
    dependencies = set()
    while pending:
        path = pending.pop()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.level:
                    parent = path.parent
                    for _ in range(node.level - 1):
                        parent = parent.parent
                    source = parent / (node.module.replace(".", "/") + ".py")
                elif node.module.startswith("experiments."):
                    source = ROOT / (node.module.replace(".", "/") + ".py")
                else:
                    continue
                if source.exists() and source not in files and source not in dependencies:
                    dependencies.add(source)
                    pending.append(source)
    tests = [str(path.relative_to(ROOT)) for path in sorted(files) if path.name.startswith("test_")]
    if not tests:
        tests = [f"{package}/test_{version}_antigravity.py"]
    nodes = []
    for path in sorted(files | dependencies):
        callers = []
        module = str(path.relative_to(ROOT).with_suffix("")).replace("/", ".")
        for candidate in sorted(files | dependencies):
            if candidate != path and (module in candidate.read_text() or
                                      (candidate.parent == path.parent and f"from .{path.stem}" in candidate.read_text())):
                callers.append(str(candidate.relative_to(ROOT)))
        nodes.append(dict(path=str(path.relative_to(ROOT)), callers=callers,
                          effect="experiment implementation" if path in files else "reused immutable dependency",
                          tests=tests, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for relative, caller, effect in (
        ("docs/superpowers/specs/2026-09-16-exp489-491-preregistration-antigravity.html", "experiment runners", "frozen protocol"),
        ("experiments/exp480_family_balanced_evidence_routing_antigravity/evidence_antigravity.sqlite3", "validated evidence loader", "historical development outcomes"),
        ("experiments/exp488_catch22_train_descriptors_antigravity/descriptor_cache_antigravity.json", "descriptor loader", "normal TRAIN descriptors"),
    ):
        path = ROOT / relative
        nodes.append(dict(path=relative, callers=[caller], effect=effect, tests=tests,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    data = (json.dumps(dict(schema=f"{version}_codemap_v1", nodes=nodes), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    page = (f'<!doctype html><html lang="ko"><meta charset="utf-8"><title>{version} 코드맵</title>'
            '<style>body{font:14px/1.6 system-ui;margin:25px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
            f'<h1>{version}: 호출·영향·검증</h1><pre>{html.escape(data.decode())}</pre></html>\n').encode()
    lock = (json.dumps(dict(json_sha256=hashlib.sha256(data).hexdigest(), html_sha256=hashlib.sha256(page).hexdigest()), indent=2) + "\n").encode()
    return dict(json=data, html=page, lock=lock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    for version, package in PACKAGES.items():
        for extension, content in generate(version, package).items():
            path = ROOT / f"docs/codemap/codemap_{version}_antigravity.{extension}"
            if args.validate:
                if path.read_bytes() != content:
                    raise SystemExit(f"stale map: {path}")
            else:
                path.write_bytes(content)
