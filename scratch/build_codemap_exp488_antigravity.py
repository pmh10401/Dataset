"""Generate and validate the Exp488 caller/effect/test map."""
import argparse
import hashlib
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "experiments/exp488_catch22_train_descriptors_antigravity"
TEST = f"{PACKAGE}/test_exp488_antigravity.py"
ENTRIES = (
    ("experiments/exp481_strict_evidence_residual_correction_antigravity/policy_antigravity.py", "Exp481/484/488 evaluator", "Frozen residual selection; optional numeric descriptor space", TEST),
    ("experiments/exp481_strict_evidence_residual_correction_antigravity/evaluation_antigravity.py", "Exp481/484/488 runners", "Nested family exclusion and unchanged qualification gates", TEST),
    (f"{PACKAGE}/features_antigravity.py", "Exp488 runner", "Read-only normal TRAIN extraction and independent unscaled descriptors", TEST),
    (f"{PACKAGE}/run_antigravity.py", "python -m experiments.exp488_catch22_train_descriptors_antigravity.run_antigravity", "Single preregistered arm, validation and separate result/manifest", TEST),
    (TEST, "pytest", "Descriptor isolation and historical behavior regression", TEST),
    ("experiments/exp480_family_balanced_evidence_routing_antigravity/evidence_antigravity.sqlite3", "Exp488 runner", "Frozen score and original seven-feature evidence", TEST),
    ("experiments/exp484_distance_qualification_antigravity/results_antigravity.json", "Exp488 runner", "Immutable baseline result identity", TEST),
    ("docs/superpowers/specs/2026-09-16-exp488-catch22-train-descriptors-antigravity.html", "Exp488 implementation and review", "Preregistered descriptor, gates and input boundaries", TEST),
    ("scratch/build_codemap_exp488_antigravity.py", "CLI", "Versioned code-map consistency", "--validate"),
)


def build():
    nodes = []
    for path, caller, effect, tests in ENTRIES:
        file = ROOT / path
        coverage = [tests]
        if path.startswith("experiments/exp481_"):
            coverage += ["experiments/exp481_strict_evidence_residual_correction_antigravity/tests",
                         "experiments/exp484_distance_qualification_antigravity/tests"]
        nodes.append(dict(path=path, caller=caller, effect=effect, tests=coverage,
                          planned=not file.exists(),
                          sha256=hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None))
    payload = dict(schema="exp488_codemap_v1", baseline="docs/codemap/codemap_exp484_antigravity.json", nodes=nodes)
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    rows = "".join("<tr>" + "".join(f"<td>{escape(str(node[k]))}</td>" for k in ("path", "caller", "effect", "tests", "planned")) + "</tr>" for node in nodes)
    html = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Exp488 코드맵</title>'
            '<style>body{font:14px/1.6 system-ui;margin:28px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px;overflow-wrap:anywhere}</style>'
            '<h1>Exp488 호출·영향·검증</h1><table><tr><th>파일</th><th>호출자</th><th>영향</th><th>검증</th><th>구현 예정</th></tr>' + rows + '</table></html>\n').encode()
    lock = (json.dumps(dict(schema="exp488_codemap_lock_v1", json_sha256=hashlib.sha256(encoded).hexdigest(), html_sha256=hashlib.sha256(html).hexdigest()), sort_keys=True, indent=2) + "\n").encode()
    return {"json": encoded, "html": html, "lock": lock}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    for extension, content in build().items():
        path = ROOT / f"docs/codemap/codemap_exp488_antigravity.{extension}"
        if args.validate:
            if not path.exists() or path.read_bytes() != content:
                raise SystemExit(f"stale code map: {path}")
        else:
            path.write_bytes(content)
