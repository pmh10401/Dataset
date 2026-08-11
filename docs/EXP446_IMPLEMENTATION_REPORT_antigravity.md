# Exp446 Task 1 implementation report

## Scope and status

Task 1 freezes the Exp446 engineering-only publication-materialization repair before product
code. `Exp446` is `preregistered_implementation_pending`; no official lifecycle command, Dataset
root, TEST label, product module, or test was accessed or changed.

Exp445 is the immutable completed calculation / failed-publication predecessor, not a failed
scientific result: `30/30`, error `0`, labels `0`, CPU/11 with inner thread `1`, aggregate
`af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`, absent publication and
evaluation, five terminal SHA256 values, and exact failure text
`Exp444 original-control replay is invalid`. Its root permits no retry, resume, publication,
inspect, or evaluation.

## Files

- `docs/EXP446_PREREGISTRATION_antigravity_KO.md`
- `docs/EXP446_RUNBOOK_antigravity_KO.md`
- `docs/EXP446_IMPLEMENTATION_REPORT_antigravity.md`
- `EXPERIMENT_CATALOG.json`, `RESEARCH_STATE.json`, `FAILURE_LEDGER.json`, `CHAMPION_LEDGER.json`
- `docs/codemap/codemap_exp446_antigravity.{html,json,lock}`

The prior `codemap_exp445_antigravity.lock` records
`0529afb3c4e6b245604c6fbb508b3e6b4fb4de78`; Task 1 created a separate Exp446 triplet rather
than modifying the Exp445 triplet. The latest map regeneration is based on the prior Task 1
commit `a4959d4a7aaa657c91d96b65732999775a76c249`.

## Initial map topology and unknowns

The Exp446 map has 20 nodes, 58 edges, and 5 flows. It retains the Exp445 structural baseline,
replaces the obsolete Exp425 checkpoint node with `exp446_governance`, and has no Exp446 product
edge or flow. Product callers, downstream effects, test coverage, imports, and runtime routing
remain explicitly unknown until product code exists.

The historical Exp445 flow now records its terminal failed-publication state rather than an
obsolete readiness/new/evaluate next action.

## Validation

JSON and diff checks:

```bash
python3 -m json.tool EXPERIMENT_CATALOG.json >/dev/null
python3 -m json.tool RESEARCH_STATE.json >/dev/null
python3 -m json.tool FAILURE_LEDGER.json >/dev/null
python3 -m json.tool CHAMPION_LEDGER.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp446_antigravity.json >/dev/null
python3 -m json.tool docs/codemap/codemap_exp446_antigravity.lock >/dev/null
git diff --check
```

Exact reproducible code-map validator:

```bash
python3 - <<'PY'
import base64, hashlib, json, re, subprocess
from pathlib import Path

root = Path('.')
map_bytes = (root / 'docs/codemap/codemap_exp446_antigravity.json').read_bytes()
map_data = json.loads(map_bytes)
html = (root / 'docs/codemap/codemap_exp446_antigravity.html').read_text()
encoded = re.search(r'const encoded="([^"]*)";', html).group(1)
assert base64.b64decode(encoded) == map_bytes
node_ids = {node['id'] for node in map_data['nodes']}
assert len(node_ids) == len(map_data['nodes']) == 20
for edge in map_data['edges']:
    assert edge['from'] in node_ids and edge['to'] in node_ids
for flow in map_data['flows']:
    for step in flow['steps']:
        assert step in node_ids
evidence = [item for container in [*map_data['nodes'], *map_data['edges']] for item in container.get('evidence', [])]
for item in evidence:
    source = root / item['path']
    assert source.is_file() and item['symbol'] in source.read_text()
lock = json.loads((root / 'docs/codemap/codemap_exp446_antigravity.lock').read_text())
modules = {}
for name in filter(None, subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')):
    if name.startswith('docs/codemap/') or any(name == excluded or name.startswith(excluded + '/') for excluded in lock['excluded_directories']):
        continue
    modules.setdefault(name.split('/', 1)[0] if '/' in name else 'root', []).append(name)
actual = {}
for module, names in sorted(modules.items()):
    digest = hashlib.sha256()
    for name in sorted(names):
        payload = (root / name).read_bytes()
        digest.update(name.encode()); digest.update(b'\0'); digest.update(str(len(payload)).encode()); digest.update(b'\0'); digest.update(payload); digest.update(b'\0')
    actual[module] = {'fingerprint': digest.hexdigest(), 'tracked_files': len(names)}
assert actual == lock['module_fingerprints']
print(f'map validation passed: evidence={len(evidence)} nodes={len(node_ids)} edges={len(map_data["edges"])} flows={len(map_data["flows"])} fingerprints={len(actual)}')
PY
```

Recorded result after fix round 1: all six JSON parsers exit 0, `git diff --check` is clean, and
the validator passes HTML byte parity, evidence paths/symbols, edge/flow references, and module
fingerprints. The validator output is recorded in the Task 1 handoff report.

## Remaining concerns

The map intentionally does not assert future Exp446 product wiring. Product implementation must
resolve these unknowns and regenerate the complete Exp446 triplet in its own bounded change.
