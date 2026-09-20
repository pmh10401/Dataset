"""Validate reviewed Exp510 sources, synchronize its map, test, and freeze the pilot."""
from datetime import datetime, timezone
from hashlib import sha256, file_digest
from importlib.metadata import version
from pathlib import Path
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

from scratch.build_codemap_exp489_491_antigravity import ROOT, generate
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity

P = ROOT / 'experiments/exp510_separated_rank_antigravity'
read = lambda path: json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def main():
    if (P / 'preflight_antigravity.json').exists():
        raise RuntimeError('Existing preflight must be verified, not overwritten')
    assert not (P / 'pilot_manifest_antigravity.json').exists(), 'Pilot already started'
    reviewed = read(P / 'reviewed_sources_antigravity.json')
    review = read(P / 'code_review_antigravity.json')
    assert reviewed['exact_packet_matches_review']
    assert reviewed['prompt_sha256'] == review['prompt_sha256']
    assert review['review']['blocking_for_frozen_pilot'] is False
    for relative, expected in reviewed['source_hashes'].items():
        path = ROOT / relative
        if path == P / 'verify_antigravity.py':
            # Only the already reviewed atomic-output correction may differ.
            original = path.read_text().replace(
                'from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity\n', '')
            original = original.replace(
                "write_json_atomic_antigravity(P/'verification_antigravity.json',output);print(json.dumps(output))",
                "(P/'verification_antigravity.json').write_text(json.dumps(output,indent=2)+'\\n');print(json.dumps(output))")
            assert sha256(original.encode()).hexdigest() == expected, relative
        else:
            assert digest(path) == expected, relative
    historical = {}
    for exp in ('exp508', 'exp509'):
        base = ROOT / f'docs/codemap/codemap_{exp}_antigravity.json'
        lock = read(base.with_suffix('.lock'))
        for ext in ('json', 'html', 'lock'):
            path = base.with_suffix('.' + ext)
            historical[str(path.relative_to(ROOT))] = digest(path)
            if ext != 'lock':
                assert digest(path) == lock[ext + '_sha256'], str(path)
        for node in read(base)['nodes']:
            assert digest(ROOT / node['path']) == node['sha256'], node['path']
    auth = read(P / 'authorization_antigravity.json')
    prior_closure = ROOT / 'experiments/exp509_rank_signal_diagnosis_antigravity/closure_antigravity.json'
    assert auth['state'] == 'authorized_exp510'
    assert digest(prior_closure) == auth['prior_exp509_closure_sha256']
    historical[str(prior_closure.relative_to(ROOT))] = digest(prior_closure)
    benchmark = read(P / 'acceleration_results_antigravity.json')
    assert benchmark['success'] and benchmark['state'] == 'complete'
    assert digest(P / 'acceleration_antigravity.py') == benchmark['source_sha256']
    assert all(version(name) == installed for name, installed in benchmark['versions'].items())
    assert benchmark['batch_parity']['parity']
    for backend in benchmark['backends']:
        assert backend['first_parity']['parity'] and backend['warm_parity']['parity']
        if backend['backend'] != 'numpy':
            assert backend['speedup'] < 1.1
    assert all(not backend['float64_supported'] for backend in benchmark['gpu'].values())
    resolution = read(P / 'review_resolution_antigravity.json')
    assert not resolution['unresolved_pilot_blockers']
    assert read(P / 'review_checks_antigravity.json')['atomic_writer_roundtrip']
    # Regenerate only the current experiment's triplet, retaining prior versions.
    for ext, content in generate('exp510', 'experiments/exp510_separated_rank_antigravity').items():
        (ROOT / f'docs/codemap/codemap_exp510_antigravity.{ext}').write_bytes(content)
    tests = [
        'experiments/exp508_tail_rank_antigravity/test_exp508_antigravity.py',
        'experiments/exp508_tail_rank_antigravity/test_acceleration_antigravity.py',
        'experiments/exp509_rank_signal_diagnosis_antigravity/test_exp509_antigravity.py',
        'experiments/exp510_separated_rank_antigravity/test_exp510_antigravity.py',
    ]
    xml = P / 'resume_tests_antigravity.xml'
    env = dict(os.environ, NUMBA_CACHE_DIR='/private/tmp/exp510_numba_cache_antigravity')
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests,
                             '--junitxml=' + str(xml)], cwd=ROOT, env=env, capture_output=True, text=True)
    (P / 'resume_tests_antigravity.log').write_text(result.stdout + result.stderr)
    write_json_atomic_antigravity(P / 'resume_tests_exit_antigravity.json',
                                 dict(exit_code=result.returncode, xml_sha256=digest(xml) if xml.exists() else None))
    print(result.stdout, flush=True)
    assert result.returncode == 0, result.stderr
    suites = list(ET.parse(xml).getroot().iter('testsuite'))
    assert sum(int(s.attrib['tests']) for s in suites) == 19
    assert all(int(s.attrib[key]) == 0 for s in suites for key in ('failures', 'errors', 'skipped'))
    for relative, expected in historical.items():
        assert digest(ROOT / relative) == expected, relative
    files = set(P.glob('*.py')) | {ROOT / path for path in reviewed['source_hashes']}
    files.update(ROOT / path for path in tests)
    files.update(ROOT / path for path in historical)
    files.update(ROOT / f'docs/codemap/codemap_exp510_antigravity.{ext}' for ext in ('json', 'html', 'lock'))
    files.update(P / name for name in (
        'protocol_antigravity.html', 'authorization_antigravity.json', 'design_review_antigravity.json',
        'code_review_antigravity.json', 'reviewed_sources_antigravity.json', 'review_resolution_antigravity.json',
        'review_checks_antigravity.json', 'acceleration_results_antigravity.json',
        'regression_tests_antigravity.xml', 'resume_tests_antigravity.xml',
        'resume_tests_antigravity.log', 'resume_tests_exit_antigravity.json'))
    files.update((Path(__file__).resolve(), ROOT / 'scratch/build_codemap_exp489_491_antigravity.py'))
    preflight = dict(state='passed', utc=datetime.now(timezone.utc).isoformat(), tests_passed=19,
                     scope='Exp510 frozen TRAIN pilot; full evaluation requires all four verified gates',
                     artifacts={str(path.relative_to(ROOT)): digest(path) for path in sorted(files)},
                     historical_artifacts=historical, selected_backend='numpy_scipy_float64',
                     reviewed_sources_verified=True, atomic_verifier_correction_verified=True,
                     test_values_read=False, test_labels_read=False)
    write_json_atomic_antigravity(P / 'preflight_antigravity.json', preflight)
    plan = P / 'plan_antigravity.html'
    text = plan.read_text()
    for step in ('implementation', 'acceleration'):
        text = text.replace(f'<label id="{step}"><input type="checkbox" disabled>',
                            f'<label id="{step}"><input type="checkbox" checked disabled>')
    plan.write_text(text)
    print(json.dumps(dict(state='passed', tests=19, bound_artifacts=len(files),
                          historical_maps_unchanged=True, pilot_started=False)), flush=True)


if __name__ == '__main__':
    main()
