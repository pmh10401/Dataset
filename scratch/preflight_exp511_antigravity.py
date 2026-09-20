"""Freeze one reviewed Exp511 candidate after existing tests and backend assessment."""
from pathlib import Path
from hashlib import file_digest
from datetime import datetime,timezone
from importlib.metadata import version
import json
import xml.etree.ElementTree as ET
from scratch.build_codemap_exp489_491_antigravity import ROOT,generate
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity,cpu_worker_policy_antigravity

P=ROOT/'experiments/exp511_crossfit_rank_ensemble_antigravity'
read=lambda p:json.loads(p.read_text())


def digest(path):
    with path.open('rb') as stream:return file_digest(stream,'sha256').hexdigest()


def main():
    target=P/'preflight_antigravity.json'
    assert not target.exists() and not (P/'pilot_manifest_antigravity.json').exists(), 'Do not overwrite a frozen experiment'
    auth=read(P/'authorization_antigravity.json');assert auth['state']=='authorized_exp511'
    assert auth['protocol_sha256']==digest(P/'protocol_antigravity.html')
    for relative,expected in auth['historical_artifacts'].items():assert digest(ROOT/relative)==expected,relative
    reviewed=read(P/'reviewed_sources_antigravity.json');review=read(P/'code_review_antigravity.json')
    assert review['prompt_sha256']==reviewed['prompt_sha256']
    assert review['review']['blocking_for_pilot'] is False
    for relative,expected in reviewed['source_hashes'].items():assert digest(ROOT/relative)==expected,relative
    resolution=read(P/'review_resolution_antigravity.json')
    assert resolution['state']=='resolved' and not resolution['unresolved_pilot_blockers']
    exits=read(P/'validation_exit_antigravity.json');assert exits['tests']['exit_code']==exits['acceleration']['exit_code']==0
    for name in ('tests','acceleration'):assert digest(P/(name+'_run_antigravity.log'))==exits[name]['log_sha256']
    suites=list(ET.parse(P/'tests_all_antigravity.xml').getroot().iter('testsuite'))
    count=sum(int(s.attrib['tests']) for s in suites);assert count==24
    assert all(int(s.attrib[k])==0 for s in suites for k in ('errors','failures','skipped'))
    benchmark=read(P/'acceleration_results_antigravity.json')
    assert benchmark['state']=='complete' and benchmark['success'] and benchmark['batch_parity']['parity']
    assert benchmark['source_sha256']==digest(P/'acceleration_antigravity.py')
    assert all(version(name)==value for name,value in benchmark['versions'].items())
    production=read(P/'production_backend_assessment_antigravity.json')
    assert production['state']=='complete' and production['success']
    assert not production['eligible_alternatives'], 'Qualify an eligible production backend before freezing'
    assert all(r['parity'] and r['frozen_five_exact'] for r in production['records'])
    assert len(production['records'])==9
    assessment=ROOT/'scratch/assess_exp511_production_backend_antigravity.py'
    assert digest(assessment)==production['source_sha256']
    assert not any(probe['float64_supported'] for probe in benchmark['gpu'].values())
    assert benchmark['selected_backend']=='numpy_scipy_float64'
    for ext,data in generate('exp511','experiments/exp511_crossfit_rank_ensemble_antigravity').items():
        (ROOT/f'docs/codemap/codemap_exp511_antigravity.{ext}').write_bytes(data)
    files=set(P.glob('*.py'))|{ROOT/relative for relative in reviewed['source_hashes']}
    files.update(ROOT/f'docs/codemap/codemap_exp511_antigravity.{ext}' for ext in ('json','html','lock'))
    files.update(P/name for name in ('authorization_antigravity.json','protocol_antigravity.html','code_review_antigravity.json',
                 'reviewed_sources_antigravity.json','review_resolution_antigravity.json','acceleration_results_antigravity.json',
                 'production_backend_assessment_antigravity.json','validation_exit_antigravity.json','tests_run_antigravity.log','acceleration_run_antigravity.log',
                 'tests_all_antigravity.xml','tests_red_antigravity.xml'))
    files.update((Path(__file__).resolve(),ROOT/'scratch/build_codemap_exp489_491_antigravity.py',assessment))
    population=read(ROOT/'experiments/exp495_normal_geometry_antigravity/diagnostics_manifest_antigravity.json')['population']
    assert len(population)==50 and len({r['source_family'] for r in population})==50
    out=dict(state='passed',utc=datetime.now(timezone.utc).isoformat(),tests_passed=count,
             artifacts={str(f.relative_to(ROOT)):digest(f) for f in sorted(files)},
             scope='One frozen TRAIN candidate; all four verified gates required before full1117 evaluation',
             selected_backend='numpy_scipy_float64',cpu_policy=cpu_worker_policy_antigravity(),
             historical_files_verified=len(auth['historical_artifacts']),
             test_values_read=False,test_labels_read=False)
    write_json_atomic_antigravity(target,out)
    plan=P/'plan_antigravity.html';text=plan.read_text()
    for step in ('implementation','preflight'):
        text=text.replace(f'id="{step}"><input type="checkbox" disabled',f'id="{step}"><input type="checkbox" checked disabled')
    plan.write_text(text)
    print(json.dumps(dict(state='passed',tests=count,artifacts=len(files),cpu_policy=out['cpu_policy'],pilot_started=False)))


if __name__=='__main__':main()
