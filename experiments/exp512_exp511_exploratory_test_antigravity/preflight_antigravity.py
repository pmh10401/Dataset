"""Bind explicit approval, immutable model, actual tests and independent review."""
from pathlib import Path
from importlib.metadata import version
import json
import xml.etree.ElementTree as ET
from scratch.build_codemap_exp489_491_antigravity import generate
from .scoring_antigravity import P,ROOT,PRIOR,BASE,EXP502,DB,read,digest,check_hashes,now,write_json_atomic_antigravity,cpu_worker_policy_antigravity
from .scoring_antigravity import assert_unbound_wal_antigravity

def main():
    assert_unbound_wal_antigravity(DB)
    target=P/'preflight_antigravity.json'
    assert not target.exists() and not (P/'manifest_antigravity.json').exists(),'Existing run must be verified, not overwritten'
    auth=read(P/'authorization_antigravity.json')
    assert auth['state']=='authorized_exploratory_test_after_train_rejection' and auth['candidate']=='crossfit_rank'
    assert not auth['train_gate_passed'] and auth['protocol_sha256']==digest(P/'protocol_antigravity.html')
    check_hashes(auth['historical_artifacts'])
    original=read(PRIOR/'pilot_results_antigravity.json');verified=read(PRIOR/'verification_antigravity.json')
    assert original['selected']=='square_joint' and not original['full_run_required']
    assert verified['state']=='passed' and verified['result_sha256']==digest(PRIOR/'pilot_results_antigravity.json')
    oldmap=read(ROOT/'docs/codemap/codemap_exp511_antigravity.json');check_hashes({n['path']:n['sha256'] for n in oldmap['nodes']})
    reviewed=read(P/'reviewed_sources_antigravity.json');review=read(P/'code_review_antigravity.json')
    assert review['source_hashes']==reviewed['source_hashes']
    assert review['review']['blocking_for_full_evaluation'] is False
    check_hashes(reviewed['source_hashes'])
    resolution=read(P/'review_resolution_antigravity.json');assert resolution['state']=='resolved' and not resolution['unresolved_blockers']
    exits=read(P/'tests_exit_antigravity.json');assert exits['exit_code']==0
    assert exits['xml_sha256']==digest(P/'tests_all_antigravity.xml')
    suites=list(ET.parse(P/'tests_all_antigravity.xml').getroot().iter('testsuite'))
    count=sum(int(s.attrib['tests']) for s in suites);assert count==31
    assert all(int(s.attrib[k])==0 for s in suites for k in ('errors','failures','skipped'))
    benchmark=read(PRIOR/'acceleration_results_antigravity.json');production=read(PRIOR/'production_backend_assessment_antigravity.json')
    assert benchmark['success'] and production['success'] and not production['eligible_alternatives']
    assert all(version(name)==value for name,value in benchmark['versions'].items())
    assert all(r['parity'] and r['frozen_five_exact'] for r in production['records'])
    assert all(not r['float64_supported'] for r in benchmark['gpu'].values())
    base=read(BASE/'run_manifest_antigravity.json');old502=read(EXP502/'full_manifest_antigravity.json')
    assert digest(BASE/'results_antigravity.json')==base['result_sha256']
    assert digest(EXP502/'full_results_antigravity.json')==old502['result_sha256']
    assert digest(DB)==base['binding']['source_binding']['database']==old502['binding']['database_sha256']
    assert_unbound_wal_antigravity(DB)
    rows=base['binding']['population'];assert len(rows)==1117 and len({r['source_family'] for r in rows})==128
    for ext,data in generate('exp512','experiments/exp512_exp511_exploratory_test_antigravity').items():
        (ROOT/f'docs/codemap/codemap_exp512_antigravity.{ext}').write_bytes(data)
    files=set(P.glob('*.py'))|{ROOT/p for p in reviewed['source_hashes']}|{ROOT/p for p in auth['historical_artifacts']}
    files.update(P/name for name in ('authorization_antigravity.json','protocol_antigravity.html','reviewed_sources_antigravity.json',
        'code_review_antigravity.json','review_resolution_antigravity.json','tests_exit_antigravity.json','tests_all_antigravity.xml','tests_run_antigravity.log'))
    files.update(ROOT/f'docs/codemap/codemap_exp512_antigravity.{ext}' for ext in ('json','html','lock'))
    files.update(PRIOR/name for name in ('acceleration_results_antigravity.json','production_backend_assessment_antigravity.json'))
    files.update((BASE/'run_manifest_antigravity.json',BASE/'results_antigravity.json',EXP502/'full_manifest_antigravity.json',EXP502/'full_results_antigravity.json'))
    out=dict(state='passed',utc=now(),tests_passed=count,artifacts={str(f.relative_to(ROOT)):digest(f) for f in sorted(files)},
             selected_backend='numpy_scipy_float64',cpu_policy=cpu_worker_policy_antigravity(),
             acceleration_scope='Unmodified Exp511 scorer; inherited whole-stage parity and no qualifying production-path speedup. New orchestration/hashing and NumPy evaluation do not replace numerical score kernels.',
             test_values_read=False,test_labels_read=False,train_gate_passed=False,model_promotion=False)
    write_json_atomic_antigravity(target,out)
    plan=P/'plan_antigravity.html';text=plan.read_text()
    for step in ('freeze','preflight'):text=text.replace(f'id="{step}"><input type="checkbox" disabled',f'id="{step}"><input type="checkbox" checked disabled')
    plan.write_text(text)
    print(json.dumps(dict(state='passed',tests=count,artifacts=len(files),cpu_policy=out['cpu_policy'])),flush=True)

if __name__=='__main__':main()
