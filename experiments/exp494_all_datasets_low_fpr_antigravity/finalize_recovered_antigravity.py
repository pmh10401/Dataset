"""Publish annotated results atomically; retry safely after any write boundary.

Use this entry point, not the frozen recovery source's unused finalize mode.
The scoring source remains unchanged because its hash identifies the recovered scores.
"""
from .constant_recovery_antigravity import *
from . import run_antigravity as frozen_run

def validate_resume(manifest,binding_digest,code_sha):
    if 'finalizer_sha256' in manifest:assert manifest['finalizer_sha256']==code_sha
    output=P/'results_antigravity.json'
    if not output.exists():return False
    existing=read(output)
    assert existing['binding_sha256']==binding_digest and existing['finalizer_sha256']==code_sha
    assert existing['state']=='complete_all_evaluable_frozen_model_comparison'
    assert existing['preprocessing_supplement']['manifest_sha256']==sha(SUPPLEMENT)
    assert existing['preprocessing_supplement']['dataset']==DATASET
    assert existing['decisions_sha256']==sha(P/'decisions_antigravity.json')
    assert existing['label_boundary_sha256']==sha(P/'labels_opened_antigravity.json')
    if manifest['state']==existing['state']:
        assert manifest['result_sha256']==sha(output)
        return True
    return False

def publish(result,recovery,code_sha):
    """The authoritative result is never an unannotated intermediate file."""
    result['preprocessing_supplement']=dict(dataset=DATASET,constant_train_rows=1,
        rule=recovery['binding']['rule'],manifest_sha256=sha(SUPPLEMENT),
        historical_constant_dataset_replay_not_claimed=True)
    result['finalizer_sha256']=code_sha
    output=P/'results_antigravity.json'
    if output.exists():assert read(output)==jsonable(result)
    else:save(output,result)

def main():
    start=perf_counter();rows,reps,v24,old,binding=context();digest=jsha(binding)
    recovery=read(SUPPLEMENT);assert recovery['state']=='scored'
    assert recovery['binding']==supplement_binding(digest) and recovery['binding_sha256']==jsha(recovery['binding'])
    assert read(P/'failures_antigravity.json')==[dict(dataset=DATASET,type='ValueError',error='finite non-constant rows are required')]
    assert read(P/'progress_antigravity.json')['completed']==1116
    meta,_=load(P/'score_checkpoints_antigravity',DATASET,digest)
    assert meta['recovery_binding_sha256']==recovery['binding_sha256'] and meta['score_sha256']==recovery['score_sha256']
    code_sha=sha(Path(__file__));manifest=read(P/'run_manifest_antigravity.json')
    if validate_resume(manifest,digest,code_sha):
        print('Already complete; binding and publication hashes verified',flush=True);return
    manifest['preprocessing_supplement']=dict(**recovery,manifest_sha256=sha(SUPPLEMENT))
    manifest['finalizer_sha256']=code_sha
    manifest['state']='freezing_routes_after_documented_constant_recovery';save(P/'run_manifest_antigravity.json',manifest)
    checks,decisions=freeze_routes(rows,reps,v24,P/'score_checkpoints_antigravity',digest)
    stage=P/'evaluation_staging_antigravity';stage.mkdir(exist_ok=True)
    for name in ('decisions_antigravity.json','labels_opened_antigravity.json'):(stage/name).write_bytes((P/name).read_bytes())
    # Frozen evaluator writes its intermediate output only inside the staging directory.
    frozen_run.P=stage
    try:result=frozen_run.evaluate(rows,checks,decisions,digest)
    finally:frozen_run.P=P
    publish(result,recovery,code_sha)
    manifest.update(state=result['state'],result_sha256=sha(P/'results_antigravity.json'),
        seconds=read(P/'progress_antigravity.json')['elapsed_seconds']+perf_counter()-start,
        finalization_seconds=perf_counter()-start)
    save(P/'run_manifest_antigravity.json',manifest)
    print(json.dumps(dict(state=manifest['state'],datasets=len(rows),recovered=DATASET)),flush=True)

if __name__=='__main__':main()
