"""One TRAIN-selected candidate, all1117 datasets, scores frozen before labels."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from time import perf_counter
import json,multiprocessing
import numpy as np
from threadpoolctl import threadpool_limits
from .pilot_antigravity import P,ROOT,DB,SEED,readonly_sqlite_antigravity,write_json_atomic_antigravity,cpu_worker_policy_antigravity,file_sha256_antigravity
from .amplitude_antigravity import AmplitudeHeads
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import _reference_indices_antigravity
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import prepare_dataset_values_antigravity
from experiments.exp479_missing_score_recovery_antigravity.evaluation_antigravity import load_retained_test_labels_antigravity
from experiments.exp494_all_datasets_low_fpr_antigravity.constant_recovery_antigravity import prepare as prepare_constant
from experiments.exp494_all_datasets_low_fpr_antigravity.evaluation_antigravity import metric,summarize
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import sha256_json_antigravity

BASE=ROOT/'experiments/exp494_all_datasets_low_fpr_antigravity'
OUT=P/'full_scores_antigravity'
read=lambda p:json.loads(p.read_text())

def baseline_replay(actual,expected):
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-8)
    assert np.array_equal(np.argsort(actual,kind='stable'),np.argsort(expected,kind='stable'))
    return float(np.max(abs(actual-expected)))

def checked(name,binding):
    path=OUT/f'{name}_meta_antigravity.json'
    if not path.exists():return None
    m=read(path);score=OUT/f'{name}_scores_antigravity.npz'
    assert m['binding_sha256']==binding and m['dataset']==name and file_sha256_antigravity(score)==m['score_sha256']
    source=read(BASE/'score_checkpoints_antigravity'/f'{name}_meta_antigravity.json')
    assert m['test_indices']==source['test_indices'] and m['n_train']==source['n_train'] and m['n_test']==source['n_test']
    assert m['source_meta_sha256']==file_sha256_antigravity(BASE/'score_checkpoints_antigravity'/f'{name}_meta_antigravity.json')
    assert m['source_family']==source['source_family']
    source_score=BASE/'score_checkpoints_antigravity'/f'{name}_scores_antigravity.npz'
    assert file_sha256_antigravity(source_score)==source['score_sha256']
    with np.load(score) as f,np.load(source_score) as old:
        assert set(f.files)=={'baseline','candidate'}
        assert all(f[k].shape==(m['n_test'],) and np.isfinite(f[k]).all() for k in f.files)
        baseline_replay(f['baseline'],old['test__0'])
    return m

def worker(payload):
    row,binding=payload;name=row['dataset'];found=checked(name,binding)
    if found:return found
    start=perf_counter();source_path=BASE/'score_checkpoints_antigravity'/f'{name}_meta_antigravity.json';source=read(source_path)
    old_scores=BASE/'score_checkpoints_antigravity'/f'{name}_scores_antigravity.npz'
    assert file_sha256_antigravity(old_scores)==source['score_sha256']
    with closing(readonly_sqlite_antigravity(DB)) as con,threadpool_limits(limits=1):
        prepared=prepare_constant(con) if name=='UWaveGestureLibraryZ_normal_8' else prepare_dataset_values_antigravity(con,name)
        train=prepared.train_rows.astype(np.float32);test=prepared.test_rows.astype(np.float32)
        assert list(prepared.test_indices)==source['test_indices']
        refs=_reference_indices_antigravity(np.arange(len(train)),seed=SEED+7919+4096)
        head=AmplitudeHeads(train[refs])
        baseline=[];candidate=[]
        for start_query in range(0,len(test),128):
            values=head.score(test[start_query:start_query+128])
            baseline.extend(values['MiniRocket']);candidate.extend(values['square_joint'])
        baseline=np.asarray(baseline);candidate=np.asarray(candidate)
    assert np.isfinite(candidate).all()
    with np.load(old_scores) as f:
        replay_error=baseline_replay(baseline,f['test__0'])
    OUT.mkdir(exist_ok=True);score=OUT/f'{name}_scores_antigravity.npz';tmp=OUT/f'{name}_temporary_antigravity.npz'
    np.savez_compressed(tmp,baseline=baseline,candidate=candidate);tmp.replace(score)
    m=dict(dataset=name,source_family=row['source_family'],binding_sha256=binding,score_sha256=file_sha256_antigravity(score),
        source_meta_sha256=file_sha256_antigravity(source_path),n_train=len(train),n_test=len(test),test_indices=list(prepared.test_indices),
        baseline_max_abs_error=replay_error,baseline_ranking_exact=True,seconds=perf_counter()-start)
    write_json_atomic_antigravity(OUT/f'{name}_meta_antigravity.json',m);return m

def main():
    start=perf_counter();pilot=read(P/'pilot_results_antigravity.json');pilot_manifest=read(P/'pilot_manifest_antigravity.json')
    assert pilot['selected']=='square_joint' and file_sha256_antigravity(P/'pilot_results_antigravity.json')==pilot_manifest['result_sha256']
    for relative,digest in pilot_manifest['code'].items():assert file_sha256_antigravity(ROOT/relative)==digest
    assert file_sha256_antigravity(P/'protocol_antigravity.html')==pilot_manifest['protocol_sha256']
    base=read(BASE/'run_manifest_antigravity.json');assert file_sha256_antigravity(DB)==base['binding']['source_binding']['database']
    assert file_sha256_antigravity(BASE/'results_antigravity.json')==base['result_sha256']
    cm=read(ROOT/'docs/codemap/codemap_exp502_antigravity.json')
    assert {str(Path(__file__).relative_to(ROOT)),'experiments/exp494_all_datasets_low_fpr_antigravity/evaluation_antigravity.py'}<={n['path'] for n in cm['nodes']}
    for node in cm['nodes']:assert file_sha256_antigravity(ROOT/node['path'])==node['sha256']
    binding=dict(source494_result_sha256=base['result_sha256'],database_sha256=base['binding']['source_binding']['database'],
        selection_sha256=file_sha256_antigravity(P/'pilot_results_antigravity.json'),protocol_sha256=file_sha256_antigravity(P/'full_protocol_antigravity.html'),
        acceleration_sha256=file_sha256_antigravity(P/'acceleration_results_antigravity.json'),code={n['path']:n['sha256'] for n in cm['nodes']},selected='square_joint')
    digest=sha256_json_antigravity(binding);path=P/'full_manifest_antigravity.json';result_path=P/'full_results_antigravity.json'
    if path.exists():assert read(path)['binding_sha256']==digest
    if result_path.exists():raise RuntimeError('full results already exist; verify rather than overwrite')
    policy=cpu_worker_policy_antigravity();manifest=dict(state='scoring',binding=binding,binding_sha256=digest,cpu_policy=policy)
    write_json_atomic_antigravity(path,manifest);rows=base['binding']['population'];assert len(rows)==1117
    pending=[r for r in rows if checked(r['dataset'],digest) is None];completed=len(rows)-len(pending);failures=[]
    with ProcessPoolExecutor(max_workers=policy['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker,(r,digest)):r['dataset'] for r in pending}
        for f in as_completed(futures):
            try:f.result();completed+=1
            except Exception as e:failures.append(dict(dataset=futures[f],error=str(e),type=type(e).__name__))
            progress=dict(completed=completed,total=len(rows),failures=failures,seconds=perf_counter()-start)
            write_json_atomic_antigravity(P/'full_progress_antigravity.json',progress)
            if completed%25==0 or failures:print(json.dumps(progress),flush=True)
    if failures:raise RuntimeError('scoring failures; no labels opened')
    metas={r['dataset']:checked(r['dataset'],digest) for r in rows};assert all(metas.values())
    boundary=dict(binding_sha256=digest,scores={d:m['score_sha256'] for d,m in metas.items()},metadata={d:sha256_json_antigravity(m) for d,m in metas.items()})
    boundary_path=P/'full_labels_opened_antigravity.json'
    if boundary_path.exists():assert read(boundary_path)==boundary
    else:write_json_atomic_antigravity(boundary_path,boundary)
    manifest['state']='evaluating';write_json_atomic_antigravity(path,manifest);records=[]
    with closing(readonly_sqlite_antigravity(DB)) as con:
        for row in rows:
            d=row['dataset'];m=metas[d];y=load_retained_test_labels_antigravity(con,d,tuple(m['test_indices']))
            with np.load(OUT/f'{d}_scores_antigravity.npz') as f:values={k:metric(y,f[k]) for k in ('baseline','candidate')}
            records.append(dict(dataset=d,source_family=row['source_family'],metrics=values))
    summaries={k:summarize([dict(source_family=r['source_family'],metrics=r['metrics'][k]) for r in records]) for k in ('baseline','candidate')}
    old=read(BASE/'results_antigravity.json')['summaries']['all_evaluable']['MiniRocket']
    for key in ('family_macro_tpr','dataset_macro_tpr','micro_tpr'):np.testing.assert_allclose(summaries['baseline'][key],old[key],rtol=0,atol=1e-12)
    assert summaries['baseline']['total_zf']==old['total_zf']
    b=np.asarray(summaries['baseline']['family_macro_tpr']);c=np.asarray(summaries['candidate']['family_macro_tpr']);gain=c/b-1
    groups=sorted({r['source_family'] for r in records});paired=np.asarray([[np.mean([r['metrics'][k]['tpr'][1] for r in records if r['source_family']==g]) for k in ('baseline','candidate')] for g in groups])
    draws=np.random.default_rng(SEED).integers(0,len(groups),size=(2000,len(groups)));boot=paired[draws].mean(axis=1)
    result=dict(state='complete_frozen_full_comparison',selected='square_joint',binding_sha256=digest,raw_datasets=1119,evaluated_datasets=1117,families=128,
        summaries=summaries,relative_gain=gain.tolist(),primary_relative_gain=float(gain[1]),primary_gain_family_bootstrap95=np.quantile(boot[:,1]/boot[:,0]-1,[.025,.975]).tolist(),
        observed_target_met=bool(gain[1]>=.1 and c[0]>=b[0]),records=records,boundary_sha256=file_sha256_antigravity(boundary_path),
        interpretation='Previously exposed TEST development benchmark; TRAIN-only candidate selection; independent verification still required',seconds=perf_counter()-start)
    write_json_atomic_antigravity(result_path,result);manifest.update(state=result['state'],result_sha256=file_sha256_antigravity(result_path),seconds=perf_counter()-start)
    write_json_atomic_antigravity(path,manifest);print(json.dumps({k:result[k] for k in ('state','primary_relative_gain','observed_target_met','seconds')}),flush=True)

if __name__=='__main__':main()
