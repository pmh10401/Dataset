"""One immutable Exp511 scoring pass; this module never loads TEST labels."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from hashlib import file_digest,sha256
from datetime import datetime,timezone
from contextlib import closing,contextmanager
from concurrent.futures import ProcessPoolExecutor,as_completed
from time import perf_counter
import multiprocessing
import json
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import ROOT,DB,SEED,readonly_sqlite_antigravity,write_json_atomic_antigravity,cpu_worker_policy_antigravity
from experiments.exp511_crossfit_rank_ensemble_antigravity.ensemble_antigravity import CrossfitRankHeads,NAMES
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import _reference_indices_antigravity
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import prepare_dataset_values_antigravity
from experiments.exp494_all_datasets_low_fpr_antigravity.constant_recovery_antigravity import prepare as prepare_constant
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import assert_unbound_wal_antigravity

P=Path(__file__).resolve().parent
OUT=P/'scores_antigravity'
BASE=ROOT/'experiments/exp494_all_datasets_low_fpr_antigravity'
EXP502=ROOT/'experiments/exp502_amplitude_envelope_antigravity'
PRIOR=ROOT/'experiments/exp511_crossfit_rank_ensemble_antigravity'
read=lambda p:json.loads(p.read_text())
canonical=lambda value:sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
now=lambda:datetime.now(timezone.utc).isoformat()

def digest(path):
    with path.open('rb') as stream:return file_digest(stream,'sha256').hexdigest()

def check_hashes(expected,root=ROOT):
    for relative,value in expected.items():
        if digest(root/relative)!=value:raise ValueError('Changed frozen artifact: '+relative)

@contextmanager
def database_snapshot(database=DB):
    assert_unbound_wal_antigravity(database)
    with closing(readonly_sqlite_antigravity(database)) as con:
        try:
            con.execute('BEGIN')
            yield con
        finally:assert_unbound_wal_antigravity(database)

def assert_scoring_allowed(directory=P):
    if (directory/'labels_opened_antigravity.json').exists() or (directory/'results_antigravity.json').exists():
        raise RuntimeError('TEST labels already accessed: rescoring is forbidden')
    manifest=directory/'manifest_antigravity.json'
    if manifest.exists():
        state=read(manifest)
        if state.get('test_labels_read') or state.get('state') in ('evaluating','complete_exploratory_test'):
            raise RuntimeError('TEST evaluation already started: rescoring is forbidden')

def replay(actual,expected):
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-8)
    np.testing.assert_array_equal(np.argsort(actual,kind='stable'),np.argsort(expected,kind='stable'))
    return float(np.max(np.abs(actual-expected)))

def score_rows(train,test,batch_size=128):
    if type(batch_size) is not int or batch_size<1:raise ValueError('positive batch size required')
    refs=_reference_indices_antigravity(np.arange(len(train)),seed=SEED+7919+4096)
    head=CrossfitRankHeads(train[refs]);parts={}
    for start in range(0,len(test),batch_size):
        values,components=head.score_components(test[start:start+batch_size])
        for key,val in {**values,**components}.items():parts.setdefault(key,[]).append(val)
    arrays={key:np.concatenate(values) for key,values in parts.items()}
    members=[]
    for j,member in enumerate(head.members):
        arrays[f'member{j}__weights']=member.weights.copy()
        members.append(dict(member=j,a=refs[member.a_indices].tolist(),b=refs[member.b_indices].tolist(),
                            parents=refs[member.parents].tolist(),diagnostics=member.diagnostics))
    info=dict(n_train=len(train),n_test=len(test),reference_indices=refs.tolist(),fallback=head.fallback,members=members)
    validate_arrays(arrays,info)
    return arrays,info

def validate_arrays(arrays,meta):
    expected=set(NAMES)
    if not meta['fallback']:
        expected|={f'member{j}__{kind}' for j in range(4) for kind in ('rank','unweighted','weights')}
    assert set(arrays)==expected
    for key,value in arrays.items():
        shape=(8064,) if key.endswith('__weights') else (meta['n_test'],)
        assert value.shape==shape and np.isfinite(value).all(),key
        if key.endswith('__weights'):assert np.all((value>=.25)&(value<=4.))
    refs=np.asarray(meta['reference_indices']);assert len(refs)==min(meta['n_train'],1000)
    assert meta['fallback']==(len(refs)<32)
    if meta['fallback']:
        assert not meta['members']
        for key in ('crossfit_unweighted','crossfit_rank','a_unweighted','separated_rank'):
            np.testing.assert_array_equal(arrays[key],arrays['square_joint'])
    else:
        assert len(meta['members'])==4
        for kind,key in (('rank','crossfit_rank'),('unweighted','crossfit_unweighted')):
            np.testing.assert_allclose(arrays[key],np.mean([arrays[f'member{j}__{kind}'] for j in range(4)],axis=0),rtol=1e-14,atol=1e-12)

def checked(name,identity):
    meta_path=OUT/f'{name}_meta_antigravity.json'
    if not meta_path.exists():return None
    m=read(meta_path);score=OUT/f'{name}_scores_antigravity.npz'
    assert m['dataset']==name and m['binding_sha256']==identity and not m['test_labels_read']
    assert digest(score)==m['score_sha256']
    for folder,key in ((BASE/'score_checkpoints_antigravity','source494'),(EXP502/'full_scores_antigravity','source502')):
        path=folder/f'{name}_meta_antigravity.json';source=read(path)
        assert digest(path)==m[key+'_meta_sha256']
        assert all(m[k]==source[k] for k in ('test_indices','n_train','n_test','source_family'))
        assert digest(folder/f'{name}_scores_antigravity.npz')==source['score_sha256']
    with np.load(score,allow_pickle=False) as current,np.load(EXP502/'full_scores_antigravity'/f'{name}_scores_antigravity.npz',allow_pickle=False) as old:
        validate_arrays({key:current[key] for key in current.files},m)
        replay(current['MiniRocket'],old['baseline']);replay(current['square_joint'],old['candidate'])
    return m

def worker(payload):
    row,identity=payload;name=row['dataset'];previous=checked(name,identity)
    if previous is not None:return previous
    started=perf_counter()
    path494=BASE/'score_checkpoints_antigravity'/f'{name}_meta_antigravity.json'
    path502=EXP502/'full_scores_antigravity'/f'{name}_meta_antigravity.json'
    source=read(path494);source502=read(path502)
    assert all(source[k]==source502[k] for k in ('test_indices','n_train','n_test','source_family'))
    oldfile=EXP502/'full_scores_antigravity'/f'{name}_scores_antigravity.npz'
    assert digest(oldfile)==source502['score_sha256']
    with database_snapshot(DB) as con,threadpool_limits(limits=1):
        prepared=prepare_constant(con) if name=='UWaveGestureLibraryZ_normal_8' else prepare_dataset_values_antigravity(con,name)
        assert list(prepared.test_indices)==source['test_indices']
        arrays,info=score_rows(prepared.train_rows.astype(np.float32),prepared.test_rows.astype(np.float32))
    with np.load(oldfile,allow_pickle=False) as old:
        errors=dict(MiniRocket=replay(arrays['MiniRocket'],old['baseline']),square_joint=replay(arrays['square_joint'],old['candidate']))
    score=OUT/f'{name}_scores_antigravity.npz';temporary=OUT/f'{name}_temporary_antigravity.npz'
    np.savez_compressed(temporary,**arrays);temporary.replace(score)
    m=dict(dataset=name,source_family=row['source_family'],binding_sha256=identity,**info,
           test_indices=list(prepared.test_indices),target_length=prepared.target_length,
           raw_train_rows=prepared.raw_train_rows,excluded_train_rows=prepared.excluded_train_rows,
           raw_test_rows=prepared.raw_test_rows,excluded_test_rows=prepared.excluded_test_rows,
           source494_meta_sha256=digest(path494),source502_meta_sha256=digest(path502),score_sha256=digest(score),
           replay_max_abs_error=errors,baseline_rankings_exact=True,test_values_read=True,test_labels_read=False,
           seconds=perf_counter()-started)
    write_json_atomic_antigravity(OUT/f'{name}_meta_antigravity.json',m)
    return m

def main():
    assert_scoring_allowed(P)
    assert_unbound_wal_antigravity(DB)
    manifest_path=P/'manifest_antigravity.json';boundary_path=P/'score_freeze_antigravity.json'
    if boundary_path.exists():raise RuntimeError('Completed score freeze exists; verify without re-scoring')
    preflight=read(P/'preflight_antigravity.json');assert preflight['state']=='passed'
    check_hashes(preflight['artifacts'])
    auth=read(P/'authorization_antigravity.json')
    assert auth['state']=='authorized_exploratory_test_after_train_rejection' and auth['candidate']=='crossfit_rank'
    check_hashes(auth['historical_artifacts'])
    base=read(BASE/'run_manifest_antigravity.json');rows=base['binding']['population']
    assert len(rows)==len({r['dataset'] for r in rows})==1117 and len({r['source_family'] for r in rows})==128
    dbhash=base['binding']['source_binding']['database'];assert digest(DB)==dbhash
    cm=read(ROOT/'docs/codemap/codemap_exp512_antigravity.json');code={n['path']:n['sha256'] for n in cm['nodes']};check_hashes(code)
    binding=dict(code=code,population=rows,database_sha256=dbhash,preflight_sha256=digest(P/'preflight_antigravity.json'),
                 protocol_sha256=digest(P/'protocol_antigravity.html'),authorization_sha256=digest(P/'authorization_antigravity.json'),
                 source494_result_sha256=digest(BASE/'results_antigravity.json'),source502_result_sha256=digest(EXP502/'full_results_antigravity.json'),
                 candidate='crossfit_rank',train_gate_passed=False,evaluation_kind='user_authorized_exploratory_test')
    identity=canonical(binding)
    if manifest_path.exists():assert read(manifest_path)['binding_sha256']==identity
    policy=cpu_worker_policy_antigravity();OUT.mkdir(exist_ok=True);started=perf_counter()
    manifest=dict(state='scoring',binding=binding,binding_sha256=identity,cpu_policy=policy,started_utc=now(),test_labels_read=False)
    write_json_atomic_antigravity(manifest_path,manifest)
    pending=[r for r in rows if checked(r['dataset'],identity) is None];completed=len(rows)-len(pending);failures=[]
    with ProcessPoolExecutor(max_workers=policy['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker,(row,identity)):row['dataset'] for row in pending}
        for future in as_completed(futures):
            try:future.result();completed+=1
            except Exception as error:failures.append(dict(dataset=futures[future],type=type(error).__name__,error=str(error)))
            progress=dict(completed=completed,total=1117,failures=failures,seconds=perf_counter()-started)
            write_json_atomic_antigravity(P/'progress_antigravity.json',progress)
            if completed%25==0 or failures:print(json.dumps(progress),flush=True)
    if failures:
        manifest.update(state='failed_scoring',failures=failures)
        write_json_atomic_antigravity(manifest_path,manifest)
        raise RuntimeError('Scoring failed; TEST labels remain unopened, no exclusions or replacements')
    check_hashes({**code,**preflight['artifacts']});assert_unbound_wal_antigravity(DB);assert digest(DB)==dbhash
    metas={row['dataset']:checked(row['dataset'],identity) for row in rows};assert all(metas.values())
    boundary=dict(binding_sha256=identity,utc=now(),datasets=1117,test_labels_read=False,
                  scores={name:meta['score_sha256'] for name,meta in metas.items()},
                  metadata={name:digest(OUT/f'{name}_meta_antigravity.json') for name in metas})
    write_json_atomic_antigravity(boundary_path,boundary)
    manifest.update(state='scores_frozen',boundary_sha256=digest(boundary_path),scoring_seconds=perf_counter()-started)
    write_json_atomic_antigravity(manifest_path,manifest)
    print(json.dumps(dict(state='scores_frozen',datasets=1117,test_labels_read=False)),flush=True)

if __name__=='__main__':main()
