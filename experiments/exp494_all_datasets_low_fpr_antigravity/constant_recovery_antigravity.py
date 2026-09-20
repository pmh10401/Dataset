"""Explicit pre-label supplement for the one exact-constant TRAIN row."""
from .run_antigravity import *
from . import scoring_antigravity as scoring
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import (
    _VALUE_SQL_ANTIGRAVITY,_decode_finite_row_antigravity,_standardize_rows_antigravity,
    resample_row_linear_antigravity,lower_median_length_antigravity,
)
from experiments.exp479_missing_score_recovery_antigravity.contracts_antigravity import PreparedDatasetAntigravity

DATASET='UWaveGestureLibraryZ_normal_8'
SUPPLEMENT=P/'constant_recovery_manifest_antigravity.json'

def normalize(rows,target_length):
    prepared=np.asarray([resample_row_linear_antigravity(r,target_length) for r in rows])
    mean=prepared.mean(axis=1,keepdims=True);scale=prepared.std(axis=1,keepdims=True)
    if not np.isfinite(prepared).all() or not np.isfinite(scale).all():raise ValueError('finite input required')
    result=(prepared-mean)/np.where(scale==0.,1.,scale)
    if not np.isfinite(result).all():raise ValueError('finite normalized output required')
    return result

def prepare(connection):
    train=[];test=[];indices=[];raw={'TRAIN':0,'TEST':0};excluded={'TRAIN':0,'TEST':0};constants=[]
    for split,index,blob in connection.execute(_VALUE_SQL_ANTIGRAVITY,(DATASET,)):
        raw[split]+=1;r=_decode_finite_row_antigravity(blob)
        if r is None:excluded[split]+=1;continue
        if np.std(r)==0:constants.append((split,index))
        (train if split=='TRAIN' else test).append(r)
        if split=='TEST':indices.append(index)
    assert constants==[('TRAIN',85)],constants
    length=lower_median_length_antigravity(tuple(len(r) for r in train))
    x=normalize(train,length);z=normalize(test,length)
    keep=np.array([np.std(r)!=0 for r in train])
    np.testing.assert_array_equal(x[keep],_standardize_rows_antigravity([r for r,k in zip(train,keep) if k],length))
    np.testing.assert_array_equal(z,_standardize_rows_antigravity(test,length))
    np.testing.assert_array_equal(x[~keep],np.zeros_like(x[~keep]))
    return PreparedDatasetAntigravity(DATASET,x,z,tuple(indices),length,raw['TRAIN'],excluded['TRAIN'],raw['TEST'],excluded['TEST'])

def supplement_binding(original):
    paths=[Path(__file__),P/'constant_recovery_protocol_antigravity.html',ROOT/'scratch/test_exp494_constant_recovery_antigravity.py']
    return dict(original_binding_sha256=original,files={str(p.relative_to(ROOT)):sha(p) for p in paths},
        dataset=DATASET,rule='Only exact scale==0 uses divisor1; all other Exp479 arithmetic and rows unchanged',
        acceleration='NumPy float64 normalization retained; existing Numba resampling reused. Added one scalar guard is not performance-critical; no new encoder or GPU stage. No additional JIT/device conversion warranted.')

def main_recovery():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['score','finalize'],required=True);a=ap.parse_args()
    start=perf_counter();rows,reps,v24,old,binding=context();digest=jsha(binding);supp=supplement_binding(digest)
    directory=P/'score_checkpoints_antigravity'
    assert not (P/'labels_opened_antigravity.json').exists() and not (P/'results_antigravity.json').exists()
    if a.mode=='score':
        assert not (directory/f'{DATASET}_meta_antigravity.json').exists()
        existing={p:sha(p) for p in directory.iterdir() if p.suffix in ('.json','.npz') and 'temporary' not in p.name}
        save(SUPPLEMENT,dict(state='scoring',binding=supp,binding_sha256=jsha(supp),expanded_labels_opened=False))
        with closing(readonly_sqlite_antigravity(DB)) as con,threadpool_limits(limits=1):
            prepared=prepare(con);arrays=score_and_describe(prepared.train_rows,prepared.test_rows)
        for p,h in existing.items():assert sha(p)==h
        row=next(r for r in rows if r.dataset==DATASET)
        meta=scoring.persist(directory,DATASET,arrays,dict(binding_sha256=digest,recovery_binding_sha256=jsha(supp),
            source_family=row.source_family,n_train=len(prepared.train_rows),n_test=len(prepared.test_rows),
            target_length=prepared.target_length,test_indices=list(prepared.test_indices),
            origin='explicit_exact_constant_TRAIN_supplement',seconds=perf_counter()-start))
        save(SUPPLEMENT,dict(state='scored',binding=supp,binding_sha256=jsha(supp),expanded_labels_opened=False,
            score_sha256=meta['score_sha256'],nonconstant_rows_bitwise_unchanged=True,constant_train_rows=1,
            existing_files_unchanged=len(existing),seconds=perf_counter()-start))
        print(json.dumps(read(SUPPLEMENT)),flush=True);return
    recovery=read(SUPPLEMENT);assert recovery['state']=='scored' and recovery['binding']==supp
    failures=read(P/'failures_antigravity.json')
    assert failures==[dict(dataset=DATASET,type='ValueError',error='finite non-constant rows are required')]
    assert read(P/'progress_antigravity.json')['completed']==1116
    meta,_=load(directory,DATASET,digest)
    assert meta['recovery_binding_sha256']==jsha(supp) and meta['score_sha256']==recovery['score_sha256']
    manifest=read(P/'run_manifest_antigravity.json');manifest['preprocessing_supplement']=dict(**recovery,manifest_sha256=sha(SUPPLEMENT))
    manifest['state']='freezing_routes_after_documented_constant_recovery';save(P/'run_manifest_antigravity.json',manifest)
    checks,decisions=freeze_routes(rows,reps,v24,directory,digest)
    result=evaluate(rows,checks,decisions,digest)
    result['preprocessing_supplement']=dict(dataset=DATASET,constant_train_rows=1,rule=supp['rule'],manifest_sha256=sha(SUPPLEMENT),historical_constant_dataset_replay_not_claimed=True)
    save(P/'results_antigravity.json',result)
    manifest.update(state=result['state'],result_sha256=sha(P/'results_antigravity.json'),
        seconds=read(P/'progress_antigravity.json')['elapsed_seconds']+perf_counter()-start,
        finalization_seconds=perf_counter()-start)
    save(P/'run_manifest_antigravity.json',manifest)
    print(json.dumps(dict(state=manifest['state'],datasets=len(rows),recovered=DATASET)),flush=True)

if __name__=='__main__':main_recovery()
