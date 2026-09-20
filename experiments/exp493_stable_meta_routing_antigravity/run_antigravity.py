"""Additive Exp493: compatible score ledgers, frozen LOFO routing, paired evaluation."""
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import (
    configure_inner_threads_antigravity, DATABASE_PATH_ANTIGRAVITY as DB, SEED_ANTIGRAVITY as SEED,
    ROOT_ANTIGRAVITY as ROOT, EXP488_CACHE_PATH_ANTIGRAVITY as CACHE24,
    load_frozen_sources_antigravity, representative_rows_antigravity,
    EXCLUDED_FIVEFOLD_FAMILIES_ANTIGRAVITY, validate_database_antigravity,
    readonly_sqlite_antigravity, file_sha256_antigravity as sha, sha256_json_antigravity as jsha,
    write_json_atomic_antigravity as write, jsonable_antigravity as jsonable,
    assess_acceleration_antigravity, _probe_subprocess_antigravity,
)
configure_inner_threads_antigravity()
import argparse, json, html, multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import METHODS_ANTIGRAVITY as METHODS
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import prepare_dataset_values_antigravity
from experiments.exp479_missing_score_recovery_antigravity.evaluation_antigravity import load_retained_test_labels_antigravity, evaluate_score_vector_antigravity
from experiments.exp489_lof_score_head_antigravity.run_antigravity import load_score_checkpoint_antigravity
from experiments.exp490_behavior_descriptors_antigravity.behavior_descriptors_antigravity import prepare_normal_train_matrix_antigravity
from experiments.exp490_behavior_descriptors_antigravity.run_antigravity import _load_24d_antigravity, validate_behavior_cache_antigravity
from experiments.exp491_gain_prediction_antigravity.gain_prediction_antigravity import target_vector_antigravity
from experiments.exp492_stable_normalization_antigravity.run_antigravity import load_antigravity, validate_alignment_antigravity
from .behavior_antigravity import extract_behavior_antigravity, score_multirocket_test_antigravity
from .meta_antigravity import predict_antigravity, routes_antigravity, summarize_route_antigravity
P=Path(__file__).resolve().parent
P489=ROOT/'experiments/exp489_lof_score_head_antigravity'
P490=ROOT/'experiments/exp490_behavior_descriptors_antigravity'
P492=ROOT/'experiments/exp492_stable_normalization_antigravity'

def read(path): return json.loads(path.read_text())
def save(path,value): write(path,jsonable(value))

def context():
    rows,_,sources=load_frozen_sources_antigravity()
    reps,excluded=representative_rows_antigravity(rows,expected_families=125,expected_excluded=EXCLUDED_FIVEFOLD_FAMILIES_ANTIGRAVITY)
    validate_database_antigravity(DB,frozen=True)
    m492=read(P492/'run_manifest_antigravity.json'); r489=read(P489/'results_antigravity.json')
    b492=m492['binding']; b489=r489['binding']
    assert sha(P492/'results_antigravity.json')==m492['result_sha256']
    assert b489['sources']['raw_database']==b492['database_sha256']==sha(DB)
    assert b489['seed']==b492['seed']==SEED
    for name,v in b489['versions'].items():
        if name in b492['versions']: assert v==b492['versions'][name],name
    for binding in (b489,b492):
        for path,digest in binding['code_sha256'].items():
            assert sha(ROOT/path)==digest,path
    old=read(P490/'descriptor_cache_antigravity.json')
    assert old['binding']['sources']['raw_database']==b492['database_sha256']
    assert b489['versions']['python']==b492['python']
    for name,value in b492['versions'].items():
        from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import package_version_antigravity
        assert package_version_antigravity(name)==value, name
    for path,digest in old['binding']['code_sha256'].items():
        assert sha(ROOT/path)==digest,path
    assert jsha(old['records'])==old['records_sha256']
    assert sha(P490/'descriptor_cache_antigravity.json')==read(P490/'results_antigravity.json')['descriptor_cache_sha256']
    validate_behavior_cache_antigravity(old['records'],reps,complete=True)
    values24=_load_24d_antigravity(CACHE24,rows,DB,sources)
    cm=read(ROOT/'docs/codemap/codemap_exp493_antigravity.json')
    code={node['path']:sha(ROOT/node['path']) for node in cm['nodes']}
    for node in cm['nodes']: assert code[node['path']]==node['sha256'],'stale map'
    frozen=[P489/'results_antigravity.json',P490/'results_antigravity.json',P490/'descriptor_cache_antigravity.json',P492/'results_antigravity.json',P492/'run_manifest_antigravity.json',CACHE24]
    upstream={}
    for package in (P489,P492):
        scores={r['dataset']:r['score_sha256'] for r in read(package/'results_antigravity.json')['records']}
        for meta in (package/'score_checkpoints_antigravity').glob('*meta_antigravity.json'):
            data=read(meta)
            assert scores[data['dataset']]==data['score_sha256'], 'checkpoint differs from frozen results'
            upstream[str(meta)]=sha(meta)
        assert len(scores)==125
    assert len(upstream)==250
    binding=dict(upstream_meta_sha256=upstream,code=code,inputs={str(p):sha(p) for p in frozen},database=b492['database_sha256'],population=[r.dataset for r in reps],seed=SEED)
    return rows,reps,values24,old['records'],b489,b492,binding

def cached(directory,dataset,binding):
    meta=directory/f'{dataset}_meta_antigravity.json'; npz=directory/f'{dataset}_scores_antigravity.npz'
    if not meta.exists():return None
    data=read(meta)
    if data['binding']!=binding or data['dataset']!=dataset or sha(npz)!=data['score_sha256']:raise ValueError('Exp493 checkpoint mismatch')
    with np.load(npz,allow_pickle=False) as f: arrays={k:f[k].copy() for k in f.files}
    if not all(a.ndim==1 and len(a) and np.isfinite(a).all() for a in arrays.values()): raise ValueError('invalid score arrays')
    if arrays['multirocket_test'].shape!=(data['n_test'],): raise ValueError('TEST shape mismatch')
    source=P492/'score_checkpoints_antigravity'/f'{dataset}_meta_antigravity.json'
    if sha(source)!=binding['upstream_meta_sha256'][str(source)]: raise ValueError('upstream metadata changed')
    expected=read(source)
    for key in ('source_family','n_train','n_test','target_length','test_indices'):
        if data[key]!=expected[key]:raise ValueError('checkpoint alignment mismatch')
    if len(data['values'])!=18 or not np.isfinite(data['values']).all(): raise ValueError('invalid descriptor')
    return data,arrays

def worker(payload):
    configure_inner_threads_antigravity();torch.set_num_threads(1)
    dataset, family, directory, binding,b489,b492,old=payload
    directory=Path(directory); found=cached(directory,dataset,binding)
    if found:return found[0]
    started=perf_counter()
    mini=load_score_checkpoint_antigravity(P489/'score_checkpoints_antigravity',dataset,b489)
    meta,a492=load_antigravity(P492/'score_checkpoints_antigravity',dataset,b492)
    with closing(readonly_sqlite_antigravity(DB)) as con,threadpool_limits(limits=1):
        prepared=prepare_dataset_values_antigravity(con,dataset)
        validate_alignment_antigravity(meta,prepared);validate_alignment_antigravity(mini['meta'],prepared)
        train,identity=prepare_normal_train_matrix_antigravity(DB,dataset)
        np.testing.assert_array_equal(train,prepared.train_rows)
        for key,value in identity.items():
            if old[key]!=value:raise ValueError('TRAIN membership mismatch')
        if meta['source_family']!=family or mini['meta']['source_family']!=family:raise ValueError('family mismatch')
        aold={METHODS[0]:mini['heads']['control']['train_oof'],**{m:a492[f'{m}__old__train_oof'] for m in METHODS[2:]}}
        astable={METHODS[0]:aold[METHODS[0]],**{m:a492[f'{m}__stable__train_oof'] for m in METHODS[2:]}}
        output=extract_behavior_antigravity(train,aold,astable,old['values'],seed=SEED)
        arrays={**output['arrays'],'multirocket_test':score_multirocket_test_antigravity(train,prepared.test_rows,seed=SEED)}
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{dataset}_scores_antigravity.npz'; temp=directory/f'{dataset}_temporary_antigravity.npz'
    np.savez_compressed(temp,**arrays);temp.replace(path)
    record=dict(dataset=dataset,source_family=family,binding=binding,values=output['values'],diagnostics=output['diagnostics'],score_sha256=sha(path),
        n_train=len(train),n_test=len(prepared.test_rows),test_indices=list(prepared.test_indices),seconds=perf_counter()-started,**identity)
    save(directory/f'{dataset}_meta_antigravity.json',record)
    return record

def acceleration():
    result=assess_acceleration_antigravity()
    result['mlx_gpu_float64']=_probe_subprocess_antigravity("import mlx.core as m,json\nm.set_default_device(m.gpu)\na=m.array([1.,2.],dtype=m.float64)\nb=a+1\nm.eval(b)\nprint(json.dumps(dict(finite=bool(m.isfinite(b).all().item()))))")
    from numba import njit
    @njit
    def route_numba(p):
        out=np.zeros((len(p),2),dtype=np.int64)
        for i in range(len(p)):
            for block in range(2):
                best=0.
                for j in range(3):
                    if p[i,block*3+j]>best:best=p[i,block*3+j];out[i,block]=j+1
        return out
    p=np.random.default_rng(SEED).normal(size=(125,6));p[0]=0
    t=perf_counter();expected=routes_antigravity(p);tn=perf_counter()-t
    t=perf_counter();actual=route_numba(p);tj=perf_counter()-t
    np.testing.assert_array_equal(actual,np.column_stack(list(expected.values())))
    result['exp493_routing']=dict(numpy_seconds=tn,numba_compile_and_execute_seconds=tj,parity=True,selected='numpy',reason='small routing stage; compilation included')
    t=perf_counter();predict_antigravity(np.random.default_rng(SEED).normal(size=(125,42)),p,[str(i) for i in range(125)])
    result['exp493_lofo42_seconds']=perf_counter()-t
    result['cpu_policy']['exp493_workers']=7
    result['stage_policy']='official aeon/LocalGap/Ridge and scipy quantile/Spearman retained; Numba cannot wrap these official Python estimators without rewriting; existing aeon Numba kernels retained'
    return result

def targets(outcomes):
    o=np.asarray(outcomes);return np.column_stack((o[:,1:,0]-o[:,0:1,0],np.sign(o[:,1:,1]-o[:,0:1,1])))

def evaluate(reps,values24,old,records,b489,b492,binding,directory):
    # All scores exist and are checked before the first fresh TEST-label query.
    verified=[cached(directory,r.dataset,binding) for r in reps]
    if any(x is None for x in verified):raise RuntimeError('incomplete score population')
    boundary=dict(binding=binding,scores={r.dataset:x[0]['score_sha256'] for r,x in zip(reps,verified)},metadata={r.dataset:jsha(x[0]) for r,x in zip(reps,verified)})
    marker=P/'labels_opened_antigravity.json'
    if marker.exists() and read(marker)!=boundary:raise ValueError('label boundary metadata changed')
    save(marker,boundary)
    tables={b:[] for b in ('old','stable')}; rows_out=[]
    expected_mini={x['dataset']:x['control'] for x in read(P489/'results_antigravity.json')['records']}
    expected_pairs={x['dataset']:x['metrics'] for x in read(P492/'results_antigravity.json')['records']}
    with closing(readonly_sqlite_antigravity(DB)) as con:
        for row,(record,arrays) in zip(reps,verified):
            labels=load_retained_test_labels_antigravity(con,row.dataset,tuple(record['test_indices']))
            mini=load_score_checkpoint_antigravity(P489/'score_checkpoints_antigravity',row.dataset,b489)
            _,prior=load_antigravity(P492/'score_checkpoints_antigravity',row.dataset,b492)
            controls=[evaluate_score_vector_antigravity(mini['heads']['control']['test_scores'],labels),evaluate_score_vector_antigravity(arrays['multirocket_test'],labels)]
            assert controls[0]==expected_mini[row.dataset], 'Mini metric replay mismatch'
            item=dict(dataset=row.dataset,source_family=row.source_family)
            for branch in tables:
                metrics=controls+[evaluate_score_vector_antigravity(prior[f'{m}__{branch}__test_scores'],labels) for m in METHODS[2:]]
                for method,metric in zip(METHODS[2:],metrics[2:]):
                    assert metric==expected_pairs[row.dataset][method][branch], 'Exp492 metric replay mismatch'
                item[branch]=metrics;tables[branch].append([[m['auc'],m['zf']] for m in metrics])
            rows_out.append(item)
    save(P/'outcomes_antigravity.json',dict(binding=binding,records=rows_out))
    tables={k:np.asarray(v) for k,v in tables.items()}; families=[r.source_family for r in reps]
    x24=np.asarray([values24[r.dataset] for r in reps],dtype=float)
    features={'24D':x24,'old42D':np.column_stack((x24,[old[r.dataset]['values'] for r in reps])),
        'stable42D':np.column_stack((x24,[records[r.dataset]['values'] for r in reps]))}
    predictions={}
    for branch,o in tables.items():
        y=targets(o)
        for name,x in features.items():
            pred,constant=predict_antigravity(x,y,families)
            predictions[f'{name}__{branch}']=dict(predictions=pred,routes=routes_antigravity(pred),constant=constant,constant_routes=routes_antigravity(constant))
    save(P/'decisions_antigravity.json',dict(binding=binding,datasets=[r.dataset for r in reps],models=predictions))
    # Only serialized, already frozen decisions are evaluated.
    frozen=read(P/'decisions_antigravity.json')['models']; summaries={}
    for name,model in frozen.items():
        fitted=name.split('__')[1];y=targets(tables[fitted]);pred=np.asarray(model['predictions'])
        summaries[name]=dict(mse_auc=float(((pred-y)**2)[:,:3].mean()),mse_zf_sign=float(((pred-y)**2)[:,3:].mean()),
            evaluations={branch:{head:summarize_route_antigravity(route,o) for head,route in model['routes'].items()} for branch,o in tables.items()},
            constant_evaluations={branch:{head:summarize_route_antigravity(route,o) for head,route in model['constant_routes'].items()} for branch,o in tables.items()})
    historical={};previous=read(P490/'results_antigravity.json')['comparison']
    for name in ('24D','old42D'):
        pred,_=predict_antigravity(features[name],np.stack([target_vector_antigravity(r) for r in reps]),families)
        key='features_24' if name=='24D' else 'features_42'
        from experiments.exp491_gain_prediction_antigravity.gain_prediction_antigravity import TARGET_KEYS_ANTIGRAVITY
        lookup={d:[fold['predictions'][k][i] for k in TARGET_KEYS_ANTIGRAVITY] for fold in previous[key]['folds'] for i,d in enumerate(fold['datasets'])}
        expected=np.asarray([lookup[r.dataset] for r in reps]); np.testing.assert_allclose(pred,expected,rtol=1e-12,atol=1e-12)
        historical[name]=float(abs(pred-expected).max())
    controls={b:{m:summarize_route_antigravity(np.full(len(reps),i,dtype=int),o) for i,m in enumerate(METHODS)} for b,o in tables.items()}
    result=dict(state='complete_paired_meta_development',families=len(reps),models=summaries,controls=controls,historical_replay_max_abs=historical,
        decisions_sha256=sha(P/'decisions_antigravity.json'),outcomes_sha256=sha(P/'outcomes_antigravity.json'),binding=binding)
    save(P/'results_antigravity.json',result)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('smoke','official'),required=True);args=parser.parse_args()
    started=perf_counter();rows,reps,v24,old,b489,b492,binding=context()
    if args.mode=='official':
        if (P/'results_antigravity.json').exists():raise RuntimeError('official results already exist')
        smoke=read(P/'smoke_results_antigravity.json')
        if smoke['binding']!=binding or smoke['state']!='passed':raise RuntimeError('matching smoke required')
        selected=reps
    else:
        selected=[r for r in reps if r.dataset in ('ACSF1_normal_0','DiatomSizeReduction_normal_1')]
        assert len(selected)==2
    directory=P/('score_checkpoints_antigravity' if args.mode=='official' else 'smoke_scores_antigravity')
    if (P/'labels_opened_antigravity.json').exists() and any(cached(directory,r.dataset,binding) is None for r in selected):raise RuntimeError('no rescoring after label boundary')
    acc=acceleration() if args.mode=='smoke' else smoke['acceleration']
    manifest=dict(state='scoring',binding=binding,workers=7,acceleration=acc,completed=0,failures={})
    path=P/('run_manifest_antigravity.json' if args.mode=='official' else 'smoke_manifest_antigravity.json');save(path,manifest)
    records={}
    with ProcessPoolExecutor(max_workers=min(7,len(selected)),mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs={pool.submit(worker,(r.dataset,r.source_family,str(directory),binding,b489,b492,old[r.dataset])):r for r in selected}
        for future in as_completed(jobs):
            row=jobs[future]
            try:records[row.dataset]=future.result();print(f'{len(records)}/{len(selected)} {row.dataset}',flush=True)
            except Exception as e:manifest['failures'][row.dataset]=repr(e);print(f'ERROR {row.dataset}: {e}',flush=True)
            manifest.update(completed=len(records),seconds=perf_counter()-started);save(path,manifest)
    if manifest['failures']:raise RuntimeError(f"dataset failures: {manifest['failures']}")
    if args.mode=='official':
        result=evaluate(reps,v24,old,records,b489,b492,binding,directory)
        manifest.update(state=result['state'],result_sha256=sha(P/'results_antigravity.json'))
    else:
        save(P/'smoke_results_antigravity.json',dict(state='passed',binding=binding,acceleration=acc,records=records))
        manifest['state']='passed'
    for name,digest in {**binding['inputs'],**binding['upstream_meta_sha256']}.items():assert sha(Path(name))==digest,'prior artifact changed'
    manifest['seconds']=perf_counter()-started;save(path,manifest)

if __name__=='__main__': main()
