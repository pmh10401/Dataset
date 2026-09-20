"""Resume-safe all-dataset evaluation; score and route files precede fresh labels."""
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
import argparse,json,multiprocessing,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import closing
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import (
    ROOT_ANTIGRAVITY as ROOT, DATABASE_PATH_ANTIGRAVITY as DB, SEED_ANTIGRAVITY as SEED,
    file_sha256_antigravity as sha,sha256_json_antigravity as jsha,
    write_json_atomic_antigravity as write,jsonable_antigravity as jsonable,
    cpu_worker_policy_antigravity,readonly_sqlite_antigravity,assess_acceleration_antigravity,
)
from experiments.exp493_stable_meta_routing_antigravity.run_antigravity import context as source_context,targets
from experiments.exp493_stable_meta_routing_antigravity.meta_antigravity import routes_antigravity
from experiments.exp479_missing_score_recovery_antigravity.evaluation_antigravity import load_retained_test_labels_antigravity
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import prepare_dataset_values_antigravity
from .scoring_antigravity import P,SCORE_NAMES,load,persist,worker,score_and_describe
from .evaluation_antigravity import GRID,GRID_NUM,metric,predict_queries,summarize
P493=ROOT/'experiments/exp493_stable_meta_routing_antigravity'
read=lambda p:json.loads(p.read_text())
def save(p,x):write(p,jsonable(x))

def context():
    rows,reps,v24,old,b489,b492,original=source_context()
    source=read(P493/'results_antigravity.json')
    assert source['binding']==original
    assert sha(P493/'decisions_antigravity.json')==source['decisions_sha256']
    assert sha(P493/'outcomes_antigravity.json')==source['outcomes_sha256']
    for path,digest in source['binding']['code'].items():assert sha(ROOT/path)==digest
    cm=read(ROOT/'docs/codemap/codemap_exp494_antigravity.json')
    for n in cm['nodes']:assert sha(ROOT/n['path'])==n['sha256'],n['path']
    binding=dict(protocol_sha256=sha(P/'protocol_antigravity.html'),source493_sha256=sha(P493/'results_antigravity.json'),source_binding=original,
        population=[dict(dataset=r.dataset,source_family=r.source_family,n_train=int(r.features[1])) for r in rows],
        code={n['path']:n['sha256'] for n in cm['nodes']},seed=SEED)
    with closing(readonly_sqlite_antigravity(DB)) as con:all_names=[r[0] for r in con.execute('SELECT name FROM datasets ORDER BY name')]
    missing=sorted(set(all_names)-{r.dataset for r in rows})
    assert missing==['CornellWhaleChallenge','Phoneme_normal_39'] and len(rows)==1117
    return rows,reps,v24,old,binding

def reuse_representatives(directory,binding,reps,old):
    boundary=read(P493/'labels_opened_antigravity.json')['metadata']
    r493=read(P493/'results_antigravity.json')
    for row in reps:
        d=row.dataset
        if load(directory,d,binding):continue
        fresh=P493/'score_checkpoints_antigravity'/f'{d}_scores_antigravity.npz'
        mini=ROOT/'experiments/exp489_lof_score_head_antigravity/score_checkpoints_antigravity'/f'{d}_antigravity_scores_antigravity.npz'
        pair=ROOT/'experiments/exp492_stable_normalization_antigravity/score_checkpoints_antigravity'/f'{d}_scores_antigravity.npz'
        meta=read(fresh.with_name(f'{d}_meta_antigravity.json'))
        assert jsha(meta)==boundary[d]
        for path in (fresh,mini,pair):
            sidecar=path.with_name(path.name.replace('_scores_antigravity.npz','_meta_antigravity.json'));m=read(sidecar)
            assert sha(path)==m['score_sha256'] and m['test_indices']==meta['test_indices']
            if path!=fresh:assert sha(sidecar)==r493['binding']['upstream_meta_sha256'][str(sidecar)]
        with np.load(fresh) as f,np.load(mini) as m,np.load(pair) as h:
            arrays={'test__0':m['control_test_scores'],'test__1':f['multirocket_test'],'test__4':m['lof_test_scores'],
                'test__2':h['hydra_local_gap__stable__test_scores'],'test__3':h['hydra_multirocket_local_gap__stable__test_scores'],
                'test__5':h['hydra_local_gap__old__test_scores'],'test__6':h['hydra_multirocket_local_gap__old__test_scores'],
                'old18':np.asarray(old[d]['values']), 'stable18':np.asarray(meta['values'])}
            persist(directory,d,arrays,dict(binding_sha256=binding,source_family=row.source_family,n_train=meta['n_train'],n_test=meta['n_test'],
                target_length=meta['target_length'],test_indices=meta['test_indices'],origin='verified_Exp489_492_493',seconds=0.))

def freeze_routes(rows,reps,v24,directory,binding):
    checks={r.dataset:load(directory,r.dataset,binding) for r in rows}
    assert all(v is not None for v in checks.values())
    row_by={r.dataset:r for r in rows};names=[r.dataset for r in rows];training=[r.dataset for r in reps]
    assert len(names)==1117 and sum(checks[d][0]['n_train']>=5 for d in names)==956
    features={}
    for kind in ('24D','old42D','stable42D'):
        covered=names if kind=='24D' else [d for d in names if checks[d][0]['n_train']>=5]
        def vector(d):return v24[d] if kind=='24D' else list(v24[d])+checks[d][1]['old18' if kind=='old42D' else 'stable18'].tolist()
        features[kind]=(covered,np.asarray([vector(d) for d in covered],float),np.asarray([vector(d) for d in training],float))
    outcomes=read(P493/'outcomes_antigravity.json')['records'];lookup={r['dataset']:r for r in outcomes}
    old_dec=read(P493/'decisions_antigravity.json'); models={};maxerr=0.
    for kind,(covered,q,x) in features.items():
        for target in ('old','stable'):
            y=targets([[[m['auc'],m['zf']] for m in lookup[d][target]] for d in training])
            pred=predict_queries(x,y,[row_by[d].source_family for d in training],q,[row_by[d].source_family for d in covered])
            route=routes_antigravity(pred);name=kind+'__'+target
            indices=[covered.index(d) for d in old_dec['datasets']];expected=np.asarray(old_dec['models'][name]['predictions'])
            np.testing.assert_allclose(pred[indices],expected,rtol=1e-12,atol=1e-12)
            maxerr=max(maxerr,float(np.max(abs(pred[indices]-expected))))
            for head in route:assert route[head][indices].tolist()==old_dec['models'][name]['routes'][head]
            models[name]=dict(datasets=covered,predictions=pred,routes=route)
    decisions=dict(binding_sha256=binding,training_datasets=training,training_population_fixed=True,same_family_excluded=True,models=models,replay_max_abs_error=maxerr)
    output=P/'decisions_antigravity.json'
    if output.exists():assert read(output)==jsonable(decisions)
    else:save(output,decisions)
    boundary=dict(binding_sha256=binding,decisions_sha256=sha(output),scores={d:checks[d][0]['score_sha256'] for d in names},metadata={d:jsha(checks[d][0]) for d in names})
    marker=P/'labels_opened_antigravity.json'
    if marker.exists():assert read(marker)==boundary
    else:save(marker,boundary)
    return checks,read(output)

def evaluate(rows,checks,decisions,binding):
    records=[]
    with closing(readonly_sqlite_antigravity(DB)) as con:
        for row in rows:
            meta,arrays=checks[row.dataset];y=load_retained_test_labels_antigravity(con,row.dataset,tuple(meta['test_indices']))
            metrics={name:metric(y,arrays[f'test__{j}']) for j,name in enumerate(SCORE_NAMES)}
            records.append(dict(dataset=row.dataset,source_family=row.source_family,n_train=meta['n_train'],metrics=metrics))
    lookup={r['dataset']:r for r in records}; current125=read(ROOT/'docs/CURRENT_LOW_FPR_COMPARISON_antigravity.json')
    previous={s['model']:s for s in current125['summary']}
    groups={'all_evaluable':list(lookup),'common_42d':[d for d in lookup if lookup[d]['n_train']>=5], 'previous_representatives':decisions['training_datasets']}
    methods={name:{d:r['metrics'][name] for d,r in lookup.items()} for name in SCORE_NAMES}
    for name,model in decisions['models'].items():
        for head,route in model['routes'].items():methods[name+' / '+head]={d:lookup[d]['metrics'][SCORE_NAMES[j]] for d,j in zip(model['datasets'],route)}
    summaries={}
    for group,names in groups.items():
        summaries[group]={}
        for name,values in methods.items():
            if not set(names).issubset(values):continue
            selected=[dict(source_family=lookup[d]['source_family'],metrics=values[d]) for d in names]
            s=summarize(selected);baseline=[lookup[d]['metrics']['MiniRocket'] for d in names]
            delta=np.array([values[d]['tpr'] for d in names])-np.array([m['tpr'] for m in baseline])
            s['dataset_wins_vs_mini']=(delta>1e-12).sum(axis=0).tolist();s['dataset_losses_vs_mini']=(delta<-1e-12).sum(axis=0).tolist()
            summaries[group][name]=s
            if group=='previous_representatives':
                np.testing.assert_allclose(s['dataset_macro_tpr'],previous[name]['macro_tpr'],rtol=0,atol=1e-12)
                assert s['total_zf']==previous[name]['total_zf']
    result=dict(state='complete_all_evaluable_frozen_model_comparison',binding_sha256=binding,raw_datasets=1119,evaluated_datasets=len(rows),families=len({r.source_family for r in rows}),
        excluded=[dict(dataset='CornellWhaleChallenge',reason='normal_class_not_mapped'),dict(dataset='Phoneme_normal_39',reason='fewer_than_two_normal_train_rows')],
        unavailable_42d=[dict(dataset=r['dataset'],source_family=r['source_family'],n_train=r['n_train'],reason='fewer_than_five_normal_train_rows') for r in records if r['n_train']<5],
        fpr_grid=GRID.tolist(),primary_aggregation='mean within source family then equal mean across source families',
        metric_scope='retrospective TEST ROC, deterministic thresholds, no interpolation; no operational calibration claim',
        false_positive_resolution={group:[sum(lookup[d]['metrics']['MiniRocket']['n_normal']*int(n)//1000>=1 for d in names) for n in GRID_NUM] for group,names in groups.items()},
        summaries=summaries,records=records,decisions_sha256=sha(P/'decisions_antigravity.json'),label_boundary_sha256=sha(P/'labels_opened_antigravity.json'),
        verification=dict(score_files_checked=len(checks),score_vectors_evaluated=len(rows)*7,all_low_fpr_points_crosschecked=True,previous125_all19_model_summaries_reproduced=True,router_replay_max_abs_error=decisions['replay_max_abs_error']))
    save(P/'results_antigravity.json',result)
    return result

def smoke(rows,reps,v24,old,binding):
    start=perf_counter(); directory=P/'smoke_scores_antigravity';reuse_representatives(directory,binding,reps,old)
    cases=['DiatomSizeReduction_normal_1','ProximalPhalanxOutlineCorrect_normal_0'];report={}
    with closing(readonly_sqlite_antigravity(DB)) as con,threadpool_limits(limits=1):
        for d in cases:
            prepared=prepare_dataset_values_antigravity(con,d);expected=load(directory,d,binding)[1]
            t=perf_counter();actual=score_and_describe(prepared.train_rows,prepared.test_rows)
            errors={}
            for k in expected:
                np.testing.assert_allclose(actual[k],expected[k],rtol=1e-10,atol=1e-10 if k.endswith('18') else 1e-8)
                errors[k]=float(np.max(abs(actual[k]-expected[k])))
            # Already evaluated representative labels only; new population labels remain unopened.
            y=load_retained_test_labels_antigravity(con,d,tuple(prepared.test_indices))
            for k in expected:
                if k.startswith('test__'):
                    a=metric(y,actual[k]);b=metric(y,expected[k]);assert a['tp']==b['tp'] and a['fp']==b['fp']
                    neg=expected[k][y==0]; threshold=np.sort(neg)[::-1][len(neg)*GRID_NUM//1000]
                    for th in threshold:np.testing.assert_array_equal(np.flatnonzero(actual[k]>th),np.flatnonzero(expected[k]>th))
            report[d]=dict(seconds=perf_counter()-t,max_abs_error=errors)
    rng=np.random.default_rng(SEED);small=score_and_describe(rng.normal(size=(2,32)),rng.normal(size=(5,32)))
    assert set(small)=={f'test__{j}' for j in range(7)}
    accel=assess_acceleration_antigravity();accel['exp494']='Frozen official CPU encoders, scaler/KNN and scipy stats reused. No new score-source operation. Prior guarded-scaler MLX/Numba benchmarks remain applicable; exact score/descriptor smoke verified. Existing aeon/preprocessing Numba retained. No added JIT for Python estimator orchestration or short ROC aggregation.'
    accel['exp492_assessment']=read(ROOT/'experiments/exp492_stable_normalization_antigravity/run_manifest_antigravity.json').get('acceleration',{})
    out=dict(state='passed',binding_sha256=binding,cases=report,small_train_test_only_passed=True,acceleration=accel,seconds=perf_counter()-start)
    save(P/'smoke_results_antigravity.json',out);print(json.dumps({'smoke':'passed','seconds':out['seconds'],'case_seconds':{k:v['seconds'] for k,v in report.items()}}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','run'],required=True);args=parser.parse_args()
    start=perf_counter();rows,reps,v24,old,binding=context();digest=jsha(binding)
    if args.mode=='smoke':smoke(rows,reps,v24,old,digest);return
    if (P/'results_antigravity.json').exists():raise RuntimeError('complete results already exist; verify instead')
    sm=read(P/'smoke_results_antigravity.json');assert sm['state']=='passed' and sm['binding_sha256']==digest
    manifest=P/'run_manifest_antigravity.json';policy=cpu_worker_policy_antigravity()
    data=dict(state='scoring',binding=binding,binding_sha256=digest,cpu_policy=policy,acceleration=sm['acceleration'])
    if manifest.exists():assert read(manifest)['binding_sha256']==digest
    save(manifest,data)
    directory=P/'score_checkpoints_antigravity';reuse_representatives(directory,digest,reps,old)
    missing=[r for r in rows if load(directory,r.dataset,digest) is None]
    completed=len(rows)-len(missing);print(json.dumps(dict(state='scoring',completed=completed,total=len(rows),workers=policy['workers'])),flush=True)
    failures=[]
    with ProcessPoolExecutor(max_workers=policy['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker,(r.dataset,r.source_family,str(directory),digest)):r.dataset for r in missing}
        for future in as_completed(futures):
            d=futures[future]
            try:record=future.result();completed+=1
            except Exception as e:failures.append(dict(dataset=d,type=type(e).__name__,error=str(e)))
            progress=dict(completed=completed,total=len(rows),failures=failures,elapsed_seconds=perf_counter()-start)
            save(P/'progress_antigravity.json',progress)
            if completed%25==0 or failures:print(json.dumps(progress),flush=True)
    if failures:save(P/'failures_antigravity.json',failures);raise RuntimeError('scoring failures; no new TEST-label evaluation')
    data['state']='freezing_routes';save(manifest,data);print('Freezing held-family-out routes',flush=True)
    checks,decisions=freeze_routes(rows,reps,v24,directory,digest)
    data['state']='evaluating';save(manifest,data);print('Evaluating complete score population',flush=True)
    result=evaluate(rows,checks,decisions,digest)
    data.update(state=result['state'],result_sha256=sha(P/'results_antigravity.json'),seconds=perf_counter()-start)
    save(manifest,data);print(json.dumps(dict(state=data['state'],datasets=len(rows),seconds=data['seconds'])),flush=True)
if __name__=='__main__':main()
