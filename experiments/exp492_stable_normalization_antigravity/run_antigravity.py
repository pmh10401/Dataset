"""Paired normalization experiment; all population scores precede TEST labels."""
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import (
    configure_inner_threads_antigravity, cpu_worker_policy_antigravity,
    DATABASE_PATH_ANTIGRAVITY, SEED_ANTIGRAVITY, ROOT_ANTIGRAVITY,
    FROZEN_SCORER_PATHS_ANTIGRAVITY, FROZEN_PREPROCESS_PATHS_ANTIGRAVITY,
    load_frozen_sources_antigravity, representative_rows_antigravity,
    EXCLUDED_FIVEFOLD_FAMILIES_ANTIGRAVITY, validate_database_antigravity,
    readonly_sqlite_antigravity, file_sha256_antigravity, file_identity_antigravity,
    write_json_atomic_antigravity, jsonable_antigravity, sha256_json_antigravity,
    package_version_antigravity, _probe_subprocess_antigravity,
    require_matching_smoke_antigravity, refuse_official_overwrite_antigravity,
)
configure_inner_threads_antigravity()
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime, timezone
import gzip
import html
import json
import multiprocessing
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import prepare_dataset_values_antigravity
from experiments.exp479_missing_score_recovery_antigravity.evaluation_antigravity import load_retained_test_labels_antigravity, evaluate_score_vector_antigravity
from experiments.exp490_behavior_descriptors_antigravity.behavior_descriptors_antigravity import prepare_normal_train_matrix_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import MethodCandidateAntigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import score_candidate_antigravity
from .scoring_antigravity import score_pair_antigravity
from .normalization_antigravity import StableRobustScalerAntigravity

PACKAGE = Path(__file__).resolve().parent
METHODS = ('hydra_local_gap', 'hydra_multirocket_local_gap')
BRANCHES = ('old', 'stable')
SMOKE_CASES = ('DiatomSizeReduction_normal_1', 'ProximalPhalanxOutlineCorrect_normal_0')


def binding_antigravity():
    rows, _, sources = load_frozen_sources_antigravity()
    population, excluded = representative_rows_antigravity(rows, expected_families=125, expected_excluded=EXCLUDED_FIVEFOLD_FAMILIES_ANTIGRAVITY)
    validate_database_antigravity(DATABASE_PATH_ANTIGRAVITY, frozen=True)
    map_path = ROOT_ANTIGRAVITY / 'docs/codemap/codemap_exp492_antigravity.json'
    paths = [ROOT_ANTIGRAVITY / n['path'] for n in json.loads(map_path.read_text())['nodes'] if n['path'].endswith('.py')]
    required = set(FROZEN_SCORER_PATHS_ANTIGRAVITY + FROZEN_PREPROCESS_PATHS_ANTIGRAVITY) | {PACKAGE/name for name in ('run_antigravity.py','scoring_antigravity.py','normalization_antigravity.py')}
    if not required.issubset(set(paths)): raise RuntimeError('code map omits required implementation')
    code = {str(p.relative_to(ROOT_ANTIGRAVITY)): file_sha256_antigravity(p) for p in paths}
    for node in json.loads(map_path.read_text())['nodes']:
        if node['path'] in code and node['sha256'] != code[node['path']]:
            raise RuntimeError(f"stale code map: {node['path']}")
    binding = dict(schema='exp492_binding_v1', protocol_sha256=file_sha256_antigravity(PACKAGE / 'protocol_antigravity.html'),
        sources=sources, database_sha256=file_sha256_antigravity(DATABASE_PATH_ANTIGRAVITY), code_sha256=code,
        population=[dict(dataset=r.dataset, source_family=r.source_family) for r in population], excluded=list(excluded),
        versions={p:package_version_antigravity(p) for p in ('numpy','scipy','scikit-learn','aeon','torch','numba','mlx','mlx-metal')},
        python=platform.python_version(), seed=SEED_ANTIGRAVITY)
    return jsonable_antigravity(binding)


def score_stats_antigravity(values):
    values=np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('nonempty finite score vector required')
    q25,q50,q75,q95,q99=np.quantile(values,[.25,.5,.75,.95,.99],method='linear')
    iqr=q75-q25
    return dict(count=len(values), finite_count=int(np.isfinite(values).sum()), max_abs=float(np.abs(values).max()),
        q95=float(q95), q99=float(q99), iqr=float(iqr),
        tail_q95=None if iqr==0 else float((q95-q50)/iqr), tail_q99=None if iqr==0 else float((q99-q50)/iqr))


def paths_antigravity(directory, dataset):
    # Population identifiers are bound and checked before constructing paths.
    if not dataset or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._' for c in dataset):
        raise ValueError('unsafe dataset identifier')
    return tuple(Path(directory)/f'{dataset}_{suffix}_antigravity.{extension}' for suffix,extension in
                 (('scores','npz'),('meta','json'),('diagnostics','json.gz')))


def persist_antigravity(directory, dataset, pairs, extra):
    npz,meta,diag=paths_antigravity(directory,dataset)
    npz.parent.mkdir(parents=True,exist_ok=True)
    arrays={f'{m}__{b}__{split}':pairs[m][b][split] for m in METHODS for b in BRANCHES for split in ('train_oof','test_scores')}
    temporary=npz.with_name(npz.stem+'_temporary_antigravity.npz')
    np.savez_compressed(temporary,**arrays); temporary.replace(npz)
    details={m:pairs[m]['diagnostics'] for m in METHODS}
    temp_diag=diag.with_name(diag.stem+'_temporary_antigravity.gz')
    temp_diag.write_bytes(gzip.compress(json.dumps(jsonable_antigravity(details),allow_nan=False).encode(),mtime=0)); temp_diag.replace(diag)
    record=dict(dataset=dataset, **extra, score_sha256=file_sha256_antigravity(npz), diagnostics_sha256=file_sha256_antigravity(diag),
        dimensions={m:pairs[m]['feature_dimension'] for m in METHODS}, representation_fits={m:pairs[m]['representation_fits'] for m in METHODS},
        score_stats={m:{b:{s:score_stats_antigravity(pairs[m][b][s]) for s in ('train_oof','test_scores')} for b in BRANCHES} for m in METHODS})
    write_json_atomic_antigravity(meta,jsonable_antigravity(record))
    return record


def load_antigravity(directory,dataset,binding):
    npz,meta,diag=paths_antigravity(directory,dataset)
    if not meta.exists():
        return None
    record=json.loads(meta.read_text())
    if record.get('dataset')!=dataset or record.get('binding')!=binding:
        raise ValueError('checkpoint binding/dataset mismatch')
    if record['score_sha256']!=file_sha256_antigravity(npz) or record['diagnostics_sha256']!=file_sha256_antigravity(diag):
        raise ValueError('checkpoint checksum mismatch')
    with np.load(npz,allow_pickle=False) as data:
        arrays={key:data[key].copy() for key in data.files}
    expected={f'{m}__{b}__{s}' for m in METHODS for b in BRANCHES for s in ('train_oof','test_scores')}
    if set(arrays)!=expected: raise ValueError('checkpoint keys mismatch')
    for key,values in arrays.items():
        count=record['n_train'] if key.endswith('train_oof') else record['n_test']
        if values.shape!=(count,) or count<1 or not np.isfinite(values).all(): raise ValueError('invalid checkpoint scores')
    if len(record['test_indices'])!=record['n_test'] or len(set(record['test_indices']))!=record['n_test']:
        raise ValueError('invalid retained TEST indices')
    if any(record['representation_fits'][m]!=6 for m in METHODS): raise ValueError('wrong fit count')
    return record,arrays


def validate_alignment_antigravity(record,prepared):
    expected=dict(n_train=len(prepared.train_rows),n_test=len(prepared.test_rows),
                  target_length=prepared.target_length,test_indices=list(prepared.test_indices))
    if any(record.get(k)!=v for k,v in expected.items()):
        raise ValueError('checkpoint score-to-row alignment mismatch')


def score_worker_antigravity(payload):
    configure_inner_threads_antigravity(); torch.set_num_threads(1)
    dataset=payload['dataset']; started=perf_counter()
    cached=load_antigravity(payload['directory'],dataset,payload['binding'])
    if cached is not None:
        if cached[0]['source_family']!=payload['source_family']: raise ValueError('cached family mismatch')
        return cached[0]
    with threadpool_limits(limits=1), closing(readonly_sqlite_antigravity(DATABASE_PATH_ANTIGRAVITY)) as connection:
        prepared=prepare_dataset_values_antigravity(connection,dataset)
        pairs={m:score_pair_antigravity(prepared.train_rows,prepared.test_rows,m,seed=SEED_ANTIGRAVITY) for m in METHODS}
    return persist_antigravity(payload['directory'],dataset,pairs,dict(binding=payload['binding'],source_family=payload['source_family'],
        n_train=len(prepared.train_rows),n_test=len(prepared.test_rows),test_indices=list(prepared.test_indices),
        target_length=prepared.target_length,seconds=perf_counter()-started))


def acceleration_antigravity():
    """Measure the complete guarded fit stage; official encoders/KNN remain CPU."""
    from numba import njit
    @njit
    def mask_numba(scale,spread,tolerance):
        result=scale.copy()
        for j in range(len(result)):
            if spread[j]<=tolerance[j] and result[j]<1: result[j]=1.
        return result
    from sklearn.preprocessing import RobustScaler
    values=np.random.default_rng(20260721).normal(size=(128,2048))
    values[:,::4]*=1e-8
    def numpy_stage(): return StableRobustScalerAntigravity(quantile_range=(10,90)).fit(values).scale_
    def numba_stage():
        model=RobustScaler(quantile_range=(10,90)).fit(values)
        q10,q90=np.percentile(values,[10,90],axis=0,method='linear')
        tolerance=10*np.finfo(np.float32).eps*np.maximum(1,np.maximum(abs(q10),abs(q90)))
        return mask_numba(model.scale_,q90-q10,tolerance)
    timings={};outputs={}
    for name,fn in (('numpy',numpy_stage),('numba',numba_stage)):
        t=perf_counter(); outputs[name]=fn(); cold=perf_counter()-t
        t=perf_counter(); repeat=fn(); warm=perf_counter()-t
        np.testing.assert_array_equal(outputs[name],repeat)
        timings[name]=dict(cold_full_fit_seconds=cold,warm_full_fit_seconds=warm,total_two_fits_seconds=cold+warm)
    np.testing.assert_array_equal(outputs['numpy'],outputs['numba'])
    speedup=timings['numpy']['total_two_fits_seconds']/timings['numba']['total_two_fits_seconds']
    if speedup>1.0: raise RuntimeError('Numba beats NumPy including compilation; review backend before official run')
    from .normalization_antigravity import StableLocalGapModelAntigravity
    model=StableLocalGapModelAntigravity(values[:80])
    expected_scores=model.score(values[80:])
    fitted=model.scaler
    alternate=mask_numba(fitted.original_scale_,fitted.original_spread_,fitted.tolerance_)
    np.testing.assert_array_equal(alternate,fitted.scale_)
    fitted.scale_=alternate
    alternate_scores=model.score(values[80:])
    np.testing.assert_array_equal(expected_scores,alternate_scores)
    threshold=np.quantile(model.score(values[:80]),.995,method='higher')
    np.testing.assert_array_equal(np.flatnonzero(expected_scores>threshold),np.flatnonzero(alternate_scores>threshold))
    # Compilation is included; no newly compiled code enters the canonical scorer.
    mlx_benchmark = dict(selected=False, numerical_parity=None, candidate_index_parity=None, speedup=None)
    try:
        import mlx.core as mx
        def mlx_mask(scale, spread, tolerance):
            scale_mx, spread_mx, tolerance_mx = (mx.array(a) for a in (scale, spread, tolerance))
            guarded = mx.where((spread_mx <= tolerance_mx) & (scale_mx < 1), mx.ones_like(scale_mx), scale_mx)
            mx.eval(guarded); mx.synchronize()
            return np.asarray(guarded).copy()
        def mlx_stage():
            fitted = RobustScaler(quantile_range=(10,90)).fit(values)
            q10,q90 = np.percentile(values,[10,90],axis=0,method='linear')
            tolerance=10*np.finfo(np.float32).eps*np.maximum(1,np.maximum(abs(q10),abs(q90)))
            return mlx_mask(fitted.scale_,q90-q10,tolerance)
        t=perf_counter(); mlx_scale=mlx_stage(); cold=perf_counter()-t
        t=perf_counter(); mlx_repeat=mlx_stage(); warm=perf_counter()-t
        np.testing.assert_array_equal(mlx_scale,mlx_repeat)
        np.testing.assert_array_equal(mlx_scale,outputs['numpy'])
        fitted=model.scaler
        fitted.scale_=mlx_mask(fitted.original_scale_,fitted.original_spread_,fitted.tolerance_)
        mlx_scores=model.score(values[80:])
        np.testing.assert_array_equal(mlx_scores,expected_scores)
        np.testing.assert_array_equal(np.flatnonzero(mlx_scores>threshold),np.flatnonzero(expected_scores>threshold))
        mlx_speedup=timings['numpy']['total_two_fits_seconds']/(cold+warm)
        mlx_benchmark.update(device=str(mx.default_device()),cold_full_fit_seconds=cold,warm_full_fit_seconds=warm,
            numerical_parity=True,candidate_index_parity=True,finite_scores=bool(np.isfinite(mlx_scores).all()),
            speedup=mlx_speedup,fallback_reason='complete guarded fit including transfer and synchronization below 1.1x speed gate')
    except (ImportError,ValueError,RuntimeError,AssertionError) as error:
        mlx_benchmark['fallback_reason']=str(error)
    if mlx_benchmark['speedup'] is not None and mlx_benchmark['speedup']>=1.1:
        raise RuntimeError('MLX passed speed gate; integrate backend before official run')
    probes={
      'torch_mps':_probe_subprocess_antigravity("import json,torch\nr={'available':True,'mps_available':torch.backends.mps.is_available()}\ntry:\n x=torch.ones(2,device='mps',dtype=torch.float64); torch.mps.synchronize(); r['float64']=True\nexcept Exception as e: r.update(float64=False,reason=str(e))\nprint(json.dumps(r))"),
      'mlx':_probe_subprocess_antigravity("import json\nr={}\ntry:\n import mlx.core as mx\n r.update(available=True,device=str(mx.default_device()))\n x=mx.array([1.,2.],dtype=mx.float64); mx.eval(x); r['float64']=True\nexcept Exception as e: r.update(float64=False,reason=str(e))\nprint(json.dumps(r))")}
    return dict(versions={p:package_version_antigravity(p) for p in ('torch','mlx','mlx-metal','numba','numpy','aeon','scikit-learn')},
        probes=probes, selected_backend='official_aeon_torch_cpu_sklearn_numpy',numba_guard_benchmark=timings,numba_numerical_parity=True,numba_finite_scores=bool(np.isfinite(alternate_scores).all()),numba_candidate_index_parity=True,numba_compilation_inclusive_speedup=speedup,
        numba_fallback='NumPy retained: additional Numba compilation in every spawned worker dominates the measured short masking stage; official percentile/scaler remains NumPy',
        mlx_guard_benchmark=mlx_benchmark,gpu_fallback='official encoders/KNN stay canonical CPU; guarded fit benchmark must pass 1.1x including transfer and synchronization; MPS lacks float64',
        tensor_eligible=['masking','dense distance arithmetic'], candidate_index_parity=None, gpu_full_stage_seconds=None,
        gpu_speedup=None, concurrent_gpu_jobs=0, numba_existing='aeon MultiRocket and preprocessing retain their existing Numba kernels')


def smoke_antigravity(binding):
    torch.set_num_threads(1)
    rng=np.random.default_rng(SEED_ANTIGRAVITY)
    cases={'synthetic':rng.normal(size=(20,40)).astype(np.float32)}
    for dataset in SMOKE_CASES:
        cases[dataset], _ = prepare_normal_train_matrix_antigravity(DATABASE_PATH_ANTIGRAVITY,dataset)
    results={}
    with threadpool_limits(limits=1):
        for dataset,train in cases.items():
            query=train[:7].copy()
            results[dataset]={}
            for method in METHODS:
                first=score_pair_antigravity(train,query,method,seed=SEED_ANTIGRAVITY)
                second=score_pair_antigravity(train,query,method,seed=SEED_ANTIGRAVITY)
                assert first['representation_fits']==6
                candidate=MethodCandidateAntigravity(method,method,(SEED_ANTIGRAVITY,),(.005,))
                canonical=score_candidate_antigravity(train,query,candidate,seed=SEED_ANTIGRAVITY)
                for branch in BRANCHES:
                    for split in ('train_oof','test_scores'):
                        np.testing.assert_array_equal(first[branch][split],second[branch][split])
                for split in ('train_oof','test_scores'):
                    np.testing.assert_allclose(first['old'][split],getattr(canonical,split),rtol=1e-10,atol=1e-8)
                # Same finite-sample upper-tail order-statistic candidate decision.
                for source in (first['old'],dict(train_oof=canonical.train_oof,test_scores=canonical.test_scores)):
                    threshold=np.quantile(source['train_oof'],.995,method='higher')
                    source['candidate_indices']=np.flatnonzero(source['test_scores']>threshold)
                np.testing.assert_array_equal(first['old']['candidate_indices'],source['candidate_indices'])
                results[dataset][method]=dict(repeatable=True,canonical_parity=True,candidate_index_parity=True,
                    stats={b:score_stats_antigravity(first[b]['train_oof']) for b in BRANCHES},diagnostics=first['diagnostics'])
                print(json.dumps(dict(smoke=dataset,method=method,status='passed')),flush=True)
    acceleration=acceleration_antigravity()
    if binding_antigravity()!=binding: raise RuntimeError('binding changed during smoke')
    payload=dict(state='smoke_nonofficial',binding=binding,cases=results,acceleration=acceleration)
    write_json_atomic_antigravity(PACKAGE/'smoke_results_antigravity.json',jsonable_antigravity(payload))
    return payload


def summarize_antigravity(records):
    summary={}
    for method in METHODS:
        rows=[r['metrics'][method] for r in records]
        summary[method]={b:dict(mean_auc=float(np.mean([r[b]['auc'] for r in rows])),mean_zf=float(np.mean([r[b]['zf'] for r in rows])),total_zf=sum(r[b]['zf'] for r in rows)) for b in BRANCHES}
        for metric in ('auc','zf'):
            delta=np.array([r['stable'][metric]-r['old'][metric] for r in rows])
            summary[method][metric+'_paired']=dict(mean_delta=float(delta.mean()),wins=int((delta>0).sum()),losses=int((delta<0).sum()),ties=int((delta==0).sum()))
    return summary


def report_antigravity(result):
    rows=''.join('<tr><td>'+html.escape(m)+'</td>'+''.join(f'<td>{s[b][k]:.9f}</td>' if k=='mean_auc' else f'<td>{s[b][k]}</td>' for k in ('mean_auc','total_zf') for b in BRANCHES)+'</tr>' for m,s in result['summary'].items())
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><title>Exp492 안정화 결과</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:35px auto;padding:20px}table{border-collapse:collapse;width:100%}th,td{padding:12px;border:1px solid #ccd}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Exp492: 정규화 안정화와 평균 성능</h1><p>125개 family 대표, 두 방법별 기존/안정화 쌍대 평가. 모든 점수 저장 후 TEST 라벨 평가. 개발 집단 재평가이며 자동 승격하지 않습니다.</p><table><tr><th>방법</th><th>기존 평균 AUC</th><th>안정화 평균 AUC</th><th>기존 총 ZF</th><th>안정화 총 ZF</th></tr>'+rows+'</table><h2>평가 범위</h2><p>수치 증폭 개선이 AUC·ZF 개선을 보장하지 않습니다. ZF는 기존 TEST-normal-max를 사용하는 사후 평가 지표입니다. Exp490의 Ridge 또는 실제 라우팅 성능은 이 실험에서 재평가하지 않았습니다. 원래 점수·정규화 진단·코드 및 입력 해시는 실험 폴더에 있습니다.</p><details><summary>전체 결과와 계열별 변화</summary><pre>'+html.escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre></details></html>'


def run_antigravity(mode):
    started=perf_counter(); binding=binding_antigravity(); before=file_identity_antigravity(DATABASE_PATH_ANTIGRAVITY)
    if mode=='smoke': return smoke_antigravity(binding)
    result_path=PACKAGE/'results_antigravity.json'; refuse_official_overwrite_antigravity(result_path)
    smoke=require_matching_smoke_antigravity(PACKAGE/'smoke_results_antigravity.json',binding)
    workers=cpu_worker_policy_antigravity(); directory=PACKAGE/'score_checkpoints_antigravity'
    manifest=dict(state='scoring',started_at=datetime.now(timezone.utc).isoformat(),binding=binding,workers=workers,acceleration=smoke['acceleration'],errors=[],completed=0,labels_opened=False)
    manifest_path=PACKAGE/'run_manifest_antigravity.json'
    previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    resumed_evaluation=bool(previous.get('labels_opened'))
    if resumed_evaluation:
        if previous.get('binding')!=binding: raise RuntimeError('cannot change binding after TEST label boundary')
        manifest=previous
    else: write_json_atomic_antigravity(manifest_path,manifest)
    records=[]
    if not resumed_evaluation:
      with ProcessPoolExecutor(max_workers=workers['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
          futures={pool.submit(score_worker_antigravity,dict(**row,directory=str(directory),binding=binding)):row for row in binding['population']}
          for future in as_completed(futures):
              row=futures[future]
              try: records.append(future.result())
              except Exception as error: manifest['errors'].append(dict(**row,error_type=type(error).__name__,error=str(error)))
              manifest['completed']=len(records); manifest['seconds']=perf_counter()-started
              write_json_atomic_antigravity(manifest_path,jsonable_antigravity(manifest))
              print(json.dumps(dict(completed=len(records),errors=len(manifest['errors']),dataset=row['dataset'],seconds=manifest['seconds'])),flush=True)
    if manifest['errors'] or (not resumed_evaluation and len(records)!=125):
        manifest['state']='failed_before_labels'; write_json_atomic_antigravity(manifest_path,manifest)
        raise RuntimeError('incomplete population; TEST evaluation refused')
    if binding_antigravity()!=binding or file_identity_antigravity(DATABASE_PATH_ANTIGRAVITY)!=before: raise RuntimeError('input/code changed during scoring')
    # Validate every score artifact before any TEST label query anywhere in this run.
    ready=[]
    for row in binding['population']:
        checkpoint=load_antigravity(directory,row['dataset'],binding)
        if checkpoint is None or checkpoint[0]['source_family']!=row['source_family']: raise ValueError('missing/misaligned population')
        with closing(readonly_sqlite_antigravity(DATABASE_PATH_ANTIGRAVITY)) as connection:
            prepared=prepare_dataset_values_antigravity(connection,row['dataset'])
        validate_alignment_antigravity(checkpoint[0],prepared)
        ready.append(checkpoint)
    population_sha=sha256_json_antigravity([{k:r[k] for k in ('dataset','score_sha256','diagnostics_sha256')} for r,_ in ready])
    if resumed_evaluation and previous.get('score_population_sha256')!=population_sha:
        raise RuntimeError('score population changed after TEST label boundary')
    manifest.update(state='evaluating',score_population_sha256=population_sha,all_scores_verified_at=datetime.now(timezone.utc).isoformat(),labels_opened=True)
    write_json_atomic_antigravity(manifest_path,manifest)
    evaluated=[]
    with closing(readonly_sqlite_antigravity(DATABASE_PATH_ANTIGRAVITY)) as connection:
        for record,arrays in ready:
            labels=load_retained_test_labels_antigravity(connection,record['dataset'],tuple(record['test_indices']))
            evaluated.append({k:v for k,v in record.items() if k not in ('binding','test_indices')} | dict(metrics={m:{b:evaluate_score_vector_antigravity(arrays[f'{m}__{b}__test_scores'],labels) for b in BRANCHES} for m in METHODS}))
    if binding_antigravity()!=binding or file_identity_antigravity(DATABASE_PATH_ANTIGRAVITY)!=before: raise RuntimeError('input/code changed during evaluation')
    result=jsonable_antigravity(dict(state='complete_development_evaluation',family_count=125,failures=0,summary=summarize_antigravity(evaluated),records=evaluated,binding_sha256=sha256_json_antigravity(binding),seconds=perf_counter()-started))
    write_json_atomic_antigravity(result_path,result)
    report=ROOT_ANTIGRAVITY/'docs/EXP492_STABLE_NORMALIZATION_antigravity.html'; report.write_text(report_antigravity(result))
    manifest.update(state='complete',seconds=perf_counter()-started,result_sha256=file_sha256_antigravity(result_path),report_sha256=file_sha256_antigravity(report))
    write_json_atomic_antigravity(manifest_path,manifest)
    print(json.dumps(result['summary']),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--mode',choices=('smoke','official'),required=True)
    run_antigravity(parser.parse_args().mode)
