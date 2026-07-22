from __future__ import annotations

import argparse, csv, hashlib, json, os, tempfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from experiments.exp206_210.core import empirical_upper_pvalues
from experiments.exp226_235_hard_alert_quality.baseline_exp137 import load_exp137_baseline
from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation, load_values_bundle, sha256_file
from experiments.exp226_235_hard_alert_quality.preprocessing import align_bundle
from experiments.exp238_rocket_dimension_confidence.core import build_confidence_lanes
from run_experiment_26_rocket import make_kernels, rocket_transform


DATA_ROOT=Path('/Users/minho/Documents/Dataset'); DB_PATH=DATA_ROOT/'univariate_ts.db'
OUTPUT_ROOT=DATA_ROOT/'exp238_rocket_dimension_confidence'
REPO_OUTPUT=Path(__file__).resolve().parents[2]/'outputs'/'exp238_rocket_dimension_confidence'
EXP236_ROOT=DATA_ROOT/'exp236_detection_coverage_expansion'; EXP237_ROOT=DATA_ROOT/'exp237_new_score_source_qualification'
EXPERIMENT_ID='experiment_238_rocket_dimension_confidence'; ALPHAS=(0.005,0.01,0.02,0.05)
BASE_SEED=20260706; STABILITY_SEED=20261706; MAX_REFERENCE=1000; MAX_LENGTH=2048
SOURCES=('rocket_256','rocket_512','rocket_1024','rocket_256_seed2')
EXTERNAL=DATA_ROOT/'rank_dashboard_external_live_run.json'; HEARTBEAT=DATA_ROOT/'rank_experiments_dashboard_heartbeat.json'


def _atomic(path:Path,data:bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as h:
        tmp=Path(h.name); h.write(data); h.flush(); os.fsync(h.fileno())
    os.replace(tmp,path)


def _json(path,payload): _atomic(path,(json.dumps(payload,sort_keys=True,indent=2)+'\n').encode())


def _status(root,status,phase,done,total,current=None,errors=0):
    now=time.time(); p={'id':EXPERIMENT_ID,'experiment_id':EXPERIMENT_ID,'label':'Exp238 - ROCKET dimension confidence','status':status,'phase':phase,'pid':os.getpid(),'child_pid':os.getpid(),'datasets_done':done,'done':done,'expected_datasets':total,'expected':total,'progress_percent':100*done/max(1,total),'current_dataset':current,'errors':errors,'updated_unix':now,'updated_at':time.strftime('%Y-%m-%d %H:%M:%S'),'detail_csv':str(root/'experiment_238_results.csv'),'stdout_log':str(root/'exp238_full.log'),'scope':'retrospective_confidence_tier_audit','autonomous_metric':'frozen_exp236_unchanged','review_metric':'priority_strata_only'}
    _json(root/'status.json',p); _json(EXTERNAL,p); _json(HEARTBEAT,p)


def _names(limit=None):
    names=tuple(sorted(load_exp137_baseline()))
    if len(names)!=1117: raise ValueError('population coverage mismatch')
    return names[:limit] if limit else names


def _z(x):
    x=np.asarray(x,dtype=np.float32); mean=x.mean(1,keepdims=True); std=x.std(1,keepdims=True)
    return (x-mean)/np.where(std>1e-8,std,1.0)


def _reference(x,target,dataset):
    if len(x)<=MAX_REFERENCE:return x
    seed=BASE_SEED+target+len(x)
    return x[np.sort(np.random.default_rng(seed).choice(len(x),MAX_REFERENCE,replace=False))]


def _fit_local_gap(features):
    scaler=RobustScaler(quantile_range=(10,90)); z=scaler.fit_transform(features)
    k=max(1,min(4,len(z))); nn=NearestNeighbors(n_neighbors=k).fit(z); dist,idx=nn.kneighbors(z)
    if dist.shape[1]>1: radius=dist[:,1:].mean(1); neighbors=idx[:,1:]
    else: radius=dist[:,0]; neighbors=idx
    local=radius[neighbors].mean(1)+1e-9; scale=np.median(radius)+1e-9
    train=(radius-local)/scale
    return scaler,NearestNeighbors(n_neighbors=max(1,min(3,len(z)))).fit(z),radius,scale,train


def _score(model,features):
    scaler,nn,radius,scale,_=model; dist,idx=nn.kneighbors(scaler.transform(features))
    return (dist.mean(1)-(radius[idx].mean(1)+1e-9))/scale


def _stem(name):return hashlib.sha256(name.encode()).hexdigest()[:20]


def _build_one(name,root_text,reset=False):
    root=Path(root_text); npz=root/'cache'/f'{_stem(name)}.npz'; meta=root/'cache'/f'{_stem(name)}.json'
    if npz.exists() and meta.exists() and not reset:
        m=json.loads(meta.read_text());
        if sha256_file(npz)==m['cache_sha256']:return m
    b=align_bundle(load_values_bundle(name,DB_PATH)); train=_z(np.asarray(b.train_normal)); test=_z(np.asarray(b.test_values)); ntrain=len(train)
    if ntrain<5:
        arrays={f'{s}__train':np.array([],float) for s in SOURCES}|{f'{s}__test':np.array([],float) for s in SOURCES}
    else:
        ref=_reference(train,int(b.target_length),name)
        kernels=make_kernels(train.shape[1],num_kernels=1024,seed=BASE_SEED)
        seed2=make_kernels(train.shape[1],num_kernels=256,seed=STABILITY_SEED)
        base_train=rocket_transform(ref,kernels); seed_train=rocket_transform(ref,seed2)
        models={'rocket_256':_fit_local_gap(base_train[:,:512]),'rocket_512':_fit_local_gap(base_train[:,:1024]),'rocket_1024':_fit_local_gap(base_train),'rocket_256_seed2':_fit_local_gap(seed_train)}
        arrays={f'{s}__train':np.asarray(models[s][4],float) for s in SOURCES}; parts={s:[] for s in SOURCES}
        for start in range(0,len(test),128):
            chunk=test[start:start+128]; base=rocket_transform(chunk,kernels); second=rocket_transform(chunk,seed2)
            feats={'rocket_256':base[:,:512],'rocket_512':base[:,:1024],'rocket_1024':base,'rocket_256_seed2':second}
            for s in SOURCES:parts[s].append(_score(models[s],feats[s]))
        arrays.update({f'{s}__test':np.concatenate(parts[s]) for s in SOURCES})
    npz.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=npz.parent,suffix='.npz',delete=False) as h:tmp=Path(h.name)
    np.savez_compressed(tmp,**arrays); os.replace(tmp,npz)
    m={'dataset':name,'cache_file':str(npz.relative_to(root)),'n_train':ntrain,'n_test':len(test),'target_length':int(b.target_length),'abstained':int(ntrain<5),'base_seed':BASE_SEED,'stability_seed':STABILITY_SEED,'contains_test_labels':False,'cache_sha256':sha256_file(npz)}
    _json(meta,m);m['manifest_file']=str(meta.relative_to(root));m['manifest_sha256']=sha256_file(meta);return m


def _write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def cache(root,limit,workers,reset):
    names=_names(limit);rows=[];errors=[];_status(root,'running','cache',0,len(names))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fs={ex.submit(_build_one,n,str(root),reset):n for n in names}
        for done,f in enumerate(as_completed(fs),1):
            n=fs[f]
            try:rows.append(f.result())
            except Exception as e:errors.append({'dataset':n,'error':repr(e)})
            _status(root,'running','cache',done,len(names),n,len(errors));print(f'progress {done}/{len(names)} last={n} errors={len(errors)}',flush=True)
    _write_csv(root/'cache_manifest.csv',sorted(rows,key=lambda r:r['dataset']))
    if errors:_write_csv(root/'cache_errors.csv',errors);raise RuntimeError(f'cache errors {len(errors)}')


def _load_exp236():
    return {r['dataset']:r for r in map(json.loads,(EXP236_ROOT/'exp236_frozen_predictions.jsonl').read_text().splitlines())}


def _load_exp237():
    out={}
    for r in map(json.loads,(EXP237_ROOT/'exp237_frozen_predictions.jsonl').read_text().splitlines()):
        if r['source'] in {'shrinkage_normal_space','elastic_normal_prototype'}:out[(r['dataset'],r['source'],float(r['alpha']))]=set(map(int,r['candidate_indices']))
    return out


def predict(root,limit):
    names=_names(limit);base=_load_exp236();aux=_load_exp237();rows=[]
    for done,n in enumerate(names,1):
        m=json.loads((root/'cache'/f'{_stem(n)}.json').read_text()); hard=set(base[n]['hard_indices']);standard=set(base[n]['standard_review_indices']);priority=set(base[n]['priority_review_indices'])
        with np.load(root/m['cache_file']) as z:
            p={s:{a:set() for a in ALPHAS} for s in SOURCES}
            if not m['abstained']:
                for s in SOURCES:
                    pv=empirical_upper_pvalues(z[f'{s}__train'],z[f'{s}__test'])
                    for a in ALPHAS:p[s][a]=set(np.flatnonzero(pv<=a).astype(int).tolist())
        for a in ALPHAS:
            independent=aux.get((n,'shrinkage_normal_space',a),set())|aux.get((n,'elastic_normal_prototype',a),set())
            lanes=build_confidence_lanes(hard=hard,standard=standard,priority=priority,rocket_256=p['rocket_256'][a],rocket_512=p['rocket_512'][a],rocket_1024=p['rocket_1024'][a],rocket_256_seed2=p['rocket_256_seed2'][a],independent_auxiliary=independent)
            rows.append({'experiment_id':EXPERIMENT_ID,'dataset':n,'alpha':a,'n_test':m['n_test'],'calibration_abstained':bool(m['abstained']),'hard_indices':sorted(hard),'standard_indices':sorted(standard),'priority_indices':sorted(priority),'rocket_256_indices':sorted(p['rocket_256'][a]),'rocket_512_indices':sorted(p['rocket_512'][a]),'rocket_1024_indices':sorted(p['rocket_1024'][a]),'rocket_256_seed2_indices':sorted(p['rocket_256_seed2'][a]),'rocket_stable_indices':sorted(lanes.rocket_stable),'independent_auxiliary_indices':sorted(lanes.independent_auxiliary),'confidence_a_indices':sorted(lanes.confidence_a),'confidence_b_indices':sorted(lanes.confidence_b),'confidence_c_indices':sorted(lanes.confidence_c),'review_priority_1_indices':sorted(lanes.review_priority_1),'review_priority_2_indices':sorted(lanes.review_priority_2),'review_priority_3_indices':sorted(lanes.review_priority_3),'contains_test_labels':False,'contains_evaluation_metrics':False,'hard_lane_changed':False,'standard_lane_changed':False,'priority_lane_changed':False})
        _status(root,'running','predict',done,len(names),n)
    data=b''.join((json.dumps(r,sort_keys=True,separators=(',',':'))+'\n').encode() for r in rows);_atomic(root/'exp238_frozen_predictions.jsonl',data);_json(root/'exp238_prediction_manifest.json',{'rows':len(rows),'sha256':hashlib.sha256(data).hexdigest(),'contains_test_labels':False,'contains_evaluation_metrics':False});return rows


def _metric(indices,truth):
    s=set(indices);tp=len(s&truth);fp=len(s-truth);return len(s),tp,fp,tp/max(1,len(s))


def evaluate(root,limit):
    manifest=json.loads((root/'exp238_prediction_manifest.json').read_text());data=(root/'exp238_frozen_predictions.jsonl').read_bytes()
    if hashlib.sha256(data).hexdigest()!=manifest['sha256']:raise ValueError('prediction hash mismatch')
    rows=[json.loads(x) for x in data.splitlines()];out=[]
    fields=('hard','confidence_a','confidence_b','confidence_c','review_priority_1','review_priority_2','review_priority_3','rocket_256','rocket_512','rocket_1024','rocket_256_seed2','rocket_stable')
    for r in rows:
        labels=load_test_labels_for_evaluation(r['dataset']);truth=set(np.flatnonzero(labels==1).astype(int));e=dict(r)
        for f in fields:
            count,tp,fp,precision=_metric(r[f'{f}_indices'],truth);e.update({f'{f}_count':count,f'{f}_tp':tp,f'{f}_fp':fp,f'{f}_precision':precision})
        out.append(e)
    summary=[]
    for a in ALPHAS:
        g=[r for r in out if float(r['alpha'])==a];row={'experiment_id':EXPERIMENT_ID,'alpha':a,'datasets':len(g),'abstained_datasets':sum(r['calibration_abstained'] for r in g),'hard_lane_changed_datasets':sum(r['hard_lane_changed'] for r in g),'standard_lane_changed_datasets':sum(r['standard_lane_changed'] for r in g)}
        for f in fields:
            tp=sum(r[f'{f}_tp'] for r in g);fp=sum(r[f'{f}_fp'] for r in g);row.update({f'{f}_count':tp+fp,f'{f}_tp':tp,f'{f}_fp':fp,f'{f}_precision':tp/max(1,tp+fp)})
        row.update({'policy_selected_from_test_metrics':0,'comparison_eligible':0,'hard_policy':'frozen_exp236_unchanged'});summary.append(row)
    _write_csv(root/'experiment_238_results.csv',out);_write_csv(root/'experiment_238_summary.csv',summary);_status(root,'completed','completed',len(_names(limit)),len(_names(limit)));return out,summary


def copy_outputs(root):
    REPO_OUTPUT.mkdir(parents=True,exist_ok=True)
    for n in ('exp238_prediction_manifest.json','experiment_238_summary.csv','status.json'):(REPO_OUTPUT/n).write_bytes((root/n).read_bytes())


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('cache','predict','evaluate','all'),default='all');p.add_argument('--output-root',type=Path,default=OUTPUT_ROOT);p.add_argument('--dataset-limit',type=int);p.add_argument('--workers',type=int,default=7);p.add_argument('--reset-cache',action='store_true');a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
    if a.stage in ('cache','all'):cache(a.output_root,a.dataset_limit,a.workers,a.reset_cache)
    if a.stage in ('predict','all'):predict(a.output_root,a.dataset_limit)
    if a.stage in ('evaluate','all'):evaluate(a.output_root,a.dataset_limit);copy_outputs(a.output_root)


if __name__=='__main__':main()
