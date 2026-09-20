"""Frozen score/descriptor extension; no TEST-label access in this module."""
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from contextlib import closing
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import (
    readonly_sqlite_antigravity, DATABASE_PATH_ANTIGRAVITY as DB,
    SEED_ANTIGRAVITY as SEED, file_sha256_antigravity as sha,
    write_json_atomic_antigravity as write, jsonable_antigravity as jsonable,
)
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import prepare_dataset_values_antigravity
from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import METHODS_ANTIGRAVITY as METHODS
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
    _REQUESTED_KERNELS_ANTIGRAVITY as KERNELS, _reference_indices_antigravity,
    _feature_matrix_antigravity, _score_chunks_antigravity, crossfit_plan_antigravity,
)
from experiments.exp489_lof_score_head_antigravity.lof_score_head_antigravity import (
    score_heads_antigravity, _fit_paired_heads_antigravity, _score_paired_chunks_antigravity,
)
from experiments.exp490_behavior_descriptors_antigravity.behavior_descriptors_antigravity import (
    assemble_behavior_antigravity, b_reference_indices_antigravity,
)
from experiments.exp492_stable_normalization_antigravity.scoring_antigravity import (
    score_pair_antigravity, _make_encoder_antigravity, _run_fold_antigravity,
)
from experiments.exp493_stable_meta_routing_antigravity.behavior_antigravity import (
    _oof_multirocket_a_antigravity, score_multirocket_test_antigravity,
)

P=Path(__file__).resolve().parent
SCORE_NAMES=('MiniRocket','MultiRocket','Hydra stable','Hydra+MultiRocket stable','MiniRocket+LOF','Hydra old','Hydra+MultiRocket old')

def score_and_describe(train,test):
    torch.set_num_threads(1)
    train=np.asarray(train,dtype=np.float32); test=np.asarray(test,dtype=np.float32)
    if train.ndim!=2 or test.ndim!=2 or len(train)<2 or train.shape[1]!=test.shape[1]:raise ValueError('aligned TRAIN>=2 required')
    if not np.isfinite(train).all() or not np.isfinite(test).all():raise ValueError('finite inputs required')
    eligible=len(train)>=5
    if eligible:heads=score_heads_antigravity(train,test,seed=SEED)
    else:
        refs=_reference_indices_antigravity(np.arange(len(train)),seed=SEED+7919+KERNELS[METHODS[0]])
        rep=make_representation_antigravity(METHODS[0],SEED)
        control,lof,dim,_=_fit_paired_heads_antigravity(rep,train[refs])
        c,l=_score_paired_chunks_antigravity(rep,control,lof,test,feature_dimension=dim)
        heads={'control':{'test_scores':c},'lof':{'test_scores':l}}
    arrays={'test__0':heads['control']['test_scores'],'test__4':heads['lof']['test_scores'],
            'test__1':score_multirocket_test_antigravity(train,test,seed=SEED)}
    old_a={}; stable_a={}
    if eligible:
        old_a[METHODS[0]]=heads['control']['train_oof']
        old_a[METHODS[1]]=_oof_multirocket_a_antigravity(train,seed=SEED)[0]
        stable_a.update(old_a)
    for j,method in enumerate(METHODS[2:],2):
        if eligible:
            pair=score_pair_antigravity(train,test,method,seed=SEED)
            old_test=pair['old']['test_scores'];stable_test=pair['stable']['test_scores']
            old_a[method]=pair['old']['train_oof'];stable_a[method]=pair['stable']['train_oof']
        else:
            refs=_reference_indices_antigravity(np.arange(len(train)),seed=SEED+7919+KERNELS[method])
            old_test,stable_test,_=_run_fold_antigravity(_make_encoder_antigravity(method,SEED),train[refs],test,fold=5)
        arrays[f'test__{j}']=stable_test;arrays[f'test__{j+3}']=old_test
    if eligible:
        old_b={m:np.empty(len(train)) for m in METHODS};stable_b={m:np.empty(len(train)) for m in METHODS}
        for method in METHODS:
            for fold,(fit,query) in enumerate(crossfit_plan_antigravity(len(train),seed=SEED)):
                a=_reference_indices_antigravity(fit,seed=SEED+1009*(fold+1)+KERNELS[method])
                b=b_reference_indices_antigravity(a,seed=SEED,fold_index=fold)
                assert not np.intersect1d(b,query).size
                if method in METHODS[2:]:
                    old,stable,_=_run_fold_antigravity(_make_encoder_antigravity(method,SEED),train[b],train[query],fold=fold)
                else:
                    rep=make_representation_antigravity(method,SEED)
                    features=_feature_matrix_antigravity(rep.fit_transform(train[b]))
                    old=_score_chunks_antigravity(rep,LocalGapModelAntigravity(features,neighbors=3),train[query],feature_dimension=features.shape[1])
                    stable=old
                old_b[method][query]=old;stable_b[method][query]=stable
        arrays['old18']=np.asarray(assemble_behavior_antigravity(old_a,old_b)[0])
        arrays['stable18']=np.asarray(assemble_behavior_antigravity(stable_a,stable_b)[0])
    return arrays

def load(directory,dataset,binding):
    meta=Path(directory)/f'{dataset}_meta_antigravity.json'
    if not meta.exists():return None
    record=json.loads(meta.read_text());npz=meta.with_name(f'{dataset}_scores_antigravity.npz')
    if record['binding_sha256']!=binding or record['dataset']!=dataset or sha(npz)!=record['score_sha256']:raise ValueError('stale checkpoint')
    with np.load(npz,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
    expected={f'test__{j}' for j in range(7)} | ({'old18','stable18'} if record['n_train']>=5 else set())
    if set(arrays)!=expected:raise ValueError('wrong checkpoint keys')
    for k,v in arrays.items():
        if v.shape!=((18,) if k.endswith('18') else (record['n_test'],)) or not np.isfinite(v).all():raise ValueError('invalid checkpoint vectors')
    if len(record['test_indices'])!=record['n_test'] or len(set(record['test_indices']))!=record['n_test']:raise ValueError('invalid indices')
    return record,arrays

def persist(directory,dataset,arrays,record):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    file=directory/f'{dataset}_scores_antigravity.npz';tmp=directory/f'{dataset}_temporary_antigravity.npz'
    np.savez_compressed(tmp,**arrays);tmp.replace(file)
    meta=dict(dataset=dataset,score_sha256=sha(file),**record)
    write(directory/f'{dataset}_meta_antigravity.json',jsonable(meta))
    return meta

def worker(payload):
    dataset,family,directory,binding=payload
    found=load(directory,dataset,binding)
    if found:return found[0]
    start=perf_counter()
    with closing(readonly_sqlite_antigravity(DB)) as con,threadpool_limits(limits=1):
        prepared=prepare_dataset_values_antigravity(con,dataset)
        arrays=score_and_describe(prepared.train_rows,prepared.test_rows)
    return persist(directory,dataset,arrays,dict(binding_sha256=binding,source_family=family,
        n_train=len(prepared.train_rows),n_test=len(prepared.test_rows),target_length=prepared.target_length,
        test_indices=list(prepared.test_indices),origin='fresh_frozen_algorithms',seconds=perf_counter()-start))
