"""Frozen-donor prediction and retrospective low-FPR aggregation."""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score,roc_curve
from experiments.exp491_gain_prediction_antigravity.gain_prediction_antigravity import (
    _family_weights_antigravity,_donor_medians_antigravity,_impute_antigravity,
    _standardize_antigravity,_apply_standardize_antigravity,
)
GRID_NUM=np.array([0,1,5,10]); GRID=GRID_NUM/1000

def predict_queries(x,y,families,q,query_families):
    x=np.asarray(x,float);y=np.asarray(y,float);f=np.asarray(families)
    q=np.asarray(q,float);g=np.asarray(query_families)
    if x.ndim!=2 or q.ndim!=2 or x.shape[1]!=q.shape[1] or y.shape!=(len(x),6) or len(g)!=len(q) or len(f)!=len(x):raise ValueError('unaligned query/training arrays')
    if np.isinf(x).any() or np.isinf(q).any() or not np.isfinite(y).all():raise ValueError('invalid arrays')
    pred=np.empty((len(q),6))
    for held in sorted(set(g)):
        donor=f!=held;query=g==held
        if not donor.any():raise ValueError('no other-family donors')
        median=_donor_medians_antigravity(x[donor])
        dx,mean,std=_standardize_antigravity(_impute_antigravity(x[donor],median))
        qx=_apply_standardize_antigravity(_impute_antigravity(q[query],median),mean,std)
        pred[query]=Ridge(alpha=10.,fit_intercept=True,solver='cholesky').fit(dx,y[donor],sample_weight=_family_weights_antigravity(f[donor])).predict(qx)
    if not np.isfinite(pred).all():raise ValueError('nonfinite prediction')
    return pred

def metric(y,s):
    y=np.asarray(y);s=np.asarray(s,float)
    if y.ndim!=1 or s.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(s).all():raise ValueError('invalid score/label vectors')
    neg=s[y==0];pos=s[y==1]
    if not len(neg) or not len(pos):raise ValueError('both TEST classes required')
    budget=len(neg)*GRID_NUM//1000;ordered=np.sort(neg)[::-1]
    thresholds=ordered[budget]
    tp=np.array([(pos>v).sum() for v in thresholds],int)
    fp=np.array([(neg>v).sum() for v in thresholds],int)
    fpr,tpr,_=roc_curve(y,s,drop_intermediate=False)
    observed=np.rint(fpr*len(neg)).astype(int)
    independent=np.array([tpr[observed<=b].max() for b in budget])
    np.testing.assert_allclose(tp/len(pos),independent,rtol=0,atol=1e-15)
    assert np.all(fp<=budget) and np.all(np.diff(tp)>=0)
    return dict(auc=float(roc_auc_score(y,s)),zf=int(tp[0]),tp=tp.tolist(),fp=fp.tolist(),n_normal=len(neg),n_anomaly=len(pos),tpr=(tp/len(pos)).tolist())

def summarize(records):
    if not records:raise ValueError('empty population')
    families=sorted({r['source_family'] for r in records})
    tpr=np.array([r['metrics']['tpr'] for r in records]); auc=np.array([r['metrics']['auc'] for r in records])
    family=np.array([r['source_family'] for r in records])
    family_tpr=np.array([tpr[family==f].mean(axis=0) for f in families])
    npositive=sum(r['metrics']['n_anomaly'] for r in records)
    return dict(datasets=len(records),families=len(families),family_macro_tpr=family_tpr.mean(axis=0).tolist(),dataset_macro_tpr=tpr.mean(axis=0).tolist(),
        micro_tpr=(np.array([r['metrics']['tp'] for r in records]).sum(axis=0)/npositive).tolist(),
        total_zf=sum(r['metrics']['zf'] for r in records),family_macro_auc=float(np.mean([auc[family==f].mean() for f in families])),dataset_macro_auc=float(auc.mean()),
        n_anomaly=npositive,n_normal=sum(r['metrics']['n_normal'] for r in records))
