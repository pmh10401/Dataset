"""Prediction and routing only; evaluation is a separate persisted-artifact stage."""
import numpy as np
from sklearn.linear_model import Ridge
from experiments.exp491_gain_prediction_antigravity.gain_prediction_antigravity import (
    _family_weights_antigravity, _donor_medians_antigravity, _impute_antigravity,
    _standardize_antigravity, _apply_standardize_antigravity, _weighted_mean_antigravity,
)

def predict_antigravity(features, targets, families):
    x=np.asarray(features,dtype=float); y=np.asarray(targets,dtype=float); f=np.asarray(families)
    if x.ndim!=2 or y.shape!=(len(x),6) or f.shape!=(len(x),) or len(set(f))<2:
        raise ValueError('aligned features, six targets and at least two families required')
    if np.isinf(x).any() or not np.isfinite(y).all(): raise ValueError('invalid values')
    pred=np.empty_like(y); constant=np.empty_like(y)
    for held in sorted(set(f)):
        donor=f!=held; query=~donor
        median=_donor_medians_antigravity(x[donor])
        dx,mean,std=_standardize_antigravity(_impute_antigravity(x[donor],median))
        qx=_apply_standardize_antigravity(_impute_antigravity(x[query],median),mean,std)
        weights=_family_weights_antigravity(f[donor])
        pred[query]=Ridge(alpha=10.,fit_intercept=True,solver='cholesky').fit(dx,y[donor],sample_weight=weights).predict(qx)
        constant[query]=_weighted_mean_antigravity(y[donor],weights)
    if not np.isfinite(pred).all(): raise ValueError('nonfinite predictions')
    return pred,constant

def routes_antigravity(predictions):
    p=np.asarray(predictions,dtype=float)
    if p.ndim!=2 or p.shape[1]!=6 or not np.isfinite(p).all(): raise ValueError('finite six-column predictions required')
    return {name:np.argmax(np.column_stack((np.zeros(len(p)),p[:,start:start+3])),axis=1)
            for name,start in (('auc',0),('zf_sign',3))}

def summarize_route_antigravity(route, outcomes):
    o=np.asarray(outcomes,dtype=float); r=np.asarray(route)
    if o.shape!=(len(r),4,2) or r.dtype.kind not in 'iu' or np.any((r<0)|(r>3)) or not np.isfinite(o).all():
        raise ValueError('invalid routes or outcomes')
    chosen=o[np.arange(len(r)),r]; delta=chosen-o[:,0]
    return dict(mean_auc=float(chosen[:,0].mean()),total_zf=int(chosen[:,1].sum()),
        delta_auc_vs_mini=float(delta[:,0].mean()),delta_zf_vs_mini=int(delta[:,1].sum()),
        auc_wins=int((delta[:,0]>1e-12).sum()),auc_losses=int((delta[:,0]<-1e-12).sum()),
        zf_wins=int((delta[:,1]>0).sum()),zf_losses=int((delta[:,1]<0).sum()),
        method_counts=np.bincount(r,minlength=4).tolist())
