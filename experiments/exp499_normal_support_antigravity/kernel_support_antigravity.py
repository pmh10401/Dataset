"""Normal-only hard and RBF soft support in frozen OAS feature geometry."""
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.svm import OneClassSVM
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix,OASGeometry
NAMES=('MiniRocket','oas_distance','oas_nearest','kernel_g025_n001','kernel_g025_n010','kernel_g4_n001','kernel_g4_n010')

def log_support(squared_distances,gamma,weights):
    d=matrix(squared_distances);w=np.asarray(weights,float)
    if w.shape!=(d.shape[1],) or not np.isfinite(w).all() or np.any(w<=0) or not np.isfinite(gamma) or gamma<=0 or np.any(d<0):raise ValueError('nonnegative distances and positive finite kernel weights required')
    logs=np.log(w);logs-=logsumexp(logs)
    scores=np.maximum(-logsumexp(logs[None,:]-gamma*d,axis=1),0)
    if not np.isfinite(scores).all():raise ValueError('nonfinite support score')
    return scores

class KernelHeads:
    names=NAMES
    def __init__(self,features):
        x=matrix(features);self.control=LocalGapModelAntigravity(x);z=self.control.scaler.transform(x)
        self.geometry=OASGeometry(z);self.reference=self.geometry.transform(z);self.models={}
        if len(x)<16:return
        d=cdist(self.reference,self.reference,'sqeuclidean');pairs=d[np.triu_indices(len(x),1)];positive=pairs[pairs>1e-12]
        if not len(positive):return
        scale=float(np.median(positive))
        for gs,factor in (('025',.25),('4',4.)):
            gamma=factor/scale;gram=np.exp(-gamma*d)
            for ns,nu in (('001',.01),('010',.1)):
                model=OneClassSVM(kernel='precomputed',nu=nu,tol=1e-5,max_iter=100000,cache_size=128).fit(gram)
                if model.fit_status_!=0 or not np.isfinite(model.dual_coef_).all() or np.any(model.dual_coef_<=0):raise ValueError('kernel fit did not produce converged positive support weights')
                self.models[f'kernel_g{gs}_n{ns}']=(gamma,model)
    def score(self,features):
        x=matrix(features);z=self.geometry.transform(self.control.scaler.transform(x));d=cdist(z,self.reference,'sqeuclidean')
        ordered=np.sort(d,axis=1);anchor=np.sqrt(ordered[:,:min(3,len(self.reference))]).mean(axis=1)
        result=dict(MiniRocket=self.control.score(x),oas_distance=anchor,oas_nearest=np.sqrt(ordered[:,0]))
        for name in self.names[3:]:
            if name not in self.models:result[name]=anchor
            else:
                gamma,model=self.models[name];result[name]=log_support(d[:,model.support_],gamma,model.dual_coef_.ravel())
        if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite score')
        return result
