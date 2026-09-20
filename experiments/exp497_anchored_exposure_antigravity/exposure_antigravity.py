"""Bounded synthetic exposure added to an immutable positive normal-distance anchor."""
import numpy as np
from sklearn.linear_model import Ridge
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix,OASGeometry,NeighborGeometry
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import SEED
from experiments.exp495_normal_geometry_antigravity.pilot_antigravity import perturbations,KINDS
NAMES=('MiniRocket','oas_distance','oe_a1_g025','oe_a1_g050','oe_a100_g025','oe_a100_g050')

def bounded_exposure(prediction,center,scale):
    x=np.asarray(prediction,float)
    if not np.isfinite(x).all() or not np.isfinite(center) or not np.isfinite(scale) or scale<0:raise ValueError('finite predictions and nonnegative scale required')
    if scale<1e-6:return np.zeros_like(x)
    positive=np.maximum(x-center,0);out=positive/(scale+positive)
    if not np.isfinite(out).all():raise ValueError('nonfinite bounded exposure')
    return out

class ExposureHeads:
    names=NAMES
    def __init__(self,features,parents,synthetics):
        x=matrix(features);self.control=LocalGapModelAntigravity(x)
        z=self.control.scaler.transform(x);self.geometry=OASGeometry(z)
        self.neighbors=NeighborGeometry(self.geometry.transform(z));self.regressors={}
        self.parents=np.asarray(parents)
        if len(x)<16:return
        if self.parents.ndim!=1 or not np.issubdtype(self.parents.dtype,np.integer) or len(self.parents)!=min(len(x),128) or not np.array_equal(self.parents,np.unique(self.parents)) or np.any(self.parents<0) or np.any(self.parents>=len(x)):raise ValueError('invalid fit-normal parent indices')
        syn=matrix(synthetics);n=len(self.parents)
        if syn.shape!=(4*n,x.shape[1]):raise ValueError('four synthetic kinds per fit parent required')
        design=np.vstack((z[self.parents],self.control.scaler.transform(syn)))
        target=np.r_[np.zeros(n),np.ones(4*n)];weights=np.r_[np.full(n,1/n),np.full(4*n,1/(4*n))]
        for alpha in (1,100):
            model=Ridge(alpha=alpha,solver='cholesky',fit_intercept=True).fit(design,target,sample_weight=weights)
            normal=model.predict(z);center=float(np.median(normal));scale=float(1.4826*np.median(abs(normal-center)))
            if not np.isfinite(model.coef_).all() or not np.isfinite(normal).all():raise ValueError('nonfinite exposure fit')
            self.regressors[alpha]=(model,center,scale)
    def score(self,features):
        x=matrix(features);z=self.control.scaler.transform(x)
        anchor=self.neighbors.score(self.geometry.transform(z))[1]/self.neighbors.scale
        result={'MiniRocket':self.control.score(x),'oas_distance':anchor}
        for alpha in (1,100):
            if alpha in self.regressors:
                model,center,scale=self.regressors[alpha];term=bounded_exposure(model.predict(z),center,scale)
            else:term=np.zeros(len(x))
            for suffix,gamma in (('025',.25),('050',.5)):result[f'oe_a{alpha}_g{suffix}']=anchor+gamma*term
        return result

def fit_heads(encoder,rows,features):
    if len(rows)!=len(features):raise ValueError('unaligned fit rows and features')
    if len(rows)<16:return ExposureHeads(features,np.array([],int),np.empty((0,features.shape[1])))
    parents=np.sort(np.random.default_rng(SEED+497).choice(len(rows),min(len(rows),128),replace=False))
    proxies=perturbations(rows[parents],SEED+1497)
    synthetic_features=np.vstack([encoder.transform(proxies[k]) for k in KINDS])
    return ExposureHeads(features,parents,synthetic_features)
