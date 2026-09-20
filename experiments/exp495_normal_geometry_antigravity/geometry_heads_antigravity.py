"""Full-space OAS whitening via the smaller TRAIN Gram matrix."""
import numpy as np
from scipy.spatial.distance import cdist
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity

def matrix(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():raise ValueError('finite nonempty matrix required')
    return x

class OASGeometry:
    def __init__(self,reference):
        x=matrix(reference);n,p=x.shape
        if n<2:raise ValueError('at least two TRAIN rows required')
        self.center=x.mean(axis=0);x=x-self.center
        gram=x@x.T/n;mu=float(np.trace(gram)/p);alpha=float(np.sum(gram*gram)/(p*p))
        den=(n+1)*(alpha-mu*mu/p)
        self.shrinkage=1. if den<=0 else min((alpha+mu*mu)/den,1.)
        self.tau=max(self.shrinkage*mu,1e-12)
        eigen,u=np.linalg.eigh(gram)
        keep=eigen>max(float(eigen[-1])*np.finfo(float).eps*max(n,p),1e-30)
        eigen=eigen[keep];self.basis=x.T@u[:,keep]/np.sqrt(n*eigen)
        self.factor=1/np.sqrt((1-self.shrinkage)*eigen+self.tau)-1/np.sqrt(self.tau)

    def transform(self,query):
        q=matrix(query)
        if q.shape[1]!=len(self.center):raise ValueError('feature dimensions differ')
        q=q-self.center
        # Keep the orthogonal residual: a truncated PCA distance would erase it.
        result=q/np.sqrt(self.tau)+((q@self.basis)*self.factor)@self.basis.T
        if not np.isfinite(result).all():raise ValueError('nonfinite transformed features')
        return result

class NeighborGeometry:
    def __init__(self,reference):
        self.reference=matrix(reference)
        n=len(self.reference)
        if n<2:raise ValueError('at least two reference rows required')
        self.k=min(3,n);d=cdist(self.reference,self.reference);np.fill_diagonal(d,np.inf)
        self.radius=np.sort(d,axis=1)[:,:min(3,n-1)].mean(axis=1)
        self.scale=float(np.median(self.radius)+1e-9)

    def score(self,query):
        q=matrix(query)
        if q.shape[1]!=self.reference.shape[1]:raise ValueError('feature dimensions differ')
        d=cdist(q,self.reference);indices=np.argsort(d,axis=1,kind='stable')[:,:self.k]
        distance=np.take_along_axis(d,indices,axis=1).mean(axis=1)
        gap=(distance-self.radius[indices].mean(axis=1)-1e-9)/self.scale
        return gap,distance

def arcsine_ppv(x):
    x=matrix(x)
    if x.min()<-1e-7 or x.max()>1+1e-7:raise ValueError('MiniRocket PPV features required')
    return 2*np.arcsin(np.sqrt(np.clip(x,0,1)))

class PairedHeads:
    names=('MiniRocket','oas_gap','oas_distance','arcsine_oas_gap')
    def __init__(self,features):
        self.control=LocalGapModelAntigravity(features)
        z=self.control.scaler.transform(features)
        self.oas=OASGeometry(z);self.neighbors=NeighborGeometry(self.oas.transform(z))
        a=arcsine_ppv(features);self.arc_oas=OASGeometry(a)
        self.arc_neighbors=NeighborGeometry(self.arc_oas.transform(a))

    def score(self,features):
        z=self.control.scaler.transform(features)
        gap,distance=self.neighbors.score(self.oas.transform(z))
        arc,_=self.arc_neighbors.score(self.arc_oas.transform(arcsine_ppv(features)))
        return dict(zip(self.names,(self.control.score(features),gap,distance,arc)))
