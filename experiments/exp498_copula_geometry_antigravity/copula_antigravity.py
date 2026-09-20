"""Reference-only tied Gaussian ranks and frozen copula distance variants."""
import numpy as np
from scipy.special import ndtri
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix,OASGeometry,NeighborGeometry
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import SEED
from sklearn.preprocessing import QuantileTransformer

NAMES=('MiniRocket','oas_distance','rank_distance','rank_oas_distance','qt_oas_distance','clipped_rank_oas_distance','atom5_rank_oas_distance')

class GaussianRanks:
    def __init__(self,reference,minimum_unique=2):
        x=matrix(reference);self.n,self.width=x.shape;values=np.sort(x,axis=0)
        self.columns=np.flatnonzero(1+np.sum(np.diff(values,axis=0)!=0,axis=0)>=minimum_unique)
        self.sorted=values[:,self.columns]
    def transform(self,query):
        x=matrix(query)
        if x.shape[1]!=self.width:raise ValueError('feature dimensions differ')
        result=np.empty((len(x),len(self.columns)))
        for j,column in enumerate(self.columns):
            l=np.searchsorted(self.sorted[:,j],x[:,column],side='left');r=np.searchsorted(self.sorted[:,j],x[:,column],side='right')
            result[:,j]=(l+.5*(r-l)+.5)/(self.n+1)
        return ndtri(np.clip(result,1/(self.n+1),self.n/(self.n+1)))

class CopulaHeads:
    names=NAMES
    def __init__(self,features):
        x=matrix(features);self.control=LocalGapModelAntigravity(x);z=self.control.scaler.transform(x)
        self.oas=OASGeometry(z);self.neighbors=NeighborGeometry(self.oas.transform(z))
        self.heads={};self.rank=None;self.atom=None
        if len(x)<16:return
        self.rank=GaussianRanks(x);self.atom=GaussianRanks(x,minimum_unique=5);spaces={}
        if len(self.rank.columns)>=8:
            r=self.rank.transform(x)
            self.bounds=ndtri([1/(len(x)+1),len(x)/(len(x)+1)])
            self.qt=QuantileTransformer(n_quantiles=min(64,len(x)),output_distribution='normal',subsample=None,random_state=SEED)
            qt=np.clip(self.qt.fit_transform(x[:,self.rank.columns]),*self.bounds)
            spaces.update(rank_distance=r,rank_oas_distance=r,qt_oas_distance=qt,clipped_rank_oas_distance=np.clip(r,-2,2))
        if len(self.atom.columns)>=8:spaces['atom5_rank_oas_distance']=self.atom.transform(x)
        for name,values in spaces.items():
            geometry=None if name=='rank_distance' else OASGeometry(values)
            self.heads[name]=(geometry,NeighborGeometry(values if geometry is None else geometry.transform(values)))
    def score(self,features):
        x=matrix(features);z=self.control.scaler.transform(x)
        anchor=self.neighbors.score(self.oas.transform(z))[1]
        result={'MiniRocket':self.control.score(x),'oas_distance':anchor};spaces={}
        if 'rank_distance' in self.heads:
            r=self.rank.transform(x)
            spaces.update(rank_distance=r,rank_oas_distance=r,qt_oas_distance=np.clip(self.qt.transform(x[:,self.rank.columns]),*self.bounds),clipped_rank_oas_distance=np.clip(r,-2,2))
        if 'atom5_rank_oas_distance' in self.heads:spaces['atom5_rank_oas_distance']=self.atom.transform(x)
        for name in self.names[2:]:
            if name not in self.heads:result[name]=anchor
            else:
                geometry,head=self.heads[name];values=spaces[name]
                result[name]=head.score(values if geometry is None else geometry.transform(values))[1]
        return result
