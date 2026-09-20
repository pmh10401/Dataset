"""Official all-reference DTW and fixed geometric fusion with OAS distances."""
import numpy as np
from aeon.distances import dtw_pairwise_distance
from scipy.spatial.distance import cdist
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix,OASGeometry

def temporal_rows(rows):
    x=matrix(rows);grid=np.linspace(0,1,128);old=np.linspace(0,1,x.shape[1])
    out=np.array([np.interp(grid,old,row) for row in x]);out-=out.mean(axis=1,keepdims=True)
    scale=out.std(axis=1,keepdims=True);out/=np.where(scale==0,1,scale)
    if not np.isfinite(out).all():raise ValueError('nonfinite temporal rows')
    return out

def distance_scores(mini,dtw):
    mini=matrix(mini);dtw=matrix(dtw)
    if mini.shape!=dtw.shape or np.any(mini<0) or np.any(dtw<0):raise ValueError('aligned nonnegative distances required')
    k=min(3,mini.shape[1])
    mean=lambda d:np.sort(d,axis=1)[:,:k].mean(axis=1)
    return dict(oas_distance=mean(mini),dtw_distance=mean(dtw),dtw_nearest=dtw.min(axis=1),joint_geometric=mean(np.sqrt(mini*dtw)))

class TemporalHeads:
    names=('MiniRocket','oas_distance','dtw_distance','dtw_nearest','joint_geometric')
    def __init__(self,features,rows):
        x=matrix(features);self.temporal=temporal_rows(rows)
        if len(x)!=len(self.temporal):raise ValueError('feature and raw reference rows differ')
        self.control=LocalGapModelAntigravity(x);z=self.control.scaler.transform(x)
        self.geometry=OASGeometry(z);self.reference=self.geometry.transform(z)

    def score(self,features,rows):
        x=matrix(features);q=temporal_rows(rows)
        if len(x)!=len(q):raise ValueError('feature and raw query rows differ')
        mini=cdist(self.geometry.transform(self.control.scaler.transform(x)),self.reference)
        dtw=np.sqrt(dtw_pairwise_distance(q,self.temporal,window=.1,n_jobs=1)/128)
        result=dict(MiniRocket=self.control.score(x),**distance_scores(mini,dtw))
        if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite temporal score')
        return result
