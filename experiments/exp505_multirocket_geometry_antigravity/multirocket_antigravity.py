"""Official MultiRocket features with TRAIN-only residual OAS geometry."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix,OASGeometry
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import SEED

def pad_multirocket_rows(rows):
    x=matrix(rows)
    need=max(10,x.shape[1])-x.shape[1]
    return np.pad(x,((0,0),(0,need))) if need else x

class MultiRocketHeads:
    names=('MiniRocket','oas_distance','multirocket_distance','multirocket_joint')
    def __init__(self,rows):
        x=matrix(rows)
        if len(x)<2:raise ValueError('at least two TRAIN rows required')
        self.length=int(x.shape[1])
        self.raw_encoder=make_representation_antigravity('minirocket_local_gap',SEED)
        self.multi_encoder=make_representation_antigravity('multirocket_local_gap',SEED)
        raw=self.raw_encoder.fit_transform(x);multi=self.multi_encoder.fit_transform(pad_multirocket_rows(x))
        self.diagnostics=dict(raw_featuredim=int(raw.shape[1]),multi_featuredim=int(multi.shape[1]),multi_zero_iqr_fraction=float(np.mean(np.diff(np.quantile(multi,[.1,.9],axis=0),axis=0)[0]<=1e-12)))
        self.control=LocalGapModelAntigravity(raw);self.models={}
        for name,features in zip(self.names[1:],(raw,multi,np.column_stack((raw,multi)))):
            scaler=self.control.scaler if name=='oas_distance' else RobustScaler(quantile_range=(10,90)).fit(features)
            scaled=scaler.transform(features);geometry=OASGeometry(scaled)
            self.models[name]=(scaler,geometry,geometry.transform(scaled))

    def score(self,rows):
        x=matrix(rows)
        if x.shape[1]!=self.length:raise ValueError('query length differs')
        raw=self.raw_encoder.transform(x);multi=self.multi_encoder.transform(pad_multirocket_rows(x))
        result=dict(MiniRocket=self.control.score(raw))
        for name,features in zip(self.names[1:],(raw,multi,np.column_stack((raw,multi)))):
            scaler,geometry,reference=self.models[name];d=cdist(geometry.transform(scaler.transform(features)),reference)
            result[name]=np.sort(d,axis=1)[:,:min(3,len(reference))].mean(axis=1)
        if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite multirocket score')
        return result
