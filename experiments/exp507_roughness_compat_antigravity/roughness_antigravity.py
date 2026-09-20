"""Official MiniRocket on mean-centered absolute differences; fixed 2**20 roughness scale; TRAIN-only residual OAS geometry."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix,OASGeometry
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import SEED

def roughness_rows(rows):
    x=matrix(rows)
    if x.shape[1]<2:y=np.zeros_like(x)
    else:
        d=np.abs(np.diff(x,axis=1));d=np.concatenate((d[:,:1],d),axis=1);y=d-d.mean(axis=1,keepdims=True)
    if not np.isfinite(y).all():raise ValueError('nonfinite roughness rows')
    y=y.astype(np.float32)
    if not np.isfinite(y).all():raise ValueError('nonfinite roughness rows')
    return y

def scale_roughness(values):
    values=np.asarray(values,dtype=np.float32)
    if not np.isfinite(values).all():raise ValueError('nonfinite scaled roughness')
    values=np.ldexp(values,20)
    if not np.isfinite(values).all():raise ValueError('nonfinite scaled roughness')
    return np.asarray(values,dtype=np.float32)

class RoughnessHeads:
    names=('MiniRocket','oas_distance','roughness_distance','roughness_joint')
    def __init__(self,rows):
        x=matrix(rows)
        if len(x)<2:raise ValueError('at least two TRAIN rows required')
        self.length=int(x.shape[1])
        self.raw_encoder=make_representation_antigravity('minirocket_local_gap',SEED)
        raw=self.raw_encoder.fit_transform(x)
        self.control=LocalGapModelAntigravity(raw);self.models={}
        rough=roughness_rows(x);constant=float(np.mean(np.ptp(rough,axis=1)==0));self.fallback=self.length<3
        if self.fallback:
            scaler=self.control.scaler;scaled=scaler.transform(raw);geometry=OASGeometry(scaled)
            self.models['oas_distance']=(scaler,geometry,geometry.transform(scaled))
            self.rough_encoder=None
            self.diagnostics=dict(fallback=True,rough_featuredim=0,rough_constant_row_fraction=constant,scale_power=20)
            return
        self.rough_encoder=make_representation_antigravity('minirocket_local_gap',SEED)
        feat=self.rough_encoder.fit_transform(scale_roughness(rough))
        self.diagnostics=dict(fallback=False,rough_featuredim=int(feat.shape[1]),rough_constant_row_fraction=constant,scale_power=20)
        for name,features in zip(self.names[1:],(raw,feat,np.column_stack((raw,feat)))):
            scaler=self.control.scaler if name=='oas_distance' else RobustScaler(quantile_range=(10,90)).fit(features)
            scaled=scaler.transform(features);geometry=OASGeometry(scaled)
            self.models[name]=(scaler,geometry,geometry.transform(scaled))

    def score(self,rows):
        x=matrix(rows)
        if x.shape[1]!=self.length:raise ValueError('query length differs')
        raw=self.raw_encoder.transform(x);result=dict(MiniRocket=self.control.score(raw))
        if self.fallback:
            scaler,geometry,reference=self.models['oas_distance']
            d=cdist(geometry.transform(scaler.transform(raw)),reference)
            oas=np.sort(d,axis=1)[:,:min(3,len(reference))].mean(axis=1)
            result['oas_distance']=oas;result['roughness_distance']=np.array(oas,copy=True);result['roughness_joint']=np.array(oas,copy=True)
        else:
            feat=self.rough_encoder.transform(scale_roughness(roughness_rows(x)))
            for name,features in zip(self.names[1:],(raw,feat,np.column_stack((raw,feat)))):
                scaler,geometry,reference=self.models[name]
                d=cdist(geometry.transform(scaler.transform(features)),reference)
                result[name]=np.sort(d,axis=1)[:,:min(3,len(reference))].mean(axis=1)
        if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite roughness score')
        return result
