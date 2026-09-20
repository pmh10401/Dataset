"""Shared-quarter MiniRocket pyramid on globally z-normalized series; no local rescaling."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from aeon.transformations.collection.convolution_based import MiniRocket
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity,_SingleRepresentationAntigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix,OASGeometry

SEED=20260721

def quarter_segments(rows):
    x=matrix(rows);n,length=x.shape
    if length<4:raise ValueError('quarter split requires length at least 4')
    width=max(9,length//4);target=np.linspace(0.0,1.0,width)
    blocks=[]
    for part in np.array_split(x,4,axis=1):
        source=np.linspace(0.0,1.0,part.shape[1])
        blocks.append(np.stack([np.interp(target,source,row) for row in part]))
    stacked=np.stack(blocks,axis=1).reshape(n*4,width)
    if not np.isfinite(stacked).all():raise ValueError('nonfinite quarter segment')
    return stacked

class SpatialHeads:
    names=('MiniRocket','oas_distance','pyramid_distance','pyramid_joint')
    def __init__(self,rows):
        x=matrix(rows);n,length=x.shape;self.length=length
        self.raw_encoder=make_representation_antigravity('minirocket_local_gap',SEED)
        raw=self.raw_encoder.fit_transform(x)
        self.control=LocalGapModelAntigravity(raw);self.models={};self.fallback=length<4
        if self.fallback:
            scaler=self.control.scaler;scaled=scaler.transform(raw);geometry=OASGeometry(scaled)
            self.models['oas_distance']=(scaler,geometry,geometry.transform(scaled))
            self.quarter_encoder=None
            self.diagnostics=dict(fallback=True,quarterlength=0,quarterfeaturedim=0)
            return
        self.quarter_encoder=_SingleRepresentationAntigravity(MiniRocket(n_kernels=1024,n_jobs=1,random_state=SEED))
        stacked=quarter_segments(x);encoded=self.quarter_encoder.fit_transform(stacked)
        if encoded.shape!=(4*n,1008):raise ValueError('quarter encoder must return shape (4*n, 1008)')
        quarter=encoded.reshape(n,4032)
        self.diagnostics=dict(fallback=False,quarterlength=int(stacked.shape[1]),quarterfeaturedim=int(encoded.shape[1]))
        for name,features in zip(self.names[1:],(raw,quarter,np.column_stack((raw,quarter)))):
            scaler=self.control.scaler if name=='oas_distance' else RobustScaler(quantile_range=(10,90)).fit(features)
            scaled=scaler.transform(features);geometry=OASGeometry(scaled)
            self.models[name]=(scaler,geometry,geometry.transform(scaled))

    def score(self,rows):
        x=matrix(rows)
        if x.shape[1]!=self.length:raise ValueError('query length must equal fit input length')
        raw=self.raw_encoder.transform(x);result=dict(MiniRocket=self.control.score(raw))
        if self.fallback:
            scaler,geometry,reference=self.models['oas_distance']
            d=cdist(geometry.transform(scaler.transform(raw)),reference)
            oas=np.sort(d,axis=1)[:,:min(3,len(reference))].mean(axis=1)
            result['oas_distance']=oas;result['pyramid_distance']=np.array(oas,copy=True);result['pyramid_joint']=np.array(oas,copy=True)
        else:
            encoded=self.quarter_encoder.transform(quarter_segments(x))
            if encoded.shape!=(4*len(x),1008):raise ValueError('quarter encoder must return shape (4*n, 1008)')
            quarter=encoded.reshape(len(x),4032)
            for name,features in zip(self.names[1:],(raw,quarter,np.column_stack((raw,quarter)))):
                scaler,geometry,reference=self.models[name]
                d=cdist(geometry.transform(scaler.transform(features)),reference)
                result[name]=np.sort(d,axis=1)[:,:min(3,len(reference))].mean(axis=1)
        if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite spatial score')
        return result
