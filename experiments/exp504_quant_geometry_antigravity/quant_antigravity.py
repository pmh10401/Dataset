"""Official QUANT interval quantiles on globally z-normalized series; no local rescaling."""
import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from aeon.transformations.collection.interval_based import QUANTTransformer
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix,OASGeometry
torch.set_num_threads(1)
SEED=20260721
QUANT_CAP=512
INTERVAL_DEPTH=6
QUANTILE_DIVISOR=4

def cap_series(rows):
    x=matrix(rows);length=x.shape[1]
    if length<=QUANT_CAP:capped=x
    else:
        target=np.linspace(0.0,1.0,QUANT_CAP);source=np.linspace(0.0,1.0,length)
        capped=np.stack([np.interp(target,source,row) for row in x])
    if not np.isfinite(capped).all():raise ValueError('nonfinite quant series')
    return np.asarray(capped,dtype=np.float32)[:,np.newaxis,:]

class QuantHeads:
    names=('MiniRocket','oas_distance','quant_distance','quant_joint')
    def __init__(self,rows):
        torch.set_num_threads(1)
        x=matrix(rows);n,length=x.shape;self.length=length
        if n<2:raise ValueError('at least two TRAIN rows required')
        self.raw_encoder=make_representation_antigravity('minirocket_local_gap',SEED)
        raw=self.raw_encoder.fit_transform(x)
        self.control=LocalGapModelAntigravity(raw);self.models={};self.fallback=length<3
        if self.fallback:
            scaler=self.control.scaler;scaled=scaler.transform(raw);geometry=OASGeometry(scaled)
            self.models['oas_distance']=(scaler,geometry,geometry.transform(scaled))
            self.quant_encoder=None
            self.diagnostics=dict(fallback=True,quantlength=0,quantfeaturedim=0)
            return
        self.quant_encoder=QUANTTransformer(interval_depth=INTERVAL_DEPTH,quantile_divisor=QUANTILE_DIVISOR)
        collection=cap_series(x)
        quant=np.asarray(self.quant_encoder.fit_transform(collection),dtype=np.float64)
        if quant.ndim!=2 or quant.shape[0]!=n or not np.isfinite(quant).all():raise ValueError('quant features must be finite two-dimensional')
        self.diagnostics=dict(fallback=False,quantlength=int(collection.shape[-1]),quantfeaturedim=int(quant.shape[1]))
        for name,features in zip(self.names[1:],(raw,quant,np.column_stack((raw,quant)))):
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
            result['oas_distance']=oas;result['quant_distance']=np.array(oas,copy=True);result['quant_joint']=np.array(oas,copy=True)
        else:
            collection=cap_series(x)
            quant=np.asarray(self.quant_encoder.transform(collection),dtype=np.float64)
            if quant.ndim!=2 or quant.shape[0]!=len(x) or quant.shape[1]!=self.diagnostics['quantfeaturedim'] or not np.isfinite(quant).all():raise ValueError('quant features must be finite two-dimensional')
            for name,features in zip(self.names[1:],(raw,quant,np.column_stack((raw,quant)))):
                scaler,geometry,reference=self.models[name]
                d=cdist(geometry.transform(scaler.transform(features)),reference)
                result[name]=np.sort(d,axis=1)[:,:min(3,len(reference))].mean(axis=1)
        if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite quant score')
        return result
