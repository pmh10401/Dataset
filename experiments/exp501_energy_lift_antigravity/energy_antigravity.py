"""Squared-amplitude MiniRocket view with TRAIN-only joint OAS geometry."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix,OASGeometry
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import SEED

def energy_rows(rows):
    y=np.square(matrix(rows));y-=y.mean(axis=1,keepdims=True)
    scale=y.std(axis=1,keepdims=True);y/=np.where(scale==0,1,scale)
    if not np.isfinite(y).all():raise ValueError('nonfinite energy rows')
    return y.astype(np.float32)

class EnergyHeads:
    names=('MiniRocket','oas_distance','energy_distance','joint_distance')
    def __init__(self,rows):
        x=matrix(rows);self.raw_encoder=make_representation_antigravity('minirocket_local_gap',SEED)
        self.energy_encoder=make_representation_antigravity('minirocket_local_gap',SEED)
        raw=self.raw_encoder.fit_transform(x);energy=self.energy_encoder.fit_transform(energy_rows(x))
        self.diagnostics=dict(energy_constant_row_fraction=float(np.mean(np.ptp(np.square(x),axis=1)==0)),energy_zero_iqr_fraction=float(np.mean(np.diff(np.quantile(energy,[.1,.9],axis=0),axis=0)[0]<=1e-12)))
        self.control=LocalGapModelAntigravity(raw);self.models={}
        for name,features in zip(self.names[1:],(raw,energy,np.column_stack((raw,energy)))):
            scaler=self.control.scaler if name=='oas_distance' else RobustScaler(quantile_range=(10,90)).fit(features)
            scaled=scaler.transform(features);geometry=OASGeometry(scaled)
            self.models[name]=(scaler,geometry,geometry.transform(scaled))

    def score(self,rows):
        x=matrix(rows);raw=self.raw_encoder.transform(x);energy=self.energy_encoder.transform(energy_rows(x))
        result=dict(MiniRocket=self.control.score(raw))
        for name,features in zip(self.names[1:],(raw,energy,np.column_stack((raw,energy)))):
            scaler,geometry,reference=self.models[name];d=cdist(geometry.transform(scaler.transform(features)),reference)
            result[name]=np.sort(d,axis=1)[:,:min(3,len(reference))].mean(axis=1)
        if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite energy score')
        return result
