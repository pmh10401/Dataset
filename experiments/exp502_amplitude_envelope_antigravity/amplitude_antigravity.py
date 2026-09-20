"""Fixed centered-square and moving-RMS views, preserving frozen normal-only fits."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import RobustScaler
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix,OASGeometry
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import SEED

def amplitude_views(rows):
    x=matrix(rows);square=x*x;square-=square.mean(axis=1,keepdims=True)
    width=min(x.shape[1],max(8,x.shape[1]//10));left=width//2;right=width-1-left
    padded=np.pad(x*x,((0,0),(left,right)),mode='edge')
    rms=np.sqrt(np.array([np.convolve(row,np.ones(width)/width,mode='valid') for row in padded]))
    rms-=rms.mean(axis=1,keepdims=True);scale=rms.std(axis=1,keepdims=True);rms/=np.where(scale==0,1,scale)
    if not np.isfinite(square).all() or not np.isfinite(rms).all():raise ValueError('nonfinite amplitude view')
    return square.astype(np.float32),rms.astype(np.float32)

class AmplitudeHeads:
    names=('MiniRocket','oas_distance','square_distance','square_joint','rms_distance','rms_joint')
    def __init__(self,rows):
        x=matrix(rows);views=(x,*amplitude_views(x))
        self.encoders=[make_representation_antigravity('minirocket_local_gap',SEED) for _ in views]
        raw,square,rms=[encoder.fit_transform(view) for encoder,view in zip(self.encoders,views)]
        self.control=LocalGapModelAntigravity(raw);self.models={}
        self.diagnostics=dict(square_constant_fraction=float(np.mean(np.ptp(views[1],axis=1)==0)),rms_constant_fraction=float(np.mean(np.ptp(views[2],axis=1)==0)))
        for name,features in zip(self.names[1:],(raw,square,np.column_stack((raw,square)),rms,np.column_stack((raw,rms)))):
            scaler=self.control.scaler if name=='oas_distance' else RobustScaler(quantile_range=(10,90)).fit(features)
            scaled=scaler.transform(features);geometry=OASGeometry(scaled)
            self.models[name]=(scaler,geometry,geometry.transform(scaled))

    def score(self,rows):
        x=matrix(rows);raw,square,rms=[encoder.transform(view) for encoder,view in zip(self.encoders,(x,*amplitude_views(x)))]
        result=dict(MiniRocket=self.control.score(raw))
        for name,features in zip(self.names[1:],(raw,square,np.column_stack((raw,square)),rms,np.column_stack((raw,rms)))):
            scaler,geometry,reference=self.models[name];d=cdist(geometry.transform(scaler.transform(features)),reference)
            result[name]=np.sort(d,axis=1)[:,:min(3,len(reference))].mean(axis=1)
        if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite amplitude score')
        return result
