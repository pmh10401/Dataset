"""Fixed uniform-feature LocalGap bags with TRAIN-only consensus selection."""
import numpy as np
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import SEED

NAMES=('MiniRocket','half_median','full_median','half_q90','full_q90')

def combine(base,bags):
    base=np.asarray(base,float);bags=matrix(bags)
    if base.shape!=(bags.shape[1],) or not np.isfinite(base).all():raise ValueError('unaligned scores')
    median,q90=np.quantile(bags,[.5,.9],axis=0)
    return dict(zip(NAMES,(base,(base+median)/2,median,(base+q90)/2,q90)))

class SubspaceHeads:
    names=NAMES
    def __init__(self,features):
        x=matrix(features);self.control=LocalGapModelAntigravity(x)
        rng=np.random.default_rng(SEED+496)
        self.columns=[rng.choice(x.shape[1],min(256,x.shape[1]),replace=False) for _ in range(32)]
        self.models=[LocalGapModelAntigravity(x[:,c]) for c in self.columns]
    def score(self,features):
        x=matrix(features)
        base=self.control.score(x)
        bags=np.asarray([model.score(x[:,c]) for model,c in zip(self.models,self.columns)])
        return combine(base,bags)
