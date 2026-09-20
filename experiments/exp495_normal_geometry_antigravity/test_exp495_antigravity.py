import numpy as np
from .diagnostics_antigravity import effective_rank

def test_duplicate_features_have_one_effective_direction():
    assert abs(effective_rank(np.array([[-1.,-1.],[0.,0.],[1.,1.]]))-1.)<1e-12
    assert effective_rank(np.ones((4,3)))==0.
