import numpy as np
from .acceleration_antigravity import stage

def test_torch_rank_arithmetic_keeps_float64_after_integer_search_indices():
    rng=np.random.default_rng(9);x=np.round(rng.uniform(size=(64,12))*16)/16;q=rng.uniform(size=(8,12))
    expected,scores=stage(x,q,'numpy');actual,other=stage(x,q,'torch_cpu')
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    np.testing.assert_allclose(other,scores,rtol=1e-10,atol=1e-8)
