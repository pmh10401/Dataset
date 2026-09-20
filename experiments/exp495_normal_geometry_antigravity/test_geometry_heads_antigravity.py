import numpy as np
from scipy.spatial.distance import cdist
from sklearn.covariance import OAS
from .geometry_heads_antigravity import OASGeometry,NeighborGeometry

def test_dual_geometry_matches_full_oas_and_preserves_unseen_directions():
    rng=np.random.default_rng(482);x=rng.normal(size=(11,19));q=rng.normal(size=(7,19))
    model=OASGeometry(x);oracle=OAS().fit(x)
    actual=cdist(model.transform(q),model.transform(x))**2
    expected=np.array([np.einsum('ij,jk,ik->i',v-x,oracle.precision_,v-x) for v in q])
    np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(model.shrinkage,oracle.shrinkage_,rtol=1e-12,atol=1e-12)
    line=np.array([[-1.,0.],[0.,0.],[1.,0.]])
    m=OASGeometry(line)
    assert np.linalg.norm(m.transform([[0.,4.]])-m.transform([[0.,0.]]))>0

def test_neighbor_gap_has_exact_reference_correction():
    head=NeighborGeometry(np.array([[0.],[2.],[4.]]))
    gap,distance=head.score(np.array([[6.]]))
    np.testing.assert_allclose(distance,[4.])
    np.testing.assert_allclose(gap,[(4.-8./3.-1e-9)/(3.+1e-9)])

def test_constant_features_remain_finite():
    model=OASGeometry(np.ones((4,7)))
    assert np.isfinite(model.transform(np.zeros((2,7)))).all()
