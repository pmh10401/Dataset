import numpy as np
import pytest
from scipy.special import ndtri
from .copula_antigravity import GaussianRanks

def test_reference_ties_query_tail_clipping_and_constant_mask():
    ref=np.column_stack(([0.,0.,1.,3.],[5.,5.,5.,5.]));queries=np.column_stack(([-1.,0.,.5,1.,2.,3.,4.],np.full(7,5.)))
    mapping=GaussianRanks(ref)
    np.testing.assert_array_equal(mapping.columns,[0])
    np.testing.assert_allclose(mapping.transform(queries).ravel(),ndtri([.2,.3,.5,.6,.7,.8,.8]),atol=1e-15,rtol=0)
    np.testing.assert_array_equal(mapping.transform(ref)[:2,0],[ndtri(.3),ndtri(.3)])
    with pytest.raises(ValueError):mapping.transform([[np.nan,5.]])

def test_heads_preserve_distance_definition_and_do_not_fit_on_query():
    from .copula_antigravity import CopulaHeads
    from sklearn.neighbors import NearestNeighbors
    from sklearn.covariance import OAS
    from scipy.spatial.distance import cdist
    rng=np.random.default_rng(498);x=rng.normal(size=(24,12));q=rng.normal(size=(4,12))
    head=CopulaHeads(x);scores=head.score(q);assert set(scores)==set(head.names)
    rank=GaussianRanks(x);rx=rank.transform(x);rq=rank.transform(q)
    expected=NearestNeighbors(n_neighbors=3).fit(rx).kneighbors(rq)[0].mean(axis=1)
    np.testing.assert_allclose(scores['rank_distance'],expected,rtol=1e-10,atol=1e-8)
    d=cdist(rq,rx,metric='mahalanobis',VI=OAS().fit(rx).precision_)
    np.testing.assert_allclose(scores['rank_oas_distance'],np.sort(d,axis=1)[:,:3].mean(axis=1),rtol=1e-10,atol=1e-8)
    more=head.score(np.vstack((q,np.full((1,12),100.))))
    for name in head.names:np.testing.assert_allclose(scores[name],more[name][:4],rtol=1e-10,atol=1e-8)

def test_tiny_or_constant_training_uses_existing_oas_anchor():
    from .copula_antigravity import CopulaHeads
    rng=np.random.default_rng(2)
    for x in [rng.normal(size=(2,10)),np.ones((20,10))]:
        head=CopulaHeads(x);scores=head.score(rng.normal(size=(4,10)))
        assert set(scores)==set(head.names)
        for name in head.names[2:]:np.testing.assert_array_equal(scores[name],scores['oas_distance'])
