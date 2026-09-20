import numpy as np
from .subspace_antigravity import combine

def test_pooling_axis_and_original_score_weight():
    result=combine(np.array([2.,4.]),np.array([[0.,2.],[2.,4.],[4.,6.],[6.,8.]]))
    assert set(result)=={'MiniRocket','half_median','full_median','half_q90','full_q90'}
    for name,want in [('MiniRocket',[2,4]),('half_median',[2.5,4.5]),('full_median',[3,5]),('half_q90',[3.7,5.7]),('full_q90',[5.4,7.4])]:
        np.testing.assert_allclose(result[name],want,rtol=0,atol=1e-12)

def test_original_score_replays_and_nonfinite_queries_are_rejected():
    import pytest
    from .subspace_antigravity import SubspaceHeads
    from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
    rng=np.random.default_rng(4);x=rng.normal(size=(12,300));q=rng.normal(size=(4,300))
    model=SubspaceHeads(x);scores=model.score(q)
    np.testing.assert_array_equal(scores['MiniRocket'],LocalGapModelAntigravity(x).score(q))
    assert set(scores)=={'MiniRocket','half_median','full_median','half_q90','full_q90'}
    assert all(np.isfinite(s).all() and s.shape==(4,) for s in scores.values())
    assert len(model.columns)==32 and all(len(np.unique(c))==256 for c in model.columns)
    bad=q.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError):model.score(bad)
