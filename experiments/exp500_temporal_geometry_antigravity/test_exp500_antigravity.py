"""Lock DTW units/window, reference isolation, tiny/constant data and anchor parity."""
import numpy as np
from aeon.distances import dtw_distance,dtw_pairwise_distance
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import PairedHeads
from experiments.exp500_temporal_geometry_antigravity.temporal_antigravity import temporal_rows,TemporalHeads,distance_scores

def test_temporal_contract():
    assert dtw_distance(np.array([1.,2.,3.]),np.array([11.,12.,13.]),window=0)==300
    np.testing.assert_array_equal(temporal_rows(np.ones((2,9))),np.zeros((2,128)))
    rng=np.random.default_rng(500);raw=rng.normal(size=(8,80));q=rng.normal(size=(7,80))
    a=temporal_rows(raw);b=temporal_rows(q)
    expected=np.array([[dtw_distance(x,y,window=.1) for y in a] for x in b])
    np.testing.assert_array_equal(dtw_pairwise_distance(b,a,window=.1,n_jobs=1),expected)
    features=rng.uniform(size=(8,40));query=rng.uniform(size=(7,40))
    for n in (2,8):
        head=TemporalHeads(features[:n],raw[:n]);score=head.score(query,q);old=PairedHeads(features[:n]).score(query)
        for name in ('MiniRocket','oas_distance'):np.testing.assert_allclose(score[name],old[name],rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(score['dtw_distance'],np.sort(np.sqrt(expected[:,:n]/128),axis=1)[:,:min(3,n)].mean(axis=1))
        before=head.reference.copy();single=head.score(query[:1],q[:1]);head.score(query*3,q*2)
        np.testing.assert_array_equal(head.reference,before)
        for name in head.names:np.testing.assert_allclose(single[name],score[name][:1],rtol=1e-10,atol=1e-8)
    d=rng.uniform(.1,5,size=(7,8));e=rng.uniform(.1,5,size=(7,8))
    np.testing.assert_allclose(distance_scores(d*9,e)['joint_geometric'],3*distance_scores(d,e)['joint_geometric'])
    assert distance_scores(np.zeros((1,2)),np.ones((1,2)))['joint_geometric'][0]==0
