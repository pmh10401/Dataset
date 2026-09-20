"""Window alignment, no variance division for square view, baseline and isolation."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
import numpy as np
from experiments.exp502_amplitude_envelope_antigravity.amplitude_antigravity import amplitude_views,AmplitudeHeads,SEED
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import PairedHeads

def test_amplitude_contract():
    rng=np.random.default_rng(502);x=rng.normal(size=(12,64));q=rng.normal(size=(5,64))
    square,rms=amplitude_views(x)
    np.testing.assert_allclose(square,x*x-(x*x).mean(axis=1,keepdims=True),rtol=1e-7,atol=1e-7)
    padded=np.pad(x*x,((0,0),(4,3)),mode='edge');expected=np.sqrt(np.stack([padded[:,j:j+8].mean(axis=1) for j in range(64)],axis=1));expected-=expected.mean(axis=1,keepdims=True);expected/=expected.std(axis=1,keepdims=True)
    np.testing.assert_allclose(rms,expected,rtol=1e-6,atol=1e-7)
    for view in amplitude_views(np.ones((2,5))):np.testing.assert_array_equal(view,np.zeros((2,5)))
    for n in (2,12):
        head=AmplitudeHeads(x[:n]);scores=head.score(q)
        encoder=make_representation_antigravity('minirocket_local_gap',SEED);old=PairedHeads(encoder.fit_transform(x[:n])).score(encoder.transform(q))
        for name in ('MiniRocket','oas_distance'):np.testing.assert_allclose(scores[name],old[name],rtol=1e-10,atol=1e-8)
        before={name:r.copy() for name,(_,_,r) in head.models.items()};single=head.score(q[:1]);head.score(q*4)
        for name in head.names:np.testing.assert_allclose(single[name],scores[name][:1],rtol=1e-10,atol=1e-8)
        for name,(_,_,reference) in head.models.items():np.testing.assert_array_equal(before[name],reference)
