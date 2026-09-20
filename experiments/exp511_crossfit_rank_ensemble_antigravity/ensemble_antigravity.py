"""Four fixed A/B roles, sharing fitted geometry between weighted/unweighted means."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
import numpy as np
from scipy.spatial.distance import cdist
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import SEED
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix
from experiments.exp495_normal_geometry_antigravity.pilot_antigravity import perturbations, KINDS
from experiments.exp502_amplitude_envelope_antigravity.amplitude_antigravity import AmplitudeHeads, amplitude_views
from experiments.exp508_tail_rank_antigravity.ranking_antigravity import distance_contributions
from experiments.exp508_tail_rank_antigravity.optimizer_antigravity import loss_gradient
from experiments.exp509_rank_signal_diagnosis_antigravity.diagnostic_antigravity import solve, certificate
from experiments.exp510_separated_rank_antigravity.separated_antigravity import SeparatedRankHeads, ranking_parents

NAMES=SeparatedRankHeads.names+('crossfit_unweighted','crossfit_rank')


def role_folds(size: int) -> tuple:
    if type(size) is not int or size<32:
        raise ValueError('four roles require at least 32 normal rows')
    order=np.random.default_rng(SEED+2510).permutation(size)
    return tuple((np.setdiff1d(np.arange(size),b),b) for b in
                 (np.sort(order[j*size//4:(j+1)*size//4]) for j in range(4)))


class RankMember:
    """One explicit A-only geometry; B contributes only to fixed ranking training."""
    def __init__(self,rows,a,b):
        x=matrix(rows);self.length=x.shape[1]
        a,b=np.asarray(a),np.asarray(b)
        if any(v.ndim!=1 or not np.issubdtype(v.dtype,np.integer) or not np.array_equal(v,np.unique(v)) for v in (a,b)):
            raise ValueError('sorted unique integer role indices required')
        if len(x)<32 or min(len(a),len(b))<8 or np.intersect1d(a,b).size or not np.array_equal(np.sort(np.r_[a,b]),np.arange(len(x))):
            raise ValueError('A and B must be disjoint and exhaust the fitting rows')
        self.a_indices,self.b_indices=a.copy(),b.copy();self.parents=ranking_parents(b)
        self.base=AmplitudeHeads(x[a])
        self.scaler,self.geometry,self.reference=self.base.models['square_joint']
        d=cdist(self.reference,self.reference);np.fill_diagonal(d,np.inf)
        self.scale=float(np.median(np.sort(d,axis=1)[:,:min(3,len(a)-1)].mean(axis=1))+1e-9)
        normal=self.contributions(x[self.parents]);proxies=perturbations(x[self.parents],SEED+1508)
        anomalous=np.vstack([self.contributions(proxies[k]) for k in KINDS])
        self.weights,solver=solve(normal,anomalous)
        initial,initial_g=loss_gradient(np.ones_like(self.weights),normal,anomalous)
        final,gradient=loss_gradient(self.weights,normal,anomalous)
        self.audit=dict(weights=self.weights.copy(),gradient=gradient,initial_gradient=initial_g,
                        normal_scores=normal@self.weights,anomaly_scores=anomalous@self.weights,
                        initial_normal_scores=normal.sum(axis=1),initial_anomaly_scores=anomalous.sum(axis=1))
        self.diagnostics=dict(fallback=False,optimizer_trained=True,n_fit=len(x),n_a=len(a),n_b=len(b),
                              n_parents=len(self.parents),scale=self.scale,initial_objective=initial,final_objective=final,
                              solver=solver,**certificate(self.weights,gradient))

    def contributions(self,rows):
        x=matrix(rows)
        if x.shape[1]!=self.length:raise ValueError('query length differs from fitted input')
        raw=self.base.encoders[0].transform(x)
        square=self.base.encoders[1].transform(amplitude_views(x)[0])
        z=self.geometry.transform(self.scaler.transform(np.column_stack((raw,square))))
        return distance_contributions(z,self.reference)/self.scale


class CrossfitRankHeads:
    names=NAMES

    def __init__(self,rows):
        x=matrix(rows);self.length=x.shape[1];self.single=SeparatedRankHeads(x)
        self.fallback=self.single.fallback;self.members=[]
        if not self.fallback:
            roles=role_folds(len(x))
            assert np.array_equal(roles[0][0],self.single.a_indices) and np.array_equal(roles[0][1],self.single.b_indices)
            # Reuse the exact Exp510 head for member zero and all five frozen controls.
            self.members=[self.single]+[RankMember(x,a,b) for a,b in roles[1:]]

    def score_components(self,rows):
        x=matrix(rows)
        if x.shape[1]!=self.length:raise ValueError('query length differs from fitted input')
        result=self.single.score(x);components={}
        if self.fallback:
            result.update(crossfit_unweighted=result['square_joint'].copy(),crossfit_rank=result['square_joint'].copy())
        else:
            components.update(member0__unweighted=result['a_unweighted'].copy(),member0__rank=result['separated_rank'].copy())
            for j,member in enumerate(self.members[1:],1):
                g=member.contributions(x)
                components.update({f'member{j}__unweighted':g.sum(axis=1),f'member{j}__rank':g@member.weights})
            for name,kind in (('crossfit_unweighted','unweighted'),('crossfit_rank','rank')):
                result[name]=np.mean([components[f'member{j}__{kind}'] for j in range(4)],axis=0)
        if not all(v.ndim==1 and np.isfinite(v).all() for v in (*result.values(),*components.values())):
            raise ValueError('nonfinite ensemble score')
        return result,components

    def score(self,rows):
        return self.score_components(rows)[0]
