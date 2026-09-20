"""TRAIN diagnostics only: frozen objective, fixed solver and matched reference controls."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.spatial.distance import cdist
from experiments.exp508_tail_rank_antigravity import ranking_antigravity as rank
from experiments.exp508_tail_rank_antigravity.optimizer_antigravity import _inputs, _objective, loss_gradient
from experiments.exp502_amplitude_envelope_antigravity.amplitude_antigravity import AmplitudeHeads, amplitude_views
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix
from experiments.exp495_normal_geometry_antigravity.pilot_antigravity import perturbations, KINDS
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import SEED

OPTIONS = dict(maxiter=2000, maxfun=5000, maxls=40, maxcor=20, gtol=1e-8, ftol=1e-15)
MODELS = ('MiniRocket', 'oas_distance', 'square_joint', 'adam', 'lbfgsb')
EXTRAS = ('weak_level', 'weak_scale', 'sparse_spike', 'time_warp')


def certificate(weights, gradient):
    w, g = np.asarray(weights, float), np.asarray(gradient, float)
    if w.ndim != 1 or not len(w) or g.shape != w.shape or not np.isfinite(w).all() or not np.isfinite(g).all():
        raise ValueError('finite aligned weight and gradient vectors required')
    if np.any(w < .25) or np.any(w > 4.): raise ValueError('weight outside frozen box')
    pg = w - np.clip(w-g, .25, 4.)
    mu = .2/len(w); d = w - np.clip(w-g/mu, .25, 4.)
    gap = float(g@d - .5*mu*(d@d))
    if gap < -1e-12: raise ValueError('negative convex gap bound')
    norm = float(np.max(abs(pg)))
    return dict(projected_gradient_inf=norm, projected_gradient_passed=bool(norm <= OPTIONS['gtol']),
                objective_gap_upper_bound=max(0., gap), rms_displacement=float(np.sqrt(np.mean((w-1)**2))))


def solve(normal, anomalous, objective=None):
    n, a = _inputs(normal, anomalous)
    if np.any(n < 0) or np.any(a < 0): raise ValueError('nonnegative distance attributions required')
    fn = _objective if objective is None else objective
    start = perf_counter()
    result = minimize(lambda w: fn(w, n, a), np.ones(n.shape[1]), method='L-BFGS-B', jac=True,
                      bounds=[(.25, 4.)]*n.shape[1], options=dict(OPTIONS))
    # Solver success can be a function-tolerance stop; independently check the gradient.
    value, g = _objective(result.x, n, a)
    info = dict(objective=value, solver_success=bool(result.success), status=int(result.status),
                message=str(result.message), iterations=int(result.nit), function_evaluations=int(result.nfev),
                seconds=perf_counter()-start, **certificate(result.x, g))
    return np.asarray(result.x, dtype=np.float64), info


def signal_statistics(normal_scores, anomalous_scores, weights):
    ns, az, w = (np.asarray(v, float) for v in (normal_scores, anomalous_scores, weights))
    if any(v.ndim != 1 or not len(v) or not np.isfinite(v).all() for v in (ns, az, w)):
        raise ValueError('finite nonempty one-dimensional scores and weights required')
    logz = logsumexp(ns/.1); tail = .1*logz; delta = .25+tail-az
    prob = expit(delta); penalty = float(.1*np.mean((w-1)**2)); losses = np.logaddexp(0., delta)
    out = dict(objective=float(losses.mean()+penalty), data_loss=float(losses.mean()), penalty=penalty,
               n_normal=len(ns), n_anomaly=len(az), normal_median=float(np.median(ns)),
               normal_p95=float(np.quantile(ns,.95)), normal_max=float(ns.max()),
               anomaly_median=float(np.median(az)), anomaly_p05=float(np.quantile(az,.05)),
               smooth_normal_upper=tail, mean_sigmoid=float(prob.mean()),
               active_001_fraction=float(np.mean(prob >= .01)), active_01_fraction=float(np.mean(prob >= .1)),
               above_normal_max_fraction=float(np.mean(az > ns.max())),
               median_margin=float(np.median(az-ns.max())), minimum_margin=float(np.min(az-ns.max())),
               effective_normal_count=float(1/np.sum(np.exp(ns/.1-logz)**2)))
    if not all(np.isfinite(v) for v in out.values()): raise ValueError('nonfinite signal statistics')
    return out


def roles(size, fold):
    if type(size) is not int or size < 8 or type(fold) is not int or fold < 0:
        raise ValueError('at least eight fit rows and nonnegative fold required')
    order = np.random.default_rng(SEED+2509+fold).permutation(size); count = size//4
    b, s, r = [np.sort(x) for x in (order[:count], order[count:2*count], order[2*count:])]
    return dict(reference=r, seen=s, unseen=b, representation_fit=np.sort(np.r_[r,s]))


def capture(rows):
    if len(rows) < 16: raise ValueError('diagnostic requires the non-fallback training path')
    saved = {}; original = rank.fit_tail_weights
    def observe(n, a):
        saved['normal'], saved['anomaly'] = n.copy(), a.copy()
        return original(n, a)
    rank.fit_tail_weights = observe
    try: head = rank.TailRankHeads(rows)
    finally: rank.fit_tail_weights = original
    return head, saved['normal'], saved['anomaly']


def participation_probe(rows, fold):
    x = matrix(rows); part = roles(len(x), fold); fit = part['representation_fit']
    head = AmplitudeHeads(x[fit]); scaler, geometry, full_ref = head.models['square_joint']
    reference = full_ref[np.searchsorted(fit, part['reference'])]
    distances = cdist(reference, reference); np.fill_diagonal(distances, np.inf)
    scale = float(np.median(np.sort(distances,axis=1)[:,:min(3,len(reference)-1)].mean(axis=1))+1e-9)
    def scores(q):
        raw = head.encoders[0].transform(q); square = head.encoders[1].transform(amplitude_views(q)[0])
        z = geometry.transform(scaler.transform(np.column_stack((raw,square))))
        return np.sort(cdist(z,reference),axis=1)[:,:min(3,len(reference))].mean(axis=1)/scale
    evidence = {}
    for name in ('seen','unseen'):
        q = x[part[name]]; proxies = perturbations(q, SEED+1508)
        evidence[name+'_normal'] = scores(q)
        evidence[name+'_anomaly'] = np.concatenate([scores(proxies[k]) for k in KINDS])
    info = dict(roles={k:v.tolist() for k,v in part.items()}, common_reference_count=len(reference),
                representation_fit_count=len(fit), same_scale=scale, feature_dimensions=full_ref.shape[1])
    return evidence, info


def score_variants(head, query, lb_weights):
    original = head.base.score(query)
    g = rank.distance_contributions(head._geometry(query),head.reference)/head.scale
    np.testing.assert_allclose(g.sum(axis=1)*head.scale, original['square_joint'],rtol=1e-10,atol=1e-8)
    return dict(MiniRocket=original['MiniRocket'],oas_distance=original['oas_distance'],
                square_joint=original['square_joint'],adam=g@head.weights,lbfgsb=g@lb_weights)


def recall_and_tail(evidence):
    recall = {}; tails = {}
    for name in MODELS:
        ns = evidence['held__normal__'+name]; maximum = ns.max()
        tails[name] = float((maximum-np.median(ns))/(np.quantile(ns,.75)-np.quantile(ns,.25)+1e-9))
        recall[name] = {kind:float(np.mean(evidence['held__'+kind+'__'+name] > maximum)) for kind in KINDS+EXTRAS}
    return recall, tails


def analyze_fold(fit_rows, held_rows, fold, previous):
    head, n, a = capture(fit_rows); w_lb, solver = solve(n, a)
    weights = dict(initial=np.ones(n.shape[1]),adam=head.weights,lbfgsb=w_lb)
    evidence = {}; training = {}
    for name,w in weights.items():
        value, gradient = loss_gradient(w,n,a)
        ns, az = n@w, a@w
        evidence.update({name+'_weights':w,name+'_gradient':gradient,
                         'train__'+name+'__normal':ns,'train__'+name+'__anomaly':az})
        stats = signal_statistics(ns,az,w); np.testing.assert_allclose(stats['objective'],value,rtol=1e-12,atol=1e-12)
        training[name] = dict(**stats,**certificate(w,gradient))
    np.testing.assert_allclose(training['initial']['objective'],previous['diagnostics'][fold]['initial_objective'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(training['adam']['objective'],previous['diagnostics'][fold]['final_objective'],rtol=1e-12,atol=1e-12)
    queries = dict(normal=held_rows, **perturbations(held_rows,SEED+1709*(fold+1)),
                   **rank.unseen_perturbations(held_rows,SEED+2508+1709*(fold+1)))
    for kind,q in queries.items():
        for name,s in score_variants(head,q,w_lb).items(): evidence['held__'+kind+'__'+name] = s
    recall,tails = recall_and_tail(evidence)
    for name in MODELS[:-1]:
        old = 'tail_rank' if name=='adam' else name
        np.testing.assert_allclose(tails[name],previous['normal_tail_by_fold'][fold][old],rtol=1e-12,atol=1e-12)
        for kind in KINDS+EXTRAS:
            key = 'folds' if kind in KINDS else 'unseen_folds'
            assert recall[name][kind] == previous[key][fold][kind][old], (name,kind,'baseline replay')
    control,probe = participation_probe(fit_rows,fold); evidence.update(control)
    probe['signal'] = {name:signal_statistics(control[name+'_normal'],control[name+'_anomaly'],np.ones(n.shape[1])) for name in ('seen','unseen')}
    info = dict(training=training,solver=solver,recall=recall,normal_tail=tails,participation=probe,
                nonworsening_rms_bound=float(np.sqrt(training['initial']['objective']/.1)),
                original_parents=head.parents.tolist(),original_scale=head.scale,baseline_replay=True)
    return evidence,info


def aggregate(records):
    folds = [f for r in records for f in r['folds']]
    metrics = {}
    for name in MODELS:
        standard = {k:float(np.mean([np.mean([f['recall'][name][k] for f in r['folds']]) for r in records])) for k in KINDS}
        extra = {k:float(np.mean([np.mean([f['recall'][name][k] for f in r['folds']]) for r in records])) for k in EXTRAS}
        metrics[name] = dict(standard=standard,unseen=extra,worst_standard=min(standard.values()),
                            mean_standard=float(np.mean(list(standard.values()))),mean_unseen=float(np.mean(list(extra.values()))),
                            normal_tail=float(np.median([np.mean([f['normal_tail'][name] for f in r['folds']]) for r in records])))
    optimizer = {name:{key:float(np.median([f['training'][name][key] for f in folds])) for key in
                 ('objective','data_loss','mean_sigmoid','active_001_fraction','projected_gradient_inf','objective_gap_upper_bound','rms_displacement')}
                 for name in ('initial','adam','lbfgsb')}
    normal_ratios = [float(np.median([f['participation']['signal']['unseen']['normal_max']/max(f['participation']['signal']['seen']['normal_max'],1e-30) for f in r['folds']])) for r in records]
    control = {name:{key:float(np.median([np.median([f['participation']['signal'][name][key] for f in r['folds']]) for r in records])) for key in
               ('normal_max','normal_median','data_loss','active_001_fraction','mean_sigmoid','above_normal_max_fraction')} for name in ('seen','unseen')}
    return dict(metrics=metrics,training_medians=optimizer,participation_family_medians=control,
                family_median_normal_max_ratio=float(np.median(normal_ratios)),families_unseen_max_higher=int(np.sum(np.array(normal_ratios)>1)),
                lbfgsb_projected_gradient_passes=sum(f['solver']['projected_gradient_passed'] for f in folds),
                lbfgsb_solver_successes=sum(f['solver']['solver_success'] for f in folds),
                lbfgsb_objective_above_initial=sum(f['training']['lbfgsb']['objective']>f['training']['initial']['objective']+1e-12 for f in folds),
                adam_objective_increased=sum(f['training']['adam']['objective']>f['training']['initial']['objective'] for f in folds),
                lbfgsb_max_iterations=max(f['solver']['iterations'] for f in folds),
                median_nonworsening_rms_bound=float(np.median([f['nonworsening_rms_bound'] for f in folds])))


def benchmark():
    """Synthetic full optimizer workflow; no private rows or official solver rewrite."""
    from experiments.exp508_tail_rank_antigravity.acceleration_antigravity import objective_numba, objective_torch, _gpu_probes
    from threadpoolctl import threadpool_limits
    from importlib.metadata import version
    rng = np.random.default_rng(509); fit=rng.normal(size=(64,128));query=rng.normal(size=(1024,128))
    fit[0]=0; query[0]=0; backends=[]; reference=None
    def stage(fn):
        start=perf_counter();head,n,a=capture(fit);w,_=solve(n,a,fn)
        scores=score_variants(head,query,w)
        return scores,perf_counter()-start
    def candidate(s):
        return {name:{str(b):np.flatnonzero(v>np.sort(v[:1000])[::-1][b]).tolist() for b in (0,1,5,10)} for name,v in s.items()}
    with threadpool_limits(limits=1):
        for name,fn in (('numpy',_objective),('numba',objective_numba),('torch_cpu',objective_torch)):
            first,cold=stage(fn);warm,seconds=stage(fn)
            if reference is None: reference=first
            parity=all(np.isfinite(s[k]).all() and np.allclose(s[k],reference[k],rtol=1e-10,atol=1e-8)
                       and np.array_equal(np.argsort(s[k],kind='stable'),np.argsort(reference[k],kind='stable')) for s in (first,warm) for k in reference)
            parity=bool(parity and candidate(first)==candidate(warm)==candidate(reference))
            backends.append(dict(backend=name,cold_seconds=cold,warm_seconds=seconds,parity=parity,
                                 max_abs_error=max(float(np.max(abs(first[k]-reference[k]))) for k in reference)))
    for row in backends: row['speedup']=backends[0]['warm_seconds']/row['warm_seconds']
    return dict(state='complete',success=backends[0]['parity'],backends=backends,gpu=_gpu_probes(),
                versions={p:version(p) for p in ('numpy','scipy','aeon','numba','torch','mlx','mlx-metal')},
                selected_backend='numpy_scipy_float64',options=OPTIONS,
                scope='synthetic complete head+capture+fixed L-BFGS-B+query, cold/warm including conversion/JIT',
                fallback_reason='Official SciPy optimizer retained; alternatives diagnostic only, no automatic promotion.',
                statistics_backend='NumPy reductions/quantiles; Numba orchestration and GPU transfer add no validated benefit',
                candidate_indices=candidate(reference),test_values_read=False,test_labels_read=False)
