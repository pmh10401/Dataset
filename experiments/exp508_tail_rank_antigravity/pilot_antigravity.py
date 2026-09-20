"""Frozen TRAIN-only tail-rank screening with four unseen validation probes."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path
from time import perf_counter
import json
import multiprocessing
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import ROOT, DB, SEED, readonly_sqlite_antigravity, write_json_atomic_antigravity, file_sha256_antigravity, cpu_worker_policy_antigravity
from experiments.exp495_normal_geometry_antigravity.pilot_antigravity import load_normal_train, perturbations, KINDS
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import crossfit_plan_antigravity
from .ranking_antigravity import TailRankHeads, unseen_perturbations

P = Path(__file__).resolve().parent
UNSEEN = ('weak_level', 'weak_scale', 'sparse_spike', 'time_warp')
NAMES = TailRankHeads.names
read = lambda p: json.loads(p.read_text())


def select_candidate(summary, tails, unseen):
    for name in NAMES:
        values = [summary[name][k] for k in KINDS] + [unseen[name][k] for k in UNSEEN]
        if not np.isfinite(values).all() or min(values) < 0 or max(values) > 1 or not np.isfinite(tails[name]) or tails[name] < 0:
            raise ValueError('invalid pilot summaries')
    criterion = lambda name: (min(summary[name].values()), np.mean(list(summary[name].values())), -NAMES.index(name))
    anchor = max(NAMES[:3], key=criterion)
    unseen_mean = {name: float(np.mean(list(unseen[name].values()))) for name in NAMES}
    gates = dict(minimum_margin=bool(criterion('tail_rank')[0] >= criterion(anchor)[0] + .02),
                 mean_nonregression=bool(criterion('tail_rank')[1] >= criterion(anchor)[1]),
                 normal_tail=bool(tails['tail_rank'] <= 3 * max(tails['MiniRocket'], 1e-9)),
                 unseen_nonregression=bool(unseen_mean['tail_rank'] >= max(unseen_mean[n] for n in NAMES[:3])))
    return dict(best_anchor=anchor, selected='tail_rank' if all(gates.values()) else anchor, gates=gates,
                candidate_margin=float(criterion('tail_rank')[0] - criterion(anchor)[0]), required_margin=.02,
                unseen_mean=unseen_mean)


def worker(meta):
    start = perf_counter(); folds = []; unseen_folds = []; tails = []; boundaries = []; audits = []
    with closing(readonly_sqlite_antigravity(DB)) as con:
        x = load_normal_train(con, meta['dataset'])
    assert len(x) == meta['n_train']
    with threadpool_limits(limits=1):
        for fold, (fit, held) in enumerate(crossfit_plan_antigravity(len(x), seed=SEED)):
            head = TailRankHeads(x[fit])
            assert not np.intersect1d(fit, held).size and not np.intersect1d(fit[head.parents], held).size
            boundaries.append(dict(fit=fit.tolist(), held=held.tolist(), parents=fit[head.parents].tolist()))
            audits.append(head.diagnostics)
            scores = head.score(x[held]); thresholds = {k: float(v.max()) for k, v in scores.items()}
            tails.append({k: float((v.max() - np.median(v)) / (np.quantile(v, .75) - np.quantile(v, .25) + 1e-9)) for k, v in scores.items()})
            for destination, proxies in ((folds, perturbations(x[held], SEED + 1709 * (fold + 1))),
                                         (unseen_folds, unseen_perturbations(x[held], SEED + 2508 + 1709 * (fold + 1)))):
                values = {}
                for kind, proxy in proxies.items():
                    ps = head.score(proxy)
                    values[kind] = {k: float(np.mean(v > thresholds[k])) for k, v in ps.items()}
                destination.append(values)
    return dict(dataset=meta['dataset'], source_family=meta['source_family'], folds=folds,
                unseen_folds=unseen_folds, normal_tail_by_fold=tails, fold_boundaries=boundaries,
                diagnostics=audits, seconds=perf_counter() - start)


def main():
    start = perf_counter(); result = P / 'pilot_results_antigravity.json'; manifest_path = P / 'pilot_manifest_antigravity.json'
    if result.exists() or manifest_path.exists():
        raise RuntimeError('existing pilot state must be inspected, not overwritten')
    source = ROOT / 'experiments/exp495_normal_geometry_antigravity/diagnostics_manifest_antigravity.json'
    population = read(source)['population']
    assert len(population) == len({r['source_family'] for r in population}) == 50
    old = read(ROOT / 'experiments/exp494_all_datasets_low_fpr_antigravity/run_manifest_antigravity.json')
    assert file_sha256_antigravity(DB) == old['binding']['source_binding']['database']
    assert read(P / 'acceleration_results_antigravity.json')['success']
    preflight = read(P / 'preflight_antigravity.json')
    assert preflight['state'] == 'passed'
    for relative, expected in preflight['artifacts'].items():
        assert file_sha256_antigravity(ROOT / relative) == expected, relative
    code = {n['path']: n['sha256'] for n in read(ROOT / 'docs/codemap/codemap_exp508_antigravity.json')['nodes']}
    for path, digest in code.items(): assert file_sha256_antigravity(ROOT / path) == digest, path
    policy = cpu_worker_policy_antigravity()
    manifest = dict(state='running_train_only', population=population, code=code, cpu_policy=policy,
                    source_population_sha256=file_sha256_antigravity(source), database_sha256=file_sha256_antigravity(DB),
                    protocol_sha256=file_sha256_antigravity(P / 'protocol_antigravity.html'),
                    acceleration_sha256=file_sha256_antigravity(P / 'acceleration_results_antigravity.json'),
                    preflight_sha256=file_sha256_antigravity(P / 'preflight_antigravity.json'),
                    test_values_read=False, test_labels_read=False, candidate='tail_rank',
                    backend='official MiniRocket plus canonical NumPy/SciPy float64 tail rank')
    write_json_atomic_antigravity(manifest_path, manifest)
    records = []; failures = []
    with ProcessPoolExecutor(max_workers=policy['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(worker, row): row['dataset'] for row in population}
        for future in as_completed(futures):
            try: records.append(future.result())
            except Exception as error: failures.append(dict(dataset=futures[future], type=type(error).__name__, error=str(error)))
            progress = dict(completed=len(records), total=len(population), failures=failures, seconds=perf_counter() - start)
            write_json_atomic_antigravity(P / 'pilot_progress_antigravity.json', progress)
            print(json.dumps(progress), flush=True)
    if failures:
        manifest.update(state='failed_train_only', failures=failures)
        write_json_atomic_antigravity(manifest_path, manifest)
        raise RuntimeError('TRAIN screening failed; no full evaluation authorized by this result')
    summary = {n: {k: float(np.mean([np.mean([f[k][n] for f in r['folds']]) for r in records])) for k in KINDS} for n in NAMES}
    unseen = {n: {k: float(np.mean([np.mean([f[k][n] for f in r['unseen_folds']]) for r in records])) for k in UNSEEN} for n in NAMES}
    tails = {n: float(np.median([np.mean([f[n] for f in r['normal_tail_by_fold']]) for r in records])) for n in NAMES}
    for path, digest in code.items(): assert file_sha256_antigravity(ROOT / path) == digest, path
    out = dict(state='complete_train_only_pilot', summary=summary, unseen_summary=unseen, normal_tail=tails,
               records=sorted(records, key=lambda r: r['dataset']), test_values_read=False, test_labels_read=False,
               seconds=perf_counter() - start, **select_candidate(summary, tails, unseen))
    write_json_atomic_antigravity(result, out)
    manifest.update(state=out['state'], result_sha256=file_sha256_antigravity(result))
    write_json_atomic_antigravity(manifest_path, manifest)
    print(json.dumps({k: out[k] for k in ('selected', 'best_anchor', 'gates', 'candidate_margin', 'summary', 'unseen_summary')}), flush=True)


if __name__ == '__main__': main()
