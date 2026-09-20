"""Recompute the TRAIN decision independently from persisted fold records."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np

P = Path(__file__).resolve().parent; ROOT = P.parents[1]
read = lambda p: json.loads(p.read_text())
digest = lambda p: sha256(p.read_bytes()).hexdigest()


def main():
    r = read(P / 'pilot_results_antigravity.json'); m = read(P / 'pilot_manifest_antigravity.json')
    assert r['state'] == m['state'] == 'complete_train_only_pilot'
    assert digest(P / 'pilot_results_antigravity.json') == m['result_sha256']
    for path, expected in m['code'].items(): assert digest(ROOT / path) == expected, path
    for name, key in [('protocol_antigravity.html', 'protocol_sha256'), ('acceleration_results_antigravity.json', 'acceleration_sha256'), ('preflight_antigravity.json', 'preflight_sha256')]:
        assert digest(P / name) == m[key]
    source = ROOT / 'experiments/exp495_normal_geometry_antigravity/diagnostics_manifest_antigravity.json'
    assert digest(source) == m['source_population_sha256'] and read(source)['population'] == m['population']
    oldp = ROOT / 'experiments/exp502_amplitude_envelope_antigravity'
    assert digest(oldp / 'pilot_results_antigravity.json') == read(oldp / 'pilot_manifest_antigravity.json')['result_sha256']
    old = {x['dataset']: x for x in read(oldp / 'pilot_results_antigravity.json')['records']}
    names = ('MiniRocket', 'oas_distance', 'square_joint', 'tail_rank')
    kinds = ('local_level', 'local_scale', 'permuted_quarters', 'phase_randomized')
    extras = ('weak_level', 'weak_scale', 'sparse_spike', 'time_warp')
    rows = r['records']; population = {x['dataset']: x for x in m['population']}
    assert len(rows) == len({x['dataset'] for x in rows}) == len({x['source_family'] for x in rows}) == 50
    assert set(population) == {x['dataset'] for x in rows}
    for row in rows:
        meta = population[row['dataset']]; n = meta['n_train']
        assert row['source_family'] == meta['source_family']
        assert all(len(row[key]) == 5 for key in ('folds', 'unseen_folds', 'normal_tail_by_fold', 'fold_boundaries', 'diagnostics'))
        held_all = []
        for f in range(5):
            b = row['fold_boundaries'][f]; fit, held, parents = map(set, (b['fit'], b['held'], b['parents']))
            assert not fit & held and fit | held == set(range(n)) and parents <= fit
            assert len(parents) == min(128, len(fit)) == len(b['parents']) and not parents & held
            rng = np.random.default_rng(20260721 + 508)
            expected = np.sort(rng.choice(len(b['fit']), min(128, len(b['fit'])), replace=False))
            assert np.asarray(b['fit'])[expected].tolist() == b['parents']
            held_all.extend(b['held'])
            audit = row['diagnostics'][f]
            assert audit['optimizer_trained'] and not audit['fallback'] and audit['steps'] == 80
            assert .25 <= audit['minimum_weight'] <= audit['maximum_weight'] <= 4.
            assert np.isfinite([audit['initial_objective'], audit['final_objective'], audit['scale']]).all() and audit['scale'] > 0
            for kind in kinds:
                for name in names:
                    value = row['folds'][f][kind][name]; assert np.isfinite(value) and 0 <= value <= 1
                    if name != 'tail_rank': assert value == old[row['dataset']]['folds'][f][kind][name]
            for kind in extras:
                for name in names:
                    value = row['unseen_folds'][f][kind][name]; assert np.isfinite(value) and 0 <= value <= 1
            assert all(np.isfinite(row['normal_tail_by_fold'][f][name]) and row['normal_tail_by_fold'][f][name] >= 0 for name in names)
        assert sorted(held_all) == list(range(n))
    summary = {name: {kind: sum(sum(f[kind][name] for f in row['folds']) / 5 for row in rows) / 50 for kind in kinds} for name in names}
    unseen = {name: {kind: sum(sum(f[kind][name] for f in row['unseen_folds']) / 5 for row in rows) / 50 for kind in extras} for name in names}
    tails = {name: float(np.median([sum(f[name] for f in row['normal_tail_by_fold']) / 5 for row in rows])) for name in names}
    for name in names:
        for kind in kinds: np.testing.assert_allclose(summary[name][kind], r['summary'][name][kind], atol=1e-14, rtol=0)
        for kind in extras: np.testing.assert_allclose(unseen[name][kind], r['unseen_summary'][name][kind], atol=1e-14, rtol=0)
        np.testing.assert_allclose(tails[name], r['normal_tail'][name], atol=1e-12, rtol=0)
    key = lambda name: (min(summary[name].values()), sum(summary[name].values()) / 4, -names.index(name))
    anchor = max(names[:3], key=key)
    means = {name: sum(unseen[name].values()) / 4 for name in names}
    gates = dict(minimum_margin=key('tail_rank')[0] >= key(anchor)[0] + .02,
                 mean_nonregression=key('tail_rank')[1] >= key(anchor)[1],
                 normal_tail=tails['tail_rank'] <= 3 * max(tails['MiniRocket'], 1e-9),
                 unseen_nonregression=means['tail_rank'] >= max(means[name] for name in names[:3]))
    selected = 'tail_rank' if all(gates.values()) else anchor
    assert gates == r['gates'] and anchor == r['best_anchor'] and selected == r['selected']
    assert not any(doc[key] for doc in (r, m) for key in ('test_values_read', 'test_labels_read'))
    out = dict(state='passed_train_screen', datasets=50, folds=250, selected=selected, gates=gates,
               baseline_fold_recalls_exact=True, fit_held_parent_isolation_verified=True, source_hashes_passed=True,
               result_sha256=digest(P / 'pilot_results_antigravity.json'), full_run_required=selected == 'tail_rank',
               test_labels_read=False, verifier_sha256=digest(Path(__file__)))
    (P / 'verification_antigravity.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))


if __name__ == '__main__': main()
