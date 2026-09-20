"""Verify retrospective overlaps with index sets against the frozen score boundaries."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from contextlib import closing
from hashlib import sha256
import json
import numpy as np
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import DATABASE_PATH_ANTIGRAVITY, readonly_sqlite_antigravity, sha256_json_antigravity
from experiments.exp479_missing_score_recovery_antigravity.evaluation_antigravity import load_retained_test_labels_antigravity

P = Path(__file__).resolve().parent; ROOT = P.parents[1]
read = lambda p: json.loads(p.read_text())
digest = lambda p: sha256(p.read_bytes()).hexdigest()


def main():
    r = read(P / 'diagnosis_results_antigravity.json')
    paths = [ROOT / 'experiments' / f for f in ('exp502_amplitude_envelope_antigravity', 'exp507_roughness_compat_antigravity')]
    boundaries = [read(p / 'full_labels_opened_antigravity.json') for p in paths]
    sources = [read(p / 'full_results_antigravity.json') for p in paths]
    for p, boundary, source in zip(paths, boundaries, sources):
        assert digest(p / 'full_results_antigravity.json') == r['source_results_sha256'][str(p.relative_to(ROOT))]
        assert digest(p / 'full_labels_opened_antigravity.json') == source['boundary_sha256']
        assert digest(DATABASE_PATH_ANTIGRAVITY) == read(p / 'full_manifest_antigravity.json')['binding']['database_sha256']
    rows = r['records']; assert len(rows) == len({x['dataset'] for x in rows}) == 1117
    normal_max_overlap = 0; sensitivity = [0, 0]
    with closing(readonly_sqlite_antigravity(DATABASE_PATH_ANTIGRAVITY)) as con:
        for row in rows:
            name = row['dataset']; arrays = []; meta = []
            for p, boundary in zip(paths, boundaries):
                m = read(p / 'full_scores_antigravity' / f'{name}_meta_antigravity.json')
                f = p / 'full_scores_antigravity' / f'{name}_scores_antigravity.npz'
                assert digest(f) == m['score_sha256'] == boundary['scores'][name]
                assert sha256_json_antigravity(m) == boundary['metadata'][name]
                with np.load(f) as values: arrays.append(values['candidate'].copy())
                meta.append(m)
            assert meta[0]['test_indices'] == meta[1]['test_indices']
            y = load_retained_test_labels_antigravity(con, name, tuple(meta[0]['test_indices']))
            pos = set(np.flatnonzero(y == 1)); neg = set(np.flatnonzero(y == 0))
            assert len(pos) == row['n_anomaly'] and len(neg) == row['n_normal']
            maxima = []
            for j, score in enumerate(arrays):
                ns = sorted((float(score[i]) for i in neg), reverse=True)
                maximum = {meta[j]['test_indices'][i] for i in neg if score[i] == ns[0]}
                assert maximum == set(row['details'][j]['max_normal_test_indices']); maxima.append(maximum)
                found = sum(ns[min(1, len(ns) - 1)] < score[i] <= ns[0] for i in pos)
                assert found == row['details'][j]['recovered_if_one_fp_allowed']; sensitivity[j] += int(found)
                q1, median, q3 = np.percentile(score[list(neg)], [25, 50, 75])
                np.testing.assert_allclose((ns[0] - median) / (q3 - q1 + 1e-9), row['details'][j]['normal_tail'], rtol=1e-12, atol=1e-12)
            normal_max_overlap += bool(maxima[0] & maxima[1])
            for index, numerator in enumerate((0, 1, 5, 10)):
                budget = len(neg) * numerator // 1000
                sets = []
                for score in arrays:
                    threshold = sorted((float(score[i]) for i in neg), reverse=True)[budget]
                    sets.append({i for i, value in enumerate(score) if value > threshold})
                a, b = sets
                expected = dict(fp_budget=budget, both=len(a & b & pos), only502=len((a - b) & pos),
                                only507=len((b - a) & pos), neither=len(pos - a - b), union_fp=len((a | b) & neg))
                assert expected == row['budgets'][index]
    assert sensitivity == r['one_fp_sensitivity'] and normal_max_overlap == r['normal_max_index_overlap_datasets']
    for index in range(4):
        expected = {key: sum(row['budgets'][index][key] for row in rows) for key in ('both', 'only502', 'only507', 'neither', 'union_fp')}
        expected['union_exceeds_dataset_budget'] = sum(row['budgets'][index]['union_fp'] > row['budgets'][index]['fp_budget'] for row in rows)
        assert expected == r['summary'][index]
    families = sorted({row['source_family'] for row in rows})
    oracle = [float(np.mean([np.mean([(x['budgets'][i]['both'] + x['budgets'][i]['only502'] + x['budgets'][i]['only507']) / x['n_anomaly'] for x in rows if x['source_family'] == f]) for f in families])) for i in range(4)]
    out = dict(state='passed', datasets=1117, families=len(families), frozen_boundaries_verified=True,
               overlaps_independently_recomputed=True, result_sha256=digest(P / 'diagnosis_results_antigravity.json'),
               verifier_sha256=digest(Path(__file__)), descriptive_oracle_union_family_macro=oracle,
               union_not_deployable_score=True, no_same_budget_claim=True, no_parameter_selection=True)
    (P / 'diagnosis_verification_antigravity.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out))


if __name__ == '__main__': main()
