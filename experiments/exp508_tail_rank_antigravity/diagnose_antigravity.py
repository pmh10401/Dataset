"""Read-only retrospective audit of frozen Exp502/507 scores; no candidate tuning."""
from pathlib import Path
from contextlib import closing
import json
import numpy as np
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import ROOT, DB, readonly_sqlite_antigravity, file_sha256_antigravity, write_json_atomic_antigravity
from experiments.exp479_missing_score_recovery_antigravity.evaluation_antigravity import load_retained_test_labels_antigravity

P = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())


def main():
    target = P / 'diagnosis_results_antigravity.json'
    if target.exists(): raise RuntimeError('diagnostic already complete; do not overwrite')
    paths = [ROOT / 'experiments' / f for f in ('exp502_amplitude_envelope_antigravity', 'exp507_roughness_compat_antigravity')]
    results = [read(p / 'full_results_antigravity.json') for p in paths]
    inputs = {}
    for p, result in zip(paths, results):
        verify = read(p / 'full_verification_antigravity.json'); manifest = read(p / 'full_manifest_antigravity.json')
        digest = file_sha256_antigravity(p / 'full_results_antigravity.json')
        assert verify['state'] == 'passed' and digest == verify['result_sha256'] == manifest['result_sha256']
        assert file_sha256_antigravity(DB) == manifest['binding']['database_sha256']
        inputs[str(p.relative_to(ROOT))] = digest
    population = {r['dataset']: r['source_family'] for r in results[0]['records']}
    assert population == {r['dataset']: r['source_family'] for r in results[1]['records']} and len(population) == 1117
    stored = [{r['dataset']: r['metrics']['candidate'] for r in result['records']} for result in results]
    records = []
    with closing(readonly_sqlite_antigravity(DB)) as con:
        for dataset, family in sorted(population.items()):
            metas = [read(p / 'full_scores_antigravity' / f'{dataset}_meta_antigravity.json') for p in paths]
            assert metas[0]['test_indices'] == metas[1]['test_indices']
            y = load_retained_test_labels_antigravity(con, dataset, tuple(metas[0]['test_indices']))
            normal = y == 0; positive = ~normal; scores = []
            for p, meta in zip(paths, metas):
                path = p / 'full_scores_antigravity' / f'{dataset}_scores_antigravity.npz'
                assert file_sha256_antigravity(path) == meta['score_sha256']
                with np.load(path) as f: scores.append(f['candidate'].copy())
            details = []
            for j, score in enumerate(scores):
                negative = score[normal]; ordered = np.sort(negative)[::-1]
                maximum = float(ordered[0]); quartiles = np.quantile(negative, [.25, .5, .75])
                details.append(dict(normal_tail=float((maximum - quartiles[1]) / (quartiles[2] - quartiles[0] + 1e-9)),
                                    recovered_if_one_fp_allowed=int(((score[positive] > ordered[min(1, len(ordered) - 1)]) & (score[positive] <= maximum)).sum()),
                                    max_normal_test_indices=np.asarray(metas[j]['test_indices'])[normal][negative == maximum].astype(int).tolist()))
            budgets = []
            for index, numerator in enumerate((0, 1, 5, 10)):
                b = int(normal.sum()) * numerator // 1000
                masks = [s > np.sort(s[normal])[::-1][b] for s in scores]
                for j, mask in enumerate(masks):
                    assert int((mask & positive).sum()) == stored[j][dataset]['tp'][index]
                    assert int((mask & normal).sum()) == stored[j][dataset]['fp'][index]
                a, c = masks
                budgets.append(dict(fp_budget=b, both=int((a & c & positive).sum()),
                                    only502=int((a & ~c & positive).sum()), only507=int((~a & c & positive).sum()),
                                    neither=int((~a & ~c & positive).sum()), union_fp=int(((a | c) & normal).sum())))
            records.append(dict(dataset=dataset, source_family=family, n_normal=int(normal.sum()), n_anomaly=int(positive.sum()),
                                details=details, budgets=budgets))
    summary = []
    for index in range(4):
        totals = {key: sum(r['budgets'][index][key] for r in records) for key in ('both', 'only502', 'only507', 'neither', 'union_fp')}
        totals['union_exceeds_dataset_budget'] = sum(r['budgets'][index]['union_fp'] > r['budgets'][index]['fp_budget'] for r in records)
        summary.append(totals)
    out = dict(state='complete_retrospective_diagnosis', datasets=len(records), families=len(set(population.values())),
               source_results_sha256=inputs, design_sha256=read(P / 'resume_record_antigravity.json')['implementation_design_sha256'],
               fpr001_equals_fp0=sum(r['n_normal'] < 1000 for r in records), summary=summary,
               one_fp_sensitivity=[sum(r['details'][j]['recovered_if_one_fp_allowed'] for r in records) for j in range(2)],
               normal_max_index_overlap_datasets=sum(bool(set(r['details'][0]['max_normal_test_indices']) & set(r['details'][1]['max_normal_test_indices'])) for r in records),
               records=records, test_labels_read=True, test_values_read=False, candidate_tuning_allowed=False,
               limitations='Union is a descriptive oracle, not a learned ensemble or same-budget result outside FP0. One-FP sensitivity changes the budget and proves no cause. Scores alone cannot locate lost waveform information. Rows across derived datasets may represent the same source instances.')
    write_json_atomic_antigravity(target, out)
    print(json.dumps({key: out[key] for key in ('state', 'datasets', 'fpr001_equals_fp0', 'summary', 'one_fp_sensitivity', 'normal_max_index_overlap_datasets')}))


if __name__ == '__main__': main()
