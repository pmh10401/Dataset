#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp201_205.core import binary_metrics, wilson_interval
from experiments.exp206_210.artifacts import (
    freeze_predictions, load_dataset_series_only, load_population_values_only,
    load_test_labels_for_evaluation, load_verified_predictions,
)
from experiments.exp206_210.core import (
    ALPHAS,
    SEED,
    empirical_upper_pvalues,
    fold_plan,
    sha256_file,
)
from experiments.exp211_214.core import (
    aggregate_fold_pvalues,
    pooled_cross_conformal_pvalues,
    source_stability,
    transport_metrics,
)
from experiments.exp215_217.core import fit_shape_source


OUTPUT_ROOT = Path('/Users/minho/Documents/Dataset/exp215_217_shapelet_stability')
META = {
    '215': 'experiment_215_fixed_dictionary_transport_control',
    '216': 'experiment_216_normal_consensus_medoid_transport',
    '217': 'experiment_217_train_normal_shapelet_risk_audit',
    '218': 'experiment_218_true_bootstrap_consensus_normal_prototypes',
    '219': 'experiment_219_fixed_dictionary_frozen_candidate_audit',
}
MODE = {
    '215': 'fixed_dictionary_control',
    '216': 'normal_consensus_medoid',
    '218': 'bootstrap_consensus_medoid',
    '219': 'fixed_dictionary_control',
}


def artifact_source_stage(stage: str) -> str:
    """Exp219 is a read-only candidate/risk audit of frozen Exp215 artifacts."""
    return '215' if stage == '219' else stage


def prediction_aggregations(stage: str) -> tuple[str, ...]:
    if stage == '219':
        return ('conservative_max', 'pooled_cross_conformal')
    return ('pooled_cross_conformal',)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader(); writer.writerows(rows)


def artifact_path(root: Path, stage: str, dataset: str) -> Path:
    return root / f'exp{stage}_cache' / f'{dataset}.npz'


def build_artifact(dataset: str, stage: str, root: Path) -> dict:
    x_train, x_test, _ = load_dataset_series_only(dataset)
    mode = MODE[stage]
    plan = fold_plan(len(x_train), 5, SEED)
    oof = np.zeros(len(x_train), dtype=np.float32)
    fold_counts, fold_sizes, normal_flags = [], [], []
    for number, (fit_idx, hold_idx) in enumerate(plan):
        fit_scores, held_scores = fit_shape_source(mode, x_train[fit_idx], x_train[hold_idx], seed=SEED + number)
        oof[hold_idx] = held_scores
        normal_flags.append(empirical_upper_pvalues(fit_scores, held_scores))
        _, test_scores = fit_shape_source(mode, x_train[fit_idx], x_test, seed=SEED + number)
        ordered = np.sort(fit_scores)
        fold_counts.append(len(ordered) - np.searchsorted(ordered, test_scores, side='left'))
        fold_sizes.append(len(ordered))
    full_train, full_test = fit_shape_source(mode, x_train, x_test, seed=SEED)
    payload = {
        'oof': oof, 'full_train': full_train, 'full_test': full_test,
        'test_counts': np.asarray(fold_counts, dtype=np.int32),
        'fit_sizes': np.asarray(fold_sizes, dtype=np.int32),
        'normal_pvalues': np.concatenate(normal_flags),
    }
    path = artifact_path(root, stage, dataset); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return payload


def load_or_build(dataset: str, stage: str, root: Path) -> dict:
    source_stage = artifact_source_stage(stage)
    path = artifact_path(root, source_stage, dataset)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return {name: np.asarray(data[name]) for name in data.files}
    return build_artifact(dataset, source_stage, root)


def _transport_row(entry: dict, stage: str, root: Path) -> dict:
    data=load_or_build(entry['dataset'], stage, root)
    metrics=transport_metrics(data['oof'], data['full_train'], data['full_test'])
    return {'dataset':entry['dataset'], 'n_train':entry['n_train'], 'n_test':entry['n_test'], **metrics,
        'stable_train_only':int(source_stability(metrics)), 'dictionary_mode':MODE[stage],
        'contains_test_labels':0, 'metric_scope':'label_free_transport_diagnostic'}


def run_transport(stage: str, entries: list[dict], root: Path, workers: int) -> list[dict]:
    rows=[]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures={pool.submit(_transport_row, entry, stage, root):entry['dataset'] for entry in entries}
        for number, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if number % 5 == 0 or number == len(entries): print(f'Progress: [{number} / {len(entries)}] rows={len(rows)} dataset={futures[future]}', flush=True)
    return sorted(rows, key=lambda row: row['dataset'])


def run_prediction(entries: list[dict], root: Path, stage: str) -> list[dict]:
    rows=[]
    for entry in entries:
        data=load_or_build(entry['dataset'], stage, root)
        excluded=set(entry['hard_indices'])|set(entry['standard_indices'])|set(entry['priority_indices'])
        for aggregation in prediction_aggregations(stage):
            if aggregation == 'pooled_cross_conformal':
                pvalues=pooled_cross_conformal_pvalues(data['test_counts'], data['fit_sizes'])
            else:
                fold_pvalues=(1.0 + np.asarray(data['test_counts'], dtype=float)) / (
                    1.0 + np.asarray(data['fit_sizes'], dtype=float)[:, None]
                )
                pvalues=aggregate_fold_pvalues(fold_pvalues, aggregation)
            for alpha in ALPHAS:
                indices=set(np.flatnonzero(pvalues <= alpha).tolist())-excluded
                rows.append({'experiment_id':META[stage], 'dataset':entry['dataset'], 'variant':f'{MODE[stage]}_{aggregation}__{alpha}', 'aggregation':aggregation, 'alpha':alpha,
                    'n_train':entry['n_train'], 'n_test':entry['n_test'], 'candidate_indices':sorted(indices), 'candidate_count':len(indices),
                    'final_lane':'research_sensitivity_only', 'autonomous_alert':False, 'contains_test_labels':False, 'contains_evaluation_metrics':False})
    return rows


def evaluate(rows: list[dict]) -> list[dict]:
    output=[]
    for row in rows:
        labels=load_test_labels_for_evaluation(row['dataset'])
        base=binary_metrics(labels, [])
        cand=binary_metrics(labels, row['candidate_indices'])
        output.append({**row, 'candidate_tp':cand['tp'], 'candidate_fp':cand['fp'], 'candidate_precision':cand['precision'], 'candidate_f1':cand['f1'], 'metric_scope':'retrospective_research_candidate', 'autonomous_exp137_unchanged':1})
    return output


def run_risk(entries: list[dict], root: Path, stage: str = '216') -> list[dict]:
    rows=[]
    for entry in entries:
        data=load_or_build(entry['dataset'], stage, root)
        normal_pvalues=np.asarray(data['normal_pvalues'], dtype=float)
        for alpha in ALPHAS:
            alerts=int(np.sum(normal_pvalues <= alpha)); total=len(normal_pvalues); _, upper=wilson_interval(alerts,total)
            rows.append({'dataset':entry['dataset'],'n_train':entry['n_train'],'alpha':alpha,'normal_pseudo_runs':total,'normal_false_alerts':alerts,'normal_false_alert_rate':alerts/max(1,total),'wilson_upper_95':upper,'contains_test_labels':0,'metric_scope':'train_normal_empirical_risk_diagnostic'})
    return rows


def summarize(rows: list[dict], stage: str) -> list[dict]:
    if stage in {'215','216','218'} and rows and 'stable_train_only' in rows[0]:
        return [{'datasets':len(rows),'stable_datasets':sum(int(r['stable_train_only']) for r in rows),'median_rank_corr':float(np.median([r['oof_full_rank_corr'] for r in rows])),'dictionary_mode':MODE[stage],'metric_scope':'label_free_transport_diagnostic'}]
    if stage in {'216', '218', '219'}:
        out=[]
        for variant in sorted({r['variant'] for r in rows}):
            selected=[r for r in rows if r['variant']==variant]
            tp=sum(r['candidate_tp'] for r in selected); fp=sum(r['candidate_fp'] for r in selected)
            primary = stage == '219' and variant == 'fixed_dictionary_control_conservative_max__0.005'
            out.append({'variant':variant,'datasets':len(selected),'candidate_count':tp+fp,'candidate_tp':tp,'candidate_fp':fp,'candidate_precision':tp/max(1,tp+fp),'metric_scope':'retrospective_research_candidate','autonomous_exp137_unchanged':1,'preregistered_primary':int(primary),'selected_for_operation':0})
        return out
    grouped={}
    for row in rows: grouped.setdefault(row['alpha'],[]).append(row)
    return [{'alpha':alpha,'datasets':len(group),'normal_false_alerts':sum(r['normal_false_alerts'] for r in group),'normal_pseudo_runs':sum(r['normal_pseudo_runs'] for r in group),'max_dataset_wilson_upper_95':max(r['wilson_upper_95'] for r in group),'metric_scope':'train_normal_empirical_risk_diagnostic'} for alpha,group in grouped.items()]


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=('215','216','217','218','219'),required=True); parser.add_argument('--dataset-limit',type=int); parser.add_argument('--workers',type=int,default=7); parser.add_argument('--output-dir',type=Path,default=OUTPUT_ROOT); args=parser.parse_args()
    entries=load_population_values_only(args.dataset_limit); root=args.output_dir
    if args.stage in {'215','216','218'}:
        audit=run_transport(args.stage,entries,root,args.workers)
        write_csv(root/f'{META[args.stage]}_transport_results.csv',audit)
        write_csv(root/f'{META[args.stage]}_transport_summary.csv',summarize(audit,args.stage))
        if args.stage == '215':
            write_csv(root/f'{META[args.stage]}_results.csv',audit)
            write_csv(root/f'{META[args.stage]}_summary.csv',summarize(audit,args.stage))
        if args.stage in {'216','218'}:
            frozen=run_prediction(entries,root,args.stage); freeze_predictions(args.stage,META[args.stage],frozen,root,{})
            evaluated=evaluate(frozen); write_csv(root/f'{META[args.stage]}_results.csv',evaluated); write_csv(root/f'{META[args.stage]}_summary.csv',summarize(evaluated,args.stage))
            risk=run_risk(entries,root,args.stage); write_csv(root/f'{META[args.stage]}_train_normal_risk_results.csv',risk); write_csv(root/f'{META[args.stage]}_train_normal_risk_summary.csv',summarize(risk,'217'))
    elif args.stage == '219':
        frozen=run_prediction(entries,root,args.stage)
        input_hashes={
            f'exp215_cache/{entry["dataset"]}.npz': sha256_file(artifact_path(root, '215', entry['dataset']))
            for entry in entries
        }
        prediction_path, manifest_path=freeze_predictions(args.stage,META[args.stage],frozen,root,input_hashes)
        verified=load_verified_predictions(prediction_path,manifest_path)
        evaluated=evaluate(verified)
        write_csv(root/f'{META[args.stage]}_results.csv',evaluated)
        write_csv(root/f'{META[args.stage]}_summary.csv',summarize(evaluated,args.stage))
        risk=run_risk(entries,root,args.stage)
        write_csv(root/f'{META[args.stage]}_train_normal_risk_results.csv',risk)
        write_csv(root/f'{META[args.stage]}_train_normal_risk_summary.csv',summarize(risk,'217'))
    else:
        rows=run_risk(entries,root)
        write_csv(root/f"{META['217']}_results.csv",rows)
        write_csv(root/f"{META['217']}_summary.csv",summarize(rows,args.stage))
    (root/f'exp{args.stage}_completion.json').write_text(json.dumps({'experiment_id':META[args.stage],'status':'completed','datasets':len(entries)},indent=2))


if __name__=='__main__': main()
