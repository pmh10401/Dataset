"""Publish only verified Exp511 outcomes and seal a rejected TRAIN candidate."""
from pathlib import Path
from hashlib import file_digest
from datetime import datetime,timezone
from html import escape
from statistics import median
import json
import re
import sys
from scratch.report_exp510_antigravity import validate_links
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity

ROOT=Path(__file__).resolve().parents[1];P=ROOT/'experiments/exp511_crossfit_rank_ensemble_antigravity'
DOC=ROOT/'docs/EXP511_CROSSFIT_RANK_ENSEMBLE_antigravity.html'
STATUS=ROOT/'docs/CURRENT_MODEL_IMPROVEMENT_STATUS_antigravity.html'
NAMES=('MiniRocket','oas_distance','square_joint','a_unweighted','separated_rank','crossfit_unweighted','crossfit_rank')
LABELS=('MiniRocket','원시 OAS','Exp502 · 전체 정상 기준','Exp510 · A 비가중','Exp510 · 단일 A/B 학습','네 분할 · 비가중 평균','Exp511 · 네 분할 학습 평균')
read=lambda p:json.loads(p.read_text())


def digest(path):
    with path.open('rb') as stream:return file_digest(stream,'sha256').hexdigest()


def main():
    assert not DOC.exists(), 'Preserve published reports'
    r=read(P/'pilot_results_antigravity.json');v=read(P/'verification_antigravity.json')
    assert v['state']=='passed' and v['result_sha256']==digest(P/'pilot_results_antigravity.json')
    assert v['gates']==r['gates'] and v['full_run_required']==r['full_run_required']
    passed=r['full_run_required'];decision='TRAIN 선별 통과' if passed else 'TRAIN 승격 조건 미달 · Exp502 유지'
    description='전체 평가를 진행할 사전 조건을 충족했습니다.' if passed else '사전 규칙에 따라 전체 1,117개 TEST 평가는 실행하지 않았습니다.'
    metrics={n:(min(r['summary'][n].values()),sum(r['summary'][n].values())/4,sum(r['unseen_summary'][n].values())/4) for n in NAMES}
    for n,values in metrics.items():
        for key,value in zip(('worst_standard','mean_standard','mean_unseen'),values):assert abs(value-r[key][n])<1e-14
    base=metrics['square_joint'];single=metrics['separated_rank'];matched=metrics['crossfit_unweighted'];candidate=metrics['crossfit_rank']
    summary=(f'50계열·250분할의 합성 TRAIN 검증을 완료했습니다. Exp502 대비 기존 4종 평균은 {base[1]:.4%} → {candidate[1]:.4%}, '
             f'학습 제외 4종 평균은 {base[2]:.4%} → {candidate[2]:.4%}입니다. {description}')
    header=f'<header><p class="meta">2026-09-18 · Exp511 평가·독립 검산</p><h1>{decision}</h1><p>{summary}</p><p class="note">합성 TRAIN 선별 결과입니다. 실제 이상 TEST 검출률이나 운영 오탐률의 개선으로 해석하지 않습니다.</p></header>'
    style='body{font:16px/1.8 system-ui;max-width:1150px;margin:32px auto;padding:0 24px;color:#182b3a}h1,h2{line-height:1.4}section{margin:30px 0}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:11px;border-bottom:1px solid #cfd8df;text-align:right}th:first-child{text-align:left}.note{background:#fff3db;padding:16px}.meta{color:#526576}a{color:#16568b}pre{white-space:pre-wrap;background:#eef3f6;padding:16px}'
    rows=''.join('<tr><th>'+escape(label)+'</th>'+''.join(f'<td>{x:.4%}</td>' for x in metrics[n])+f'<td>{r["normal_tail"][n]:.4f}</td></tr>' for n,label in zip(NAMES,LABELS))
    page=f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp511 교차 분할 앙상블 검증</title><style>{style}</style></head><body>{header}'
    page+='<section><h2>일곱 점수의 동일 조건 비교</h2><p>검출은 보류 정상 C의 최대 점수를 엄격히 넘을 때입니다. 검출률은 5분할 평균 후 50계열 동일 가중 평균입니다. 최저값은 기존 네 변형의 계열 평균 중 최솟값입니다. 꼬리 지표는 분할 평균 후 계열 중앙값입니다.</p><div class="scroll"><table><thead><tr><th>모델</th><th>기존 4종 최저</th><th>기존 4종 평균</th><th>학습 제외 4종 평균</th><th>정상 꼬리</th></tr></thead><tbody>'+rows+'</tbody></table></div></section>'
    page+='<section><h2>분할·학습·평균의 기여</h2><p>각 정상은 네 모델에서 B에 한 번, A에 세 번 배정됩니다. 각 모델은 A만으로 표현·정상 기준을 만들고 B로 순위 가중치를 학습합니다. 자기 A의 고정 척도로 나눈 점수 네 개를 같은 비중으로 평균합니다. 같은 네 A를 쓰는 비가중 평균도 함께 측정했습니다.</p><div class="scroll"><table><thead><tr><th>Exp511과의 비교 대상</th><th>최저 차이</th><th>기존 평균 차이</th><th>학습 제외 평균 차이</th></tr></thead><tbody>'
    for label,values in (('Exp502',base),('Exp510 단일 분할',single),('같은 네 분할 비가중 평균',matched)):
        page+='<tr><th>'+label+'</th>'+''.join(f'<td>{(x-y)*100:+.4f}%p</td>' for x,y in zip(candidate,values))+'</tr>'
    page+='</tbody></table></div><p>같은 분할의 비가중 평균과의 차이는 해당 구성에서 가중치 학습을 추가한 효과입니다. 겹치는 표본으로 적합한 네 모델이므로 독립 표본 네 배의 효과를 의미하지 않습니다.</p></section>'
    required=(max(metrics[n][0] for n in NAMES[:-1])+.02,max(metrics[n][1] for n in NAMES[:-1]),max(metrics[n][2] for n in NAMES[:-1]),3*max(r['normal_tail']['MiniRocket'],1e-9))
    actual=(*candidate,r['normal_tail']['crossfit_rank'])
    gate_keys=('minimum_margin','mean_nonregression','unseen_nonregression','normal_tail')
    gate_labels=('기존 변형 최저 +2%p','기존 4종 평균 비악화','학습 제외 4종 평균 비악화','정상 꼬리 상한')
    page+='<section><h2>사전에 고정한 네 조건</h2><div class="scroll"><table><thead><tr><th>조건</th><th>관측값</th><th>필요값</th><th>결과</th></tr></thead><tbody>'
    for i,(key,label) in enumerate(zip(gate_keys,gate_labels)):
        valid=actual[i]>=required[i] if i<3 else actual[i]<=required[i];assert valid==r['gates'][key]
        observed=f'{actual[i]:.4%}' if i<3 else f'{actual[i]:.4f}'
        limit=f'≥ {required[i]:.4%}' if i<3 else f'≤ {required[i]:.4f}'
        page+=f'<tr><th>{label}</th><td>{observed}</td><td>{limit}</td><td>{"통과" if valid else "미달"}</td></tr>'
    page+='</tbody></table></div><p>세 검출 조건은 여섯 대조군을 모두 대상으로 합니다. 결과를 본 뒤 개선 폭·분할·가중치·변형을 변경하지 않았습니다.</p></section>'
    members=[m for row in r['records'] for f in row['folds'] for m in f['members']];assert len(members)==1000
    gaps=[m['diagnostics']['objective_gap_upper_bound'] for m in members]
    preflight=read(P/'preflight_antigravity.json');acc=read(P/'production_backend_assessment_antigravity.json')
    page+=f'<section><h2>검증과 실행 근거</h2><p>회귀 검사 {preflight["tests_passed"]}개를 통과했습니다. 독립 검산에서 {v["arrays_verified"]:,}개 저장 배열, {v["baseline_score_vectors_replayed"]:,}개 기존 점수 벡터, A/B 역할, 네 점수의 평균, 손실·제약 반영 기울기 조건·전체 집계를 확인했습니다.</p><p>구성 모델 1,000개 중 최적화기 성공은 {v["solver_successes"]}개, 제약 반영 기울기 기준 통과는 {v["projected_gradient_passes"]}개입니다. 나머지 {1000-v["projected_gradient_passes"]}개는 사전 규칙대로 유한한 최종 가중치를 사용했습니다. 목적함수 오차 상한은 중앙값 {median(gaps):.6g}, 최댓값 {max(gaps):.6g}입니다.</p>'
    page+='<p>처음 전체 가속 비교의 준비 후 속도는 약 1.14배였으나 기존 다섯 대조군까지 가속 경로로 계산한 측정이었습니다. 다섯 대조군을 그대로 보존하는 실제 경로를 별도로 세 번씩 비교했습니다. 전체 단계 중앙값 속도 배수는 '+', '.join(f'{name} {s["median_speedup"]:.4f}배' for name,s in acc['summary'].items())+f'입니다. 최종 선택은 {escape(preflight["selected_backend"])}입니다. 최초 호출 비용과 반복 측정·점수·순위·후보 일치를 함께 판단했습니다.</p></section>'
    page+='<section><h2>해석의 한계</h2><p>이 실험은 이 고정된 후보의 합성 TRAIN 성능을 평가합니다. 역할 분리 또는 앙상블 일반의 성패, 새로운 장비에서의 일반화, 운영 오탐률 보장을 입증하지 않습니다. 전처리 목표 길이는 C를 포함하는 기존 전체 정상 TRAIN 기준을 상속했습니다. 행 인덱스는 분리했지만 동일 파형 내용의 중복을 제거하지 않았습니다.</p><p>독립 검산은 저장된 점수와 기울기로 손실·제약 조건·평균·검출률·결정을 재계산했습니다. 원본 기여 행렬부터 표현·기울기를 독립 재생성한 검증은 아닙니다. 과거 전체 TEST 벤치마크는 반복 노출된 개발 자료입니다.</p></section>'
    prefix='../experiments/exp511_crossfit_rank_ensemble_antigravity/'
    page+='<section><h2>확인 자료</h2><p>'+' · '.join(f'<a href="{prefix}{filename}">{label}</a>' for filename,label in (('protocol_antigravity.html','사전 규칙'),('pilot_results_antigravity.json','전체 TRAIN 결과'),('verification_antigravity.json','독립 검산'),('preflight_antigravity.json','실행 전 동결'),('code_review_antigravity.json','독립 코드 검토'),('review_resolution_antigravity.json','검토 판단'),('production_backend_assessment_antigravity.json','실제 경로 가속 검증'),('plan_antigravity.html','실행 계획')))
    page+='</p><pre>NUMBA_CACHE_DIR=/private/tmp/exp511_numba_cache_antigravity .venv/bin/python -m experiments.exp511_crossfit_rank_ensemble_antigravity.verify_antigravity</pre><p><a href="CURRENT_MODEL_IMPROVEMENT_STATUS_antigravity.html">현재 연구 상태</a> · <a href="EXP510_SEPARATED_RANK_REVIEW_antigravity.html">이전 Exp510 보고서</a></p></section></body></html>'
    DOC.write_text(page+'\n');old=STATUS.read_text();assert 'id="exp511"' not in old
    changed,count=re.subn(r'<header>.*?</header>',header,old,count=1,flags=re.S);assert count==1
    section='<section id="exp511"><h2>Exp511: 네 분할 순위 앙상블</h2><p>'+summary+'</p><p><a href="EXP511_CROSSFIT_RANK_ENSEMBLE_antigravity.html">일곱 모델·네 조건·독립 검산 보고서</a></p></section>'
    STATUS.write_text(changed.replace('</header>','</header>'+section,1))
    print(json.dumps(dict(report=str(DOC),links={str(f.relative_to(ROOT)):validate_links(f) for f in (DOC,STATUS)}),ensure_ascii=False))


def close():
    target=P/'closure_antigravity.json';assert not target.exists()
    r=read(P/'pilot_results_antigravity.json');v=read(P/'verification_antigravity.json');m=read(P/'pilot_manifest_antigravity.json')
    p=read(P/'preflight_antigravity.json');a=read(P/'authorization_antigravity.json');exits=read(P/'execution_exit_antigravity.json')
    assert not r['full_run_required'] and r['selected']==v['selected']=='square_joint'
    assert v['state']=='passed' and m['state']==r['state']=='complete_train_only_pilot'
    assert exits['pilot']['exit_code']==exits['verification']['exit_code']==0
    assert digest(P/'pilot_results_antigravity.json')==m['result_sha256']==v['result_sha256']
    assert digest(P/'preflight_antigravity.json')==m['binding']['preflight_sha256']
    for relative,expected in {**m['binding']['code'],**p['artifacts'],**a['historical_artifacts']}.items():assert digest(ROOT/relative)==expected,relative
    assert not any(doc[key] for doc in (r,v,m) for key in ('test_values_read','test_labels_read'))
    assert not list(P.glob('full_*'))
    links={str(f.relative_to(ROOT)):validate_links(f) for f in (DOC,STATUS)}
    plan=P/'plan_antigravity.html';text=plan.read_text()
    for step in ('pilot','decision','report'):text=text.replace(f'id="{step}"><input type="checkbox" disabled',f'id="{step}"><input type="checkbox" checked disabled')
    assert text.count('checked disabled')==6;plan.write_text(text)
    files=[P/name for name in ('pilot_results_antigravity.json','pilot_manifest_antigravity.json','verification_antigravity.json',
           'preflight_antigravity.json','authorization_antigravity.json','execution_exit_antigravity.json','plan_antigravity.html')]
    files.extend((DOC,STATUS,Path(__file__).resolve()))
    out=dict(state='complete_verified_train_rejection',utc=datetime.now(timezone.utc).isoformat(),datasets=50,families=50,folds=250,members=1000,
             tests_passed=p['tests_passed'],arrays_verified=v['arrays_verified'],baseline_score_vectors_replayed=v['baseline_score_vectors_replayed'],
             gates=r['gates'],selected=r['selected'],test_values_read=False,test_labels_read=False,full_test_evaluation_executed=False,model_promotion=False,
             source_code_changed_after_run=False,historical_files_preserved=len(a['historical_artifacts']),
             projected_gradient_passes=v['projected_gradient_passes'],solver_successes=v['solver_successes'],
             report=str(DOC.relative_to(ROOT)),report_links_verified=links,browser_preview_verified=False,
             browser_preview_limitation='Previous local-file browser security restriction respected; source structure and links checked without browser workaround.',
             artifacts={str(f.relative_to(ROOT)):digest(f) for f in files})
    write_json_atomic_antigravity(target,out);print(json.dumps({k:out[k] for k in ('state','gates','tests_passed','arrays_verified','report')},ensure_ascii=False))


if __name__=='__main__':
    if sys.argv[1:]==['--close']:close()
    else:
        assert not sys.argv[1:];main()
