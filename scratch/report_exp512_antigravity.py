"""Render verified full TEST evidence, complete CSV comparisons, and close this run."""
from pathlib import Path
from hashlib import file_digest
from datetime import datetime,timezone
from html import escape
from html.parser import HTMLParser
import csv
import json
import re

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'experiments/exp512_exp511_exploratory_test_antigravity'
DOC=ROOT/'docs/EXP512_EXP511_EXPLORATORY_TEST_antigravity.html'
STATUS=ROOT/'docs/CURRENT_MODEL_IMPROVEMENT_STATUS_antigravity.html'
read=lambda p:json.loads(p.read_text())
LABELS={'MiniRocket':'MiniRocket','square_joint':'Exp502','crossfit_rank':'Exp511 고정 앙상블','crossfit_unweighted':'같은 분할 비가중 평균'}
BUDGETS=('FP0','FPR ≤0.1%','FPR ≤0.5%','FPR ≤1%')

def digest(path):
    with path.open('rb') as stream:return file_digest(stream,'sha256').hexdigest()

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.extend(v for k,v in attrs if k=='href')

def check_links(path):
    parser=Links();parser.feed(path.read_text())
    for link in parser.links:
        if not link.startswith(('https:','http:','#')):assert (path.parent/link.split('#')[0]).exists(),link
    return len(parser.links)

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def validate_execution_evidence(exits,manifest,verification,boundary,boundary_sha256):
    """An unavailable historical exit code is not a claimed successful process exit."""
    assert all(exits[key]['exit_code']==0 for key in ('evaluation','verification'))
    scoring=exits['scoring']
    assert scoring['exit_code'] in (None,0)
    if scoring['exit_code'] is None:
        assert scoring.get('exit_code_observed') is False
    assert boundary_sha256==manifest['boundary_sha256']==scoring['boundary_sha256']
    assert boundary['binding_sha256']==manifest['binding_sha256']
    assert not boundary['test_labels_read'] and boundary['datasets']==1117
    assert len(boundary['scores'])==len(boundary['metadata'])==1117
    assert set(boundary['scores'])==set(boundary['metadata'])
    assert verification['state']=='passed' and verification['datasets']==1117
    assert verification['population_and_scores_frozen'] is True

def main():
    assert not DOC.exists() and not (P/'closure_antigravity.json').exists(),'Do not overwrite completed report'
    result=read(P/'results_antigravity.json');v=read(P/'verification_antigravity.json');m=read(P/'manifest_antigravity.json')
    assert v['state']=='passed' and m['state']==result['state']=='complete_exploratory_test'
    assert digest(P/'results_antigravity.json')==v['result_sha256']==m['result_sha256']
    assert not result['model_promotion'] and not result['train_gate_passed']
    for rel,expected in {**m['binding']['code'],**read(P/'preflight_antigravity.json')['artifacts'],**read(P/'authorization_antigravity.json')['historical_artifacts']}.items():
        assert digest(ROOT/rel)==expected,rel
    exits=read(P/'execution_exit_antigravity.json')
    boundary=P/'score_freeze_antigravity.json'
    validate_execution_evidence(exits,m,v,read(boundary),digest(boundary))
    for key in ('scoring','evaluation','verification'):
        assert digest(P/(key+'_execution_antigravity.log'))==exits[key]['log_sha256']
    family_csv=P/'family_comparison_antigravity.csv';dataset_csv=P/'dataset_comparison_antigravity.csv'
    assert digest(family_csv)==result['family_csv_sha256']
    with dataset_csv.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['dataset','source_family','model','budget','tpr','tp','fp','n_normal','n_anomaly'])
        for row in result['records']:
            for name,metric in row['metrics'].items():
                for j,budget in enumerate(BUDGETS):writer.writerow([row['dataset'],row['source_family'],name,budget,metric['tpr'][j],metric['tp'][j],metric['fp'][j],metric['n_normal'],metric['n_anomaly']])
    s=result['summaries'];candidate=s['crossfit_rank']['family_macro_tpr'];comparison=result['comparisons']['square_joint']
    summary=(f'동일한 1,117개 데이터셋·128계열에서 고정 Exp511을 평가했습니다. 주지표인 오탐률0.1% 이하 계열 평균 검출률은 '
             f'MiniRocket {s["MiniRocket"]["family_macro_tpr"][1]:.4%}, Exp502 {s["square_joint"]["family_macro_tpr"][1]:.4%}, '
             f'Exp511 {candidate[1]:.4%}입니다. Exp502 대비 차이는 {comparison["absolute_delta"][1]*100:+.4f}%p입니다.')
    header='<header><p class="meta">2026-09-18 · Exp512 · 고정 Exp511 탐색 TEST 평가</p><h1>실제 TEST 평가와 독립 검산 완료</h1><p>'+summary+'</p><p class="note">사용자 승인에 따른 별도 탐색 평가입니다. 기존 TRAIN 승격 조건 미달 판정을 보존했으며, 모델·설정 재조정이나 자동 승격은 수행하지 않았습니다.</p></header>'
    style='body{font:16px/1.8 system-ui;max-width:1150px;margin:36px auto;padding:0 24px;color:#182b3a}h1,h2{line-height:1.35}section{margin:34px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:11px;border-bottom:1px solid #ccd7e0;text-align:right}td:first-child,th:first-child{text-align:left}.scroll{overflow-x:auto}.note{background:#fff3db;padding:18px}.meta{color:#526576}pre{white-space:pre-wrap;background:#eef3f6;padding:16px}a{color:#16568b}'
    page=f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp512 고정 Exp511 실제 TEST 평가</title><style>{style}</style></head><body>{header}'
    for key,title,detail in (('family_macro_tpr','계열 평균','각 계열 안의 데이터셋을 먼저 평균한 뒤128계열을 같은 비중으로 평균합니다. 주지표입니다.'),('dataset_macro_tpr','데이터셋 평균','1,117개 데이터셋을 같은 비중으로 평균합니다.'),('micro_tpr','이상 표본 수 가중 평균','검출된 이상 표본 수를 전체 이상 표본 수로 나눕니다. 각 데이터셋의 오탐 한도는 개별 적용합니다.')):
        rows=[[label]+[f'{value:.4%}' for value in s[name][key]] for name,label in LABELS.items()]
        page+='<section><h2>'+title+'</h2><p>'+detail+'</p>'+table(['모델',*BUDGETS],rows)+'</section>'
    ci_rows=[]
    for control,item in result['comparisons'].items():
        for i,budget in enumerate(BUDGETS):
            lo,hi=item['absolute_bootstrap95'][i]
            ci_rows.append([LABELS[control],budget,f'{item["absolute_delta"][i]*100:+.4f}%p',f'{lo*100:+.4f} ~ {hi*100:+.4f}%p',f'{item["relative_delta"][i]:+.4%}',f'{item["family_improved"][i]} / {item["family_unchanged"][i]} / {item["family_worsened"][i]}'])
    page+='<section><h2>차이와 불확실성</h2><p>모든 차이는 Exp511−비교 모델입니다. 동일한128계열을 짝지어2000회 재표집한95% 백분위 신뢰구간이며 seed20260721을 사용했습니다. 구간이0을 포함하면 해당 차이의 방향이 이 계산에서 명확하지 않습니다. 반복 실험과 후보 선택 전체에 대한 다중비교 보정은 아닙니다.</p>'+table(['비교 모델','오탐 한도','절대 차이','95% 구간','상대 차이','계열 개선/동일/악화'],ci_rows)+'</section>'
    item=result['comparisons']['square_joint'];ranked=sorted(item['family_deltas'].items(),key=lambda kv:(kv[1][1],kv[0]))
    page+='<section><h2>Exp502 대비 계열별 변화</h2><p>주지표 기준으로 변화가 큰 계열의 양 끝을 표시합니다. 전체128계열과 모든 오탐 한도는 CSV에 빠짐없이 보존했습니다.</p>'+table(['계열','FPR≤0.1% 차이'],[[name,f'{values[1]*100:+.4f}%p'] for name,values in ranked[:10]+ranked[-10:]])+'</section>'
    target='관측 목표 달성' if result['observed_target_met'] else '관측 목표 미달성'
    page+=f'<section><h2>고정 목표: {target}</h2><p>MiniRocket 대비 주지표 상대10% 개선(검출률60.0475723544% 이상)과 FP0 계열 평균 비악화를 함께 확인했습니다. Exp511의 주지표는 {candidate[1]:.4%}, FP0는 {candidate[0]:.4%}입니다. 이 목표 충족 여부와 기존 TRAIN 승격 판정은 별개입니다.</p></section>'
    same=sum(row['metrics']['MiniRocket']['n_normal']//1000==0 for row in result['records'])
    page+=f'<section><h2>검증과 해석 범위</h2><p>회귀 검사 {read(P/"preflight_antigravity.json")["tests_passed"]}개, 실제 TEST 점수 벡터 {v["score_vectors"]:,}개, 기존 두 대조군 벡터 {v["baseline_vectors_replayed"]:,}개의 재현을 확인했습니다. 네 오탐 한도·세 평균·계열 차이·paired bootstrap을 저장 점수로 독립 재계산했습니다.</p><p>정상 TRAIN 표본 부족으로 사전 폴백을 적용한 데이터셋은 {v["fallback_datasets"]}개입니다. 구성 모델 {v["member_count"]:,}개 중 최적화기 성공은 {v["solver_successes"]:,}개, 제약 반영 기울기 기준 통과는 {v["projected_gradient_passes"]:,}개입니다. 나머지도 사전 정의대로 유한한 최종 가중치를 사용했으며 재최적화하지 않았습니다.</p><p>실제 TEST는 이미 여러 번 사용한 개발 벤치마크입니다. TEST 라벨로 오탐 한도에 대응하는 ROC 검출률을 회고적으로 계산했으므로 운영 문턱의 오탐률을 보장하지 않습니다. {same:,}개 데이터셋에서는 정상 TEST 수가1000 미만이라0.1% 한도가FP0와 같습니다. 새로운 장비·미사용 holdout의 성능은 검증하지 않았습니다. 독립 검산은 저장된 점수를 대상으로 하며 원본 표현·기울기의 전면 독립 재학습은 아닙니다.</p></section>'
    if exits['scoring']['exit_code'] is None:
        page+='<section><h2>재개 시 실행 기록</h2><p>이전 점수 생성 프로세스의 원래 종료 코드는 확인하지 못했습니다. 이를 성공 코드로 대체하지 않고, 완료 로그·라벨 조회 전 점수 동결·1,117개 데이터셋의 독립 검산을 완료 근거로 보존했습니다. 이번 평가와 독립 검산 프로세스의 종료 코드는 모두 0입니다. 점수를 다시 생성하지 않았습니다.</p></section>'
    prefix='../experiments/exp512_exp511_exploratory_test_antigravity/'
    page+='<section><h2>근거 파일</h2><p>'+' · '.join(f'<a href="{prefix}{name}">{label}</a>' for name,label in (('protocol_antigravity.html','고정 평가 규칙'),('authorization_antigravity.json','별도 탐색 승인 기록'),('score_freeze_antigravity.json','라벨 조회 전 점수 동결'),('results_antigravity.json','전체 결과'),('verification_antigravity.json','독립 검산'),('family_comparison_antigravity.csv','전체 계열별 차이 CSV'),('dataset_comparison_antigravity.csv','전체 데이터셋·모델별 결과 CSV'),('plan_antigravity.html','실행 계획')))
    page+='</p><p><a href="EXP511_CROSSFIT_RANK_ENSEMBLE_antigravity.html">보존된 Exp511 TRAIN 기각 보고서</a> · <a href="CURRENT_MODEL_IMPROVEMENT_STATUS_antigravity.html">현재 연구 상태</a></p></section></body></html>'
    DOC.write_text(page+'\n')
    current=STATUS.read_text();assert 'id="exp512"' not in current
    current,n=re.subn(r'<header>.*?</header>',header,current,count=1,flags=re.S);assert n==1
    section='<section id="exp512"><h2>고정 Exp511의 실제 TEST 탐색 평가</h2><p>'+summary+'</p><p><a href="EXP512_EXP511_EXPLORATORY_TEST_antigravity.html">전체 비교·신뢰구간·계열별 변화 보고서</a></p></section>'
    STATUS.write_text(current.replace('</header>','</header>'+section,1))
    link_counts={str(path.relative_to(ROOT)):check_links(path) for path in (DOC,STATUS)}
    plan=P/'plan_antigravity.html';text=plan.read_text()
    for step in ('scores','evaluation','report'):text=text.replace(f'id="{step}"><input type="checkbox" disabled',f'id="{step}"><input type="checkbox" checked disabled')
    assert text.count('checked disabled')==5;plan.write_text(text)
    files=[DOC,STATUS,family_csv,dataset_csv,plan,Path(__file__).resolve()]+[P/name for name in ('results_antigravity.json','verification_antigravity.json','manifest_antigravity.json','score_freeze_antigravity.json','preflight_antigravity.json','execution_exit_antigravity.json')]
    closure=dict(state='complete_verified_exploratory_test',utc=datetime.now(timezone.utc).isoformat(),evaluation_model='unchanged_exp511',
        datasets=1117,families=128,train_gate_passed=False,model_promotion=False,observed_target_met=result['observed_target_met'],
        source_code_changed_after_scoring=False,report=str(DOC.relative_to(ROOT)),links_verified=link_counts,
        browser_preview_verified=False,browser_preview_limitation='Previous local-file URL policy restriction respected; no browser workaround.',
        artifacts={str(path.relative_to(ROOT)):digest(path) for path in files})
    (P/'closure_antigravity.json').write_text(json.dumps(closure,indent=2)+'\n')
    print(json.dumps(dict(state=closure['state'],report=str(DOC),links_verified=link_counts)),flush=True)

if __name__=='__main__':main()
