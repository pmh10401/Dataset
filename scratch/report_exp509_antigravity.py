"""Publish the verified Exp509 diagnostic and retain all historical experiment records."""
from pathlib import Path
from hashlib import sha256
from html import escape
from html.parser import HTMLParser
import datetime
import json
import re
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];P=ROOT/'experiments/exp509_rank_signal_diagnosis_antigravity'
read=lambda p:json.loads(p.read_text());digest=lambda p:sha256(p.read_bytes()).hexdigest()
r=read(P/'results_antigravity.json');v=read(P/'verification_antigravity.json');s=r['summary'];folds=[f for row in r['records'] for f in row['folds']]
assert v['state']=='passed' and v['result_sha256']==digest(P/'results_antigravity.json')
assert read(P/'run_exit_antigravity.json')['exit_code']==0
suite=ET.parse(P/'tests_all_antigravity.xml').getroot().find('testsuite')
assert suite.attrib['tests']=='13' and suite.attrib['errors']==suite.attrib['failures']==suite.attrib['skipped']=='0'
prior=ROOT/'experiments/exp502_amplitude_envelope_antigravity';best=read(prior/'full_results_antigravity.json')
assert read(prior/'full_verification_antigravity.json')['result_sha256']==digest(prior/'full_results_antigravity.json')
assert not r['exp510_started'] and not r['model_promotion'] and not r['test_values_read'] and not r['test_labels_read']
report=ROOT/'docs/EXP509_SIGNAL_DIAGNOSIS_antigravity.html';assert not report.exists()
pct=lambda x:f'{100*x:.2f}%'
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
style='body{font:16px/1.75 system-ui;max-width:1080px;margin:36px auto;padding:0 24px;color:#1b2939;background:#f4f6fa}header,section{background:white;border:1px solid #d7e0ea;border-radius:12px;margin:18px 0;padding:26px}h1{font-size:30px;line-height:1.4}h2{font-size:23px;line-height:1.4}.meta{font-size:14px;color:#52667c}.note{background:#fff3d9;padding:16px;border-left:4px solid #a97b26}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:12px;border-bottom:1px solid #dae2ea;text-align:right}td:first-child,th:first-child{text-align:left}a{color:#24599f}pre{white-space:pre-wrap;background:#f0f4f8;padding:14px;font-size:13px}code{overflow-wrap:anywhere}'
body='<header><p class="meta">2026-09-18 · Exp509 · 정상 TRAIN 50계열·250분할 · 독립 검산 완료</p><h1>최적화 교체보다 학습용 정상의 역할 분리가 우선입니다</h1><p>같은 특징·손실·가중치 범위에서 최적화를 개선했지만 보류 합성 변형의 검출률은 기존 Exp502를 넘지 못했습니다. 반면 같은 거리 참조와 표본 수를 사용한 비교에서, 표현 학습 참여 여부에 따른 정상 점수 차이가 50개 계열 모두에서 관측됐습니다.</p><p class="note">Exp509는 학습 신호를 점검하는 진단입니다. 실제 이상 TEST를 평가하지 않았고 새 모델을 승격하지 않았습니다. 아래 검출률은 보류 정상 TRAIN에서 만든 합성 변형의 검출률입니다. 전체 1,117개에서의 기존 최고 관측 개선은 Exp502의 상대 +3.73%로 유지되며, 10% 목표는 아직 미달성입니다.</p></header>'
control=s['participation_family_medians'];ratio=s['family_median_normal_max_ratio']
body+='<section><h2>1. 표현 학습 참여 여부에서 큰 차이가 확인됐습니다</h2><p>외부 적합 F를 거리 참조 R, 표현 학습 참여 비교군 S, 표현 학습 제외 비교군 B로 나눴습니다. 표현·스케일러·OAS는 R+S에서만 맞추고, S와 B 모두 같은 R만 거리 참조로 사용했습니다. 두 비교군의 표본 수와 변형 시드도 같으며 어느 쪽도 거리 참조 자체에 포함되지 않았습니다.</p>'
body+=table(['진단 지표','표현 학습 참여 S','표현 학습 제외 B'],[
    ('정상 상단 점수',f"{control['seen']['normal_max']:.4f}",f"{control['unseen']['normal_max']:.4f}"),
    ('정상 점수 중앙값',f"{control['seen']['normal_median']:.4f}",f"{control['unseen']['normal_median']:.4f}"),
    ('초기 데이터 손실',f"{control['seen']['data_loss']:.8g}",f"{control['unseen']['data_loss']:.8g}"),
    ('sigmoid≥0.01 합성 표본 비율',pct(control['seen']['active_001_fraction']),pct(control['unseen']['active_001_fraction']))])
body+=f'<p>표의 값은 계열 안에서 다섯 분할의 중앙값을 구한 뒤 50계열의 중앙값을 구했습니다. 별도로 분할별 B/S 비율을 같은 방식으로 집계하면 정상 상단 점수 비율은 <strong>{ratio:.2f}배</strong>입니다. B의 정상 상단이 더 높은 방향은 <strong>{s["families_unseen_max_higher"]}/50계열</strong>에서 일치했습니다.</p><p>표현 학습에 참여한 정상은 합성 이상과 이미 쉽게 분리되는 반면, 참여하지 않은 정상에서는 학습 손실에 유의하게 기여하는 합성 표본이 더 많았습니다. 이는 순위 학습용 정상의 역할 분리를 시험할 근거입니다. 학습 신호가 정확히 0이라는 뜻은 아니며, sigmoid 기준은 사전에 고정한 진단 정의입니다.</p><p class="note">표현·스케일러·OAS 중 특정 구성 요소만의 인과 효과를 분리한 실험은 아닙니다. 원래 정상 TRAIN에서 결정한 길이·행별 정규화는 상속했습니다. 이 차이가 Exp510의 성능 향상을 보장하지 않으며, 센서의 원시 측정값이 15배 달라졌다는 의미도 아닙니다.</p></section>'
med=s['training_medians']
body+='<section><h2>2. 최적화는 좋아졌지만 검출률은 개선되지 않았습니다</h2>'
body+=table(['최적화 상태','목적함수 중앙값','제약 반영 기울기 중앙값','목적함수 오차 상한 중앙값'],[(label,f"{med[name]['objective']:.9g}",f"{med[name]['projected_gradient_inf']:.3g}",f"{med[name]['objective_gap_upper_bound']:.3g}") for name,label in [('initial','초기 가중치 1'),('adam','기존 Adam 80회'),('lbfgsb','고정 L-BFGS-B')]])
unresolved=[f for f in folds if not f['solver']['projected_gradient_passed']]
body+=f'<p>기존 Adam은 {s["adam_objective_increased"]}/250분할에서 초기값보다 손실이 증가했습니다. L-BFGS-B에서는 초기값 대비 1e-12를 넘는 손실 증가가 {s["lbfgsb_objective_above_initial"]}개였습니다. 최대 반복 횟수는 {s["lbfgsb_max_iterations"]}회이며, 설정한 2,000회 한도를 확대하거나 재시작하지 않았습니다.</p><p>최적화기는 250분할 모두 성공을 반환했지만, 독립 계산한 projected-gradient ≤1e-8 조건을 통과한 것은 <strong>{s["lbfgsb_projected_gradient_passes"]}/250분할</strong>입니다. 남은 {len(unresolved)}개를 수렴 확정으로 처리하지 않았습니다. 작은 기울기나 목적함수 오차 상한은 이 학습 대리지표에 대한 수치적 근거이며 일반화 성능의 증명은 아닙니다.</p>'
validation_rows=[]
for name,label in [('square_joint','기존 Exp502'),('adam','Exp508 · Adam'),('lbfgsb','Exp509 · 최적화기만 변경')]:
    x=s['metrics'][name];validation_rows.append((label,pct(x['worst_standard']),pct(x['mean_standard']),pct(x['mean_unseen'])))
validation_table=table(['동일 보류 분할 비교','기존 변형 최저 검출률','기존 4종 평균','검증 전용 4종 평균'],validation_rows)
body+=validation_table
body+='<p>분할별 검출률을 먼저 평균하고 50계열을 동일 가중했습니다. L-BFGS-B의 검증 전용 평균은 Adam보다 극히 소폭 높았지만, 기존 Exp502보다 낮았습니다. 손실을 더 정확히 줄이는 것만으로 기존 후보를 넘지 못했다는 결과이며, 모든 최적화 방식이 불필요하다는 결론은 아닙니다.</p></section>'
body+=f'<section><h2>3. 다음 설계의 우선순위</h2><p><strong>Exp510에서는 정상 기준 집합 A, 순위 학습 집합 B, 외부 검증 집합 C를 분리하는 방향을 우선합니다.</strong> A에서만 표현·스케일러·OAS·참조를 만들고, A에 참여하지 않은 B의 정상과 그 변형으로 가중치를 학습해야 합니다. 학습 후 A+B로 표현을 다시 맞추어 점수 공간을 바꾸지 않는 첫 비교가 필요합니다.</p><p>참조 감소의 영향을 구분하기 위해 기존 Exp502, A만 사용하는 비가중 모델, 같은 A와 별도 B를 사용하는 가중 모델을 함께 비교해야 합니다. 약해진 대조군만 이겨서는 성공으로 보지 않습니다. 분할 비율·표본 하한·종료 조건을 실행 전에 고정하고 기존 선별 기준을 유지합니다.</p><p>분할별 초기 손실에서 계산한 비악화 해의 RMS 가중치 변화 상한의 중앙값은 {100*s["median_nonworsening_rms_bound"]:.2f}%입니다. 이는 현재 학습 문제가 이미 쉬운 상태에서 반복 횟수만 늘려도 큰 변화가 나오기 어려울 수 있음을 설명합니다. 일부 특징의 큰 변화나 검출률 개선 가능성 자체를 배제하는 수치는 아닙니다.</p><p class="note">이번에 Exp510은 실행하지 않았습니다. Exp509의 승인 범위인 진단과 후속 설계 근거 정리까지만 완료했습니다. 전체 목표는 FPR≤0.1% 계열 평균 60.0476% 이상, FP0 비악화이며 변경하지 않았습니다.</p></section>'
body+='<section><h2>4. 검증 범위와 산출물</h2><p>기존 Exp508 회귀 검사 8개와 새 진단 검사 5개, 총 13개가 통과했습니다. 새 검사는 경계 제약 최적화, 수치적 정류 조건, 강볼록 목적함수 오차 상한, 손실 통계, 역할 분리, 원래 최적화 관측과 거리 항등식을 확인합니다. 모든 분할에서 기존 Exp508의 네 모델·여덟 변형 검출률과 정상 꼬리·학습 목적함수를 재현했습니다.</p><p>별도 검산기가 15,250개 저장 배열의 해시를 확인하고 손실·제약 반영 기울기 조건·역할 분리·검출률·집계를 다시 계산했습니다. 전체 기울기 벡터 자체는 원래 검증된 목적함수가 생성했으며, 새로 모든 특징 행렬을 저장해 기울기를 독립 재생성한 것은 아닙니다. 원래 기울기의 수치 미분 검사를 회귀 검사에 포함했습니다.</p><p>Grok이 수학과 진단 설계를 검토했습니다. 작은 TRAIN, 분할·꼬리 정의, 자기 제외 척도, 가속 경계, 해석 범위에 관한 지적을 실제 모집단과 구현으로 확인하고 해소 기록을 남겼습니다. 새 진단 코드 통합·실행·수치 검산의 책임은 Codex에 있습니다.</p><p>NumPy·Numba·Torch CPU의 전체 진단 최적화 경로에서 점수·순위·검출 인덱스가 같았으나 1.1배 속도 기준을 넘지 못해 NumPy/SciPy float64를 유지했습니다. 실제 MLX/MPS GPU float64 연산은 지원되지 않았습니다. 공식 SciPy 최적화기나 MiniRocket을 재작성하지 않았고, 논리 CPU−1인 11개 작업자와 내부 1스레드로 실행했습니다.</p>'
links=[('사전 규칙','protocol_antigravity.html'),('완료 계획','plan_antigravity.html'),('진단 결과','results_antigravity.json'),('독립 검산','verification_antigravity.json'),('Grok 검토','design_review_antigravity.json'),('지적 해소 근거','review_resolution_antigravity.json'),('실행 전 동결','preflight_antigravity.json'),('가속 비교','acceleration_antigravity.json')]
body+='<p>'+' · '.join(f'<a href="../experiments/exp509_rank_signal_diagnosis_antigravity/{file}">{label}</a>' for label,file in links)+'</p><h3>저장된 결과 재검산</h3><pre>NUMBA_CACHE_DIR=/private/tmp/exp509_numba_cache_antigravity .venv/bin/python -m experiments.exp509_rank_signal_diagnosis_antigravity.verify_antigravity</pre><p>작업 폴더에서 실행합니다. 완료된 진단을 재학습하는 명령이 아닙니다.</p><p>수치 최적화의 종료 조건: <a href="https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html">SciPy 공식 문서</a>. 실제 검사한 설치 버전은 SciPy 1.17.1입니다.</p></section>'
report.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exp509 학습 신호 진단 결과</title><style>'+style+'</style><body>'+body+'</body></html>')
current=ROOT/'docs/CURRENT_MODEL_IMPROVEMENT_STATUS_antigravity.html';text=current.read_text();assert 'id="exp509"' not in text
header='<header><p class="meta">2026-09-18 · Exp509 TRAIN 진단과 독립 검산 완료</p><h1>학습용 정상의 역할 분리가 다음 우선순위입니다</h1><p>최적화기 교체는 기존 Exp502를 넘지 못했습니다. 동일 참조 조건에서 표현 학습 제외군의 정상 상단 점수는 50계열 모두 더 높았고, 계열별 비율의 중앙값은 15.57배였습니다. 전체 1,117개에서의 기존 최고 관측 개선은 Exp502의 +3.73%이며 10% 목표는 미달성입니다.</p><p class="note">Exp509는 정상 TRAIN 진단입니다. 실제 TEST 평가·새 모델 승격·Exp510 실행은 하지 않았습니다. 역할 분리의 성능 효과는 후속 실험으로 검증해야 합니다.</p></header>'
text=re.sub(r'<header>.*?</header>',header,text,count=1,flags=re.S)
block='<section id="exp509"><h2>Exp509: 최적화와 학습 신호 진단</h2>'+validation_table+'<p>13개 검사와 50계열·250분할 독립 검산을 통과했습니다. L-BFGS-B의 제약 반영 기울기 기준은 246/250분할에서 통과했습니다. <a href="EXP509_SIGNAL_DIAGNOSIS_antigravity.html">동일 참조 비교와 최적화 진단 보고서</a></p></section>'
current.write_text(text.replace('</header>','</header>'+block,1))
plan=(P/'plan_antigravity.html').read_text()
for item in ('run','verification'):plan=plan.replace(f'id="{item}"><input type="checkbox" disabled>',f'id="{item}"><input type="checkbox" checked disabled>')
(P/'plan_antigravity.html').write_text(plan)
class Links(HTMLParser):
    def __init__(self,parent):super().__init__();self.parent=parent
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key=='href' and not value.startswith(('http:','https:','#')):assert (self.parent/value).exists(),value
for path in (report,current,P/'protocol_antigravity.html',P/'plan_antigravity.html'):Links(path.parent).feed(path.read_text())
closure=dict(state='complete_verified_train_diagnostic',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             datasets=50,folds=250,tests_passed=13,result_sha256=v['result_sha256'],verification_sha256=digest(P/'verification_antigravity.json'),
             report=str(report.relative_to(ROOT)),report_sha256=digest(report),source_code_changed_after_run=False,
             exp510_started=False,test_evaluation_executed=False,model_promotion=False,ten_percent_goal_achieved=False,
             best_full_experiment='exp502',best_full_primary_relative_gain=best['primary_relative_gain'],
             recommended_next_direction='Separate normal representation-reference fit A, ranking fit B and external validation C',
             unresolved_projected_gradient_folds=len(unresolved),limitations='Matched normal-TRAIN diagnostic; not component-level causal proof or generalization improvement.')
(P/'closure_antigravity.json').write_text(json.dumps(closure,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(closure,ensure_ascii=False))
