"""Publish only independently verified Exp502 full-benchmark results."""
from pathlib import Path
from hashlib import sha256
from html.parser import HTMLParser
import json,re
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'experiments/exp502_amplitude_envelope_antigravity'
read=lambda p:json.loads(p.read_text())
r=read(P/'full_results_antigravity.json');v=read(P/'full_verification_antigravity.json')
assert v['state']=='passed' and v['result_sha256']==sha256((P/'full_results_antigravity.json').read_bytes()).hexdigest()
current=ROOT/'docs/CURRENT_MODEL_IMPROVEMENT_STATUS_antigravity.html';old=current.read_text();head=old[:old.index('<body>')+6]
models=[('MiniRocket',r['summaries']['baseline'])]
for label,folder in [('Exp495 OAS','exp495_normal_geometry_antigravity'),('Exp496 부분 특징 결합','exp496_subspace_consensus_antigravity')]:models.append((label,read(ROOT/'experiments'/folder/'full_results_antigravity.json')['summaries']['candidate']))
models.append(('Exp502 원시·제곱 결합',r['summaries']['candidate']))
def table(key):
    return '<div class="scroll"><table><thead><tr><th>모델</th><th>오탐 0건</th><th>FPR≤0.1%</th><th>≤0.5%</th><th>≤1%</th></tr></thead><tbody>'+''.join('<tr><th>'+name+'</th>'+''.join(f'<td>{x*100:.2f}%</td>' for x in values[key])+'</tr>' for name,values in models)+'</tbody></table></div>'
b=r['summaries']['baseline']['family_macro_tpr'];c=r['summaries']['candidate']['family_macro_tpr'];gain=r['primary_relative_gain'];lo,hi=r['primary_gain_family_bootstrap95'];met=r['observed_target_met']
conclusion=f'Exp502 주지표: {b[1]*100:.2f}% → {c[1]*100:.2f}%, {(c[1]-b[1])*100:+.2f}%p·상대 {gain*100:+.2f}%'
body=f'<header><p class="meta">2026-09-17 · 동일1,117개/128계열 · 전체 점수 및 집계 독립 검증</p><h1>{conclusion}</h1><p>상대10% 개선·오탐0건 비악화 조건: {"관측 개발 벤치마크에서 충족" if met else "미달성"}. 목표 검출률은 {b[1]*110:.2f}%입니다.</p><p>Exp502는 원시 신호와 중심화한 제곱 신호의 MiniRocket 특징을 결합합니다. 정상 TRAIN에서만 표현·스케일러·OAS를 학습하고, 합성 검증으로 선택한 후보 하나를 동결한 뒤 전체 평가했습니다.</p></header>'
body+='<section><h2>계열을 동일하게 가중한 평균</h2>'+table('family_macro_tpr')+f'<p>상대 개선의 계열 paired bootstrap95% 구간: {lo*100:+.2f}%~{hi*100:+.2f}% (2,000회). {"0을 포함하므로 확실한 우위가 입증됐다고 말할 수 없습니다." if lo<=0<=hi else "이 구간은 계열 재표집의 불확실성이며 미사용 데이터 일반화 보장은 아닙니다."}</p></section>'
body+='<section><h2>데이터셋 동일 가중 평균</h2>'+table('dataset_macro_tpr')+'</section><section><h2>이상 표본 수 가중 평균</h2>'+table('micro_tpr')+'<p>오탐0건 검출 이상 수(ZF): '+', '.join(f'{name} {values["total_zf"]:,}개' for name,values in models)+'.</p></section>'
body+='<section><h2>범위와 검증</h2><p>원본1,119개 중 기존 정의로 평가 가능한1,117개 전체를 사용했습니다. 정상 클래스 매핑이 없는 CornellWhaleChallenge와 정상 TRAIN1개인 Phoneme_normal_39를 기존과 동일하게 제외했습니다. 기준 MiniRocket의 모든 점수와 순위가 재현됐고 2,234점수벡터의ROC·네오탐한도·세평균·bootstrap·코드/입력/결과 해시를 독립 확인했습니다.</p><p class="note">이미 노출된 개발TEST에서 라벨로 각 오탐 한도의 검출률을 계산한 결과입니다. 미사용 holdout 검증이나 실제 운영오탐 보장이 아닙니다. 1,088개 데이터셋에서 FPR0.1%는 표본 수가 적어 FP0와 같습니다. 반복 TRAIN 탐색의 편향도 남아 있습니다.</p><p><a href="../experiments/exp502_amplitude_envelope_antigravity/full_verification_antigravity.json">전체 독립검증</a> · <a href="../experiments/exp502_amplitude_envelope_antigravity/full_results_antigravity.json">전체 결과</a> · <a href="../experiments/exp502_amplitude_envelope_antigravity/protocol_antigravity.html">TRAIN 사전규칙</a> · <a href="../experiments/exp502_amplitude_envelope_antigravity/full_protocol_antigravity.html">전체평가 규칙</a></p></section>'
page=head.replace('<title>현재 모델 개선 연구 상태</title>','<title>Exp502 전체 비교</title>')+body+'</body></html>'
report=ROOT/'docs/EXP502_FULL_COMPARISON_antigravity.html';report.write_text(page)
block=f'<section id="exp502"><h2>Exp502 전체 평가 완료</h2><p>{conclusion}. 상대10% 목표는 {"관측값 기준 충족했습니다" if met else "아직 달성하지 못했습니다"}.</p>'+table('family_macro_tpr')+f'<p>주지표 상대 개선의 계열 bootstrap95% 구간은 {lo*100:+.2f}%~{hi*100:+.2f}%입니다. 전체1,117개·2,234개점수벡터·네오탐한도·세평균·bootstrap 및해시검증을통과했습니다. 모든 MiniRocket 대조 점수·순위가재현됐습니다.</p><p><a href="EXP502_FULL_COMPARISON_antigravity.html">세 가지 평균과 전체 검증 보고서</a></p></section>'
old=re.sub(r'<section id="exp502">.*?</section>',block,old,flags=re.S)
best=max(models[1:],key=lambda x:x[1]['family_macro_tpr'][1]);bestgain=best[1]['family_macro_tpr'][1]/b[1]-1
old=old.replace('현재 최고 관측 개선은 1.65%입니다',f'현재 최고 관측 개선은 {bestgain*100:.2f}%입니다')
if met:old=old.replace('MiniRocket 대비 상대 10% 개선 목표는 아직 달성하지 못했습니다.','Exp502가 개발 벤치마크에서 상대10% 개선·FP0비악화 조건을 충족했습니다. 독립holdout 검증은 아닙니다.')
header=f'<header><p class="meta">2026-09-17 · 동일1,117개/128계열 · 독립검증 완료 결과</p><h1>현재 최고 관측 개선은 {bestgain*100:.2f}%입니다</h1><p>여덟 가지 가설 중 Exp495·496·502를 전체 평가했고 나머지 다섯 가설은 사전 TRAIN 조건을 통과하지 못했습니다. 최고 관측 주지표는 {best[0]}입니다. 상대10% 개선·FP0비악화 목표는 {"Exp502 개발 관측값에서 충족했습니다" if met else "아직 미달성입니다"}.</p><p class="note">모든 수치는 이미 노출된 개발TEST의 관측 성능입니다. 반복 탐색과 계열 간 편차가 남아 있으므로 관측 순위를 확실한 일반화 우위로 해석하지 않습니다. 각 실험의 불확실성 구간을 함께 확인해야 합니다.</p></header>'
old=re.sub(r'<header>.*?</header>',header,old,count=1,flags=re.S)
current.write_text(old)
class Links(HTMLParser):
    def __init__(self,parent):super().__init__();self.parent=parent
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key=='href' and not value.startswith(('http:','https:','#')):assert (self.parent/value).exists(),value
for path in (report,current):Links(path.parent).feed(path.read_text())
print(json.dumps(dict(report=str(report),target_met=met,primary_relative_gain=gain)))
