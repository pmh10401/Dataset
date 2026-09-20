"""Render frozen Exp493 outputs; no fitting or selection."""
from pathlib import Path
import json,html
root=Path(__file__).resolve().parents[1];p=root/'experiments/exp493_stable_meta_routing_antigravity'
r=json.loads((p/'results_antigravity.json').read_text())
mini=r['controls']['stable']['minirocket_local_gap'];a=r['models']['old42D__stable']['evaluations']['stable']['auc'];b=r['models']['stable42D__stable']['evaluations']['stable']['auc']
intro=[f"<h2>결론: stable 목표에서 특징 갱신은 평균 AUC를 낮췄다</h2><p><strong>같은 stable 목표·stable 평가표에서 old42D → stable42D로 특징만 바꾸면 평균 AUC {a['mean_auc']:.9f} → {b['mean_auc']:.9f}, ZF {a['total_zf']} → {b['total_zf']}로 하락했다.</strong> 최종 stable42D AUC-head는 MiniRocket 고정 기준선(AUC {mini['mean_auc']:.9f}, ZF {mini['total_zf']})에도 못 미친다. 점수 안정화의 개선과 특징 갱신의 개선을 구분해야 한다. 이 결과로 라우터를 승격하지 않는다.</p><h2>전체 변경 전후</h2>"]
for head in ('auc','zf_sign'):
 old=r['models']['old42D__old']['evaluations']['old'][head];new=r['models']['stable42D__stable']['evaluations']['stable'][head]
 intro.append(f"<p><strong>{head} 선택:</strong> 평균 AUC {old['mean_auc']:.9f} → {new['mean_auc']:.9f} (Δ {new['mean_auc']-old['mean_auc']:+.9f}), ZF {old['total_zf']} → {new['total_zf']}.</p>")
rows=[]
for model,values in r['models'].items():
 for outcome,heads in values['evaluations'].items():
  for head,v in heads.items():
   rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in [model,outcome,head,f"{v['mean_auc']:.9f}",v['total_zf'],f"{v['delta_auc_vs_mini']:+.9f}",v['delta_zf_vs_mini'],f"{v['auc_wins']}/{v['auc_losses']}",str(v['method_counts'])])+'</tr>')
contrasts=[]
for head in ('auc','zf_sign'):
 def value(model,outcome):return r['models'][model]['evaluations'][outcome][head]
 steps=[('출발: old42/old 목표/old 평가',value('old42D__old','old')),
        ('점수만 안정화: 선택 고정',value('old42D__old','stable')),
        ('stable 목표로 재학습: 특징 고정',value('old42D__stable','stable')),
        ('stable42 특징 적용',value('stable42D__stable','stable'))]
 previous=None
 for label,v in steps:
  delta='' if previous is None else f"AUC {v['mean_auc']-previous['mean_auc']:+.9f}; ZF {v['total_zf']-previous['total_zf']:+d}"
  contrasts.append(f"<tr><td>{head}</td><td>{label}</td><td>{v['mean_auc']:.9f}</td><td>{v['total_zf']}</td><td>{delta}</td></tr>");previous=v
mse=''.join(f"<tr><td>{k}</td><td>{v['mse_auc']:.9g}</td><td>{v['mse_zf_sign']:.9g}</td></tr>" for k,v in r['models'].items())
donor=''.join(f"<tr><td>{target}</td><td>{evaluation}</td><td>{head}</td><td>{v['mean_auc']:.9f}</td><td>{v['total_zf']}</td><td>{v['method_counts']}</td></tr>" for target in ('old','stable') for evaluation,heads in r['models'][f'24D__{target}']['constant_evaluations'].items() for head,v in heads.items())
controls=''.join(f"<tr><td>{branch}</td><td>{method}</td><td>{v['mean_auc']:.9f}</td><td>{v['total_zf']}</td></tr>" for branch,methods in r['controls'].items() for method,v in methods.items())
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Exp493 평균 성능 검증</title><style>body{font:16px/1.65 system-ui;max-width:1320px;margin:36px auto;padding:0 24px;color:#172632;background:#f7f9fb}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:16px 0 32px}th,td{padding:9px;border-bottom:1px solid #dce4ec;text-align:left}th{background:#e5edf4}.scroll{overflow-x:auto}.note{background:#fff4d5;padding:18px;border-radius:8px}code{overflow-wrap:anywhere}</style><h1>Exp493 · 안정화가 방법 선택의 평균 성능도 높이는가?</h1><p>125개 family, 24D/old42D/stable42D × 두 목표 버전, 두 고정 선택 head를 비교했다. 같은 선택 결과를 old/stable 성능표 모두에 적용했다.</p><p class="note">기존 개발 cohort를 재사용한 family-held-out meta-development이다. 독립 신규 TEST 검증이나 자동 승격 근거로 해석하지 않는다. ZF-sign head는 개선 여부의 부호를 예측하며 ZF 개수의 기대이득을 최적화하지 않는다. ZF는 TEST 정상 최대값을 초과한 이상 개수로, 배포 임계값 성능은 아니다.</p>'''+''.join(intro)+'''<h2>효과 분리: 고정된 순서의 쌍대 비교</h2><p>점수 변경 → 목표 재학습 → 특징 변경 순서로 계산했다. 상호작용이 있으므로 다른 순서의 기여분은 달라질 수 있다.</p><table><tr><th>Head</th><th>단계</th><th>평균 AUC</th><th>ZF 합계</th><th>직전 단계 대비</th></tr>'''+''.join(contrasts)+'''</table><h2>고정 방법 기준선</h2><table><tr><th>평가표</th><th>방법</th><th>평균 AUC</th><th>ZF 합계</th></tr>'''+controls+'''</table><h2>전체 선택 결과</h2><p>승/패는 MiniRocket 대비 family 수다. 선택 빈도 순서는 MiniRocket, MultiRocket, Hydra, hybrid다. 두 head 중 결과가 좋은 것만 사후 선택하지 않는다.</p><div class="scroll"><table><tr><th>특징__학습 목표</th><th>평가표</th><th>Head</th><th>평균 AUC</th><th>ZF</th><th>Δ AUC/Mini</th><th>Δ ZF/Mini</th><th>AUC 승/패</th><th>방법 빈도</th></tr>'''+''.join(rows)+'''</table></div><h2>Donor-constant 선택 기준선</h2><p>held-family를 제외한 donor 목표 평균만 사용한다.</p><table><tr><th>목표</th><th>평가표</th><th>Head</th><th>평균 AUC</th><th>ZF</th><th>선택 빈도</th></tr>'''+donor+'''</table><h2>Ridge 예측 오차</h2><p>해당 모델의 목표 버전에 대한 MSE. 서로 다른 목표 버전 간 MSE를 특징 효과로 해석하지 않는다.</p><table><tr><th>모델</th><th>AUC 차이 MSE</th><th>ZF 부호 MSE</th></tr>'''+mse+'''</table><h2>검증</h2><p>관련 회귀 검사 30개 통과. 1,000개 방법·분기 평가를 순위합 공식으로 재계산해 AUC 최대 오차 2.22×10⁻¹⁶, ZF 완전 일치. 저장된 선택 결과 24개 요약을 별도로 재계산했다. 원본 산출물과 라벨 경계 메타데이터 해시가 유지됐다.</p><h2>출처와 재현</h2><p>과거 목표와 현재 재계산 값에서 77/250개 불일치를 확인하여 성능표를 섞지 않았다. MiniRocket/새 MultiRocket를 공유하는 현재 old/stable 완전한 표를 구성했다. 모든 신규 점수 저장 후 라벨을 읽었고, donor-only 전처리와 전체 family 제외를 적용했다. 예측/선택을 저장한 후 평가했다.</p><p>과거 Exp490 예측 재현 최대 오차: '''+html.escape(str(r['historical_replay_max_abs']))+'''</p><p>원자료: <a href="../experiments/exp493_stable_meta_routing_antigravity/results_antigravity.json">결과 JSON</a> · <a href="../experiments/exp493_stable_meta_routing_antigravity/protocol_antigravity.html">사전 규약</a> · <a href="../experiments/exp493_stable_meta_routing_antigravity/run_manifest_antigravity.json">실행 기록</a></p></html>'''
(root/'docs/EXP493_REVIEW_antigravity.html').write_text(page)
