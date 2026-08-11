# Exp449 잔여 FN 기법 자격평가 설계

## 1. 한 문장 설명

Exp449는 Exp448의 잔여 false negative가 집중된 여덟 데이터셋 군에 대해 문헌에서
강했던 시계열 표현 기법을 TRAIN-only 이상점수 경계에 연결하고, 모든 예측을 먼저
동결한 뒤 단 한 번의 사후 label 평가로 `FN < 1,988`과 `FP <= 314`를 동시에 만족하는
기법만 이후 라우팅 연구 후보로 자격 부여한다.

## 2. 연구 성격과 고정 신원

```text
experiment_id = Exp449
artifact_schema = exp449
kind = residual_fn_method_qualification
run_identity = exp449-residual-fn-method-qualification-antigravity-20260812
run_root = /Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_antigravity_20260812
package = experiments/exp449_residual_fn_method_qualification_antigravity
launcher = scratch/run_exp449_residual_fn_method_qualification_antigravity.py
dashboard_schema = exp449.public-method-qualification.v1
```

Exp449는 새 champion이나 router를 만드는 실험이 아니다. Exp448 TEST 결과를 보고
대상 데이터셋 군을 골랐기 때문에 탐색적이고 설명적인 post-hoc method qualification이다.
성공한 후보도 즉시 예측, champion, promotion, production routing에 사용할 수 없다.
라우팅은 TRAIN-only 선택 규칙과 새로운 독립 확인 경계를 갖는 별도 Exp450에서만 연구한다.

새로 만드는 모든 파일명에는 `antigravity`가 포함되어야 하며 `__init__.py`는 만들지
않는다. Exp448 package, official root, evaluation leaves, dashboard result는 byte-immutable이다.

## 3. 불변 기준선과 성공 기준

Exp448 `hard_plus_stable_26`가 유일한 기준선이다.

```text
TP = 1,930
FP = 314
FN = 1,988
TN = 191,668
F1 = 0.6264199935
datasets = 1,117
label rows = 195,900
```

후보 하나의 counterfactual은 대상 데이터셋 군에서만 Exp448 stable-26 예측을 후보
예측으로 교체하고, 나머지 데이터셋에서는 Exp448 bytes를 그대로 사용해 계산한다.

후보의 필수 자격 조건은 다음과 같다.

1. 전역 `FN < 1,988`.
2. 전역 `FP <= 314`. FP 315는 FN을 아무리 많이 줄여도 실패다.
3. 전역 F1이 `0.6264199935`보다 작지 않다.
4. 대상 데이터셋 군의 FN이 Exp448보다 감소한다.
5. TRAIN-only prediction이 같은 환경·seed에서 byte-deterministic하고 모든 score가 finite다.
6. prediction freeze 이전에 TEST label, Exp448 dataset metric outcome, candidate TEST confusion을
   threshold, feature, method, hyperparameter 선택에 사용하지 않는다.

통계 요약은 paired dataset bootstrap과 정확한 변화량을 함께 기록하지만, 이 post-hoc
실험의 p-value나 신뢰구간은 promotion 권한을 만들지 않는다. 필수 자격 조건은 위의
결정적 confusion 기준이다.

## 4. 대상 데이터셋 군

Exp448 잔여 FN 상위 여덟 군을 미리 동결한다.

| 순서 | 데이터셋 군 | Exp448 잔여 FN |
|---:|---|---:|
| 1 | Crop | 246 |
| 2 | ElectricDevices | 200 |
| 3 | FordA | 88 |
| 4 | FordB | 75 |
| 5 | Yoga | 63 |
| 6 | ECG5000 | 56 |
| 7 | ChlorineConcentration | 56 |
| 8 | StarLightCurves | 53 |

상위 다섯 군은 672 FN으로 전체 1,988 FN의 33.803%다. 이 수치는 평가 대상을 설명하는
사후 증거일 뿐, 후보 prediction의 입력이나 router feature가 아니다. 개별 TEST instance
결과는 prediction package에 복사하지 않는다.

## 5. 문헌 후보와 해석 경계

UCR classification 결과는 후보 표현을 찾는 외부 근거이지 Exp449 이상탐지 성능의
대체물이 아니다. 분류 정확도와 point-level anomaly FN/FP는 다른 지표다.

| 데이터셋 군 | 우선 조사 기법 | 보조 조사 기법 |
|---|---|---|
| Crop | InceptionTime 계열 | DrCIF |
| ElectricDevices | Hydra + MultiROCKET | MultiROCKET |
| FordA | WEASEL 계열 | DrCIF |
| FordB | InceptionTime 계열 | WEASEL 계열 |
| Yoga | Hydra + MultiROCKET | MultiROCKET |
| ECG5000 | TS-CHIEF | HC2 / ROCKET 계열 |
| ChlorineConcentration | InceptionTime 계열 | ROCKET 계열 |
| StarLightCurves | HIVE-COTE 2.0 | MultiROCKET |

근거 자료는 UCR/UEA 공식 결과와 저자 공개 결과로 한정한다.

- UCR/UEA results: <https://timeseriesclassification.com/results.php>
- Bake Off Redux: <https://doi.org/10.1007/s10618-024-01022-1>
- MultiROCKET: <https://doi.org/10.1007/s10618-022-00844-1>
- MultiROCKET author code/results: <https://github.com/ChangWeiTan/MultiRocket>
- HIVE-COTE 2.0: <https://doi.org/10.1007/s10994-021-06057-9>
- HC2 supporting results: <https://timeseriesclassification.com/HC2.php>

## 6. 세 단계 자격평가

### 6.1 단계 A: 공식 구현과 score compatibility audit

각 후보는 실행 전에 다음 중 하나를 만족해야 한다.

- 공식 구현이 TRAIN series를 받아 label-free feature representation을 내보낸다.
- 기존 저장소의 동결된 공식 transform 경계가 같은 동작을 제공한다.
- classifier-native decision output을 쓰는 경우, 필요한 supervision이 TRAIN-only이고
  Exp449 point anomaly target과 의미가 정확히 맞는다는 별도 계약이 있다.

단순 classification accuracy만 제공하거나, 공식 구현의 내부 표현을 임의로 재작성해야
하거나, TEST label이 있어야 anomaly score를 만들 수 있는 기법은
`not_score_compatible`로 기록한다. 이를 비슷한 자체 구현으로 대체하거나 원래 기법
이름으로 보고하지 않는다.

현재 환경 확인 결과 Exp448의 고정 Python에는 `aeon`이 없다. 그 환경의 package set은
변경하지 않는다. MiniROCKET/MultiROCKET은 기존 Exp249의 공식 aeon CPU 경계를 재검증해
재사용한다. Hydra, WEASEL, DrCIF, InceptionTime, HC2, TS-CHIEF는 정확한 공식 source,
version, license, environment, transform/decision interface가 확인된 뒤에만 활성화한다.

### 6.2 단계 B: 공통 TRAIN-only anomaly head

score-compatible representation에는 같은 LocalGap/kNN one-class head를 적용한다.

1. TRAIN series에서만 representation을 fit/transform한다.
2. TRAIN out-of-fold feature로 LocalGap 기준과 threshold를 만든다.
3. TEST series에는 label-free transform과 고정 score/threshold만 적용한다.
4. method별 score 방향, finite 값, feature dimension, candidate index를 manifest에 기록한다.
5. 대상 데이터셋 군마다 Exp448 stable-26을 대체할 binary prediction bytes를 동결한다.

공통 head는 representation의 효과를 비교하기 위한 통제 장치다. method-native anomaly
head를 후보마다 다르게 쓰는 것은 representation과 scoring 효과를 섞으므로 Exp449에
포함하지 않는다. 필요하면 별도 새 실험으로 분리한다.

### 6.3 단계 C: 단일 label snapshot counterfactual evaluation

모든 method × target-family prediction과 SHA manifest가 먼저 완성되고 immutable해진 뒤,
한 번의 ordered label session에서 평가한다. 후보별로 대상 군의 confusion을 계산하고
Exp448의 나머지 confusion과 결합해 전역 counterfactual TP/FP/FN/TN/F1을 만든다.

같은 label snapshot으로 모든 사전등록 후보를 평가하며, 결과를 본 뒤 threshold나
candidate를 다시 만들지 않는다. prediction 단계 실패 후보는 평가에서 제외하되 재시도나
완화된 fallback을 사용하지 않는다.

## 7. 후보 행렬과 계산 예산

전체 기법 × 전체 데이터셋의 무제한 교차곱은 사용하지 않는다. 먼저 위 문헌 행렬의
우선/보조 후보만 audit하고, score-compatible 후보만 실행한다. 각 target-family에 최대
두 개 representation을 두며 동일 representation의 hyperparameter sweep은 하지 않는다.

모든 seed, transform 설정, OOF fold, neighbor 수, score 방향, threshold rule은 결과를
보기 전에 preregistration에 기록한다. 공식 third-party transformer의 기본값을 바꾸는
경우 그 이유와 exact delta를 별도 후보 이름으로 표시한다.

## 8. MLX와 병렬 실행 정책

각 새 score source는 full run 전에 CPU-versus-MLX를 평가한다. MLX는 다음 조건을 모두
만족할 때만 사용한다.

1. 동결 algorithm/feature/score/threshold/routing 의미가 변하지 않는다.
2. 같은 input과 seed를 사용한다.
3. 모든 출력이 finite이고 모든 사전등록 alpha에서 candidate index가 동일하다.
4. score 차이가 사전등록 tolerance 안이다.
5. transfer와 synchronization을 포함해 정확한 단계가 CPU보다 최소 1.1배 빠르다.

하나라도 실패하면 CPU를 유지한다. 공식 CPU transformer를 GPU용으로 다시 작성하지
않는다. CPU는 dataset worker 최대 7개를 사용하고, MLX/MPS-heavy process는 동시에 하나만
실행한다. manifest는 MLX/mlx-metal version, device, timing, parity와 fallback 이유를 담는다.

## 9. 불변성과 실패 처리

- Exp448 official artifacts와 dashboard payload를 수정하지 않는다.
- Exp449 official root는 single-use create-only다. partial/existing/symlink root는 실패다.
- source DB와 predecessor artifacts는 descriptor-pinned, no-follow, SHA/inode/nlink로 읽는다.
- nonfinite score, shape drift, representation dimension drift, nondeterminism, duplicate
  candidate, missing candidate, manifest mismatch는 label access 전에 실패한다.
- prediction failure, label-open/query/close failure, post-label source drift, artifact write
  failure는 고유한 immutable audit로 남기고 성공 결과를 만들지 않는다.
- official readiness/evaluate/inspect 중 하나가 nonzero이면 Exp449를 중단하며 같은 번호로
  retry/resume하지 않는다.
- `not_score_compatible`는 구현 실패와 구별되는 정상적인 audit 결과다.

## 10. 산출물

prediction authority는 최소한 다음을 고정한다.

- candidate registry와 method/source/version/environment identity
- target-family membership SHA
- TRAIN-only transform and LocalGap configuration
- CPU/MLX parity and benchmark decision
- per-candidate frozen prediction SHA
- exact candidate inventory and aggregate manifest SHA

evaluation authority는 다음을 포함한다.

- Exp448 baseline identity와 confusion
- candidate별 target confusion과 전역 counterfactual confusion/F1
- FN delta, FP delta, target-family FN delta
- deterministic qualification decision과 rejection reason
- paired dataset bootstrap summary
- immutable artifact map과 source/audit identity

원시 TEST label, instance index, 개별 dataset name 목록은 공개 dashboard payload에 넣지 않는다.

## 11. 대시보드 계약

owner-only 기존 dashboard를 Exp449의 진행 상황과 결과 표시 경계로 계속 사용한다. 과학
실행과 site publication은 분리하고, dashboard 실패가 실험 retry나 artifact 변경 권한을
만들지 않는다.

진행 화면에는 다음만 표시한다.

- experiment id, phase, status, reviewed commit
- target family 수, audited/compatible/running/completed/rejected candidate 수
- 현재 method family와 집계 progress/ETA
- CPU 또는 MLX 결정과 fallback 사유 범주
- label access가 아직 0인지, evaluation이 시작됐는지

완료 화면 `exp449.public-method-qualification.v1`에는 다음 aggregate만 표시한다.

- 각 target family와 method의 compatibility/qualification 상태
- target FN delta와 전역 FN/FP/F1
- `FP <= 314`, `FN < 1,988`, F1 non-regression gate 결과
- runtime과 deterministic repeat 상태
- 통과 후보 목록과 `routing_research_eligible` boolean

원시 label, instance index, 내부 dataset 이름/목록, per-instance 결과, local path, hash,
log, DB, checkpoint, secret은 공개 payload에서 제외한다. 문헌 후보와 실제 Exp449 성능을
같은 값처럼 보이지 않게 별도 표로 표시한다.

## 12. 검증 전략

TDD는 최소 다음을 포함한다.

- candidate registry exact inventory/order와 target-family freeze
- 공식 implementation identity와 unavailable/not-compatible fail-closed
- TRAIN-only import/data-access firewall
- LocalGap common-head parity와 threshold immutability
- deterministic prediction repeat와 nonfinite/shape/dimension rejection
- FP 314 pass / FP 315 fail, FN 1,987 pass / FN 1,988 fail 경계
- target replacement만 바뀌고 나머지 Exp448 prediction bytes가 같은지 검증
- one label snapshot, fixed candidate inventory, no post-result rerun
- source/DB/root symlink, replacement, nlink, namespace race adversarial tests
- partial/audit-write/session-close failure preservation
- dashboard aggregate allowlist와 금지 필드 검사
- Exp448 package/artifact/dashboard byte guard와 predecessor regression

## 13. 구현 경계

구현은 다음 순서의 bounded tasks로 나눈다.

1. governance, preregistration, runbook, state/catalog, versioned Exp449 code map.
2. candidate registry, official implementation/interface audit, environment manifests.
3. common representation-to-LocalGap score boundary와 Exp249 ROCKET reuse.
4. score-compatible method adapters와 frozen target predictions.
5. single-session counterfactual evaluator, artifact replay, failure audit.
6. closed CLI/launcher, full regression, fresh final review.
7. official readiness → prediction → evaluation → inspect, truth/map closeout.
8. immutable result의 owner-only dashboard publication.

각 code-changing task 시작 전에 versioned code map lock을 확인하고 callers/effects/tests를
답한다. module boundary가 바뀌면 `codemap_exp449_antigravity.html/json/lock`을 함께 갱신한다.
구현은 TDD로 진행하고, 공식 실행 전 clean commit에 대한 fresh final review가 `ship`을
명시해야 한다.

## 14. 이후 라우팅 연구의 문

Exp449에서 한 개 이상의 후보가 모든 자격 조건을 통과할 때만 Exp450 설계를 시작한다.
Exp450 router는 TEST family name이나 Exp449 TEST outcome을 feature로 쓰지 않는다. TRAIN-only
series/meta-feature로 method를 선택하고, Exp449 성공 기법은 고정 후보 집합으로만 사용한다.
독립 확인 경계에서 다시 `FP <= 314`와 FN 개선을 증명하기 전에는 champion/promotion이 없다.
