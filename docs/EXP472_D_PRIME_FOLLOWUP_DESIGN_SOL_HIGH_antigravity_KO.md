# Exp472 D′ 후속 대조 실험 설계

## 상태

- 작성 목적: Exp472의 역사적 D와 신규 E 비교에서 보간법 효과와 실행 환경 효과를 분리한다.
- 판정: **fix-first**
- 구현 상태: Exp473 D′ 점수·내구성 저장·상태·검증 경로 준비 완료. 공식 P0 smoke는 미실행이다.
- 이 문서는 실행 승인이 아니다. 공식 실행은 별도 승인 후 시작한다.

## 현재 확인된 문제

Exp472의 핵심 대비는 PCHIP 조건 D와 Makima 조건 E이다. 그러나 D는 Exp471의 동결 결과를 재사용했고 E만 Exp472 환경에서 새로 계산했다. 따라서 현재 E-D에는 보간법 차이 외에 실행 시점과 환경 차이가 섞일 수 있다.

공식 산출물에서 Makima 업샘플링 노출이 0인 다음 세 데이터셋에서도 D와 E의 지표가 달랐다.

| 데이터셋 | AP 차이 E-D | AUROC 차이 E-D |
|---|---:|---:|
| `Yoga_normal_2` | -0.013258 | -0.000795 |
| `CricketY_normal_3` | 0 | -0.010204 |
| `FaceFour_normal_4` | 0 | -0.010204 |

Makima가 호출되지 않은 데이터에서 발생한 차이는 Makima의 효과로 설명할 수 없다. 그러므로 동시대 PCHIP 대조군 D′가 필요하다.

## 연구 질문

> 동일한 Exp472 코드, 환경, 시드, 데이터 순서와 백엔드에서 업샘플링 함수만 PCHIP와 Makima로 달리할 때 AP와 AUROC가 달라지는가?

## 조건

- D′: polyphase 다운샘플링 + PCHIP 업샘플링 + area/local-mean 영상 구성
- E: polyphase 다운샘플링 + Makima 업샘플링 + area/local-mean 영상 구성
- D′와 E의 유일한 의도된 차이는 업샘플링 함수이다.
- E는 기존 Exp472 결과를 재사용하되, 권한·코드·환경·입력 정체성이 정확히 결속됐을 때만 비교한다.

## 필수 실행 순서

### P0. 무노출 불일치 smoke

1. `Yoga_normal_2`, `CricketY_normal_3`, `FaceFour_normal_4`를 포함한다.
2. 실제 업샘플링이 발생한 12개 원본 계열에서 대표 데이터셋을 하나씩 포함한다.
3. D′와 E에 동일한 입력, 시드, 데이터 순서와 백엔드를 사용한다.
4. 업샘플링이 없는 데이터에서는 특징, 후보 인덱스, 연속 점수와 지표가 사전 정의한 허용 범위 안에서 같아야 한다.

중단 기준: 무노출 데이터의 차이를 제거하거나 설명하지 못하면 E-D′의 인과적 해석과 전체 실행을 중단한다.

고정 P0 목록은 15개이다. 무노출 진단 3개는 `Yoga_normal_2`,
`CricketY_normal_3`, `FaceFour_normal_4`이고, 노출 대표 12개는
`AllGestureWiimoteX_normal_1`, `AllGestureWiimoteY_normal_1`,
`AllGestureWiimoteZ_normal_1`, `GestureMidAirD1_normal_1`,
`GestureMidAirD2_normal_1`, `GestureMidAirD3_normal_1`,
`GesturePebbleZ1_normal_1`, `GesturePebbleZ2_normal_1`,
`MelbournePedestrian_normal_2`, `PLAID_normal_0`,
`PickupGestureWiimoteZ_normal_1`, `ShakeGestureWiimoteZ_normal_1`이다.

입력 권한은 현재 원본 DB의 전체 SHA-256과 values-only catalog를 다시
검증한 뒤, 각 TEST 값 해시를 완료된 Exp472 E shard와 직접 대조한다. 과거
Exp467 실행 백엔드는 당시 MPS 사용 가능 상태에 결속되어 있으므로 현재
MPS 가용성을 과거 값으로 위조하지 않는다.

### P1. D′ 전체 실행

- 대상: 1,117개 전체
- 성공 조건: 1,117 completed, 0 failed, 완전한 manifest와 provenance
- 실패 비용: 데이터셋 단위 내구성 checkpoint로 한 데이터셋 이상의 재계산을 피한다.
- 자동 재시작을 반복하지 않고 정확한 실패를 보존한다.

### P2. 통계 분석

- 주 대비: E-D′
- 환경 이동 진단: D′-D
- 주 지표: AP
- 보조 지표: AUROC
- 전체 1,117개 분석은 보조 분석으로 보고한다.
- 실제 업샘플링 153개는 12개 원본 계열의 변형이므로 원본 계열 단위 cluster bootstrap을 주 분석으로 사용한다.
- leave-one-family-out 분석으로 특정 계열 의존성을 확인한다.
- 데이터셋 단위 결과는 보조적으로 함께 보고한다.

## 결론 규칙

1. 원본 계열 단위 신뢰구간이 0보다 크면 제한된 우월성 근거로 해석한다.
2. 신뢰구간이 0보다 작으면 Makima의 평균 저하 근거로 해석한다.
3. 신뢰구간이 0을 포함하면 결정적 평균 차이가 없다고 쓴다.
4. 동등성은 사전에 실용적 허용 한계를 정하고 신뢰구간 전체가 그 안에 포함될 때만 주장한다.
5. 데이터셋 단위와 원본 계열 단위 결론이 충돌하면 일반화 판단에는 원본 계열 단위를 우선한다.

## 가속 정책

- SciPy의 불규칙 1차원 PCHIP/Makima는 CPU 경로를 유지한다.
- 현재 provenance: 논리 CPU 12개, worker 11개, 내부 수치 스레드 1개, PyTorch 2.13.0, SciPy 1.17.1, MPS 사용 가능, MLX 사용 불가.
- PyTorch/MPS 또는 MLX로 보간을 다시 구현하면 실험 요인이 추가되므로 이번 D′에는 사용하지 않는다.
- 별도 가속 구현은 동일 후보 인덱스, 유한 출력, 수치 허용 범위와 종단 간 1.1배 이상의 속도 향상을 모두 통과할 때만 검토한다.

## 범위 제외

- quadratic, cubic spline, Akima 등 보간법 추가 탐색
- 결과를 본 뒤 유리한 데이터셋만 선택하는 사후 부분집합
- TEST 라벨을 사용한 보간법 또는 동등성 한계 선택
- 결정론 문제를 해결하기 전의 의미 없는 다중 시드 전체 재실행
- 공개 기준선과 E의 차이를 Makima 효과로 해석하는 것

## 제출 전 완료 조건

- 무노출 동일성 smoke 통과
- D′ 1,117개 전체 완료 및 실패 0
- E-D′, D′-D 분석 완료
- 12개 원본 계열 단위 cluster bootstrap 완료
- 논문 표·그림·주장에 D′ 결과 반영
