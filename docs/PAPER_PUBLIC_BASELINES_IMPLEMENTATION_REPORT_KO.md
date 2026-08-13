# 공개 이상 탐지 비교 구현 보고서

## 구현 범위

논문 외부 비교군을 저장소의 미공개 사내 버전에서 분리하고, 다음 여섯 공개
기법을 독립 package에 구현했다.

1. kNN distance
2. Local Outlier Factor
3. RBF One-Class SVM
4. Isolation Forest
5. Deep SVDD
6. DROCC

모든 방법은 정상 TRAIN만 학습하고, shuffled 5-fold TRAIN OOF score로
conformal reference를 만든다. TEST label은 score freeze 전에는 접근할 수
없다. 제안기술은 Exp448 최종 구성으로 별도 표시하며 Exp404/416, Exp442,
Exp448은 공개 baseline 표에 사용하지 않는다.

## 재현성 및 안전장치

- 실제 DB SHA-256과 Exp446의 라벨 비접근 1,117개 모집단을 고정한다.
- canonical JSON, create-only 파일, shard/meta hash, 코드 hash를 사용한다.
- 완료 shard는 재사용하고 다른 byte는 덮어쓰지 않는다.
- 각 dataset-method 완료 직후 shard를 게시해 중단 후 resume한다.
- 누락·추가·변조된 shard가 있으면 freeze와 evaluate가 실패한다.
- score artifact에서 label 및 평가 지표 필드를 금지한다.
- alpha `0.005`, `0.01`, `0.02`와 matched-alert-budget을 함께 보고한다.
- AP, AUROC, micro/macro confusion 지표와 2,000회 shared-draw paired
  bootstrap을 생성한다.
- 공개 비교 표와 제안기술 내부 ablation 표를 분리한다.

## MLX 판정

AGENTS.md 정책에 따라 여섯 방법 모두 CPU를 선택했다. 고전 4종은 공식
scikit-learn estimator를 MLX로 다시 쓰면 구현이 달라져 조건 1을 위반한다.
Deep SVDD와 DROCC는 동결된 PyTorch 초기화, optimizer, autograd 및 학습
경로와 정확히 같은 MLX 경로가 없어 조건 1을 충족하지 못했다. 따라서
parity와 1.1배 속도 조건을 거짓으로 측정하지 않고 fallback 사유와
Python/NumPy/scikit-learn/PyTorch/MLX/mlx-metal 버전을 기록한다.

## 검증 상태

- unit/boundary/integration 테스트: **69개 통과**
- 실데이터 3개 × 6방법 smoke: **18 pair 완료, failure 0**
- smoke post-freeze 평가와 byte 재생 보고서: **통과**
- 공식 DB prepare: **1,117개, SHA-256 일치**
- 공식 고전 공개 baseline: **4,468 pair 완료, abstain 5, failure 0**
- 공식 고전 비교 full population/common support: **1,117 / 1,115개**
- Deep SVDD 공식 50 epochs partial: **107 / 1,117개 게시 후 보존**
- DROCC 공식 50 epochs full run: **pilot 예상 비용 때문에 미실행**
- 수상 가능성 최종 판단: **[평가 완료 후 기입]**

smoke 결과는 2 epochs와 3개 데이터셋으로 수행했으므로 논문 성능 수치에
사용하지 않는다. 공식 비교가 완료되기 전에는 우승, 현장 일반화 또는 운영
효과를 주장하지 않는다.

## 공식 고전 비교 결과

공식 immutable root는
`/Users/mac/Documents/Dataset/paper_public_baselines_classical_v1`이다.
matched-alert-budget의 full-population 결과는 다음과 같다. abstain은 경보
없음으로 집계했으며 common support에서도 별도 표를 생성했다.

| 방법 | Coverage | Alerts | Micro F1 | Macro F1 | baseline−proposal macro F1 95% CI |
|---|---:|---:|---:|---:|---:|
| 제안기술 | 1.0000 | 2,244 | 0.6264 | 0.6214 | 기준 |
| kNN distance | 0.9991 | 2,243 | 0.5019 | 0.4530 | [-0.1928, -0.1437] |
| LOF | 0.9982 | 2,242 | 0.3705 | 0.3586 | [-0.2886, -0.2374] |
| RBF One-Class SVM | 0.9991 | 2,243 | 0.5071 | 0.4642 | [-0.1814, -0.1341] |
| Isolation Forest | 0.9991 | 2,243 | 0.2662 | 0.2208 | [-0.4278, -0.3737] |

공통 지원 1,115개에서는 모든 방법의 경보 수가 2,242개로 동일했다. 이때
제안기술 micro/macro F1은 0.6262/0.6207, 공개 최고 RBF One-Class SVM은
0.5070/0.4641이었다. 따라서 주 비교의 차이는 전체 경보 수 차이로 설명되지
않는다.

연속 score의 micro AP/AUROC는 데이터셋 간 원시 score scale을 직접 섞지
않고 `1 - TRAIN-OOF upper-tail conformal p-value`로 계산했다. 데이터셋별
macro AP/AUROC는 각 데이터셋 안에서 원시 anomaly score 순위를 사용했다.
제안기술의 연속 score는 frozen Exp448 산출물에 없으므로 제안기술 AP/AUROC를
사후 복원하지 않고 미기재했다.

## Artifact identity

- Frozen score manifest SHA-256:
  `c546c841c86c8f488e22d0b0b844a81bc17cdd36b1bfcec3b41869d04a774355`
- Evaluation artifact-map SHA-256:
  `814282a3fe2f1c678790a42d1ac336cf377c27b38298616ccd2d0993a4f8fe0e`
- Report artifact-map SHA-256:
  `e7468091d826a6881318c6160211d0ed3454366dcf161f4cc5edf720fa6c1c44`
- Shared bootstrap draw SHA-256:
  `bcca8c2d03c30359242d73df445d54bd39fd20374ac0659a7e33b3b2bebfece1`

Deep SVDD partial 산출물은
`/Users/mac/Documents/Dataset/paper_public_baselines_deep_partial_v1`에
보존했다. dispatch source hash가 누락됐던 초기 실행도 삭제하지 않고
`/Users/mac/Documents/Dataset/paper_public_baselines_v1_superseded_unbound_dispatch_20260812`
에 구분 보존했다. 두 root 모두 논문 성능표에는 사용하지 않는다.

## 보류 심층 실행 준비 상태

- Deep SVDD deferred launcher: **prepared, not started, 새 공식 root 0/1,117**
- DROCC deferred launcher: **prepared, not started, 새 공식 root 0/1,117**
- Canonical backend: **deterministic PyTorch CPU, worker 1, epochs 50**
- Operational support: **independent roots, create-only authority, PID/lock,
  live log, partial verified status, exact resume, freeze and verify**

향후 공식 root는 각각
`/Users/mac/Documents/Dataset/paper_public_baselines_deep_svdd_1d_full_v1_antigravity`와
`/Users/mac/Documents/Dataset/paper_public_baselines_drocc_1d_full_v1_antigravity`이다.
현재 두 root는 생성하지 않았고 어떤 50-epoch 학습 process도 시작하지 않았다.
이 준비 상태는 심층 방법의 성능 결과를 뜻하지 않으며 기존 고전 결과를
변경하지 않는다.
