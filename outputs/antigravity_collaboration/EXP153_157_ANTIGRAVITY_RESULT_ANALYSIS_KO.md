# Exp153-157 TRAIN-only 개선 실험 최종 감사 보고서
**대외 보안 등급: 내부 검토용 (retrospective sensitivity research)**

본 보고서는 저장소([timesries project](file:///Users/minho/Documents/timesries%20project))의 증거 파일과 정합성 검증 로그를 바탕으로 작성된 독립 결과 검토서입니다. 본 검토는 read-only 모드로 작성되었으며, 임의의 소스 코드 수정, 실험 스크립트 실행, 큐 및 대시보드 조작은 일절 배제되었습니다.

---

### 1. 실행 완결성 및 데이터 정합성 판정 (PASS)
* **검증 내용**: [29a_consistency_checks.csv](file:///Users/minho/Documents/timesries%20project/outputs/exp153_157_train_only_improvements/29a_consistency_checks.csv) 및 [30_consistency_report.md](file:///Users/minho/Documents/timesries%20project/outputs/exp153_157_train_only_improvements/30_consistency_report.md)에 기록된 정합성을 검토한 결과, Exp154~157의 예측(prediction) 및 결과(result) 행의 수가 완벽하게 일치합니다.
  * *Exp154*: 17,872행 / *Exp155*: 13,404행 / *Exp156*: 13,404행 / *Exp157*: 3,351행
* **라벨 무결성**: 금지된 예측 필드(y_test, hard_tp 등)의 침투가 없고, 자율 경보와 검토 레인 간의 중복 건수(`lane_overlap_rows`) 및 후보 불일치(`candidate_union_mismatch_rows`)가 모두 **0**으로 검증되어 통계적 격리 프로토콜이 정상 작동했음을 확인했습니다.

---

### 2. Confirmed Facts (확인된 사실)
* **D2 설정 출처성(Provenance) 누설 재확인**:
  * [29_d2_configuration_provenance_audit.md](file:///Users/minho/Documents/timesries%20project/outputs/exp153_157_train_only_improvements/29_d2_configuration_provenance_audit.md) 분석에 따르면, 정책 실행 자체는 TEST 라벨을 보지 않으나, 4종의 고정 소스 모델 설정(ROCKET Exp40, Exp55/56, Exp84)은 과거에 TEST 지표를 보고 수동 튜닝된 "best" 모델들입니다. 따라서 end-to-end 무결한 TRAIN-only는 성립되지 않습니다.
* **극심한 소샘플 기각 현상**:
  * Exp157 리스크 제어 정책([25_exp157_summary.md](file:///Users/minho/Documents/timesries%20project/outputs/exp153_157_train_only_improvements/25_exp157_summary.md))에서 `risk_target_0.005_per_run` (0.5% 오탐 제한) 적용 시, 전체 1,117개 데이터셋 중 **1,095개(98.0%)**가 기각(abstained)되어 경보 감시 장치 자체가 꺼진 것으로 확인되었습니다.

---

### 3. Critical Findings (핵심 발견 사항)

#### 🚨 소형 샘플의 통계적 해상도 벽 (Resolution Block)
* [04_exp153_summary.md](file:///Users/minho/Documents/timesries%20project/outputs/exp153_157_train_only_improvements/04_exp153_summary.md)에 기록된 pseudo-normal 146,796회 실행 결과, alpha=0.005의 미세 오탐 통제를 가하고자 할 때 **940개(84.1%)**의 데이터셋이 Resolution-blocked(해상도 차단)로 분류됩니다.
* **해석 및 운영 한계**: 학습 데이터 개수 $n_{train}$이 부족하여 $1/(n_{train}+1) > \alpha$가 되면 신뢰 한계를 수식화할 수 없습니다. 이로 인해 false alarm을 억제하기 위해 alpha를 보수적으로 낮출수록 자율 감시 커버리지가 물리적으로 붕괴하는 치명적인 상용 배포 제약이 확인되었습니다.

#### 🚨 연속형 퓨전(Continuous Fusion)의 한계와 소스 독립성 붕괴
* [15_exp155_summary.md](file:///Users/minho/Documents/timesries%20project/outputs/exp153_157_train_only_improvements/15_exp155_summary.md)에서 alpha=0.05 기준, 복잡한 연속형 가중 퓨전(`corr_weighted_top2_meta`)은 F1 0.2750 및 precision 78.16%를 보여, 단순 이산형 다수결 합의인 `discrete_2of4` (F1 0.2925, precision 80.0%) 대비 오히려 하락하였습니다.
* **해석 및 시사점**: [10_exp155_source_independence_audit.csv](file:///Users/minho/Documents/timesries%20project/outputs/exp153_157_train_only_improvements/10_exp155_source_independence_audit.csv) 감사 결과, 소스 간의 Spearman correlation이 강한 양의 상관성을 보입니다. 즉, 피처와 이미지 모델들이 완전히 새로운 정보가 아닌 중복 오차 신호를 대량 공유하고 있어, 복잡한 가중 결합이 무의미한 잡음만 전파시키는 결과를 낳았습니다.

---

### 4. Variant별 Trade-Off (대안별 상충 관계)

#### A. Exp154: Cross-Dataset Pooling 효과
* `local_baseline` (alpha=0.05): precision 80.02%, recall 29.38%로 보수적이나 해상도 벽에 막힙니다.
* `global_pool` (alpha=0.05): recall을 49.23%로 올리지만, 전역적인 오차 스케일 왜곡으로 인해 standard review candidates가 **1,679건**으로 가장 크게 폭증하여 엔지니어의 경보 피로도를 높입니다.
* `hybrid_fallback` (alpha=0.05): local의 해상도 한계 시에만 상위 pool을 사용하는 타협안으로, precision 80.41%, recall 45.58%를 확보하고 수동 검토 대상을 **1,359건**으로 통제하는 비교적 균형 잡힌 민감도 경향을 보입니다.

#### B. Exp156: Confirmation Triage의 제로섬(Zero-Sum) Workload 관계
* `conformal_strict` (Block-B 0.01, Block-C 0.02): 자율 precision은 **84.75%**(FP 221건)로 높으나, 자율 recall이 **8.76%**로 수직 낙하하며 2,136건에 달하는 수동 검토 대상(priority 698건 + standard 1,438건)을 양산합니다.
* `conformal_balanced` (Block-B 0.02, Block-C 0.05): 자율 recall은 22.47%로 회복되고 standard review 대상을 **709건**으로 대폭 절감하여, 기계 자율과 인간 개입의 균형점을 형성합니다.

---

### 5. Safe / Prohibited Claims (발표 가능한/금지된 주장)

#### 🟢 Safe Claims (발표 가능 주장)
* *"본 연구는 shared cache 프로토콜을 도입하여 예측 생성과 라벨 평가 단계를 격리하고, TEST 라벨 정보의 누설 없이 cleanly OOF 통계로만 작동하는 retrospective policy 민감도 분석을 수립하였다."*
* *"Wilson conformal bound 및 pooling 기법이 소샘플 시계열 데이터셋의 임계치 해상도 제약 및 오탐 통제에 미치는 정량적 민감도와 그 Trade-off를 규명하였다."*

#### 🔴 Prohibited Claims (사용 금지 주장)
* **"본 시스템의 하위 소스 모델을 포함한 전체 개발 히스토리는 end-to-end strict TRAIN-only로 완성되었다."** (과거에 TEST 지표를 보고 소스 configuration을 sweep/채택한 provenance 누설 이력이 D2 감사 결과로 확인되었기 때문)
* **"$\alpha=0.005$ 혹은 특정 리스크 target이 실제 설비 배포를 위한 최적의 운영 상수로 입증되었다."** (retrospective counterfactual 민감도 수치일 뿐이며 현업의 구체적 제약 및 shadow run 검증을 거치지 않았기 때문)
* **"본 시스템이 실제 반도체 설비에서 false alarm rate를 1% 이하로 완벽히 보장한다."** (prospective validation 단계가 아니며 UCR 가상 retrospective benchmark 결과이기 때문)

---

### 6. Next Decision Gates (향후 의사결정 게이트)

#### 🚪 의사결정 게이트 1: Source Configuration의 순수 TRAIN-only HPO 재선정
* **내용**: 과거 TEST 지표의 관찰을 전면 중단하고, 오직 TRAIN OOF stability 및 pseudo-anomaly score만으로 4종의 소스 피처 하이퍼파라미터를 재튜닝하여 provenance를 영구 세척해야 합니다.
* **프로토콜**: 이 HPO 실험은 반드시 별도의 사전등록(Preregistration) 문서를 생성하고, 신규 실험 ID(예: **Exp158**)를 독립 발급받아 진행해야 합니다.

#### 🚪 의사결정 게이트 2: Locked Prospective Dataset shadow run 평가
* **내용**: 가상 retrospective benchmark 단계를 종결하고, 실제 설비 sensor 파형으로부터 획득한 잠금 prospective 데이터셋을 동결 정책 해시에 흘려보내는 shadow run 검증을 수행해야 합니다.
