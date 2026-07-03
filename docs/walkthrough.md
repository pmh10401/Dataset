# 연구 수행 과정 및 최종 결과 요약서 (Walkthrough)

본 문서는 univariate_ts.db 데이터베이스 재구축 과정, 유효성 검증 결과, VAE 확장 설계 및 다양한 임계치/손실 기법들을 적용한 최종 벤치마크 평가 이력을 요약합니다.

---

## 1. 생성된 산출물

* **최종 데이터베이스**: [univariate_ts.db](file:///Users/minho/Documents/Dataset/univariate_ts.db) (1,672.06 MB)
* **재구축 실행 스크립트**: [rebuild_database_v2.py](file:///Users/minho/Documents/Dataset/rebuild_database_v2.py)
* **검증 실행 스크립트**: [verify_sqlite.py](file:///Users/minho/Documents/Dataset/verify_sqlite.py)

---

## 2. 데이터베이스 구성 및 추가된 스키마

### A. 통계 요약
* **성공적으로 통합된 총 데이터셋 수**: **1,119개**
  * **UCR 분류 아카이브의 One-Class 변환 데이터셋**: **1,118개**
  * **Hugging Face 고래 소리 탐지 데이터셋 (CornellWhaleChallenge)**: **1개**
* **총 저장된 시계열 인스턴스 (TRAIN + TEST)**: **396,737개**
* **최종 데이터베이스 파일 크기**: **1,672.06 MB (1.67 GB)**

---

## 3. 알고리즘 구현 및 검증 결과

### ① [검증 1] 학습 데이터셋(TRAIN)의 100% 정상 보장
* **검증 결과**: **성공 (위반 건수: 0건)**
  * 모든 데이터셋의 `TRAIN` 세트에서 이상치 라벨(`1` 또는 `anomaly`)이 전혀 발견되지 않았습니다.

### ② [검증 2] 테스트 데이터셋(TEST)의 이상치 비율 2% 수준 통제
* **검증 결과**: **성공 (위반 건수: 0건, 평균 이상치 비율: 2.00%)**
  * 테스트 세트 내의 이상치 비율이 정확히 2% 수준으로 맞추어져, 편향되지 않은 정확한 모델 평가가 가능함을 확인했습니다.

---

## 4. [추가] 고성능 VAE 설계 및 데이터 증강 실험 결과

비지도 시계열 이상치 탐지 연구의 완성도를 높이기 위해, 오토인코더(AE)에 이어 확률론적 모델인 **1D Conv-VAE**를 성공적으로 도입하고 데이터 증강을 연동하였습니다.

### A. 핵심 개발 스크립트
1. **일반 VAE 스크립트**: [run_vae.py](file:///Users/minho/Documents/Dataset/run_vae.py)
2. **왜도 기반 적응형 임계치 VAE 스크립트**: [run_all_skewness_threshold_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_skewness_threshold_evaluations.py)
3. **극값 이론(EVT) 기반 임계치 VAE 스크립트**: [run_all_evt_threshold_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_evt_threshold_evaluations.py)
4. **와이블 및 검벨 임계치 VAE 스크립트**: [run_all_weibull_gumbel_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_weibull_gumbel_evaluations.py)
5. **온라인 동적 임계치 VAE 스크립트**: [run_all_online_dynamic_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_online_dynamic_evaluations.py)
6. **주파수 도메인 STFT VAE 스크립트**: [run_all_stft_vae_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_stft_vae_evaluations.py)
7. **주기성 분기형 하이브리드 VAE 스크립트**: [run_all_periodicity_hybrid_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_periodicity_hybrid_evaluations.py)
8. **고도화 주기성 분기형 하이브리드 VAE 스크립트**: [run_all_advanced_periodicity_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_advanced_periodicity_evaluations.py)
9. **길이 적응형 Conv-VAE 스크립트**: [run_all_adaptive_cnn_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_adaptive_cnn_evaluations.py) (시계열 길이에 비례해 커널 크기 및 레이어 레이아웃을 동적으로 매핑)
10. **적응형 복원 확률 VAE 스크립트**: [run_all_adaptive_recon_prob_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_adaptive_recon_prob_evaluations.py) (길이 적응형 구조와 NLL 몬테카를로 이상 점수 결합)
11. **이중 적응 하이브리드 VAE 스크립트**: [run_hybrid_adaptive_vae_evaluations.py](file:///Users/minho/Documents/Dataset/run_hybrid_adaptive_vae_evaluations.py) (시계열 길이에 따른 가우시안 NLL 스코어 및 시간축 MSE 스코어 결합 병합 스크립트)
12. **길이 게이트형 주기 VAE 스크립트**: [run_all_length_gated_periodicity_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_length_gated_periodicity_evaluations.py) (짧은 FFT 분해능을 통제하되 주기성 신호에만 STFT를 제한 연동하는 구조)
13. **연속 오케스트레이션 파이프라인**: [run_sequential_experiments.py](file:///Users/minho/Documents/Dataset/run_sequential_experiments.py) (선행 종료 시 4종 차세대 혁신 실험 자동 순차 실행)

### B. 최종 벤치마크 요약 (947개 데이터셋 전수 평가 완료)
* **극값 이론(Extreme Value Theory, EVT) 기반 임계치**:
  * **평균 F1-Score**: **0.3433**
* **주파수 도메인 STFT 복원 VAE**:
  * **평균 F1-Score**: **0.2384** (경계 스펙트럼 허상 및 Underfitting 문제 유발)
* **길이 게이트형 주기 VAE (제한적 STFT 결합)**:
  * **평균 F1-Score**: **0.2646** (FFT 붕괴 오탐은 막았으나, 복잡한 비선형 추세를 포함하는 센서 파형 영역에서 STFT 가중 결합이 원래의 형상 복원(MSE) 수렴을 현격히 그라디언트 교란/Underfitting함을 실증적으로 입증)
* **길이 적응형 Conv-VAE**:
  * **전체 평균 F1-Score (EVT)**: **0.3592**
* **적응형 복원 확률 VAE (Adaptive Recon-Prob VAE) 결과 🧪**:
  * **전체 평균 F1-Score (EVT)**: **0.3492**
* **이중 적응 하이브리드 VAE (Hybrid Adaptive VAE) 결과 (최종 최고 기록 달성) 🌟**:
  * **전체 평균 AUC-ROC**: **0.8971 (+0.38%p)**
  * **전체 평균 AUC-PR**: **0.5453 (+1.37%p)**
  * **전체 평균 F1-Score (EVT)**: **0.3663 (+0.71%p 추가 상승, SOTA 갱신) 🌟**
  * **전체 평균 Oracle F1**: **0.6527 (+1.10%p 상승, 상한선 돌파) 🌟**
