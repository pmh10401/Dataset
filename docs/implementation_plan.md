# 실행 계획서: [우선순위 2 - 길이 게이트형 주기 VAE (Length-Gated Periodicity VAE) 실험]

본 계획서는 **"길이 적응형 VAE(SOTA: F1 0.3592)"**에 주파수 복원 기법(STFT Loss)을 결합하되, 과거 실패 요인이었던 짧은 시계열 상의 주파수 분해능 한계와 ACF 노이즈 왜곡을 방어하기 위해 **길이 제한 게이트(Length-Gate)** 필터를 결합하는 방안을 기술합니다.

---

## 1. 아키텍처 및 논리적/통계적 설계 근거

### 💡 통계적 한계와 게이트의 필요성
* **FFT 주파수 해상도 한계**:
  시계열 길이 $L$이 짧은 소형 데이터셋($L < 128$)에서 FFT를 연산하게 되면 스펙트럼의 주파수 Bin 개수가 너무 작아 확률분포 엔트로피 수식상 비주기 신호조차 엔트로피가 $0.65$ 미만으로 쉽게 붕괴해 주기성 신호로 잘못 분류되는 기형적 오탐이 유발됩니다.
* **디트렌딩 후 ACF Peak 왜곡**:
  Z-정규화된 짧은 시계열을 디트렌딩(Detrend)할 시 남은 초미세 노이즈 진폭의 위상 왜곡으로 인해 특정 Lag에서 우연히 $0.45$ 이상의 상관 피크가 팽창하여 오판 확률이 매우 높습니다.
* **길이 제한 게이트 (Length-Gate) 라우팅**:
  이 오류를 원천 차단하기 위해 **시계열 길이 $L \ge 128$인 데이터셋만 주기성 판별기의 대상으로 허용**합니다. $L < 128$인 초소형/짧은 시계열군은 주기성 진단 자체를 거치지 않고 무조건 비주기성으로 매핑하여 시간축 단독 복원 학습(lambda_spec = 0.0)을 수행합니다.

### 📐 복합 분기 손실 함수 구조:
1. **$L < 128$ 또는 비주기성 판정 시**:
   * 시간축 등분산 MSE 손실 단독 사용 ($\lambda_{spec} = 0.0$)
2. **$L \ge 128$ 이며 주기성 판정 통과 시 (True Periodic Long Series)**:
   * 시간 MSE와 STFT 2D Spectrogram Magnitude 오차를 절반씩 융합 ($\lambda_{spec} = 0.5$)

---

## 2. 제안되는 변경 내용

### 평가 및 벤치마크 스크립트 작성
#### [NEW] [run_all_length_gated_periodicity_evaluations.py](file:///Users/minho/Documents/Dataset/run_all_length_gated_periodicity_evaluations.py)
* 길이 적응형 네트워크 토폴로지를 기반으로 하는 주기성 하이브리드 VAE 벤치마크 코드를 작성합니다.
* 전처리로 선형 디트렌딩을 거친 신호에 대해 PSD Normalized Spectral Entropy 및 ACF 피크 판정을 집행하되, 시계열 길이 $L < 128$인 데이터셋은 자동으로 판정을 배제합니다.
* 947개 데이터셋 전수에 평가를 수행하고 극값 이론(EVT-GPD) 임계치 및 왜도 적응형 임계치를 매핑하여 성능을 도출합니다.
* 결과를 [vae_results_length_gated_periodicity.csv](file:///Users/minho/Documents/Dataset/vae_results_length_gated_periodicity.csv)에 저장합니다.

---

## 3. 검증 계획

### 자동 테스트
동적 레이어 생성 시 PyTorch가 계산하는 인코더 아웃풋 플랫 디멘션(`flat_dim`)과 디코더의 업샘플링 보간 연산 간의 수학적 차원이 유효한지 1차 자가진단(Dry run)을 수행합니다:
```bash
/Users/minho/Documents/Dataset/.venv/bin/python /Users/minho/Documents/Dataset/run_all_length_gated_periodicity_evaluations.py
```

### 학술적/실무적 성능 분석
* 최종 [vae_results_length_gated_periodicity.csv](file:///Users/minho/Documents/Dataset/vae_results_length_gated_periodicity.csv)와 이전의 단순 적응형 VAE(`vae_results_adaptive_cnn.csv`) 및 고도화 주기성 VAE(`vae_results_advanced_periodicity.csv`)를 교차 대조합니다.
* 게이트 필터 장착으로 인해 주기성 오판 사태가 통제되었는지 확인하고, 실제 주기성 신호들의 도메인 지표 결합이 전역 F1 성능 상승을 이끄는지 입증합니다.
* 분석 수치를 `research_report.md` 및 `walkthrough.md`에 업데이트합니다.
