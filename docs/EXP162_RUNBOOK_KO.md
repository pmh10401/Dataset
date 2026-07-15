# Exp162 Alternative Score Cache 실행 가이드 (Runbook)

본 가이드는 Exp162 스모크 테스트 및 검증 프로세스를 수행하기 위한 실행 지침서입니다.

## 1. 실행 전 준비사항
- **데이터베이스 확인:** 데이터베이스 파일 `/Users/minho/Documents/Dataset/univariate_ts.db`가 지정된 경로에 존재해야 합니다.
- **의존성 확인:** Python 3 및 필수 라이브러리(`numpy`, `scikit-learn`)가 설치되어 있어야 합니다.

## 2. 병렬 처리 및 자원 제약 조건
- 워커 내부의 NumPy, OpenBLAS, MKL 등의 스레드는 CPU 경쟁 방지를 위해 **1스레드**로 고정해야 합니다.
- 본 실행 러너는 상단에서 자동으로 아래 환경 변수를 1로 설정하여 실행합니다:
  ```python
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"
  os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
  os.environ["NUMEXPR_NUM_THREADS"] = "1"
  ```

## 3. 테스트 명령어

### (1) 컴파일 테스트 및 단위 테스트
코드가 정상적으로 동작하는지 사전 검증하기 위해 컴파일과 테스트 스위트를 수행합니다.
```bash
python3 -m py_compile experiments/exp162/*.py
python3 -m unittest \
  experiments/exp162/test_exp162_full_coverage_score_cache.py \
  experiments/exp162/test_exp162_structure.py
```

### (2) 10-Dataset Smoke Run (초기 실행)
초기 실행 시 checkpoint를 리셋하고 10개의 데이터셋에 대해 2개의 워커로 스모크 테스트를 진행합니다.
```bash
python3 experiments/exp162/run_experiment_162_full_coverage_alternative_score_cache.py \
  --dataset-limit 10 \
  --workers 2 \
  --reset-checkpoint \
  --output-dir outputs/exp162_full_coverage_alternative_score_cache_smoke
```

### (3) Checkpoint 재사용 검증 (재실행)
동일한 명령어에서 `--reset-checkpoint`만 제외하고 실행하여 checkpoint가 완벽히 재사용되는지, 캐시 SHA256 해시가 흔들림 없이 유지되는지 검증합니다.
```bash
python3 experiments/exp162/run_experiment_162_full_coverage_alternative_score_cache.py \
  --dataset-limit 10 \
  --workers 2 \
  --output-dir outputs/exp162_full_coverage_alternative_score_cache_smoke
```
*예상 결과:* `Resume summary: 10 datasets reused from checkpoints. 0 datasets to compute.`가 표시되고 해시 검증이 성공해야 합니다.

## 4. 완료 조건 (Smoke 완료 보고 요구사항)
스모크 테스트를 마친 후, 완료 보고서 작성을 위해 아래 항목들을 수집 및 검증하십시오:
1. 생성/수정 파일 목록
2. 단위 테스트 실행 결과
3. Smoke 실행 결과 및 에러 건수 (0이어야 성공)
4. 세 소스의 Cache array shape 및 finite 여부 독립 검사
5. Checkpoint resume 재사용 횟수 및 해시 안정성 (재실행 전후로 해시가 완벽히 일치하는지 여부)
6. 발견된 이슈 및 해결 방안
7. Full run을 고의적으로 수행하지 않았다는 최종 확약
