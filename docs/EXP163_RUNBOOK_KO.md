# Exp163: 대안 점수원 꼬리 합의 검토 실행 가이드 (Runbook)

본 가이드는 Exp163 1차 구현의 실행, 테스트, 그리고 Smoke 테스트를 위한 명령 조합 및 결과물 확인 절차를 설명한다.

## 1. 사전 요구사항

- 프로젝트 저장소 위치: `/Users/minho/Documents/timesries project`
- 사용 가상환경: 프로젝트 내 구성된 기본 파이썬 환경 (`numpy` 탑재)
- 필수 데이터 파일 준비 상태 확인:
  - `outputs/exp160_candidate_zero_similarity_recovery/02_exp160_predictions.jsonl`
  - `outputs/exp162_full_coverage_alternative_score_cache/02_cache_manifest.csv`
  - `outputs/exp162_full_coverage_alternative_score_cache/01_source_contract.json`

## 2. 모듈 구조

실험 모듈은 `experiments/exp163/` 폴더 내에 배치되어 있다.
- `alternative_tail_core.py`: 분위수 임계값, strict 초과 연산, 2-of-3/3-of-3 합의 및 레인 통합 논리.
- `exp163_io.py`: 입력 파일 해시 검증, atomic write 및 manifest 쓰기 관리.
- `run_experiment_163_alternative_source_tail_agreement_review.py`: CLI 실행 프론트엔드.
- `test_exp163_alternative_tail_review.py`: 단위 논리 검증 테스트.
- `test_exp163_structure.py`: 패키지 구조 및 prediction 인자 제약(y_test 배제) 테스트.

## 3. 실행 단계 안내

`run_experiment_163_alternative_source_tail_agreement_review.py`는 `--stage` 인자를 통해 다음의 단계들을 순차적 혹은 독립적으로 실행할 수 있다.

### A. Audit 단계
동결 해시들과 1,117개 데이터셋(혹은 smoke의 10개) 캐시 파일의 무결성 및 구조(finiteness, config hash 등)를 사전 검증한다.
```bash
python3 experiments/exp163/run_experiment_163_alternative_source_tail_agreement_review.py --stage audit --smoke
```

### B. Predict 단계
검증을 통과한 캐시 파일을 이용하여 대안 후보를 산출하고 레인을 병합한 후, JSONL 형식의 예측 결과와 무결성 manifest.csv를 함께 저장한다.
```bash
python3 experiments/exp163/run_experiment_163_alternative_source_tail_agreement_review.py --stage predict --smoke
```
Smoke 검증이 통과한 뒤 full prediction은 `--smoke` 없이 실행한다. Full prediction은 1,117개 데이터셋과 4개 alpha, 2개 variant로 총 8,936행을 생성해야 한다.

### C. Manifest 단계
생성된 예측 파일의 SHA256 해시값 검증용 manifest 파일만을 단독으로 재생성한다.
```bash
python3 experiments/exp163/run_experiment_163_alternative_source_tail_agreement_review.py --stage manifest --smoke
```

### D. Evaluate 단계
예측 결과 manifest 해시 검증을 완료한 후, SQLite DB에서 테스트 라벨을 불러와 비교 평가한다.
```bash
python3 experiments/exp163/run_experiment_163_alternative_source_tail_agreement_review.py \
  --stage evaluate
```

평가 전에 `03_prediction_manifest.json`의 SHA256과 행 수를 검증한다. 평가 단계는 prediction 파일을 수정하지 않아야 한다.

### E. Full 실행
```bash
python3 experiments/exp163/run_experiment_163_alternative_source_tail_agreement_review.py \
  --stage audit --stage predict

python3 experiments/exp163/independent_reaggregate_exp163.py \
  --output-dir outputs/exp163_alternative_source_tail_agreement_review

python3 experiments/exp163/run_experiment_163_alternative_source_tail_agreement_review.py \
  --stage evaluate

python3 experiments/exp163/independent_reaggregate_exp163.py \
  --include-evaluation \
  --output-dir outputs/exp163_alternative_source_tail_agreement_review
```

## 4. 단위 및 구조 테스트 실행

모든 단위 기능 및 구조적 불변성 검증을 확인하기 위해 다음의 테스트 명령어를 활용한다:
```bash
python3 -m unittest discover -s experiments/exp163 -p "test_*.py"
```

## 5. Smoke 테스트 확인 절차

`--smoke` 옵션 실행 시, 출력 경로는 `outputs/exp163_alternative_source_tail_agreement_review_smoke/` 가 된다.
성공적인 실행 후 아래의 메트릭과 규칙들이 만족하는지 최종 체크한다:
1. **생성된 행의 개수**: `10 datasets x 4 alphas x 2 variants = 80 rows` 가 올바르게 수록되었는가?
2. **평가 필드 부재**: JSONL 각 행 내에 `tp`, `fp`, `fn`, `f1`, `y_test` 등 테스트 라벨이나 메트릭 관련 필드가 0 또는 부재 상태인가?
3. **레인 상호 배타성**: `Hard`, `Priority`, `Standard` 세 레인의 상호 교집합이 공집합인가?
4. **베이스라인 일치**: `hard_indices`, `priority_review_indices`가 Exp160의 오리지널 예측값과 일치하는가?

## 6. 결과 해석

- Hard 지표는 autonomous 성능이다.
- Standard/Priority와 `mean_combined_f1`은 human-assisted 진단이다.
- `tail_rank_2of3`과 `tail_rank_3of3`, 네 alpha를 모두 보고하며 TEST 결과로 하나를 선택하지 않는다.
- 후보 0 gate는 TEST label을 사용하지 않지만 frozen TEST score 결과에 의존하므로 end-to-end strict TRAIN-only로 부르지 않는다.
