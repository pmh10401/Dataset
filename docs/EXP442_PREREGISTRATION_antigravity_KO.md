# Exp442 종료 ETA 저장 수리 사전등록

Exp442는 새 탐지 방법이 아니다. Exp441의 선택 규칙은 그대로 두고, 실행 종료 시
남은 시간을 `null`로 저장하도록 기술 계약만 맞춘 뒤 새 폴더에서 다시 실행한다.

동결 설계는
`docs/superpowers/specs/2026-08-08-exp442-terminal-eta-publication-repair-antigravity-design.md`
이며 구현 전에 확인한 SHA256은 다음과 같다.

```text
reviewed_design_sha256 = cf0b6e916a85cdf21c8cc28c735894f927b63e4c33aa4d62e95753540bd06a7b
```

```text
experiment_id = Exp442
run_identity = exp442-terminal-eta-publication-repair-antigravity-20260808
run_root = /Users/mac/Documents/Dataset/exp442_terminal_eta_publication_repair_antigravity_20260808
dataset_count = 1117
backend = cpu
cpu_workers = 11
candidate_budget = 30
candidate_per_dataset = 1
selected_coordinate_identity_sha256 = db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048
bootstrap_seed = 44120260808
bootstrap_samples = 20000
numpy_version = 2.4.6
```

다음은 바꾸지 않는다.

- Exp440 동결 source와 모든 SHA256
- Exp404 Hard 좌표와 SHA256
- Hard-empty dataset만 허용하는 규칙
- dataset마다 후보 하나, 전체 최대 30개
- 후보 순위와 동률 처리
- TEST label 차단선과 사후 평가 순서
- review-only 비승격 해석

Exp441의 기존 root와 `attempt_0000_antigravity`는 실패 증거로 보존한다. Exp442는
그 파일을 수정·복사·resume하지 않고 readiness부터 1,117개를 새로 읽는다.

prediction 단계에서는 TEST label, 실제 이상 위치, 과거 family별 TEST 성능,
Exp440 evaluation을 읽지 않는다. prediction publication과 SHA256이 완전히 동결된
뒤에만 Exp442 evaluation이 TEST label을 연다. 같은 실험에서 평가 결과를 보고
threshold, source, ranking, budget을 다시 고르지 않는다.

Exp442 결과는 적응형 사후 재현이므로 성공해도 자동 경보나 독립 검증, 외부
일반화, 과학적 champion으로 승격하지 않는다.
