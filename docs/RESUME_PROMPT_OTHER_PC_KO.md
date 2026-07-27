# 다른 Mac Codex 재개 프롬프트

아래 내용을 새 Mac의 Codex에 전달한다.

```text
저장소: /Users/minho/Documents/timesries project
데이터: /Users/minho/Documents/Dataset
기준 브랜치: codex/exp403-408-feature-ml-validation

먼저 다음 문서를 읽어라.

1. docs/HANDOFF_20260728_EXP403_408_KO.md
2. docs/EXP403_408_PREREGISTRATION_KO.md
3. docs/EXP403_408_RUNBOOK_KO.md
4. .superpowers/sdd/2026-07-27-exp403-408-feature-ml-validation/task-13-report.md

그다음 아래 검증을 수행하고 결과를 보고하라.

1. git status --short --branch
2. git rev-parse HEAD
3. shasum -a 256 /Users/minho/Documents/Dataset/univariate_ts.db
4. sqlite3 /Users/minho/Documents/Dataset/univariate_ts.db "PRAGMA quick_check;"
5. python3 -m unittest discover -s experiments/exp403_408/tests -p "test_*.py"
6. Google Drive HANDOFF_MANIFEST.json에 기록된 Git commit, DB SHA256,
   artifact archive SHA256과 현재 파일을 비교

현재 Exp403-408 전체 실행은 1,117개 데이터셋, 오류 0이며 동결 lane은
Hard 2,218 / Standard 806 / Priority 31이다. Exp406-408 새 근거는
review-only다. prediction SHA256은
437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70이다.

프로젝트 목표는 단일 mean F1 최대화가 아니라 정상 TRAIN 데이터만으로 후보를
만들고 false alarm을 억제하여 사용자 신뢰를 높이는 것이다.

절대 지켜야 할 원칙:
- prediction에서 TEST label, 실제 이상 위치, 과거 family별 TEST 성능을 사용하지 않는다.
- mean_combined_f1은 human-assisted 지표로만 다룬다.
- Priority review와 Shadow 후보를 자동 경보로 승격하지 않는다.
- tail replacement를 사용하지 않는다.
- TEST 결과를 보고 alpha, threshold, source, budget을 재선택하지 않는다.
- 새 개선은 새 실험 번호와 사전등록으로 분리한다.

전체 파이프라인을 다시 시작하지 말고 우선 복원 상태와 현재 checkpoint의
재사용 가능성을 확인하라. 장시간 실행은 사용자의 명시적 지시 후 시작한다.
```
