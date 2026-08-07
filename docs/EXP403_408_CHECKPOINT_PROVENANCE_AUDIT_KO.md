# Exp403-408 checkpoint provenance 복원 감사

작성일: 2026-07-28

## 결론

복원된 최종 산출물과 checkpoint 파일은 자체 SHA256 검증을 통과하지만, 현재
Git 이력만으로 checkpoint의 `input_sha256`을 재현할 수 없다. 따라서 이
checkpoint는 결과 감사와 review-only 분석에는 사용할 수 있지만, 현재
브랜치에서 재계산 없는 직접 resume의 근거로 사용하면 안 된다.

## 확인한 사실

- DB SHA256은 handoff manifest와 일치한다.
- Exp291/292 동결 lane 파일과 주요 Exp403-408 산출물 SHA256이 manifest와
  일치한다.
- checkpoint의 `checkpoint_sha256` 자체 무결성은 유지된다.
- 최종 `MANIFEST.json`의 `code_provenance.files`에 기록된 Exp403-408 원본
  소스 해시는 기준 커밋 `d493263`의 파일과 일치한다.
- 경로 이식 전 기준 커밋을 임시 detached worktree로 재현하고, 역사적
  `/Users/minho/Documents/Dataset` 경로만 현재의 동일 해시 데이터로
  치환해도 표본 5개 단계의 `input_sha256`이 일치하지 않는다.
- `pipeline.py`의 checkpoint 입력 해시 로직이 존재했던 다음 Git 버전을
  같은 방식으로 대조했지만 모두 불일치했다.

```text
045cbfe 41e0b19 bbeea9a fca9a1d d302d46 b6647da ea6e005
f864ac2 a5f3ff7 e65b1fb f943579 9794b05 19eca50 3db5349
```

초기 커밋 `ba811b4`에는 `_stage_input_hash`가 아직 존재하지 않는다.

## 원인 범위

checkpoint에는 최종 `input_sha256`만 있고, 그 값을 구성한 다음 항목의
개별 값은 저장되지 않는다.

- 단계 실행 시점의 각 코드 파일 SHA256
- 전처리된 dataset 값 SHA256
- feature source별 SHA256
- frozen lane 구성
- upstream checkpoint SHA256

한편 최종 `MANIFEST.json`의 code provenance는 report 작성 시점의 소스를
기록한다. 그러므로 checkpoint 생성과 report 생성 사이에 코드가 달랐는지를
manifest만으로 판별할 수 없다. 현재 Git 버전이 모두 불일치한다는 사실은
checkpoint가 Git에 남지 않은 working-tree 상태 또는 별도로 보존되지 않은
실행 소스에서 생성됐을 가능성과 일치한다. 단일 합성 SHA256만으로는 해당
소스를 역산할 수 없다.

## 허용되는 사용

- 해시가 일치하는 최종 산출물의 retrospective 검토
- 기존 결과의 review-only 분석
- 동결된 Hard/Standard/Priority 수량과 보고서의 무결성 감사

## 금지되는 주장과 동작

- 현재 코드가 기존 checkpoint를 안전하게 직접 resume할 수 있다는 주장
- `input_sha256` 불일치를 무시하도록 validator를 우회하는 변경
- 기존 Exp403-408 번호 아래에서 checkpoint를 재생성한 뒤 같은 실행으로 취급
- TEST 결과를 보고 새 설정을 고르는 재실행

## 다음 실행 조건

원본 실행 디렉터리 `/private/tmp/timeseries-exp393-396-design`의 보존본이나
checkpoint 생성 당시 소스 snapshot을 찾으면 다시 구성요소 해시를 대조한다.
이를 찾지 못한 상태에서 재검증이 필요하면 새 실험 번호와 사전등록으로
분리하고, checkpoint마다 합성 해시뿐 아니라 구성요소별 provenance를 함께
저장해야 한다.
