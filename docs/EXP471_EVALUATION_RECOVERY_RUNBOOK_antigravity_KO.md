# Exp471 평가 복구 운영 런북

이 문서는 Exp467의 고정된 결과를 다시 실행하지 않고 Exp471 점수 캐시,
평가 산출물, 논문 보고서를 안전하게 생성·재개하는 운영 절차다. 운영자는
반드시 아래의 공식 Python과 닫힌 launcher만 사용한다.

## 1. 고정 경로와 운영 원칙

- 작업 디렉터리: `/Users/mac/Documents/Time Series/.worktrees/exp471-exp467-evaluation-recovery-antigravity`
- 공식 Python: `/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python`
- Exp467 읽기 전용 소스 루트: `/Users/mac/Documents/Dataset/exp467_numba_calibration_resume_official_antigravity_20260823`
- Exp471 루트: `/Users/mac/Documents/Dataset/exp471_exp467_evaluation_recovery_antigravity_20260824`
- 점수 캐시: `<Exp471 루트>/score_cache_antigravity`
- 평가 산출물: `<Exp471 루트>/evaluation_antigravity`
- 보고서: `<Exp471 루트>/report_antigravity`
- 진행 로그: `<Exp471 루트>/progress_antigravity.jsonl`
- 전처리 진행 로그: `<Exp471 루트>/preprocessing_progress_antigravity.jsonl`

Exp467 루트는 읽기 전용 권위다. Exp471 명령으로 Exp467 파일, manifest,
checkpoint, progress log를 수정해서는 안 된다. Exp471의 모든 출판은
create-only이며, 이미 있는 바이트가 정확히 같을 때만 재사용한다.

경로나 산출물의 소유권·정체를 확실히 증명할 수 없으면 즉시 중단한다.
운영자는 알 수 없는 산출물, 외부에서 만든 산출물, marker가 없는 산출물을
삭제하거나 덮어쓰거나 이름을 바꾸지 않는다. symlink, 예상과 다른 파일
종류, 예상 밖의 기존 destination도 임의로 고치지 말고 보존한 채 조사한다.

한 시점에 상태를 바꾸는 공식 Exp471 명령은 하나만 실행한다. Task 7에서는
그 명령이 `report` 하나다. 단, `report` 실행을 관찰하기 위한 read-only
`status`는 별도 터미널에서 동시에 실행할 수 있다. `status`는 공식 작업을
시작·중단·변경하지 않는다. `report` 실행 중에는 `verify-source`,
`benchmark-prepare`, `prepare`, 또 다른 `report` 같은 명령을 시작하지 않는다.

## 2. 공식 명령

먼저 고정 작업 디렉터리로 이동한다.

```sh
cd '/Users/mac/Documents/Time Series/.worktrees/exp471-exp467-evaluation-recovery-antigravity'
```

도움말과 허용 명령을 확인한다.

```sh
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py --help
```

읽기 전용 상태 확인은 실행 전·후뿐 아니라 `report` 실행 중에도 별도
터미널에서 수행할 수 있다.

```sh
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py status
```

선택적인 사전 점검 명령은 다음과 같다. `verify-source`는 고정 Git 바이트와
Exp467 권위를 검증하며 TEST label을 열지 않는다. `benchmark-prepare`는
동일 바이트 조건과 1.1배 속도 기준으로 CPU worker 수를 결정한다.
`prepare`는 label-free 점수 shard만 준비한다.

```sh
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py verify-source

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py benchmark-prepare

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py prepare
```

### Task 7의 유일한 실행 명령

Task 7에서는 아래의 재개 가능한 `report` 명령을 한 번만 시작한다.

```sh
/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py report
```

`report`는 source 검증 후 완전한 cache를 재개 또는 재사용하고, 완전한 평가
산출물을 재사용하거나 없을 때만 평가한 뒤, 보고서를 원자적으로 출판한다.
별도의 `evaluate` 명령은 공개 CLI에 없으며 운영자는 이를 만들거나 실행하지
않는다. 특히 `report`와 동시에 별도 `evaluate` 프로세스를 시작해서는 안
된다.

정상 완료 때 cache의 정확한 typed partition은 총 11,170개 중 scored 10,775개,
abstained 395개, failed 0개다. 열 개 방법 모두의 scored 교집합(common support)은
922개 데이터셋이다. 이 수는 연속 점수 지표의 분모이며, 전체 1,117개 데이터셋
결정 표의 분모와 다르므로 서로 바꾸어 쓰지 않는다.

### 별도 터미널의 실시간 read-only 모니터링

터미널 1에서는 위의 `report` 명령 하나를 계속 실행한다. 터미널 2에서는
같은 고정 worktree로 이동한 뒤 다음 `status` 명령만 필요할 때 반복한다.

```sh
cd '/Users/mac/Documents/Time Series/.worktrees/exp471-exp467-evaluation-recovery-antigravity'

/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
  scratch/run_exp471_exp467_evaluation_recovery_antigravity.py status
```

`status`는 cache, evaluation, report, progress를 관찰할 뿐 새 작업을
예약하거나 공식 프로세스를 시작·중단·재시작하지 않는다. progress writer는
exclusive lock으로 append하고 `status`는 shared lock으로 완전한 최신 기록을
읽는다. 따라서 실행 중 진행률, 실패 수, ETA는 이 명령의 stdout JSON으로만
확인한다. 실행 중인 `progress_antigravity.jsonl`을 `cat`, `tail` 또는 다른
도구로 직접 읽어 잠깐 보이는 미완성 tail을 권위로 해석해서는 안 된다.

다음은 `status`만 반복해 별도의 임시 모니터 로그에 남기는 안전한 예다.
`/tmp`의 로그는 공식 산출물이 아니며, `Ctrl-C`는 이 모니터만 멈춘다.
실행 중인 `report`에는 영향을 주지 않는다.

```sh
while true; do
  /Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python \
    scratch/run_exp471_exp467_evaluation_recovery_antigravity.py status
  sleep 15
done | tee /tmp/exp471_status_monitor_antigravity.log
```

전처리가 진행 중이면 `status.preprocessing_progress`에서 `completed`, `total`,
`percent`, `elapsed_seconds`, `eta_seconds`, `workers`를 확인한다. 첫 측정 전에는
ETA가 `null`일 수 있다. 실제 데이터의 동일성 검사에 통과하고 11개 작업자가
1개 작업자보다 1.1배 이상 빠를 때만 병렬 전처리를 사용하며, 그렇지 않으면
결과를 바꾸지 않기 위해 자동으로 1개 작업자를 사용한다.

`acceleration_backend`가 `numba_serial_safe`이면 ASPE 내부 반복문에
`parallel=False`, `fastmath=False`, `nogil=True` Numba 커널을 사용한다. 실제
11개 데이터셋 사전 측정에서 CPU 37.64초, Numba 0.55초였고 최종 점수 구조,
선택 후보, 수치 결과가 같았다. `parallel=True`와 `fastmath=True` 후보는 같은
결과를 냈지만 더 빠르지 않아 선택하지 않았다. Numba 적용 후에도 데이터셋
단위 11작업자 경로는 같은 결과와 1.1배 속도 기준을 다시 통과해야 채택된다.

## 3. 표준 출력과 오류 출력

각 명령은 성공 시 stdout에 canonical finite JSON 한 줄만 출력한다. 명령
실패와 잘못된 인자는 stderr에 민감한 상세 메시지나 실제 데이터 이름을
넣지 않은 JSON 한 줄만 출력하고 종료 코드 `2`를 반환한다.

성공 레코드의 최상위 필드는 다음과 같다.

| 명령 | 예상 필드 |
| --- | --- |
| `--help` | `schema`, `experiment`, `command`, `commands` |
| `verify-source` | `schema`, `experiment`, `command`, `authority_sha256`, `inventory_sha256`, `total_successes`, `root` |
| `benchmark-prepare` | `schema`, `experiment`, `command`, `decision` |
| `prepare` | `schema`, `experiment`, `command`, `manifest_sha256`, `total_shards`, `root` |
| `status` | `schema`, `experiment`, `command`, `durable_completed`, `total`, `scored_shards`, `abstained_shards`, `failed_shards`, `remaining`, `remaining_identities`, `cache_complete`, `evaluation_complete`, `report_complete`, `elapsed_seconds`, `rate_per_second`, `eta_seconds`, `failures`, `verified_stage`, `paths`, `worker_decision`, `latest_progress`, `preprocessing_progress`, `failure` |
| `report` | `schema`, `experiment`, `command`, `source_authority_sha256`, `cache_manifest_sha256`, `evaluation_sha256`, `report_manifest_sha256`, `report_root` |

`benchmark-prepare.decision`에는 다음 필드가 모두 있어야 한다.

`schema`, `cache_context_sha256`, `detected_logical_cpus`,
`default_cpu_workers`, `requested_parallel_workers`, `selected_workers`,
`inner_numerical_threads`, `single_worker_seconds`, `parallel_seconds`,
`speedup_threshold`, `speedup`, `byte_parity`, `parallel_accepted`,
`parallel_error_type`, `parallel_error_message`, `fallback_reason`,
`acceleration`, `decision_sha256`.

잘못된 인자는 `schema`, `error`만 가지며 `error`는 `invalid_arguments`다.
실행 중 오류는 `schema`, `error`, `command`, `exception_type`만 가지며
`error`는 `command_failed`다. stderr의 `exception_type`만 보고 파일을
수정하거나 삭제하지 않는다. 상태와 보존된 증거를 함께 조사한다.

## 4. 상태 해석

`status.verified_stage`는 다음 네 값 중 하나다.

- `cache-incomplete`: 완전한 cache manifest가 아직 없다. 이미 검증된 shard
  수는 `durable_completed`, 남은 수는 `remaining`이다.
- `cache-complete`: 완전한 cache manifest가 있다.
- `evaluation-present-unverified`: 평가 파일의 존재를 관찰했다. 이 문자열만
  보고 평가 내용을 최종 승인해서는 안 되며, `report`가 권위를 다시 열어
  검증하게 한다.
- `report-present-unverified`: 보고서 complete marker의 존재를 관찰했다.
  이 문자열만으로 출판 완료를 선언하지 말고, `report` 성공 JSON과 report
  manifest digest까지 확인한다.

`worker_decision.benchmark`는 측정된 worker 결정이고
`worker_decision.actual`은 실제 실행 결정이다. cache 작업이 시작되기
전에는 둘 다 `null`일 수 있다. 기본 정책은 감지된 logical CPU 수에서
하나를 뺀 worker 수(최소 1)이며, 병렬 경로는 shard 바이트가 같고 정확한
stage에서 1.1배 이상 빠른 경우에만 선택된다.

`scored_shards`, `abstained_shards`, `failed_shards`는 cache shard의 현재
durable 분류다. abstained는 입력 부족 때문에 점수를 만들지 않는 정상 완료이며,
실패도 오류도 아니다. 따라서 abstained는 `durable_completed`에는 포함되지만
`failed_shards`나 누적 `failures`에는 포함되지 않는다. `evaluation_complete`와
`report_complete`는 source-free 구조·digest 검증을 통과한 durable 파일 또는
complete marker를 read-only로 관찰한 값이다. source 및 TEST label에 다시
결속한 최종 과학적 승인까지 뜻하지는 않으며, 최종 권위 확인은 `report`의
성공 JSON과 manifest digest로 한다.

다음은 구조 설명용 합성 상태 레코드다. 실제 경로, dataset 이름, label,
digest를 포함하지 않으며 공식 산출물로 사용할 수 없다.

```json
{
  "command": "status",
  "durable_completed": 3,
  "experiment": "Exp471",
  "failure": null,
  "failures": 0,
  "latest_progress": {
    "artifact_path": "/synthetic/exp471_antigravity/score_cache_antigravity",
    "attempts": 4,
    "durable_completed": 3,
    "elapsed_seconds": 12.5,
    "eta_seconds": 46570.833333,
    "failures": 0,
    "remaining": 11167,
    "root_path": "/synthetic/exp471_antigravity",
    "schema": "exp471-progress-antigravity.v1",
    "stage": "score-cache",
    "timestamp_utc": "2099-01-01T00:00:00.000+00:00",
    "total": 11170
  },
  "paths": {
    "evaluation": "/synthetic/exp471_antigravity/evaluation_antigravity",
    "progress": "/synthetic/exp471_antigravity/progress_antigravity.jsonl",
    "report": "/synthetic/exp471_antigravity/report_antigravity",
    "root": "/synthetic/exp471_antigravity",
    "score_cache": "/synthetic/exp471_antigravity/score_cache_antigravity"
  },
  "remaining": 11167,
  "schema": "exp471-status-antigravity.v1",
  "total": 11170,
  "verified_stage": "cache-incomplete",
  "worker_decision": {
    "actual": null,
    "benchmark": null
  }
}
```

## 5. 진행 로그와 실패 원장

진행 로그는 다음 경로의 append-only JSON Lines 파일이다.

`/Users/mac/Documents/Dataset/exp471_exp467_evaluation_recovery_antigravity_20260824/progress_antigravity.jsonl`

각 줄에는 다음 12개 필드가 정확히 있어야 한다.

`schema`, `stage`, `durable_completed`, `total`, `failures`, `attempts`,
`remaining`, `elapsed_seconds`, `eta_seconds`, `timestamp_utc`,
`artifact_path`, `root_path`.

- `attempts`는 JSONL에 추가된 전체 progress 레코드 수다. 특정 shard의
  재시도 횟수가 아니다.
- `failures`는 실패 progress 이벤트의 누적 수다. 성공적으로 재개해도
  과거 실패를 지우지 않는다.
- `remaining`은 항상 `total - durable_completed`다.
- `eta_seconds`는 완료된 작업이 없으면 `null`, stage가 끝나면 `0.0`이다.
- `timestamp_utc`는 timezone이 포함된 UTC 시각이다.

stage별 고정 total과 의미는 다음과 같다.

| stage | total | 완료 의미 |
| --- | ---: | --- |
| `verify-source` | 4,468 | 고정 Exp467 source authority가 durable하게 출판됨 |
| `score-cache` | 11,170 | 10개 method × 1,117개 dataset shard가 durable함 |
| `evaluation` | 10 | 고정 method 순서의 평가 산출물이 durable함 |
| `report` | 15 | 14개 typed 보고서 파일과 provenance leaf가 digest로 결속됨 |

로그를 직접 편집, 잘라내기, 교체, 삭제하거나 실패 수를 0으로 고치지 않는다.
직접 읽어야 한다면 공식 프로세스가 완전히 종료된 뒤 forensic 목적으로만
read-only 확인하고, 마지막 줄이 불완전하거나 schema/path/count 관계가
맞지 않으면 즉시 중단한다. 실행 중에는 직접 JSONL을 읽지 않고 공식
`status`의 locked snapshot만 사용한다.
`status`가 progress 파일을 malformed, noncanonical, truncated, unsafe로
판정하면 파일을 복구하려고 쓰지 말고 원본을 보존해 조사 담당자에게 넘긴다.

점수 worker가 실패하면 다른 첫 시도는 계속되고, 첫 wave가 끝난 뒤 실패한
항목만 한 번 재시도한다. 두 번째도 실패하면 완전한 cache manifest를
출판하지 않고 명령이 실패한다. 이때 이미 durable한 shard와 progress
원장은 그대로 둔다. 환경 또는 코드 권위 문제를 해결한 뒤 같은 `report`
명령을 다시 실행하면 기존 shard를 검증하고 누락된 작업만 예약한다.

`status.failures > 0`은 과거 실패가 원장에 남아 있다는 뜻이며 그 자체로
현재 실행이 실패 중이라는 뜻은 아니다. `status.failure`은 최근 progress
stage와 누적 실패 수를 보여준다. 완료 판정은 실패 수를 지우는 방식이
아니라, 재개한 `report`의 성공 JSON과 durable manifest digest를 확인하는
방식으로 한다. progress 이전 또는 사이에서 발생한 command 오류는 stderr
JSON도 별도 운영 증거로 보존한다.

## 6. 안전한 중단과 재개

1. 가능한 경우 실행 중인 공식 명령에 `Ctrl-C`를 한 번 보내 정상 중단을
   요청한다. 새 mutating Exp471 명령을 동시에 시작하지 않는다. 중단 상태를
   관찰하는 read-only `status`만 별도 터미널에서 허용한다.
2. 기존 프로세스가 완전히 끝났음을 확인한다. 강제 종료는 정상 중단이
   불가능할 때만 사용하며, 종료 직후 파일을 정리하지 않는다.
3. 공식 `status` 명령을 단독으로 실행해 durable count, latest progress,
   오류 여부를 기록한다.
4. symlink, 외부 산출물, markerless destination, malformed JSONL이 없고
   중단 원인이 해소되었으면 Task 7의 동일한 `report` 명령을 다시 실행한다.

중단 시점에 완성된 cache shard는 재검증 후 재사용한다. 완전한 평가
산출물이 있으면 metric 계산을 반복하지 않고 다시 연다. 보고서가 원자적
rename 전에 중단되면 durable 평가에서 보고서만 다시 만든다. 부분 파일을
완료 파일로 이름 변경하거나 manifest를 수작업으로 만들지 않는다.

## 7. retained quarantine과 외부 산출물

보고서 출판·lock 정리 중 pathname의 inode 정체를 확실히 증명할 수 없으면
구현은 해당 객체를 Exp471 루트 아래의 비권위 sibling 이름으로 보존할 수
있다. 이름 형식은 `.retained_quarantine_<24자리 16진 토큰>_antigravity`다.

이 retained quarantine은 최종 보고서나 lock 권위가 아니며 canonical
`report_antigravity`를 대신하지 않는다. 반복된 실패로 누적되어 디스크를
사용할 수 있지만 운영자가 임의 삭제해서는 안 된다. 다음과 같이 처리한다.

1. 이름, `lstat` 메타데이터, 발견 시각, 직전 명령의 sanitized stderr JSON을
   기록한다. 내용을 실행하거나 symlink를 따라가지 않는다.
2. canonical report, lock, stage 이름으로 되돌리거나 재사용하지 않는다.
3. 알 수 없는 객체와 함께 그대로 보존하고 조사 담당자에게 전달한다.
4. 삭제가 필요하면 별도의 검토를 거친 inode-bound 회수 절차가 승인될
   때까지 기다린다. 이 런북은 자동 또는 수동 삭제 권한을 부여하지 않는다.

canonical report destination이 이미 있지만 complete marker가 없거나,
기존 바이트가 예상 digest와 다르면 외부 산출물로 취급한다. 덮어쓰기,
삭제, 이동, 격리 이름의 재사용을 하지 말고 `command_failed` 증거와 상태를
보존한 채 중단한다.

## 8. 완료 체크리스트

- `report`가 종료 코드 `0`으로 끝났고 stdout에 정확히 한 개의
  `exp471-command-result-antigravity.v1` JSON 레코드가 있다.
- JSON의 `command`가 `report`이며 네 digest 필드가 64자리 SHA-256이다.
- `report_root`가 고정 Exp471 report 경로다.
- 별도 동시 `verify-source`, `benchmark-prepare`, `prepare`, `evaluate` 또는
  또 다른 `report` 프로세스를 시작하지 않았다. 별도 터미널의 read-only
  `status`만 실시간 관찰자로 사용했다.
- Exp467 루트와 파일을 변경하지 않았다.
- 알 수 없는 산출물과 retained quarantine을 삭제하거나 덮어쓰지 않았다.
- 실패가 있었다면 stderr JSON, 마지막 유효 progress 레코드, 재개 결과를
  append-only 운영 기록에 함께 남겼다.
