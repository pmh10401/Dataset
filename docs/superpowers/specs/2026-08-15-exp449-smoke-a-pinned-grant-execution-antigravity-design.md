# Exp449 Smoke A Pinned-Grant Execution Design

## 목적

논문용 결과 산출을 위해 이미 검증된 Exp449 제안 알고리즘을 변경하지 않고 Smoke A를 안전하게 시작·재개·동결·검증할 수 있게 한다. 보안과 운영 메타데이터는 잘못된 실행, 다른 루트 재사용, 동시 실행, 직접 API 우회를 막는 최소 범위로 제한한다.

이 설계는 Smoke A만 다룬다. Smoke B, official 1,117-dataset 실행, TEST label 접근, 공개 baseline 및 AP/AUROC 평가는 별도 승인과 후속 계획 없이는 계속 차단한다.

## 절대 불변 조건

- Task4 TRAIN-only preprocessing, Task5 label-free TRAIN+TEST-value length-transductive preprocessing, Task4–Task7 scoring, Candidate C `hard_plus_stable`, threshold, alpha, seed, route, promotion, global top-30 규칙을 변경하지 않는다.
- 기존 Exp449 typed shard, outer manifest, checkpoint, policy score 및 alert 직렬화 형식을 변경하지 않는다.
- 실행 연결 전후의 `ProposalScore` 모든 과학 필드와 canonical shard/manifest bytes가 기존 순수 composition/freeze 경로와 같아야 한다.
- CPU dataset stage는 감지한 logical CPU 수에서 1을 뺀 worker 수를 사용하며 현재 host의 고정 기대값은 12→11이다.
- MLX/MPS/PyTorch 정책은 기존 Exp449 readiness를 그대로 사용한다. 새 가속 구현이나 benchmark는 이 설계 범위가 아니다. CPU가 canonical fallback이다.
- TEST label module을 import하거나 TEST label을 읽지 않는다.
- 새 파일명은 모두 `antigravity`를 포함한다.

## 최소 신뢰 모델

사용자는 키 기반 서명 대신 1회용 고정 digest 방식을 선택했다. 이 방식의 신뢰 기준은 사용자가 명시적으로 승인한 정확한 Git authorization commit SHA다.

이 방식이 방어하는 범위:

- 오래된 grant, 다른 target, 다른 commit, 다른 context/catalog/database/backend/worker, 다른 root로의 재사용;
- grant 또는 digest 변조;
- authorization commit에 허용되지 않은 코드나 문서 변경 추가;
- dirty worktree, symlink/traversal destination, 이미 사용된 root, 동시 worker;
- CLI를 우회한 공식 mutating API 호출.

이 방식이 방어하지 않는 범위:

- 저장소를 악의적으로 수정하고 새로운 커밋을 만든 사람이 사용자의 승인까지 가장하는 경우;
- 같은 OS 계정의 악의적 프로세스에 대한 운영체제 수준 격리.

따라서 실행 직전 사용자에게 exact authorization commit SHA, grant SHA, implementation parent SHA, target root, worker 수를 제시하고 `Smoke A 실행 승인`을 별도로 받아야 한다.

## 두 커밋 경계

### 구현 커밋 I

구현 커밋은 grant schema/verifier, process-local capability, claim/lock, worker launch, progress log, resume, Task4–Task7/freeze 연결과 테스트를 포함한다. grant와 digest는 포함하지 않으며 실행은 `execution_authority_not_configured`로 차단된다.

I는 Exp449 codemap fingerprint에서 `docs/authorities/`를 runtime authorization-data exclusion으로 명시한다. 이 경로에는 Python/source 파일을 둘 수 없고 아래 두 canonical JSON만 허용한다. Grant는 자체 digest와 authorization commit delta로 검증되므로 code-map source fingerprint에 포함하지 않는다.

I는 독립 Sol review에서 SHIP 판정을 받아야 한다.

### 승인 커밋 A

I의 clean HEAD에서 read-only readiness를 실행해 Smoke A context를 확정한 뒤 다음 두 데이터 파일만 추가한다.

- `docs/authorities/smoke_a_execution_grant_antigravity.json`
- `docs/authorities/smoke_a_execution_grant_digest_antigravity.json`

A는 다음을 만족해야 한다.

- 부모 commit이 정확히 I다.
- `I..A` 변경은 위 두 새 파일뿐이다.
- 두 파일은 canonical JSON이며 symlink가 아닌 tracked regular file이다.
- digest record의 SHA-256이 exact grant bytes와 일치한다.
- 현재 production/scientific source blobs, lineage lock 및 Exp449 codemap lock은 I와 동일하다.

독립 review가 A의 수정 필요성을 발견하면 A 위에 fix commit을 쌓아 승인하지 않는다. 구현을 수정한 새 I를 만든 뒤 grant와 direct-child A를 새로 생성한다.

A도 독립 Sol review에서 SHIP 판정을 받아야 한다. 이후 사용자가 exact A SHA를 승인해야만 launch할 수 있다.

## Grant schema

Grant schema는 `exp449-smoke-a-execution-grant-antigravity.v1`이다. 정확히 다음 권한을 묶는다.

- experiment `Exp449`, target `smoke-a`, policy `hard_plus_stable`;
- implementation commit I;
- Smoke A scientific root, operations root 및 고정 claim-lock path;
- database path/SHA/size, smoke catalog SHA 및 23 dataset identity;
- Smoke A `Exp449Context` SHA;
- I의 complete tracked scientific closure, explicit authority-code closure, package inventory, lineage lock, codemap lock, seeds/constants 및 acceleration assessment SHA;
- logical CPU count, dataset worker count 및 canonical backend;
- maximum waves `2`;
- label access `false`;
- unique canonical nonce;
- grant creation timestamp는 감사 정보로만 기록하고 유효성이나 결과 계산에 사용하지 않는다.

Digest record schema는 `exp449-smoke-a-execution-grant-digest-antigravity.v1`이며 grant relative path, grant byte size, grant SHA-256, implementation commit I만 포함한다.

Grant에는 authorization commit A를 넣지 않는다. A는 grant가 생성된 뒤 만들어지므로 순환 해시가 되기 때문이다. Runtime은 `HEAD == 사용자 승인 A`, `parent(A) == I`, exact two-file delta를 별도로 검증한다.

Nonce는 `secrets.token_hex(16)`으로 한 번 생성한 lowercase 32-hex이며 grant identity와 consumption receipt에만 사용한다. 과학 seed나 score 계산에는 사용하지 않는다.

## Readiness와 grant 준비

`prepare-smoke-a-grant`는 clean reviewed I에서만 실행한다. 이 명령은 기존 Exp449 readiness를 호출하므로 frozen DB의 values/catalog identity를 read-only로 확인하지만 scoring, root 생성, checkpoint, label 접근은 수행하지 않는다.

명령은 canonical grant와 digest 두 파일을 생성한 뒤 종료한다. grant 준비 자체는 실행 권한이 아니며 Smoke A root나 claim 파일을 만들 수 없다.

두 destination은 사전에 absent여야 하며 create-only로 발행한다. 기존 파일의 reuse나 overwrite는 허용하지 않는다. 준비가 끝나면 working tree에는 정확히 두 untracked authority JSON만 있어야 하고, 이 둘을 추가한 direct-child commit이 A가 된다.

## Fresh launch 흐름

1. CLI가 `--approved-authority-commit A`가 lowercase 40-hex이며 현재 clean HEAD와 같은지 확인한다.
2. A의 부모 I와 exact two-file delta를 확인한다.
3. grant/digest를 no-follow single snapshot으로 읽어 canonical schema와 SHA를 검증한다.
4. 현재 scientific blobs와 locks가 I와 같은지 확인한다.
5. exact authorization-delta 검증이 발급한 내부 read-only capability로만 기존 HEAD equality check를 I에 대해 좁게 대체한다. grant의 I를 기준으로 readiness의 비-root 사실을 다시 계산하고 grant의 database/catalog/context/closure/environment/backend/worker와 비교한다. 일반 readiness나 caller-supplied commit에는 이 예외를 허용하지 않는다.
6. Smoke A scientific/operations roots와 claim path가 모두 absent, non-symlink이고 정확한 frozen destination인지 확인한다.
7. fixed parent 아래 `/Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_a_antigravity_20260814_claim_antigravity.lock`을 create-only로 발행하고 `flock(LOCK_EX|LOCK_NB)`을 획득한다. 경쟁 실행은 즉시 실패한다.
8. operations root를 원자적으로 생성하고 `execution_authority_consumed_antigravity.json`을 create-only로 기록한다.
9. exact authority claims와 owner PID를 가진 process-local capability를 발급한다.
10. parent worker가 lock을 유지하고 11 dataset workers를 관리한다. child scorer는 공식 artifact를 쓰지 않으며 검증된 결과의 publication은 parent만 수행한다.
11. exact catalog completion 후에만 Task6/7/Candidate C를 실행하고 Exp449 shard/outer manifest를 freeze한다.
12. scorer-free verifier가 완전한 Smoke A root를 검증한 뒤 terminal complete record를 기록한다.

## Process-local capability

공식 root를 변경하는 API는 exact internal capability를 요구한다. capability는 다음 성질을 가진다.

- 외부에서 생성할 수 없는 exact class/token 조합;
- grant SHA, A, I, context SHA, target, roots, owner PID 및 claim-lock inode 결합;
- copy/pickle/serialization 금지;
- 다른 PID, root, target 또는 authority에서 사용 금지;
- root claim, parent publication, freeze, terminal publication 경계에서 재검증.

`run_exp449_wave_antigravity`, freeze wrapper 및 terminal writer는 capability 없는 공식 destination mutation을 거부한다. 순수 scoring/composition 함수의 과학 인터페이스는 변경하지 않는다.

## 실패·재개

### Wave 처리

- Wave 1은 개별 dataset 실패를 append-only attempt ledger에 기록하고 다른 dataset을 계속한다.
- Wave 2는 실패하거나 아직 시도하지 못한 dataset만 실행한다.
- Wave 2 후 exact completion이 아니면 `proposal_score_incomplete` terminal을 create-only로 기록하고 Task6/7/freeze를 실행하지 않는다.

### 프로세스 중단

`resume-smoke-a`는 새 root를 만들지 않는다. 다음이 모두 같을 때만 기존 root를 재개한다.

- approved A, I, grant/digest와 consumed authority;
- context/catalog/database/closure/environment/backend/worker;
- claim-lock path/inode와 inactive flock;
- complete success namespace, attempt ledger 및 terminal 상태;
- waves가 `{1, 2}` 범위를 넘지 않음.

재개 시 verified successes는 다시 계산하지 않는다. 코드, DB, dependency 또는 context가 바뀌면 기록을 이전하거나 수정하지 않고 재개를 거부한다.

freeze가 완료됐지만 complete terminal 기록 전에 중단된 경우 scorer-free verification을 다시 수행하고 terminal만 기록한다. 재-scoring은 하지 않는다.

## 운영 로그

운영 root의 최소 파일은 다음과 같다.

- `execution_authority_consumed_antigravity.json`: authoritative create-only grant consumption;
- `progress_antigravity.jsonl`: append-only dataset/wave progress;
- `status_antigravity.json`: atomic-replace derived status cache;
- `worker_stdout_antigravity.log`, `worker_stderr_antigravity.log`;
- create-only complete 또는 incomplete terminal record.

Progress event는 timestamp, event, wave, dataset, completed/failed/pending, elapsed, throughput, ETA, latest error 및 worker PID를 기록한다. TEST label이나 과학 payload 전체를 로그에 기록하지 않는다.

`progress_antigravity.jsonl` append/fsync가 실패하면 다음 dataset을 시작하지 않고 안전하게 중단한다. `status_antigravity.json`은 파생 캐시이며 검증 권한으로 사용하지 않는다.

`status`는 DB나 scoring 모듈을 열지 않고 claim, operations log, checkpoint 및 terminal의 검증 가능한 상태만 읽는다.

## 과학 결과 불변성

보안/운영 wrapper는 다음 값에 참여하지 않는다.

- raw TRAIN/TEST values;
- feature 또는 score 계산;
- p-value, threshold, alpha, native pass;
- Task4/5 route, Task6 evidence, Task7 stability;
- Candidate C selection 및 continuous score;
- shard와 outer manifest의 과학 payload.

Wrapper가 추가하는 grant, claim, PID, timestamp, progress 및 terminal metadata는 operations root에만 저장한다. scientific root의 기존 Exp449 schema에는 새 필드를 넣지 않는다.

작은 typed fixture에서 기존 직접 composition/freeze와 capability-bound fresh run, interruption/resume run의 모든 `ProposalScore` 과학 필드, shard bytes 및 outer manifest bytes가 같아야 한다. Root-bound checkpoint/context metadata만 기존 Task4–Task6에서 preregister한 방식으로 명시적으로 정규화할 수 있다.

## TDD와 공격 테스트

구현은 production code 전에 RED를 관찰한다. 최소 검증 목록은 다음과 같다.

- grant/digest/schema/size/SHA/nonce/target/root/context 변조;
- A의 잘못된 부모, extra/modified delta, dirty tree 및 승인되지 않은 HEAD;
- 다른 interpreter, dependency inventory, database/catalog, closure, locks, backend 또는 workers;
- symlink, traversal, existing root 및 동시 claim/flock 경쟁;
- grant로 두 번째 root 생성 또는 target 재사용;
- direct Python mutating API call과 forged/copied/cross-PID capability;
- claim/consumption/progress append 각 경계의 interruption;
- Wave 1 실패 계속 처리, Wave 2 exact queue, incomplete terminal;
- verified success no-recompute와 code/context drift resume rejection;
- freeze-after-crash terminal-only recovery;
- progress append failure safe stop;
- direct composition, fresh capability run 및 interruption/resume의 full scientific equality;
- label import firewall 및 no-label static call graph;
- 모든 실제 Exp449 root가 implementation test 동안 absent임.

## 단계별 승인

1. 이 design을 커밋하고 사용자 검토를 받는다.
2. 별도 implementation plan을 작성하고 사용자 실행 방식을 선택한다.
3. 구현 커밋 I를 TDD로 작성하고 독립 Sol review SHIP을 받는다.
4. clean I에서 read-only readiness와 grant preparation을 수행한다.
5. 두 grant data file만 추가한 authorization commit A를 만들고 독립 Sol review SHIP을 받는다.
6. 사용자에게 A, I, grant SHA, roots, 12→11 workers를 제시한다.
7. 사용자가 정확히 `Smoke A 실행 승인`이라고 승인한 뒤에만 fresh launch한다.

## 비범위

- Ed25519, SSH/GPG 서명, HMAC secret, remote approval service 또는 범용 권한 프레임워크;
- Smoke B, official 및 label/evaluation 권한;
- 새로운 가속 구현이나 과학 알고리즘 최적화;
- 과학 payload schema 변경;
- 실패 root 삭제, 덮어쓰기 또는 다른 root로 record migration.
