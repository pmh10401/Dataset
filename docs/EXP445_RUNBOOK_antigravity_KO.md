# Exp445 Exp444 publication source authority 복구 실행 안내서

## 고정 경로와 실행 전 조건

```text
RUN_ROOT=/Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810
ATTEMPTS=$RUN_ROOT/attempts_antigravity
ATTEMPT=$ATTEMPTS/attempt_0000_antigravity
CHECKPOINT=$ATTEMPT/checkpoint_antigravity
PROGRESS=$ATTEMPT/progress_antigravity.json
EVENTS=$ATTEMPT/events_antigravity.jsonl
TEXT_LOG=$ATTEMPT/text_antigravity.log
PID_FILE=$ATTEMPT/pid_antigravity.json
PUBLICATION=$ATTEMPT/publication_antigravity
EVALUATION=$RUN_ROOT/evaluation_antigravity
```

Exp445는 Exp444 terminal predecessor를 읽기 전용으로 검증하는 단일-use 새 root다.
Exp444/Exp442 artifact는 어떤 경우에도 수정하지 않는다. root allocation 전 실패는
mutation-free이고 root를 남기지 않는다. allocation 뒤 실패는 create-only evidence로
보존하며, 같은 root의 resume/retry/overwrite는 없다. 단, 같은 run의 checkpoint state,
progress, PID는 기존 동결 방식대로 atomic replacement를 허용한다. 이 예외는
create-only record, failure, publication, evaluation artifact에는 적용되지 않는다.

```text
same-run mutable artifacts = checkpoint state, progress, PID (atomic replacement only)
create-only artifacts = record, failure, publication, evaluation
```

## 정확히 네 명령

공식 명령은 implementation regression, read-only smoke, fresh Sol High
commitment-boundary review, candidate commit, 그리고 exact clean HEAD fresh Sol High
final review가 통과하기 전에는 실행하지 않는다. 이 최종 review는 아래 candidate에 대해
`ship`으로 통과했지만, 아직 official readiness는 실행하지 않았다.

## 구현 검증 현황

구현은 `implementation_verified_awaiting_official_readiness` 상태다. 기능 commit은
`c8a0ad351d701b11df69fac66bd7c95284e644be`, map commit은
`477399062818b49dcc01101e3a83e723c8ec7d44`이다. 컨트롤러가 캡처한 전체 회귀는
Exp444 205개(`195.955`초), Exp445 39개(`1.397`초), 총 244개이며 모두 성공했고
sentinel과 outer exit code도 모두 0이다.

비공식 읽기 전용 확인으로 CLI help는 exit 0, 금지된 `resume`은 exit 2로 거절됐고,
30개 임시 입력의 sequential/CPU 11-worker canonical SHA256은
`267449fa3a4d00e3d214ca3fa5b5f38c1f73ecd07ec3d33545003e0b92a76b30`로 일치했다.
DB inode `1048138`의 link count는 1이며 fixed Exp445 root는 없다. 별도
real-composition 시도는 출력 handle 유실로 독립 성공 근거로 쓰지 않으며, 전체 회귀에
포함된 동일 real-composition 테스트를 근거로 사용한다. TEST label은 열지 않았다.

fresh Sol High final review는 exact candidate
`09903c608fb78508ae8fdd343594e3d99dfee660`에 대해 `ship` verdict를 냈다. 따라서
다음 단계는 official readiness이지만, 그 명령은 이 상태 전환에서 실행하지 않았다.
아래 명령은 여전히 closed profile과 reviewed commit 경계를 지켜야 한다.

```bash
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py readiness --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py new --reviewed-commit REVIEWED_COMMIT
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py inspect
/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python scratch/run_exp445_exp444_publication_source_authority_repair_antigravity.py evaluate
```

`new`는 single-use이며 resume 명령이 없다. public CLI는 attempt/path/profile/source/seed/
mask/science/worker/backend/evaluation override를 제공하지 않는다. `readiness`와 `new`만
`--reviewed-commit`을 받고, clean HEAD와 일치하는 40-hex 값을 요구한다.

## 각 단계의 닫힌 계약

- `readiness`는 Exp445 identity/root/document hash, inherited environment/scientific/backend authority, Exp444 five-condition predecessor, source database link count 1, labels 0, CPU/11 inner-1와 absent root를 검증한다. evidence는 root에 쓰지 않는 detached stdout JSON뿐이다.
- `new`는 descriptor-pinned Exp440 `dataset_summary_antigravity.jsonl` bytes, manifest와 artifact map을 한 retained descriptor authority로 검증한 뒤 capture한다. canonical 1,117 ordered summary row의 label-free schema를 pure parser로 확인하고, values-only handoff가 SQLite session을 닫은 뒤에만 root를 만든다.
- `new`는 frozen 30 coordinate를 재계산하고 모든 create-only record write 전에 same-index Exp444 canonical wrapper SHA256 parity를 요구한다. 마지막 write 전 predecessor terminal hashes와 aggregate `af24dd48f7ebe51881637c2ee584c82e7477240b4088d868b7876a61babab034`를 재검증한다.
- `inspect`는 Exp445 publication만 read-only로 확인한다. Exp444 publication을 찾거나 해석하지 않고, worker·SQLite·TEST label을 열지 않는다.
- `evaluate`는 Exp445 publication을 서로 독립적으로 두 번 inspect한 뒤에만 lazy evaluator import를 허용한다. 그 전에는 label-capable module과 TEST label을 import/open/query하지 않는다.

새 score source가 아니므로 MLX/PyTorch/MPS/Numba backend를 재benchmark·재선택하지
않는다. inherited CPU backend, 12 logical CPU의 `max(1, logical CPU - 1)=11` outer
workers, inner numeric thread 1, and predecessor parity authority만 SHA로 재검증한다.

## publication과 descriptive-only governance

publication은 complete 30/30, error 0, labels 0, index별 record parity, aggregate parity,
descriptor-pinned summary bytes, predecessor authority가 모두 성공한 뒤 create-only로
만든다. 그 밖의 실패는 publication/evaluation을 만들지 않는다.

```text
scientific_success_criterion = none
evaluation_status = descriptive_complete_no_promotion
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
same_experiment_reselection_allowed = false
test_informed_reselection_allowed = false
```

Exp445 결과는 champion, pending candidate, operational alert, source replacement 또는
same-experiment reselection의 근거가 될 수 없다.
