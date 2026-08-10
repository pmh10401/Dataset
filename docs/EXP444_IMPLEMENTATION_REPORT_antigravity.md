# Exp444 구현 검증 기록

## 현재 상태

`Exp444`는 동결된 Exp442 30개 좌표를 normal-TRAIN omission view로만 재계산하는
review-only 구현을 마쳤다. 현재 상태는
`implementation_verified_awaiting_fresh_review`이며, 이 기록은 구현 검증을
요약할 뿐 공식 readiness, prediction, publication, evaluation 또는 과학적 승격을
승인하지 않는다.

다음 명령은 정확히 다음과 같다.

```text
fresh_sol_high_read_only_review_current_clean_head
```

## 범위와 거버넌스

- 설계 SHA256: `1bb76f19d07ef312d5ac5498414d0c914ce94b99c5912e0bbcf3f0197c1400e8`
- 환경 lock SHA256: `8835702b9793a5cc08b5e95621b49c0a4a4136325354da7a655761f6b91f3a94`
- backend: CPU, workers: 11 (`max(1, os.cpu_count() - 1)`)
- `scientific_success_criterion = none`; `evaluation_status = descriptive_complete_no_promotion`
- review-only이며 champion, pending candidate, operational alert로 승격하지 않는다.

공식 run root
`/Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810`
은 존재하지 않는다. `official_readiness=false`, `prediction_started=false`,
`prediction_sha256=null`, `evaluation_sha256=null`, `test_labels_opened=false`다.
이 Task는 official readiness/new/resume/inspect/evaluate를 실행하지 않았고,
prediction, publication, evaluation 및 TEST label 접근을 수행하지 않았다.

## 구현 이력과 수정 근거

아래 SHA는 Task 1–10의 구현·수정·맵 동기화 근거다. 각 범위의 완전한 순서는
`git log --reverse --format='%H%x09%s' b6354c52794fc787c916d434e7264e57eba8f1a3..6f2972c81c8bb606722acafd29a56ea6ee390a3d`
로 재발견할 수 있다.

| Task | 정확한 커밋/범위 | 주요 수정 또는 닫힌 경계 |
| --- | --- | --- |
| 1 | `8b9201ff2ceedbc45c9b86bac28a368dc28ed10a`, `99a9d0373ec283fc6ba565ecb8160fdb644449cf` | TRAIN-only 계약, 문서와 map evidence anchor |
| 2 | `7ceab729331b6bb992cf49bb889260750eee2880`..`74eae2d8034c655856860cc04a68491a26906152` | 환경/계약, numeric type·pre-mutation·map cap 보완 |
| 3 | `127f0dbfbac168357f1eba8db927c7904bcfbd4e`..`637966f58b37c4f422563cadbe322e755121e427` | frozen 30-coordinate source, DB descriptor, provenance와 pin evidence 보완 |
| 4 | `f6cf8151e5ac16e65349f8c1e3e2c90c4ffda54a`..`8d2bb81644f56c4366f8f8dfc57a45748c19add9`; corrective `d6ae18e2ee19c70750ba0bf67c1a6d30dc822485`, `a680b99f5837ac989ff34a35c5d56f4e0d860eab` | deterministic masks/stability lane conservation, valid empty view와 ordered-score guard |
| 5 | `76f930304955b66f38f116ec2987c90d03d63904`..`53f63a1f992b91080b72e0cc34de824d17d72d2a` | fail-closed CPU/MLX/Numba gate와 measured MLX provenance |
| 6 | `d9ef121d9a76c65f1c535d20bb9c36c74e563e21`..`ca0501cad641feb40665679e05cafd98c648b034` | checkpoint/progress atomic recovery, PID ownership와 stale map metadata 제거 |
| 7 | `1bf1de35572b75cfab1aff13836ec301379906b9`..`347a462d4b56dff17cdab0cd3f93be8a8c669339` | ordered 11-worker runtime, recovery preflight와 orphan reconciliation |
| 8 | `c171687de013599d050eccb3be441372fff9f000`..`c7763467fcd26498c030a1c60be7d009c51728c7` | create-only label-free publication, pinned authority와 rejected-publication quarantine |
| 9 | `a8fc88678a6520c349649e8a8871047fcd1b2641`..`7e0d3885b5bc2f4c3a4bfc94186b0f27d467d78a` | lazy post-freeze evaluator, descriptor/authorizer, replay/audit hardening |
| 10 | `303c5b1fb756178a7b7c81b89609ac4057fda486`..`6f2972c81c8bb606722acafd29a56ea6ee390a3d` | fixed five-command lifecycle, reviewed-HEAD identity, document SHA binding와 final payload-parity map |

## 회귀 검증

부모 검증 근거는 모두 failure/error 0이다.

| Suite | 결과 | 경과 시간 |
| --- | ---: | ---: |
| Exp444 | 188/188 | 61.245s |
| Exp440 inherited | 35/35 | 0.274s |
| Exp441 inherited | 197/197 | 81.690s |
| Exp442 inherited | 7/7 | 0.163s |

Task 5 acceleration gate는 Numba exact-stage end-to-end ratio가 `0.9968x`였고
MLX float64를 사용할 수 없어서 CPU를 선택했다. 이는 동결된 점수, mask, threshold,
routing을 바꾸지 않는 CPU/11-worker 경로를 보존한 결정이다.

## Worker parity 권한

```text
worker_parity_evidence=satisfied_by_revalidated_task7_authority
fresh_task11_worker_parity_rerun=not_completed_not_required
```

재검증된 실제 Task 7 authority는 30 candidates, 11 workers, TEST labels 0이며,
sequential/parallel canonical SHA256은
`729b31c0698fda6b60c17012a3fd2cae3f4b0899e01e5a3bf4d569902e814857`다.
authority file SHA256은
`79b6a5847a01f594d140c73aa587589413b4933ac7d53dcf07d2bb478a5f771e`다.

Task 11에서 새 parity smoke를 시도했으나, 관련 없는 승인된 Exp236 7-worker run과
동시 실행하지 않기 위해 sequential rank 7 중단 전에 안전하게 interrupted했다.
parallel phase 전에 중단됐고 artifact나 새로운 evidence를 만들지 않았으므로 완료로
주장하지 않는다. Fresh Sol High는 parity-relevant diff를 read-only로 검토하여
reuse-acceptable이라고 판단했다. smoke script, source loader, stability scorer,
masks, backend, sequential reference, 11-worker dispatch, canonical serializer가
Task 7 이후 byte-unchanged였기 때문에 위 Task 7 authority를 재사용한다.

## 다음 게이트

fresh Sol High read-only review가 현재 clean HEAD에 대해 `ship`이고 unresolved
Critical/Important finding이 없어야 official readiness가 가능하다. 그 전에는
공식 실행·예측·평가·승격을 하지 않는다.

