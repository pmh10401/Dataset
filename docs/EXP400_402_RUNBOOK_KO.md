# Exp400-402 Exp397 Shadow 검증 Runbook

## 1. 검증 범위

이번 실행은 기존 1,117개 데이터셋을 이용한 과거 재생 검증이다.

- 자동 경보 2,218건은 변경하지 않는다.
- Priority review 31건은 변경하지 않는다.
- Standard review 806건의 검토 순위만 비교한다.
- TEST 라벨은 Exp400 prediction 완료와 SHA256 동결 후 Exp401에서만 읽는다.
- 실제 신규 데이터가 없으므로 prospective 진행률은 0이다.
- 특정 alpha나 검토량 K를 운영값으로 선택하지 않는다.

## 2. 로컬 테스트

```bash
cd "/Users/minho/Documents/timesries project"
python3 -m py_compile experiments/exp400_402/*.py \
  scratch/run_exp400_402_shadow_validation.py \
  scratch/upload_exp400_shadow_replay.py
python3 -m unittest discover \
  -s experiments/exp400_402 \
  -p 'test_*.py' -v
python3 scratch/test_exp400_402_queue.py
```

Dashboard:

```bash
cd vercel-dashboard
npm ci
npm test
npx tsc --noEmit
npm run build
```

## 3. 10개 데이터셋 smoke

```bash
python3 scratch/run_exp400_402_shadow_validation.py \
  --stage all \
  --dataset-limit 10 \
  --output-dir \
  "/Users/minho/Documents/Dataset/exp400_402_exp397_shadow_validation_smoke"
```

확인:

- dataset rows 10
- prediction의 TEST label/evaluation field 0
- independent mismatch 0
- prospective run 0

## 4. 전체 과거 재생

```bash
python3 scratch/run_exp400_402_shadow_validation.py \
  --stage all \
  --output-dir \
  "/Users/minho/Documents/Dataset/exp400_402_exp397_shadow_validation"
```

확인:

- datasets 1,117
- Standard candidates 806
- automatic Hard 2,218
- Priority review 31
- independent mismatch 0

## 5. Queue 실행

```bash
python3 run_rank_experiments_sequential.py \
  queue experiment_400_402_exp397_shadow_validation
python3 run_rank_experiments_sequential.py run-queue
```

한 번에 하나의 batch만 실행한다. `status.json`은 Exp400 predict, Exp401
evaluate, Exp402 audit 순서로 갱신된다.

## 6. Supabase migration

```bash
cd vercel-dashboard
supabase link --project-ref "$SUPABASE_PROJECT_REF"
supabase migration list --linked
supabase db push --linked --dry-run
```

dry-run에서
`202607270004_exp397_shadow_ranking.sql` 외 예상하지 못한 migration이 보이면
중단한다. 확인 후에만 적용한다.

```bash
supabase db push --linked
supabase migration list --linked
```

## 7. Replay enqueue와 업로드

먼저 로컬 spool에만 넣는다.

```bash
python3 scratch/upload_exp400_shadow_replay.py --enqueue-only
```

전송 값은 환경변수로만 설정한다.

```bash
export SHADOW_RANKING_INGEST_URL="https://YOUR_DOMAIN/api/shadow/rankings/ingest"
export DASHBOARD_INGEST_TOKEN="..."
export DASHBOARD_SUPABASE_ACCESS_TOKEN="..."
python3 scratch/upload_exp400_shadow_replay.py --once
```

토큰을 명령 인자나 로그에 기록하지 않는다.

## 8. 원장 검증

- Exp397 ranking runs: 1,117
- Exp397 ranking candidates: 806
- 같은 run과 같은 hash 재전송: idempotent
- 같은 run과 다른 hash: HTTP 409
- 비로그인 `/api/shadow/rankings`: HTTP 401
- `/shadow`에서 `과거 데이터 재생 · 실제 운영 검증 아님` 표시
- 기존 Exp265 화면과 자료는 그대로 유지

## 9. 중단 조건

다음 조건에서는 upload 또는 배포를 중단하고 blocker를 기록한다.

- Exp400 prediction SHA256 불일치
- 1,117 dataset 또는 806 candidate coverage 불일치
- 자동 경보 또는 Priority review 수 변경
- migration dry-run에 예상하지 못한 변경 포함
- Vercel 또는 Supabase 인증 환경 누락
- prediction에 TEST label 또는 평가 지표 포함

## 10. 해석 제한

말할 수 있는 것:

- label-free Exp397 순위 artifact를 과거 데이터로 재생했다.
- 자동 경보와 기존 검토 후보를 바꾸지 않았다.
- 향후 신규 run을 기록할 기술적 경계를 준비했다.

말할 수 없는 것:

- Exp397이 실제 설비에서 prospective 검증됐다.
- 특정 alpha가 운영 최적값이다.
- Exp397이 자동 경보 성능을 개선했다.
