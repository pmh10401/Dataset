# Exp265 Shadow 검증 운영 Runbook

## 1. 현재 시작 조건

코드 완료는 수집 활성화가 아니다. 실제 수집은 신규 label-free upstream feed가 다음을 모두 제공할 때만 시작한다.

- 안정적인 `run_id`와 `collection_mode="prospective_shadow"`
- Exp253의 Hard/Standard/Priority baseline lane
- 통계·구조·변화점·마스킹 복원 점수원의 후보 index, score, TRAIN threshold
- 품질 flag와 정책/config SHA256
- TEST 라벨, 실제 이상 위치, retrospective 표식이 전혀 없는 입력

마스킹 복원 점수원은 `n_train < 100` 때문에만 명시적 미적격 처리가 가능하다. 이 feed가 없으면 상태는 **BLOCKED: prospective upstream feed unavailable**이다. 과거 1,117개 TEST 데이터셋을 신규 run처럼 재생하지 않는다.

## 2. Supabase 마이그레이션

Supabase CLI의 현재 공식 흐름을 사용한다.

```bash
cd "/Users/minho/Documents/timesries project/vercel-dashboard"
supabase link --project-ref "$SUPABASE_PROJECT_REF"
supabase migration list --linked
supabase db push --linked --dry-run
supabase db push --linked
supabase migration list --linked
```

적용 파일은 `supabase/migrations/202607210001_shadow_validation.sql`이다. `--dry-run` 결과에 이 파일 외 예상하지 못한 migration이 있으면 중단한다.

## 3. RLS 확인

1. 비로그인 상태에서 `/api/shadow/report`와 `/api/shadow/runs`가 `401`인지 확인한다.
2. 읽기 전용 dashboard member는 조회할 수 있지만 ingest와 feedback insert가 거부되는지 확인한다.
3. `can_write=true` member만 예측 ingest와 feedback append가 가능한지 확인한다.
4. authenticated role로 prediction 또는 feedback `UPDATE/DELETE`가 거부되는지 확인한다.
5. service-role key는 서버 관리 작업 외에는 사용하지 않으며 브라우저·스크립트 인자·로그에 넣지 않는다.

## 4. Vercel 빌드와 배포

```bash
cd "/Users/minho/Documents/timesries project/vercel-dashboard"
npm ci
npm test
npx tsc --noEmit
npm run build
vercel deploy --prod
vercel list --prod
```

Vercel 환경에는 `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY`, `DASHBOARD_INGEST_TOKEN`을 설정한다. `SUPABASE_SERVICE_ROLE_KEY`는 설정하지 않는다.

## 5. 격리된 3-run smoke

실제 활성화 전에 테스트 Supabase 프로젝트 또는 트랜잭션으로 격리된 schema에서만 수행한다.

1. 모든 ID를 `smoke-`로 시작하게 만든다.
2. 후보 0개 run, Hard A 후보 1개 run, Shadow+Priority 후보 run을 각각 전송한다.
3. 동일 payload 재전송이 idempotent인지 확인한다.
4. 동일 `run_id`의 다른 hash가 `409`인지 확인한다.
5. 실제 이상→오탐 정정 이벤트, run 누락 판정을 append한다.
6. 진행률 `3/100`, 후보-only 상태의 Recall/F1 숨김, Exp253 Hard 불변을 확인한다.

활성화 전 관리자 전용 SQL로 smoke 행만 삭제한다. 아래 명령은 service-role 또는 DB 관리자 세션에서만 실행한다.

```sql
begin;
delete from public.shadow_validation_runs where run_id like 'smoke-%';
commit;
```

FK cascade로 smoke 후보와 feedback도 함께 삭제된다. 실제 ID에는 이 명령을 사용하지 않는다.

## 6. 실제 exporter 시작

먼저 실제 upstream이 생성한 신규 JSONL을 enqueue-only로 검증한다.

```bash
cd "/Users/minho/Documents/timesries project"
python3 scratch/run_exp265_shadow_exporter.py \
  --prediction-jsonl "/absolute/path/to/new_prospective_runs.jsonl" \
  --enqueue-only
```

전송 토큰은 환경변수로만 제공한다.

```bash
export SHADOW_INGEST_URL="https://YOUR_DOMAIN/api/shadow/ingest"
export DASHBOARD_INGEST_TOKEN="..."
export DASHBOARD_SUPABASE_ACCESS_TOKEN="..."
python3 scratch/run_exp265_shadow_exporter.py --once
```

토큰을 명령행 인자로 직접 기록하거나 로그에 출력하지 않는다.

## 7. spool 점검

```bash
sqlite3 "/Users/minho/Documents/Dataset/exp265_shadow_validation/exp265_spool.sqlite" \
  "select state,count(*) from shadow_predictions group by state order by state;"
sqlite3 "/Users/minho/Documents/Dataset/exp265_shadow_validation/exp265_spool.sqlite" \
  "select run_id,attempt_count,last_error,updated_at from shadow_predictions where state='dead_letter' order by updated_at desc;"
```

`dead_letter`는 원인을 고친 뒤 새 run ID로 숨기지 않는다. 동일 run의 hash 충돌은 수집을 중단하고 upstream을 조사한다.

## 8. 보고서 내보내기

```bash
export SUPABASE_ACCESS_TOKEN="..."
python3 scratch/export_exp265_shadow_report.py \
  --dashboard-url "https://YOUR_DOMAIN" \
  --output-dir "/Users/minho/Documents/Dataset/exp265_shadow_validation/report" \
  --expected-policy-version "exp265-shadow-v1"
```

언제든 실행할 수 있으며 100 run에서도 동일 명령을 사용한다. 생성물은 `shadow_report.json`, `SHADOW_REPORT.md`, `MANIFEST.json`이다. 100 run 도달은 자동 정책 변경을 유발하지 않는다.

## 9. 수집 중지

exporter 프로세스만 정상 종료한다. Supabase prediction, feedback, local spool을 삭제하지 않는다. 중지 시각·이유·마지막 delivered run ID를 운영 기록에 남긴다.

## 10. 활성화 체크리스트

- [ ] migration 및 RLS 검증 완료
- [ ] Vercel production 배포와 `/shadow` 인증 확인
- [ ] 격리 smoke 완료 및 smoke 행 삭제
- [ ] 실제 신규 upstream feed 계약 확인
- [ ] 활성화 UTC 시각 기록
- [ ] 현재 unique prospective run 수가 0임을 확인

하나라도 충족하지 않으면 prospective collection은 시작하지 않는다.
