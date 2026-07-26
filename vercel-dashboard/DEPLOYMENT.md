# 배포 절차

## 1. Supabase

1. 프로젝트를 활성화한다.
2. 다음 마이그레이션을 순서대로 적용한다.
   - `supabase/migrations/202607200001_dashboard_schema.sql`
   - `supabase/migrations/202607210001_shadow_validation.sql`
   - `supabase/migrations/202607210002_shadow_feedback_indexes.sql`
   - `supabase/migrations/202607270001_operational_alpha_settings.sql`
   - `supabase/migrations/202607270002_operational_alpha_acknowledgement.sql`
3. Auth에 이메일/암호 사용자를 만든다.
4. 사용자 UUID를 `public.dashboard_members`에 추가하고 exporter 사용자에만 쓰기 권한을 준다.
5. Security Advisor에서 RLS와 공개 권한 경고가 없는지 확인한다.

## 2. Vercel

`vercel-dashboard` 폴더를 새 Vercel 프로젝트의 root directory로 연결하고 다음 환경
변수를 Production, Preview에 등록한다.

- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY`
- `DASHBOARD_INGEST_TOKEN`

`DASHBOARD_INGEST_TOKEN`은 충분히 긴 임의 문자열을 사용하며 Supabase 사용자 암호와
같게 만들지 않는다.

## 3. Mac exporter

Supabase 로그인 정보, Supabase URL/publishable key와 Vercel 수집 주소를 환경 변수로
설정한 뒤 먼저 한 번만 전송한다. exporter는 실험 상태 파일과 실행 프로세스를 직접
읽으므로 로컬 대시보드 서버를 실행하지 않는다.

```bash
python3 scratch/push_vercel_dashboard_snapshot.py --once
```

성공 후 `--once`를 빼고 10초 반복 모드로 실행한다. exporter heartbeat는
`/Users/minho/Documents/Dataset/vercel_dashboard_exporter_heartbeat.json`에 기록된다.
같은 반복 과정에서 Alpha 변경 요청을 확인하며, 적용된 설정은
`/Users/minho/Documents/Dataset/operational_alpha_setting.json`에 기록된다.
다른 Mac 식별자를 사용할 때만 `DASHBOARD_ALPHA_SOURCE_ID`를 설정한다.

`--source-url`은 진단 호환용 옵션이다. 운영에서는 지정하지 않으며
`http://127.0.0.1:8765`를 상태 원본으로 사용하지 않는다.

## 4. 검증

- 비로그인 `/` 접근이 `/login`으로 이동하는지 확인한다.
- 등록된 사용자만 로그인되는지 확인한다.
- 잘못된 ingest token이 401인지 확인한다.
- one-shot 전송이 200인지 확인한다.
- 30초 이상 전송이 끊기면 `업데이트 지연`이 표시되는지 확인한다.
- 전송 snapshot에 TEST label, 원본 로그, 로컬 경로, PID가 없는지 확인한다.
- 비로그인 `POST /api/shadow/ingest`가 로그인 화면으로 이동하지 않고 API 자체 인증의
  `401`을 반환하는지 확인한다.
- `shadow_validation_*` 네 테이블에 RLS가 활성화되고 익명 조회가 차단되는지 확인한다.
- Alpha 요청 직후 `pending`, Mac exporter 확인 후 `applied`로 바뀌는지 확인한다.
- 잘못된 버전 또는 Alpha 확인 응답이 거부되는지 확인한다.
- Alpha 설정을 바꿔도 자동 경보 후보 index와 수가 변하지 않는지 확인한다.

운영 상태 확인 화면은 Vercel 배포본만 사용한다.

## 5. 신규 데이터가 없는 개발 단계

개발 중 신규 run이 없으면 `/shadow`의 `0/100`은 오류가 아니다. 화면은
`개발 준비 완료 · 신규 데이터 대기`로 표시하고 prospective 성능 검증은 `시작 전`으로
유지한다.

- 과거 1,117개 TEST 결과를 신규 prospective run으로 재전송하지 않는다.
- 개발 smoke 자료는 `smoke-` 접두사를 사용하고 검증 직후 삭제한다.
- 실제 신규 run이 들어오기 전에는 precision, recall, F1을 운영 성능으로 해석하지 않는다.
- Shadow 후보는 자동 경보가 아니며 Exp253 자동 경보 정책을 변경하지 않는다.
