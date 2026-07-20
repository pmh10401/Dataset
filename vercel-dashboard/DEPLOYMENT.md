# 배포 절차

## 1. Supabase

1. 프로젝트를 활성화한다.
2. `supabase/migrations/202607200001_dashboard_schema.sql`을 적용한다.
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

로컬 대시보드 자격 증명, Supabase 로그인 정보, Supabase URL/publishable key와 Vercel
수집 주소를 환경 변수로 설정한 뒤 먼저 한 번만 전송한다.

```bash
python3 scratch/push_vercel_dashboard_snapshot.py --once
```

성공 후 `--once`를 빼고 10초 반복 모드로 실행한다. exporter heartbeat는
`/Users/minho/Documents/Dataset/vercel_dashboard_exporter_heartbeat.json`에 기록된다.

## 4. 검증

- 비로그인 `/` 접근이 `/login`으로 이동하는지 확인한다.
- 등록된 사용자만 로그인되는지 확인한다.
- 잘못된 ingest token이 401인지 확인한다.
- one-shot 전송이 200인지 확인한다.
- 30초 이상 전송이 끊기면 `업데이트 지연`이 표시되는지 확인한다.
- 전송 snapshot에 TEST label, 원본 로그, 로컬 경로, PID가 없는지 확인한다.

새 대시보드 검증이 끝날 때까지 기존 Cloudflare 터널은 유지한다.
