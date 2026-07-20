# Vercel + Supabase Experiment Dashboard

로컬 Mac에서 실행되는 실험의 정제된 운영 상태만 외부에서 확인하는 미러 대시보드다.
원본 SQLite DB, 시계열, TEST label, 전체 로그와 로컬 경로는 전송하지 않는다.

## 환경 변수

`.env.example`의 세 변수를 Vercel과 로컬 개발 환경에 설정한다. 브라우저에는 publishable
key만 노출하며 `DASHBOARD_INGEST_TOKEN`은 Vercel 수집 route에서만 사용한다. 상태
전송기는 쓰기 권한이 있는 Supabase 사용자 session을 함께 제시해야 한다.

## Supabase 준비

1. Supabase 프로젝트에서 `supabase/migrations/202607200001_dashboard_schema.sql`을 실행한다.
2. Auth에서 이메일/암호 사용자를 만든다.
3. 해당 사용자의 UUID를 `dashboard_members`에 추가하고 exporter 사용자만 `can_write=true`로 설정한다.
4. Vercel 환경 변수를 등록한 뒤 `vercel-dashboard`를 배포한다.

```sql
insert into public.dashboard_members (user_id, can_write) values ('AUTH_USER_UUID', true);
```

## 로컬 검증

```bash
npm install
npm test
npm run build
```

상태 전송기는 저장소 루트의 `scratch/push_vercel_dashboard_snapshot.py`를 사용한다.
