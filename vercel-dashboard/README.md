# Vercel + Supabase Experiment Dashboard

로컬 Mac에서 실행되는 실험의 정제된 운영 상태만 외부에서 확인하는 미러 대시보드다.
원본 SQLite DB, 시계열, TEST label, 전체 로그와 로컬 경로는 전송하지 않는다.

## 환경 변수

`.env.example`의 세 변수를 Vercel과 로컬 개발 환경에 설정한다. 브라우저에는 publishable
key만 노출하며 `DASHBOARD_INGEST_TOKEN`은 Vercel 수집 route에서만 사용한다. 상태
전송기는 쓰기 권한이 있는 Supabase 사용자 session을 함께 제시해야 한다.

## Supabase 준비

1. Supabase 프로젝트에서 다음 migration을 순서대로 실행한다.
   - `supabase/migrations/202607200001_dashboard_schema.sql`
   - `supabase/migrations/202607270001_operational_alpha_settings.sql`
2. Auth에서 이메일/암호 사용자를 만든다.
3. 해당 사용자의 UUID를 `dashboard_members`에 추가한다. Alpha 변경 요청 권한이 필요한 운영자와 exporter만 `can_write=true`로 설정한다.
4. Vercel 환경 변수를 등록한 뒤 `vercel-dashboard`를 배포한다.

```sql
insert into public.dashboard_members (user_id, can_write) values ('AUTH_USER_UUID', true);
```

## Alpha 신뢰도 설정

인증된 사용자는 대시보드 상단의 설정 아이콘에서 `/settings/alpha`로 이동할 수 있다.
운영자는 사전 등록된 `0.005`(엄격) 또는 `0.010`(균형)만 선택할 수 있으며, 변경 사유와
명시적 확인을 입력해야 한다. Alpha는 기존 자동 경보를 추가하거나 제거하지 않고, 이미
생성된 자동 경보의 강한 신뢰 표시 범위만 조정한다.

설정 화면은 `요청값(desired_alpha)`과 `현재 적용값(effective_alpha)`을 분리해 보여준다.
이번 1단계에서는 변경 요청과 감사 이력을 Supabase에 저장하고 상태를 `pending`으로
유지한다. 탐지기가 요청을 읽어 실제 적용한 뒤 `effective_alpha`와 상태를 갱신하는
동기화/확인 절차는 후속 단계다. 따라서 요청 직후 화면에 적용 완료로 표시하지 않는다.

## 로컬 검증

```bash
npm install
npm test
npm run build
```

상태 전송기는 저장소 루트의 `scratch/push_vercel_dashboard_snapshot.py`를 사용한다.
