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
   - `supabase/migrations/202607270002_operational_alpha_acknowledgement.sql`
   - `supabase/migrations/202607270003_operational_alpha_function_permissions.sql`
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
변경 요청은 먼저 `pending`으로 저장된다. Mac 상태 전송기가 정확한 요청 버전과 Alpha를
확인한 뒤 `/Users/minho/Documents/Dataset/operational_alpha_setting.json`을 원자적으로
갱신하고 성공 응답을 보내야만 `effective_alpha`와 상태가 `applied`로 바뀐다. 설정
화면은 10초마다 이 상태를 다시 확인한다.

탐지 코드는 다음 공통 함수로 로컬에서 확인된 값만 읽는다.

```python
from operational.alpha_setting import load_effective_alpha

alpha = load_effective_alpha()
```

이 설정은 자동 경보 후보를 추가하거나 제거하는 입력으로 사용하지 않는다. 기존 자동
경보 내부의 강한 신뢰 표시와 검토 우선순위에만 사용한다.

## 로컬 검증

```bash
npm install
npm test
npm run build
```

상태 전송기는 저장소 루트의 `scratch/push_vercel_dashboard_snapshot.py`를 사용한다.
