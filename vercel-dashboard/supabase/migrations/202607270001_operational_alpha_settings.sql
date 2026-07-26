create table if not exists public.operational_alpha_settings (
  source_id text primary key check (char_length(source_id) between 1 and 120),
  desired_alpha numeric(5,3) not null check (desired_alpha in (0.005, 0.010)),
  effective_alpha numeric(5,3) check (effective_alpha is null or effective_alpha in (0.005, 0.010)),
  status text not null default 'pending' check (status in ('pending', 'applied', 'failed')),
  version bigint not null default 1 check (version > 0),
  requested_by uuid not null references auth.users(id),
  reason text not null check (char_length(btrim(reason)) between 5 and 500),
  requested_at timestamptz not null default now(),
  applied_at timestamptz,
  updated_at timestamptz not null default now()
);

create table if not exists public.operational_alpha_setting_events (
  id bigint generated always as identity primary key,
  source_id text not null,
  previous_desired_alpha numeric(5,3) check (
    previous_desired_alpha is null or previous_desired_alpha in (0.005, 0.010)
  ),
  requested_alpha numeric(5,3) not null check (requested_alpha in (0.005, 0.010)),
  status text not null default 'requested' check (status in ('requested', 'applied', 'failed')),
  requested_by uuid not null references auth.users(id),
  reason text not null check (char_length(btrim(reason)) between 5 and 500),
  created_at timestamptz not null default now()
);

create index if not exists operational_alpha_events_source_created_idx
  on public.operational_alpha_setting_events (source_id, created_at desc);

alter table public.operational_alpha_settings enable row level security;
alter table public.operational_alpha_setting_events enable row level security;

create policy "members can read alpha settings"
  on public.operational_alpha_settings for select
  to authenticated
  using (exists (
    select 1 from public.dashboard_members m
    where m.user_id = (select auth.uid())
  ));

create policy "members can read alpha setting events"
  on public.operational_alpha_setting_events for select
  to authenticated
  using (exists (
    select 1 from public.dashboard_members m
    where m.user_id = (select auth.uid())
  ));

create or replace function public.request_operational_alpha_change(
  p_source_id text,
  p_alpha numeric,
  p_reason text
)
returns public.operational_alpha_settings
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user_id uuid := auth.uid();
  v_previous_alpha numeric(5,3);
  v_setting public.operational_alpha_settings;
begin
  if v_user_id is null then
    raise exception 'authentication required' using errcode = '42501';
  end if;
  if not exists (
    select 1 from public.dashboard_members m
    where m.user_id = v_user_id and m.can_write
  ) then
    raise exception 'write permission required' using errcode = '42501';
  end if;
  if p_source_id is null or char_length(btrim(p_source_id)) not between 1 and 120 then
    raise exception 'invalid source id' using errcode = '22023';
  end if;
  if p_alpha not in (0.005, 0.010) then
    raise exception 'invalid alpha preset' using errcode = '22023';
  end if;
  if p_reason is null or char_length(btrim(p_reason)) not between 5 and 500 then
    raise exception 'invalid reason' using errcode = '22023';
  end if;

  select desired_alpha into v_previous_alpha
  from public.operational_alpha_settings
  where source_id = btrim(p_source_id);

  insert into public.operational_alpha_settings (
    source_id,
    desired_alpha,
    effective_alpha,
    status,
    version,
    requested_by,
    reason,
    requested_at,
    applied_at,
    updated_at
  )
  values (
    btrim(p_source_id),
    p_alpha,
    null,
    'pending',
    1,
    v_user_id,
    btrim(p_reason),
    now(),
    null,
    now()
  )
  on conflict (source_id) do update
  set desired_alpha = excluded.desired_alpha,
      status = 'pending',
      version = operational_alpha_settings.version + 1,
      requested_by = excluded.requested_by,
      reason = excluded.reason,
      requested_at = excluded.requested_at,
      applied_at = null,
      updated_at = excluded.updated_at
  returning * into v_setting;

  insert into public.operational_alpha_setting_events (
    source_id,
    previous_desired_alpha,
    requested_alpha,
    status,
    requested_by,
    reason
  )
  values (
    v_setting.source_id,
    v_previous_alpha,
    p_alpha,
    'requested',
    v_user_id,
    btrim(p_reason)
  );

  return v_setting;
end;
$$;

revoke all on public.operational_alpha_settings from anon;
revoke all on public.operational_alpha_setting_events from anon;
revoke all on public.operational_alpha_settings from authenticated;
revoke all on public.operational_alpha_setting_events from authenticated;
grant select on public.operational_alpha_settings to authenticated;
grant select on public.operational_alpha_setting_events to authenticated;

revoke all on function public.request_operational_alpha_change(text, numeric, text) from public;
grant execute on function public.request_operational_alpha_change(text, numeric, text) to authenticated;
