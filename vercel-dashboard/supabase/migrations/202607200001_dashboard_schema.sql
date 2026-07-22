create table if not exists public.dashboard_members (
  user_id uuid primary key references auth.users(id) on delete cascade,
  can_write boolean not null default false,
  created_at timestamptz not null default now()
);

create table if not exists public.experiment_status_latest (
  source_id text primary key,
  generated_at timestamptz not null,
  received_at timestamptz not null default now(),
  snapshot jsonb not null
);

create table if not exists public.experiment_status_history (
  id bigint generated always as identity primary key,
  source_id text not null,
  generated_at timestamptz not null,
  received_at timestamptz not null default now(),
  active_experiment text,
  running boolean not null,
  datasets_done integer not null check (datasets_done >= 0),
  error_count integer not null check (error_count >= 0),
  snapshot jsonb not null
);

create index if not exists experiment_status_history_source_received_idx
  on public.experiment_status_history (source_id, received_at desc);

alter table public.dashboard_members enable row level security;
alter table public.experiment_status_latest enable row level security;
alter table public.experiment_status_history enable row level security;

create policy "members can read own membership"
  on public.dashboard_members for select
  to authenticated
  using ((select auth.uid()) = user_id);

create policy "members can read latest status"
  on public.experiment_status_latest for select
  to authenticated
  using (exists (
    select 1 from public.dashboard_members m where m.user_id = (select auth.uid())
  ));

create policy "members can read status history"
  on public.experiment_status_history for select
  to authenticated
  using (exists (
    select 1 from public.dashboard_members m where m.user_id = (select auth.uid())
  ));

create policy "writers can insert latest status"
  on public.experiment_status_latest for insert
  to authenticated
  with check (exists (
    select 1 from public.dashboard_members m
    where m.user_id = (select auth.uid()) and m.can_write
  ));

create policy "writers can update latest status"
  on public.experiment_status_latest for update
  to authenticated
  using (exists (
    select 1 from public.dashboard_members m
    where m.user_id = (select auth.uid()) and m.can_write
  ))
  with check (exists (
    select 1 from public.dashboard_members m
    where m.user_id = (select auth.uid()) and m.can_write
  ));

create policy "writers can insert status history"
  on public.experiment_status_history for insert
  to authenticated
  with check (exists (
    select 1 from public.dashboard_members m
    where m.user_id = (select auth.uid()) and m.can_write
  ));

grant select on public.dashboard_members to authenticated;
grant select on public.experiment_status_latest to authenticated;
grant select on public.experiment_status_history to authenticated;
grant insert, update on public.experiment_status_latest to authenticated;
grant insert on public.experiment_status_history to authenticated;
grant usage, select on sequence public.experiment_status_history_id_seq to authenticated;
revoke all on public.dashboard_members from anon;
revoke all on public.experiment_status_latest from anon;
revoke all on public.experiment_status_history from anon;

alter publication supabase_realtime add table public.experiment_status_latest;
