create table if not exists public.shadow_validation_runs (
  run_id text primary key,
  source_id text not null,
  observed_at timestamptz not null,
  dataset_key text not null,
  equipment_id text,
  recipe_id text,
  sensor_id text,
  step_id text,
  n_train integer not null check (n_train >= 0),
  n_test integer not null check (n_test > 0),
  has_missing boolean not null,
  missing_quality_status text not null,
  collection_mode text not null check (collection_mode = 'prospective_shadow'),
  policy_version text not null check (policy_version = 'exp265-shadow-v1'),
  config_sha256 text not null check (config_sha256 ~ '^[0-9a-f]{64}$'),
  prediction_sha256 text not null unique check (prediction_sha256 ~ '^[0-9a-f]{64}$'),
  series_preview jsonb not null,
  candidate_count integer not null check (candidate_count >= 0),
  hard_count integer not null check (hard_count >= 0),
  shadow_count integer not null check (shadow_count >= 0),
  priority_count integer not null check (priority_count >= 0),
  standard_count integer not null check (standard_count >= 0),
  ingested_at timestamptz not null default now(),
  check (candidate_count = hard_count + shadow_count + priority_count + standard_count)
);

create table if not exists public.shadow_validation_candidates (
  candidate_id text primary key,
  run_id text not null references public.shadow_validation_runs(run_id) on delete cascade,
  candidate_index integer not null check (candidate_index >= 0),
  final_lane text not null check (final_lane in ('Hard', 'Shadow', 'Priority', 'Standard')),
  confidence_grade text check (confidence_grade in ('A', 'B', 'C')),
  support_count integer not null check (support_count between 0 and 4),
  support_sources jsonb not null,
  quality_flagged boolean not null,
  source_evidence jsonb not null,
  created_at timestamptz not null default now(),
  unique (run_id, candidate_index),
  check (
    (final_lane = 'Hard' and confidence_grade is not null) or
    (final_lane <> 'Hard' and confidence_grade is null)
  )
);

create table if not exists public.shadow_feedback_events (
  id bigint generated always as identity primary key,
  candidate_id text not null references public.shadow_validation_candidates(candidate_id) on delete cascade,
  verdict text not null check (verdict in ('true_anomaly', 'false_alarm', 'deferred')),
  reviewer_id uuid not null references auth.users(id) on delete restrict,
  note text check (char_length(note) <= 2000),
  corrects_event_id bigint,
  created_at timestamptz not null default now(),
  unique (id, candidate_id),
  foreign key (corrects_event_id, candidate_id)
    references public.shadow_feedback_events(id, candidate_id) on delete restrict
);

create table if not exists public.shadow_run_feedback_events (
  id bigint generated always as identity primary key,
  run_id text not null references public.shadow_validation_runs(run_id) on delete cascade,
  missed_anomaly_state text not null check (missed_anomaly_state in ('none', 'exists', 'unknown')),
  missed_anomaly_count integer,
  reviewer_id uuid not null references auth.users(id) on delete restrict,
  note text check (char_length(note) <= 2000),
  corrects_event_id bigint,
  created_at timestamptz not null default now(),
  unique (id, run_id),
  foreign key (corrects_event_id, run_id)
    references public.shadow_run_feedback_events(id, run_id) on delete restrict,
  check (
    (missed_anomaly_state = 'none' and missed_anomaly_count = 0) or
    (missed_anomaly_state = 'exists' and missed_anomaly_count > 0) or
    (missed_anomaly_state = 'unknown' and missed_anomaly_count is null)
  )
);

create index if not exists shadow_validation_runs_observed_idx
  on public.shadow_validation_runs (observed_at desc, run_id);
create index if not exists shadow_validation_candidates_run_lane_idx
  on public.shadow_validation_candidates (run_id, final_lane, candidate_index);
create index if not exists shadow_feedback_candidate_created_idx
  on public.shadow_feedback_events (candidate_id, created_at desc, id desc);
create index if not exists shadow_run_feedback_run_created_idx
  on public.shadow_run_feedback_events (run_id, created_at desc, id desc);

alter table public.shadow_validation_runs enable row level security;
alter table public.shadow_validation_candidates enable row level security;
alter table public.shadow_feedback_events enable row level security;
alter table public.shadow_run_feedback_events enable row level security;

create policy "members can read shadow runs"
  on public.shadow_validation_runs for select to authenticated
  using (exists (
    select 1 from public.dashboard_members m where m.user_id = (select auth.uid())
  ));
create policy "members can read shadow candidates"
  on public.shadow_validation_candidates for select to authenticated
  using (exists (
    select 1 from public.dashboard_members m where m.user_id = (select auth.uid())
  ));
create policy "members can read candidate feedback"
  on public.shadow_feedback_events for select to authenticated
  using (exists (
    select 1 from public.dashboard_members m where m.user_id = (select auth.uid())
  ));
create policy "members can read run feedback"
  on public.shadow_run_feedback_events for select to authenticated
  using (exists (
    select 1 from public.dashboard_members m where m.user_id = (select auth.uid())
  ));

create policy "writers can insert candidate feedback"
  on public.shadow_feedback_events for insert to authenticated
  with check (
    reviewer_id = (select auth.uid()) and exists (
      select 1 from public.dashboard_members m
      where m.user_id = (select auth.uid()) and m.can_write
    )
  );
create policy "writers can insert run feedback"
  on public.shadow_run_feedback_events for insert to authenticated
  with check (
    reviewer_id = (select auth.uid()) and exists (
      select 1 from public.dashboard_members m
      where m.user_id = (select auth.uid()) and m.can_write
    )
  );

grant select on public.shadow_validation_runs to authenticated;
grant select on public.shadow_validation_candidates to authenticated;
grant select, insert on public.shadow_feedback_events to authenticated;
grant select, insert on public.shadow_run_feedback_events to authenticated;
grant usage, select on sequence public.shadow_feedback_events_id_seq to authenticated;
grant usage, select on sequence public.shadow_run_feedback_events_id_seq to authenticated;
revoke all on public.shadow_validation_runs from anon;
revoke all on public.shadow_validation_candidates from anon;
revoke all on public.shadow_feedback_events from anon;
revoke all on public.shadow_run_feedback_events from anon;

create or replace view public.shadow_effective_candidate_feedback
with (security_invoker = true) as
select id, candidate_id, verdict, reviewer_id, note, corrects_event_id, created_at
from (
  select event.*, row_number() over (
    partition by candidate_id order by created_at desc, id desc
  ) as position
  from public.shadow_feedback_events event
) ranked
where position = 1;

create or replace view public.shadow_effective_run_feedback
with (security_invoker = true) as
select id, run_id, missed_anomaly_state, missed_anomaly_count, reviewer_id, note,
       corrects_event_id, created_at
from (
  select event.*, row_number() over (
    partition by run_id order by created_at desc, id desc
  ) as position
  from public.shadow_run_feedback_events event
) ranked
where position = 1;

grant select on public.shadow_effective_candidate_feedback to authenticated;
grant select on public.shadow_effective_run_feedback to authenticated;

create or replace function public.ingest_shadow_prediction(payload jsonb)
returns jsonb
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  caller uuid := auth.uid();
  run_key text := payload->>'run_id';
  incoming_hash text := payload->>'prediction_sha256';
  existing_hash text;
  candidate jsonb;
  preview_indices jsonb := payload->'series_preview'->'indices';
  preview_values jsonb := payload->'series_preview'->'values';
  supplied_count integer := jsonb_array_length(coalesce(payload->'candidates', '[]'::jsonb));
begin
  if caller is null or not exists (
    select 1 from public.dashboard_members m where m.user_id = caller and m.can_write
  ) then
    raise exception 'shadow ingest requires dashboard writer';
  end if;
  if payload->>'collection_mode' <> 'prospective_shadow'
     or payload->>'policy_version' <> 'exp265-shadow-v1'
     or coalesce((payload->>'automatic_policy_changed')::boolean, true)
     or coalesce((payload->>'contains_test_labels')::boolean, true)
     or coalesce((payload->>'contains_evaluation_metrics')::boolean, true) then
    raise exception 'invalid prospective shadow policy contract';
  end if;
  if run_key is null or incoming_hash !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid run id or prediction hash';
  end if;
  select prediction_sha256 into existing_hash
    from public.shadow_validation_runs where run_id = run_key;
  if existing_hash is not null then
    if existing_hash <> incoming_hash then
      raise exception 'prediction hash conflict for run %', run_key using errcode = '23505';
    end if;
    return jsonb_build_object('ok', true, 'idempotent', true, 'run_id', run_key,
                              'prediction_sha256', incoming_hash);
  end if;
  if jsonb_typeof(preview_indices) <> 'array'
     or jsonb_typeof(preview_values) <> 'array'
     or jsonb_array_length(preview_indices) <> jsonb_array_length(preview_values)
     or jsonb_array_length(preview_indices) < 1
     or jsonb_array_length(preview_indices) > 1024 then
    raise exception 'invalid series preview';
  end if;
  if supplied_count <> (payload->>'candidate_count')::integer then
    raise exception 'candidate count mismatch';
  end if;
  if supplied_count <> coalesce((payload->>'hard_count')::integer, 0)
                         + coalesce((payload->>'shadow_count')::integer, 0)
                         + coalesce((payload->>'priority_count')::integer, 0)
                         + coalesce((payload->>'standard_count')::integer, 0) then
    raise exception 'lane count mismatch';
  end if;

  insert into public.shadow_validation_runs (
    run_id, source_id, observed_at, dataset_key, equipment_id, recipe_id, sensor_id,
    step_id, n_train, n_test, has_missing, missing_quality_status, collection_mode,
    policy_version, config_sha256, prediction_sha256, series_preview, candidate_count,
    hard_count, shadow_count, priority_count, standard_count
  ) values (
    run_key, payload->>'source_id', (payload->>'observed_at')::timestamptz,
    payload->>'dataset_key', payload->>'equipment_id', payload->>'recipe_id',
    payload->>'sensor_id', payload->>'step_id', (payload->>'n_train')::integer,
    (payload->>'n_test')::integer, (payload->>'has_missing')::boolean,
    payload->>'missing_quality_status', payload->>'collection_mode',
    payload->>'policy_version', payload->>'config_sha256', incoming_hash,
    payload->'series_preview', supplied_count, (payload->>'hard_count')::integer,
    (payload->>'shadow_count')::integer, (payload->>'priority_count')::integer,
    (payload->>'standard_count')::integer
  );

  for candidate in select value from jsonb_array_elements(payload->'candidates') loop
    if (candidate->>'candidate_index')::integer < 0
       or (candidate->>'candidate_index')::integer >= (payload->>'n_test')::integer
       or not (preview_indices @> jsonb_build_array((candidate->>'candidate_index')::integer)) then
      raise exception 'candidate index missing from run or preview';
    end if;
    insert into public.shadow_validation_candidates (
      candidate_id, run_id, candidate_index, final_lane, confidence_grade,
      support_count, support_sources, quality_flagged, source_evidence
    ) values (
      candidate->>'candidate_id', run_key, (candidate->>'candidate_index')::integer,
      candidate->>'final_lane', nullif(candidate->>'confidence_grade', ''),
      (candidate->>'support_count')::integer, candidate->'support_sources',
      (candidate->>'quality_flagged')::boolean, candidate->'source_evidence'
    );
  end loop;
  return jsonb_build_object('ok', true, 'idempotent', false, 'run_id', run_key,
                            'prediction_sha256', incoming_hash);
end;
$$;

revoke all on function public.ingest_shadow_prediction(jsonb) from public, anon;
grant execute on function public.ingest_shadow_prediction(jsonb) to authenticated;

alter publication supabase_realtime add table public.shadow_validation_runs;
alter publication supabase_realtime add table public.shadow_validation_candidates;
alter publication supabase_realtime add table public.shadow_feedback_events;
alter publication supabase_realtime add table public.shadow_run_feedback_events;
