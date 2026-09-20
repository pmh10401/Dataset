create table public.shadow_ranking_runs (
  run_id text primary key,
  dataset_key text not null unique,
  n_test integer not null check (n_test > 0),
  collection_mode text not null check (collection_mode = 'retrospective_replay'),
  policy_version text not null check (policy_version = 'exp397-review-ranking-replay-v1'),
  prediction_sha256 text not null unique check (prediction_sha256 ~ '^[0-9a-f]{64}$'),
  automatic_hard_count integer not null check (automatic_hard_count >= 0),
  priority_review_count integer not null check (priority_review_count >= 0),
  standard_review_count integer not null check (standard_review_count >= 0),
  contains_test_labels boolean not null check (contains_test_labels = false),
  contains_evaluation_metrics boolean not null check (contains_evaluation_metrics = false),
  automatic_policy_changed boolean not null check (automatic_policy_changed = false),
  retrospective boolean not null check (retrospective = true),
  prospective boolean not null check (prospective = false),
  ingested_at timestamptz not null default now()
);

create table public.shadow_ranking_candidates (
  candidate_id text primary key,
  run_id text not null references public.shadow_ranking_runs(run_id) on delete cascade,
  candidate_index integer not null check (candidate_index >= 0),
  exp292_tier integer not null check (exp292_tier between 1 and 3),
  exp292_rank integer not null check (exp292_rank > 0),
  tier_preserving_rank jsonb not null,
  control_rank jsonb not null,
  grammar_support_count jsonb not null,
  p_values jsonb not null,
  created_at timestamptz not null default now(),
  unique (run_id, candidate_index)
);

create index shadow_ranking_runs_dataset_idx
  on public.shadow_ranking_runs (dataset_key);
create index shadow_ranking_candidates_run_rank_idx
  on public.shadow_ranking_candidates (run_id, exp292_rank, candidate_index);

alter table public.shadow_ranking_runs enable row level security;
alter table public.shadow_ranking_candidates enable row level security;

create policy "members can read shadow ranking runs"
  on public.shadow_ranking_runs for select to authenticated
  using (exists (
    select 1
    from public.dashboard_members member
    where member.user_id = (select auth.uid())
  ));

create policy "members can read shadow ranking candidates"
  on public.shadow_ranking_candidates for select to authenticated
  using (exists (
    select 1
    from public.dashboard_members member
    where member.user_id = (select auth.uid())
  ));

grant select on public.shadow_ranking_runs to authenticated;
grant select on public.shadow_ranking_candidates to authenticated;
revoke all on public.shadow_ranking_runs from anon;
revoke all on public.shadow_ranking_candidates from anon;

create or replace function public.ingest_shadow_ranking_prediction(payload jsonb)
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
  candidate_count integer := jsonb_array_length(coalesce(payload->'candidates', '[]'::jsonb));
  alpha_keys text[] := array['0.005', '0.01', '0.02', '0.05'];
begin
  if caller is null or not exists (
    select 1
    from public.dashboard_members member
    where member.user_id = caller and member.can_write
  ) then
    raise exception 'shadow ranking ingest requires dashboard writer';
  end if;

  if payload->>'collection_mode' <> 'retrospective_replay'
     or payload->>'policy_version' <> 'exp397-review-ranking-replay-v1'
     or coalesce((payload->>'contains_test_labels')::boolean, true)
     or coalesce((payload->>'contains_evaluation_metrics')::boolean, true)
     or coalesce((payload->>'automatic_policy_changed')::boolean, true)
     or not coalesce((payload->>'retrospective')::boolean, false)
     or coalesce((payload->>'prospective')::boolean, true) then
    raise exception 'invalid Exp397 retrospective Shadow contract';
  end if;

  if run_key is null or incoming_hash !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid run id or prediction hash';
  end if;
  if jsonb_typeof(payload->'candidates') <> 'array'
     or candidate_count <> (payload->>'standard_review_count')::integer then
    raise exception 'candidate count mismatch';
  end if;

  select prediction_sha256
    into existing_hash
    from public.shadow_ranking_runs
    where run_id = run_key;
  if existing_hash is not null then
    if existing_hash <> incoming_hash then
      raise exception 'prediction hash conflict for run %', run_key using errcode = '23505';
    end if;
    return jsonb_build_object(
      'ok', true,
      'idempotent', true,
      'run_id', run_key,
      'prediction_sha256', incoming_hash
    );
  end if;

  insert into public.shadow_ranking_runs (
    run_id,
    dataset_key,
    n_test,
    collection_mode,
    policy_version,
    prediction_sha256,
    automatic_hard_count,
    priority_review_count,
    standard_review_count,
    contains_test_labels,
    contains_evaluation_metrics,
    automatic_policy_changed,
    retrospective,
    prospective
  ) values (
    run_key,
    payload->>'dataset',
    (payload->>'n_test')::integer,
    payload->>'collection_mode',
    payload->>'policy_version',
    incoming_hash,
    (payload->>'automatic_hard_count')::integer,
    (payload->>'priority_review_count')::integer,
    (payload->>'standard_review_count')::integer,
    (payload->>'contains_test_labels')::boolean,
    (payload->>'contains_evaluation_metrics')::boolean,
    (payload->>'automatic_policy_changed')::boolean,
    (payload->>'retrospective')::boolean,
    (payload->>'prospective')::boolean
  );

  for candidate in
    select value from jsonb_array_elements(payload->'candidates')
  loop
    if (candidate->>'candidate_index')::integer < 0
       or (candidate->>'candidate_index')::integer >= (payload->>'n_test')::integer
       or candidate->>'candidate_id' <> run_key || ':' || candidate->>'candidate_index'
       or jsonb_typeof(candidate->'tier_preserving_rank') <> 'object'
       or not (candidate->'tier_preserving_rank' ?& alpha_keys)
       or jsonb_object_length(candidate->'tier_preserving_rank') <> 4
       or jsonb_typeof(candidate->'grammar_support_count') <> 'object'
       or not (candidate->'grammar_support_count' ?& alpha_keys)
       or jsonb_object_length(candidate->'grammar_support_count') <> 4
       or jsonb_typeof(candidate->'control_rank') <> 'object'
       or not (candidate->'control_rank' ?& array['additive_grammar_family', 'grammar_only_control'])
       or jsonb_typeof(candidate->'p_values') <> 'object'
       or not (candidate->'p_values' ?& array['transition', 'dwell', 'recovery', 'composite']) then
      raise exception 'invalid Shadow ranking candidate';
    end if;

    insert into public.shadow_ranking_candidates (
      candidate_id,
      run_id,
      candidate_index,
      exp292_tier,
      exp292_rank,
      tier_preserving_rank,
      control_rank,
      grammar_support_count,
      p_values
    ) values (
      candidate->>'candidate_id',
      run_key,
      (candidate->>'candidate_index')::integer,
      (candidate->>'exp292_tier')::integer,
      (candidate->>'exp292_rank')::integer,
      candidate->'tier_preserving_rank',
      candidate->'control_rank',
      candidate->'grammar_support_count',
      candidate->'p_values'
    );
  end loop;

  return jsonb_build_object(
    'ok', true,
    'idempotent', false,
    'run_id', run_key,
    'prediction_sha256', incoming_hash
  );
end;
$$;

revoke all on function public.ingest_shadow_ranking_prediction(jsonb)
  from public, anon;
grant execute on function public.ingest_shadow_ranking_prediction(jsonb)
  to authenticated;

alter publication supabase_realtime add table public.shadow_ranking_runs;
