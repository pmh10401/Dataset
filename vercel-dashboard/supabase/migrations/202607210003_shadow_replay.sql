alter table public.shadow_validation_runs
  drop constraint if exists shadow_validation_runs_collection_mode_check;
alter table public.shadow_validation_runs
  drop constraint if exists shadow_validation_runs_policy_version_check;
alter table public.shadow_validation_runs
  add constraint shadow_validation_runs_mode_policy_check check (
    (collection_mode = 'prospective_shadow' and policy_version = 'exp265-shadow-v1') or
    (collection_mode = 'retrospective_replay' and policy_version = 'exp265-replay-v1')
  );

alter table public.shadow_validation_candidates
  add column if not exists series_preview jsonb;

create index if not exists shadow_validation_runs_mode_observed_idx
  on public.shadow_validation_runs (collection_mode, observed_at desc);
