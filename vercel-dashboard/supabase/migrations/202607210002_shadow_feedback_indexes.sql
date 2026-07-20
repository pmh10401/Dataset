create index if not exists shadow_feedback_reviewer_idx
  on public.shadow_feedback_events (reviewer_id);
create index if not exists shadow_feedback_correction_idx
  on public.shadow_feedback_events (corrects_event_id, candidate_id);
create index if not exists shadow_run_feedback_reviewer_idx
  on public.shadow_run_feedback_events (reviewer_id);
create index if not exists shadow_run_feedback_correction_idx
  on public.shadow_run_feedback_events (corrects_event_id, run_id);
