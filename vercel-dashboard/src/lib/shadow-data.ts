import type { SupabaseClient } from "@supabase/supabase-js";

import type { ShadowReportRun } from "./shadow-metrics";

type RunRow = {
  run_id: string;
  candidate_count: number;
  hard_count: number;
  shadow_count: number;
  priority_count: number;
  standard_count: number;
};

export async function loadShadowReportRows(supabase: SupabaseClient): Promise<ShadowReportRun[]> {
  const { data: runs, error: runError } = await supabase.from("shadow_validation_runs")
    .select("run_id,candidate_count,hard_count,shadow_count,priority_count,standard_count")
    .order("observed_at", { ascending: false })
    .limit(5000);
  if (runError) throw new Error(runError.message);
  const runRows = (runs ?? []) as RunRow[];
  if (!runRows.length) return [];
  const runIds = runRows.map((run) => run.run_id);
  const { data: candidates, error: candidateError } = await supabase.from("shadow_validation_candidates")
    .select("candidate_id,run_id,final_lane")
    .in("run_id", runIds)
    .limit(20000);
  if (candidateError) throw new Error(candidateError.message);
  const candidateIds = (candidates ?? []).map((candidate) => candidate.candidate_id);
  const candidateFeedback = candidateIds.length
    ? await supabase.from("shadow_effective_candidate_feedback").select("candidate_id,verdict").in("candidate_id", candidateIds).limit(20000)
    : { data: [], error: null };
  if (candidateFeedback.error) throw new Error(candidateFeedback.error.message);
  const runFeedback = await supabase.from("shadow_effective_run_feedback")
    .select("run_id,missed_anomaly_state,missed_anomaly_count")
    .in("run_id", runIds)
    .limit(5000);
  if (runFeedback.error) throw new Error(runFeedback.error.message);
  const verdictByCandidate = new Map((candidateFeedback.data ?? []).map((row) => [row.candidate_id, row.verdict]));
  const feedbackByRun = new Map((runFeedback.data ?? []).map((row) => [row.run_id, row]));
  return runRows.map((run) => ({
    ...run,
    candidates: (candidates ?? []).filter((candidate) => candidate.run_id === run.run_id).map((candidate) => ({
      candidate_id: candidate.candidate_id,
      final_lane: candidate.final_lane,
      verdict: verdictByCandidate.get(candidate.candidate_id) ?? null,
    })),
    run_feedback: feedbackByRun.get(run.run_id) ?? null,
  })) as ShadowReportRun[];
}
