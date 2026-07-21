import type { SupabaseClient } from "@supabase/supabase-js";

import type { ShadowReportRun } from "./shadow-metrics";
import type { ShadowCollectionMode } from "./shadow-mode";

const PAGE_SIZE = 1000;

type QueryPage<T> = { data: T[] | null; error: { message: string } | null };

type RunRow = {
  run_id: string;
  observed_at: string;
  collection_mode: string;
  policy_version: string;
  config_sha256: string;
  candidate_count: number;
  hard_count: number;
  shadow_count: number;
  priority_count: number;
  standard_count: number;
};

type CandidateRow = { candidate_id: string; run_id: string; final_lane: string };
type CandidateFeedbackRow = { candidate_id: string; verdict: string };
type RunFeedbackRow = { run_id: string; missed_anomaly_state: string; missed_anomaly_count: number | null };

export async function collectPages<T>(
  queryPage: (from: number, to: number) => PromiseLike<QueryPage<T>>,
  pageSize = PAGE_SIZE,
): Promise<T[]> {
  const rows: T[] = [];
  for (let from = 0; ; from += pageSize) {
    const { data, error } = await queryPage(from, from + pageSize - 1);
    if (error) throw new Error(error.message);
    const page = data ?? [];
    rows.push(...page);
    if (page.length < pageSize) return rows;
  }
}

export async function loadShadowReportRows(supabase: SupabaseClient, mode: ShadowCollectionMode): Promise<ShadowReportRun[]> {
  const runRows = await collectPages<RunRow>((from, to) => supabase.from("shadow_validation_runs")
    .select("run_id,observed_at,collection_mode,policy_version,config_sha256,candidate_count,hard_count,shadow_count,priority_count,standard_count")
    .eq("collection_mode", mode)
    .order("observed_at", { ascending: false })
    .range(from, to));
  if (!runRows.length) return [];

  const runIds = new Set(runRows.map((run) => run.run_id));
  const candidates = (await collectPages<CandidateRow>((from, to) => supabase.from("shadow_validation_candidates")
    .select("candidate_id,run_id,final_lane")
    .order("candidate_id")
    .range(from, to))).filter((candidate) => runIds.has(candidate.run_id));
  const candidateIds = new Set(candidates.map((candidate) => candidate.candidate_id));
  const candidateFeedback = (await collectPages<CandidateFeedbackRow>((from, to) => supabase.from("shadow_effective_candidate_feedback")
    .select("candidate_id,verdict")
    .order("candidate_id")
    .range(from, to))).filter((row) => candidateIds.has(row.candidate_id));
  const runFeedback = (await collectPages<RunFeedbackRow>((from, to) => supabase.from("shadow_effective_run_feedback")
    .select("run_id,missed_anomaly_state,missed_anomaly_count")
    .order("run_id")
    .range(from, to))).filter((row) => runIds.has(row.run_id));

  const verdictByCandidate = new Map(candidateFeedback.map((row) => [row.candidate_id, row.verdict]));
  const feedbackByRun = new Map(runFeedback.map((row) => [row.run_id, row]));
  const candidatesByRun = new Map<string, CandidateRow[]>();
  for (const candidate of candidates) {
    const group = candidatesByRun.get(candidate.run_id) ?? [];
    group.push(candidate);
    candidatesByRun.set(candidate.run_id, group);
  }
  return runRows.map((run) => ({
    ...run,
    candidates: (candidatesByRun.get(run.run_id) ?? []).map((candidate) => ({
      candidate_id: candidate.candidate_id,
      final_lane: candidate.final_lane,
      verdict: verdictByCandidate.get(candidate.candidate_id) ?? null,
    })),
    run_feedback: feedbackByRun.get(run.run_id) ?? null,
  })) as ShadowReportRun[];
}
