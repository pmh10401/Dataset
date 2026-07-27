import type { SupabaseClient } from "@supabase/supabase-js";

import type { ShadowRankingAlpha, ShadowRankingCandidate } from "./shadow-ranking";

const PAGE_SIZE = 1000;

type QueryPage<T> = { data: T[] | null; error: { message: string } | null };

export async function collectRankingPages<T>(
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

export type RankMovement = { delta: number; direction: "up" | "same" | "down" };

export function rankMovement(exp292Rank: number, shadowRank: number): RankMovement {
  const delta = exp292Rank - shadowRank;
  return { delta, direction: delta > 0 ? "up" : delta < 0 ? "down" : "same" };
}

type RunRow = {
  run_id: string;
  dataset_key: string;
  n_test: number;
  automatic_hard_count: number;
  priority_review_count: number;
  standard_review_count: number;
  prediction_sha256: string;
  ingested_at: string;
};

type CandidateRow = Omit<ShadowRankingCandidate, "tier_preserving_rank" | "control_rank" | "grammar_support_count" | "p_values"> & {
  run_id: string;
  tier_preserving_rank: Record<ShadowRankingAlpha, number>;
  control_rank: ShadowRankingCandidate["control_rank"];
  grammar_support_count: Record<ShadowRankingAlpha, number>;
  p_values: ShadowRankingCandidate["p_values"];
};

export type ShadowRankingRunView = RunRow & { candidates: CandidateRow[] };

export async function loadShadowRankingRows(supabase: SupabaseClient): Promise<ShadowRankingRunView[]> {
  const runs = await collectRankingPages<RunRow>((from, to) => supabase
    .from("shadow_ranking_runs")
    .select("run_id,dataset_key,n_test,automatic_hard_count,priority_review_count,standard_review_count,prediction_sha256,ingested_at")
    .order("dataset_key")
    .range(from, to));
  if (!runs.length) return [];
  const runIds = new Set(runs.map((row) => row.run_id));
  const candidates = (await collectRankingPages<CandidateRow>((from, to) => supabase
    .from("shadow_ranking_candidates")
    .select("candidate_id,run_id,candidate_index,exp292_tier,exp292_rank,tier_preserving_rank,control_rank,grammar_support_count,p_values")
    .order("exp292_rank")
    .range(from, to))).filter((row) => runIds.has(row.run_id));
  const byRun = new Map<string, CandidateRow[]>();
  for (const candidate of candidates) {
    const group = byRun.get(candidate.run_id) ?? [];
    group.push(candidate);
    byRun.set(candidate.run_id, group);
  }
  return runs.map((run) => ({ ...run, candidates: byRun.get(run.run_id) ?? [] }));
}
