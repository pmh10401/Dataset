import type { ShadowRankingAlpha } from "./shadow-ranking";
import { rankMovement, type RankMovement } from "./shadow-ranking-data";

export const DEFAULT_SHADOW_RANKING_ALPHA: ShadowRankingAlpha = "0.005";

type CandidateInput = {
  candidate_id: string;
  run_id: string;
  candidate_index: number;
  exp292_tier: number;
  exp292_rank: number;
  tier_preserving_rank: Record<ShadowRankingAlpha, number>;
  control_rank: Record<string, Record<ShadowRankingAlpha, number>>;
  grammar_support_count: Record<ShadowRankingAlpha, number>;
  p_values: Record<"transition" | "dwell" | "recovery" | "composite", number>;
};

type RunInput = {
  run_id: string;
  dataset_key: string;
  candidates: CandidateInput[];
};

export type ShadowRankingRow = CandidateInput & {
  dataset: string;
  shadowRank: number;
  movement: RankMovement;
  supportCount: number;
  minimumPValue: number;
};

export function buildRankingRows(
  runs: RunInput[],
  alpha: ShadowRankingAlpha,
): ShadowRankingRow[] {
  return runs
    .flatMap((run) => run.candidates.map((candidate) => {
      const shadowRank = candidate.tier_preserving_rank[alpha];
      return {
        ...candidate,
        dataset: run.dataset_key,
        shadowRank,
        movement: rankMovement(candidate.exp292_rank, shadowRank),
        supportCount: candidate.grammar_support_count[alpha],
        minimumPValue: Math.min(
          candidate.p_values.transition,
          candidate.p_values.dwell,
          candidate.p_values.recovery,
        ),
      };
    }))
    .sort((left, right) => left.shadowRank - right.shadowRank
      || left.dataset.localeCompare(right.dataset)
      || left.candidate_index - right.candidate_index);
}
