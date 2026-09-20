import { describe, expect, it } from "vitest";

import {
  assertValidShadowRankingPrediction,
  rankingPredictionHash,
} from "../src/lib/shadow-ranking";

export function validShadowRankingPayload(): Record<string, unknown> {
  const runId = "replay-exp397-demo";
  const unsigned: Record<string, unknown> = {
    schema_version: 1,
    collection_mode: "retrospective_replay",
    policy_version: "exp397-review-ranking-replay-v1",
    run_id: runId,
    dataset: "Demo",
    n_test: 8,
    automatic_hard_count: 2,
    priority_review_count: 1,
    standard_review_count: 1,
    candidates: [{
      candidate_id: `${runId}:2`,
      candidate_index: 2,
      exp292_tier: 1,
      exp292_rank: 4,
      tier_preserving_rank: { "0.005": 3, "0.01": 2, "0.02": 1, "0.05": 4 },
      control_rank: {
        additive_grammar_family: { "0.005": 4, "0.01": 3, "0.02": 2, "0.05": 1 },
        grammar_only_control: { "0.005": 2, "0.01": 3, "0.02": 4, "0.05": 1 },
      },
      grammar_support_count: { "0.005": 1, "0.01": 1, "0.02": 2, "0.05": 3 },
      p_values: { transition: 0.002, dwell: 0.4, recovery: 0.3, composite: 0.01 },
    }],
    contains_test_labels: false,
    contains_evaluation_metrics: false,
    automatic_policy_changed: false,
    retrospective: true,
    prospective: false,
  };
  return { ...unsigned, prediction_sha256: rankingPredictionHash(unsigned) };
}

function rehash(payload: Record<string, unknown>) {
  const unsigned = { ...payload };
  delete unsigned.prediction_sha256;
  payload.prediction_sha256 = rankingPredictionHash(unsigned);
  return payload;
}

describe("Exp397 Shadow ranking contract", () => {
  it("accepts the canonical label-free payload", () => {
    expect(assertValidShadowRankingPrediction(validShadowRankingPayload()).standard_review_count).toBe(1);
  });

  it.each(["y_test", "tp", "fp", "f1", "actual_anomaly_positions"])(
    "rejects forbidden field %s",
    (field) => {
      const payload = validShadowRankingPayload();
      payload[field] = [];
      expect(() => assertValidShadowRankingPrediction(payload)).toThrow(/forbidden|unknown/);
    },
  );

  it("requires exactly four preregistered alpha keys", () => {
    const payload = validShadowRankingPayload();
    const candidate = (payload.candidates as Array<Record<string, unknown>>)[0];
    delete (candidate.tier_preserving_rank as Record<string, number>)["0.05"];
    expect(() => assertValidShadowRankingPrediction(rehash(payload))).toThrow("alpha");
  });

  it("detects prediction hash tampering", () => {
    const payload = validShadowRankingPayload();
    payload.automatic_hard_count = 3;
    expect(() => assertValidShadowRankingPrediction(payload)).toThrow("SHA256");
  });
});
