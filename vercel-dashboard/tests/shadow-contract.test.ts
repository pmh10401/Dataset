import { describe, expect, it } from "vitest";

import {
  assertValidShadowPrediction,
  predictionHash,
  type ShadowPrediction,
} from "../src/lib/shadow";

export function validShadowPayload(): Record<string, unknown> {
  const unsigned: Record<string, unknown> = {
    schema_version: 1,
    collection_mode: "prospective_shadow",
    policy_version: "exp265-shadow-v1",
    run_id: "run-001",
    source_id: "equipment-a",
    observed_at: "2026-07-21T00:00:00+00:00",
    dataset_key: "recipe-a/sensor-1/step-2",
    equipment_id: "eq-a",
    recipe_id: "recipe-a",
    sensor_id: "sensor-1",
    step_id: "step-2",
    n_train: 120,
    n_test: 4,
    has_missing: false,
    missing_quality_status: "not_applicable",
    config_sha256: "a".repeat(64),
    series_preview: { indices: [0, 1, 2, 3], values: [0, 1, 0.5, -0.2] },
    baseline_hard_indices: [0],
    baseline_standard_indices: [1, 2],
    baseline_priority_indices: [3],
    hard_indices: [0],
    shadow_indices: [1],
    priority_indices: [3],
    standard_indices: [2],
    confidence: { "0": "A" },
    candidates: [
      candidate("run-001", 0, "Hard", "A", 3),
      candidate("run-001", 1, "Shadow", null, 3),
      candidate("run-001", 2, "Standard", null, 0),
      candidate("run-001", 3, "Priority", null, 0),
    ],
    candidate_count: 4,
    hard_count: 1,
    shadow_count: 1,
    priority_count: 1,
    standard_count: 1,
    contains_test_labels: false,
    contains_evaluation_metrics: false,
    automatic_policy_changed: false,
  };
  return { ...unsigned, prediction_sha256: predictionHash(unsigned) };
}

function candidate(runId: string, index: number, lane: string, confidence: string | null, support: number) {
  const sources = ["statistical", "structural", "change_point", "reconstruction"];
  return {
    candidate_id: `${runId}:${index}`,
    candidate_index: index,
    final_lane: lane,
    confidence_grade: confidence,
    support_count: support,
    support_sources: sources.slice(0, support),
    quality_flagged: false,
    source_evidence: Object.fromEntries(
      sources.map((name, position) => [name, { agrees: position < support, score: position < support ? 2 : null, threshold: 1 }]),
    ),
  };
}

function rehash(payload: Record<string, unknown>) {
  const unsigned = { ...payload };
  delete unsigned.prediction_sha256;
  payload.prediction_sha256 = predictionHash(unsigned);
  return payload;
}

describe("Exp265 shadow prediction contract", () => {
  it("uses the cross-runtime canonical hash", () => {
    expect(predictionHash({ score: 2 })).toBe("b467c14af78ba52cd8339ebcaf8c6e28b636b90d1c69c216cf96821b371b8ca1");
  });

  it.each(["y_test", "tp", "fp", "f1", "actual_anomaly_positions"])(
    "rejects forbidden field %s",
    (field) => {
      const payload = validShadowPayload();
      payload[field] = [];
      expect(() => assertValidShadowPrediction(payload)).toThrow("unknown field");
    },
  );

  it("accepts the canonical payload", () => {
    const payload: ShadowPrediction = assertValidShadowPrediction(validShadowPayload());
    expect(payload.candidate_count).toBe(4);
  });

  it("requires baseline Hard to remain unchanged", () => {
    const payload = validShadowPayload();
    payload.hard_indices = [0, 2];
    expect(() => assertValidShadowPrediction(rehash(payload))).toThrow("baseline Hard changed");
  });

  it("requires pairwise-disjoint final lanes", () => {
    const payload = validShadowPayload();
    payload.standard_indices = [1, 2];
    expect(() => assertValidShadowPrediction(rehash(payload))).toThrow("lane overlap");
  });

  it("requires Shadow policy eligibility", () => {
    const payload = validShadowPayload();
    payload.n_train = 99;
    expect(() => assertValidShadowPrediction(rehash(payload))).toThrow("Shadow eligibility");
  });

  it("recomputes every Standard review routing decision", () => {
    const payload = validShadowPayload();
    const candidates = payload.candidates as Array<Record<string, unknown>>;
    candidates.find((item) => item.candidate_index === 1)!.final_lane = "Priority";
    payload.shadow_indices = [];
    payload.priority_indices = [1, 3];
    payload.shadow_count = 0;
    payload.priority_count = 2;
    expect(() => assertValidShadowPrediction(rehash(payload))).toThrow("routing policy mismatch");
  });

  it("derives Hard confidence from source agreement", () => {
    const payload = validShadowPayload();
    payload.confidence = { "0": "C" };
    const candidates = payload.candidates as Array<Record<string, unknown>>;
    candidates.find((item) => item.candidate_index === 0)!.confidence_grade = "C";
    expect(() => assertValidShadowPrediction(rehash(payload))).toThrow("Hard confidence policy mismatch");
  });
});
