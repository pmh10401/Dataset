import { createHash } from "node:crypto";

export const SHADOW_RANKING_SCHEMA_VERSION = 1;
export const SHADOW_RANKING_POLICY_VERSION = "exp397-review-ranking-replay-v1";
export const SHADOW_RANKING_COLLECTION_MODE = "retrospective_replay";
export const SHADOW_RANKING_MAX_BYTES = 2 * 1024 * 1024;
export const SHADOW_RANKING_ALPHAS = ["0.005", "0.01", "0.02", "0.05"] as const;
export type ShadowRankingAlpha = typeof SHADOW_RANKING_ALPHAS[number];
export const SHADOW_RANKING_CONTROLS = ["additive_grammar_family", "grammar_only_control"] as const;

export type ShadowRankingCandidate = {
  candidate_id: string;
  candidate_index: number;
  exp292_tier: 1 | 2 | 3;
  exp292_rank: number;
  tier_preserving_rank: Record<ShadowRankingAlpha, number>;
  control_rank: Record<string, Record<ShadowRankingAlpha, number>>;
  grammar_support_count: Record<ShadowRankingAlpha, number>;
  p_values: Record<"transition" | "dwell" | "recovery" | "composite", number>;
};

export type ShadowRankingPrediction = {
  schema_version: 1;
  collection_mode: "retrospective_replay";
  policy_version: "exp397-review-ranking-replay-v1";
  run_id: string;
  dataset: string;
  n_test: number;
  automatic_hard_count: number;
  priority_review_count: number;
  standard_review_count: number;
  candidates: ShadowRankingCandidate[];
  contains_test_labels: false;
  contains_evaluation_metrics: false;
  automatic_policy_changed: false;
  retrospective: true;
  prospective: false;
  prediction_sha256: string;
};

const TOP_LEVEL_FIELDS = [
  "schema_version", "collection_mode", "policy_version", "run_id", "dataset", "n_test",
  "automatic_hard_count", "priority_review_count", "standard_review_count", "candidates",
  "contains_test_labels", "contains_evaluation_metrics", "automatic_policy_changed",
  "retrospective", "prospective", "prediction_sha256",
];

const CANDIDATE_FIELDS = [
  "candidate_id", "candidate_index", "exp292_tier", "exp292_rank",
  "tier_preserving_rank", "control_rank", "grammar_support_count", "p_values",
];

const FORBIDDEN = new Set([
  "y_test", "true_label", "is_true_anomaly", "actual_anomaly_positions",
  "tp", "fp", "fn", "precision", "recall", "f1", "combined_f1",
]);

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(
      Object.keys(value as Record<string, unknown>)
        .sort()
        .map((key) => [key, canonical((value as Record<string, unknown>)[key])]),
    );
  }
  return value;
}

export function rankingPredictionHash(value: Record<string, unknown>) {
  return createHash("sha256").update(JSON.stringify(canonical(value))).digest("hex");
}

function object(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function exactFields(value: Record<string, unknown>, expected: readonly string[], path: string) {
  const actual = Object.keys(value);
  const unknown = actual.filter((key) => !expected.includes(key));
  const missing = expected.filter((key) => !actual.includes(key));
  if (unknown.length) throw new Error(`unknown field: ${path}.${unknown[0]}`);
  if (missing.length) throw new Error(`missing field: ${path}.${missing[0]}`);
}

function scanForbidden(value: unknown, path = "root") {
  if (Array.isArray(value)) {
    value.forEach((child, index) => scanForbidden(child, `${path}[${index}]`));
    return;
  }
  if (typeof value !== "object" || value === null) return;
  for (const [key, child] of Object.entries(value)) {
    const lower = key.toLowerCase();
    if (FORBIDDEN.has(lower) || /_(tp|fp|fn|f1|precision|recall)$/.test(lower)) {
      throw new Error(`forbidden field: ${path}.${key}`);
    }
    scanForbidden(child, `${path}.${key}`);
  }
}

function stringValue(value: unknown, path: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${path} must be a non-empty string`);
  return value;
}

function integer(value: unknown, path: string, minimum = 0): number {
  if (!Number.isInteger(value) || (value as number) < minimum) {
    throw new Error(`${path} must be an integer >= ${minimum}`);
  }
  return value as number;
}

function finite(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${path} must be finite`);
  return value;
}

function exactAlphaMap(value: unknown, path: string, minimum: number): Record<ShadowRankingAlpha, number> {
  const map = object(value, path);
  exactFields(map, SHADOW_RANKING_ALPHAS, `${path} alpha`);
  for (const alpha of SHADOW_RANKING_ALPHAS) integer(map[alpha], `${path}.${alpha}`, minimum);
  return map as Record<ShadowRankingAlpha, number>;
}

export function assertValidShadowRankingPrediction(raw: unknown): ShadowRankingPrediction {
  scanForbidden(raw);
  const payload = object(raw, "prediction");
  exactFields(payload, TOP_LEVEL_FIELDS, "prediction");
  if (payload.schema_version !== SHADOW_RANKING_SCHEMA_VERSION) throw new Error("unsupported schema_version");
  if (payload.collection_mode !== SHADOW_RANKING_COLLECTION_MODE) throw new Error("unsupported collection_mode");
  if (payload.policy_version !== SHADOW_RANKING_POLICY_VERSION) throw new Error("unsupported policy_version");
  const runId = stringValue(payload.run_id, "run_id");
  stringValue(payload.dataset, "dataset");
  const nTest = integer(payload.n_test, "n_test", 1);
  integer(payload.automatic_hard_count, "automatic_hard_count");
  integer(payload.priority_review_count, "priority_review_count");
  const standardCount = integer(payload.standard_review_count, "standard_review_count");
  if (
    payload.contains_test_labels !== false
    || payload.contains_evaluation_metrics !== false
    || payload.automatic_policy_changed !== false
    || payload.retrospective !== true
    || payload.prospective !== false
  ) throw new Error("prediction scope flags are invalid");
  if (!Array.isArray(payload.candidates) || payload.candidates.length !== standardCount) {
    throw new Error("candidate count mismatch");
  }
  const seen = new Set<number>();
  for (const [position, rawCandidate] of payload.candidates.entries()) {
    const candidate = object(rawCandidate, `candidates[${position}]`);
    exactFields(candidate, CANDIDATE_FIELDS, `candidates[${position}]`);
    const index = integer(candidate.candidate_index, "candidate_index");
    if (index >= nTest) throw new Error("candidate index outside TEST range");
    if (seen.has(index)) throw new Error("candidate index duplicated");
    seen.add(index);
    if (candidate.candidate_id !== `${runId}:${index}`) throw new Error("candidate_id mismatch");
    const tier = integer(candidate.exp292_tier, "exp292_tier", 1);
    if (![1, 2, 3].includes(tier)) throw new Error("exp292_tier must be 1, 2, or 3");
    integer(candidate.exp292_rank, "exp292_rank", 1);
    exactAlphaMap(candidate.tier_preserving_rank, "tier_preserving_rank", 1);
    const controls = object(candidate.control_rank, "control_rank");
    exactFields(controls, SHADOW_RANKING_CONTROLS, "control_rank");
    for (const variant of SHADOW_RANKING_CONTROLS) exactAlphaMap(controls[variant], `control_rank.${variant}`, 1);
    const support = exactAlphaMap(candidate.grammar_support_count, "grammar_support_count", 0);
    for (const value of Object.values(support)) if (value > 3) throw new Error("grammar support count must be between 0 and 3");
    const pValues = object(candidate.p_values, "p_values");
    exactFields(pValues, ["transition", "dwell", "recovery", "composite"], "p_values");
    for (const [name, value] of Object.entries(pValues)) {
      const number = finite(value, `p_values.${name}`);
      if (number < 0 || number > 1) throw new Error("p-value must be between 0 and 1");
    }
  }
  const statedHash = stringValue(payload.prediction_sha256, "prediction_sha256");
  if (!/^[0-9a-f]{64}$/.test(statedHash)) throw new Error("prediction SHA256 is invalid");
  const unsigned = { ...payload };
  delete unsigned.prediction_sha256;
  if (rankingPredictionHash(unsigned) !== statedHash) throw new Error("prediction SHA256 mismatch");
  return payload as unknown as ShadowRankingPrediction;
}
