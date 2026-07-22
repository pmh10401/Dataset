import { createHash } from "node:crypto";

export const SHADOW_MAX_BYTES = 512 * 1024;
export const SHADOW_SCHEMA_VERSION = 1;
export const SHADOW_POLICY_VERSION = "exp265-shadow-v1";
export const SHADOW_COLLECTION_MODE = "prospective_shadow";
export const SHADOW_SOURCE_NAMES = ["statistical", "structural", "change_point", "reconstruction"] as const;

export type ShadowLane = "Hard" | "Shadow" | "Priority" | "Standard";
export type ConfidenceGrade = "A" | "B" | "C";

export interface ShadowSourceEvidence {
  agrees: boolean;
  score: number | null;
  threshold: number;
}

export interface ShadowCandidate {
  candidate_id: string;
  candidate_index: number;
  final_lane: ShadowLane;
  confidence_grade: ConfidenceGrade | null;
  support_count: number;
  support_sources: string[];
  quality_flagged: boolean;
  source_evidence: Record<string, ShadowSourceEvidence>;
}

export interface ShadowPrediction {
  schema_version: number;
  collection_mode: string;
  policy_version: string;
  run_id: string;
  source_id: string;
  observed_at: string;
  dataset_key: string;
  equipment_id: string | null;
  recipe_id: string | null;
  sensor_id: string | null;
  step_id: string | null;
  n_train: number;
  n_test: number;
  has_missing: boolean;
  missing_quality_status: string;
  config_sha256: string;
  series_preview: { indices: number[]; values: number[] };
  baseline_hard_indices: number[];
  baseline_standard_indices: number[];
  baseline_priority_indices: number[];
  hard_indices: number[];
  shadow_indices: number[];
  priority_indices: number[];
  standard_indices: number[];
  confidence: Record<string, ConfidenceGrade>;
  candidates: ShadowCandidate[];
  candidate_count: number;
  hard_count: number;
  shadow_count: number;
  priority_count: number;
  standard_count: number;
  contains_test_labels: false;
  contains_evaluation_metrics: false;
  automatic_policy_changed: false;
  prediction_sha256: string;
}

const TOP_LEVEL_FIELDS = [
  "schema_version", "collection_mode", "policy_version", "run_id", "source_id", "observed_at",
  "dataset_key", "equipment_id", "recipe_id", "sensor_id", "step_id", "n_train", "n_test",
  "has_missing", "missing_quality_status", "config_sha256", "series_preview",
  "baseline_hard_indices", "baseline_standard_indices", "baseline_priority_indices", "hard_indices",
  "shadow_indices", "priority_indices", "standard_indices", "confidence", "candidates",
  "candidate_count", "hard_count", "shadow_count", "priority_count", "standard_count",
  "contains_test_labels", "contains_evaluation_metrics", "automatic_policy_changed", "prediction_sha256",
] as const;

const FORBIDDEN = new Set([
  "y_test", "label", "labels", "true_label", "actual_anomaly_positions", "anomaly_positions",
  "tp", "fp", "fn", "precision", "recall", "f1", "hard_tp", "hard_fp", "hard_f1", "combined_f1",
]);

function object(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error(`${path} must be an object`);
  return value as Record<string, unknown>;
}

function exactFields(value: Record<string, unknown>, fields: readonly string[], path: string) {
  const allowed = new Set(fields);
  for (const key of Object.keys(value)) if (!allowed.has(key)) throw new Error(`unknown field: ${path}.${key}`);
  for (const key of fields) if (!(key in value)) throw new Error(`missing field: ${path}.${key}`);
}

function scanForbidden(value: unknown, path = "root") {
  if (Array.isArray(value)) return value.forEach((item, index) => scanForbidden(item, `${path}[${index}]`));
  if (typeof value !== "object" || value === null) return;
  for (const [key, child] of Object.entries(value)) {
    if (FORBIDDEN.has(key.toLowerCase())) throw new Error(`unknown field: ${path}.${key}`);
    scanForbidden(child, `${path}.${key}`);
  }
}

function stringValue(value: unknown, path: string, nullable = false): string | null {
  if (nullable && value === null) return null;
  if (typeof value !== "string" || !value.trim()) throw new Error(`${path} must be a non-empty string`);
  return value;
}

function integer(value: unknown, path: string, minimum = 0): number {
  if (!Number.isInteger(value) || (value as number) < minimum) throw new Error(`${path} must be an integer >= ${minimum}`);
  return value as number;
}

function finite(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${path} must be finite`);
  return value;
}

function sha256(value: unknown, path: string): string {
  const text = stringValue(value, path) as string;
  if (!/^[0-9a-f]{64}$/.test(text)) throw new Error(`${path} must be lowercase SHA256`);
  return text;
}

function indices(value: unknown, path: string, nTest: number): number[] {
  if (!Array.isArray(value)) throw new Error(`${path} must be a list`);
  const result = value.map((item) => integer(item, path));
  if (new Set(result).size !== result.length) throw new Error(`${path} contains duplicate index`);
  if (result.some((item) => item >= nTest)) throw new Error(`${path} contains candidate outside run range`);
  return result;
}

function sameSet(left: number[], right: number[]) {
  return left.length === right.length && left.every((value) => right.includes(value));
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (typeof value === "object" && value !== null) {
    return Object.fromEntries(Object.keys(value as Record<string, unknown>).sort().map((key) => [key, canonical((value as Record<string, unknown>)[key])]));
  }
  return value;
}

export function predictionHash(unsignedPayload: Record<string, unknown>) {
  return createHash("sha256").update(JSON.stringify(canonical(unsignedPayload))).digest("hex");
}

export function assertValidShadowPrediction(raw: unknown): ShadowPrediction {
  scanForbidden(raw);
  const payload = object(raw, "prediction");
  exactFields(payload, TOP_LEVEL_FIELDS, "prediction");
  if (payload.schema_version !== SHADOW_SCHEMA_VERSION) throw new Error("unsupported schema_version");
  if (payload.collection_mode !== SHADOW_COLLECTION_MODE) throw new Error("collection_mode must be prospective_shadow");
  if (payload.policy_version !== SHADOW_POLICY_VERSION) throw new Error("unsupported policy_version");
  const runId = stringValue(payload.run_id, "run_id") as string;
  stringValue(payload.source_id, "source_id");
  stringValue(payload.dataset_key, "dataset_key");
  for (const field of ["equipment_id", "recipe_id", "sensor_id", "step_id"] as const) stringValue(payload[field], field, true);
  const observedAt = stringValue(payload.observed_at, "observed_at") as string;
  if (!/(Z|\+00:00)$/.test(observedAt) || !Number.isFinite(Date.parse(observedAt))) throw new Error("observed_at must use UTC");
  const nTrain = integer(payload.n_train, "n_train");
  const nTest = integer(payload.n_test, "n_test", 1);
  if (typeof payload.has_missing !== "boolean") throw new Error("has_missing must be boolean");
  stringValue(payload.missing_quality_status, "missing_quality_status");
  sha256(payload.config_sha256, "config_sha256");
  if (payload.contains_test_labels !== false || payload.contains_evaluation_metrics !== false || payload.automatic_policy_changed !== false) {
    throw new Error("label, evaluation, and automatic-policy flags must be false");
  }

  const baselineHard = indices(payload.baseline_hard_indices, "baseline_hard_indices", nTest);
  const baselineStandard = indices(payload.baseline_standard_indices, "baseline_standard_indices", nTest);
  const baselinePriority = indices(payload.baseline_priority_indices, "baseline_priority_indices", nTest);
  const baselineLanes = [baselineHard, baselineStandard, baselinePriority];
  for (let left = 0; left < baselineLanes.length; left++) for (let right = left + 1; right < baselineLanes.length; right++) {
    if (baselineLanes[left].some((index) => baselineLanes[right].includes(index))) throw new Error("baseline lane overlap");
  }
  const hard = indices(payload.hard_indices, "hard_indices", nTest);
  const shadow = indices(payload.shadow_indices, "shadow_indices", nTest);
  const priority = indices(payload.priority_indices, "priority_indices", nTest);
  const standard = indices(payload.standard_indices, "standard_indices", nTest);
  if (!sameSet(hard, baselineHard)) throw new Error("baseline Hard changed");
  const lanes = [hard, shadow, priority, standard];
  for (let left = 0; left < lanes.length; left++) for (let right = left + 1; right < lanes.length; right++) {
    if (lanes[left].some((index) => lanes[right].includes(index))) throw new Error("lane overlap");
  }
  const baselineUnion = new Set([...baselineHard, ...baselineStandard, ...baselinePriority]);
  const finalUnion = new Set(lanes.flat());
  if (baselineUnion.size !== finalUnion.size || [...baselineUnion].some((index) => !finalUnion.has(index))) throw new Error("final lanes must preserve baseline candidates");

  const confidence = object(payload.confidence, "confidence");
  if (Object.keys(confidence).length !== hard.length || hard.some((index) => !["A", "B", "C"].includes(String(confidence[String(index)])))) {
    throw new Error("confidence must cover exactly the Hard lane");
  }
  const preview = object(payload.series_preview, "series_preview");
  exactFields(preview, ["indices", "values"], "series_preview");
  const previewIndices = indices(preview.indices, "series_preview.indices", nTest);
  if (!Array.isArray(preview.values) || preview.values.length !== previewIndices.length || previewIndices.length < 1 || previewIndices.length > 1024) {
    throw new Error("series_preview arrays must have equal length between 1 and 1024");
  }
  if (previewIndices.some((value, index) => index > 0 && previewIndices[index - 1] >= value)) throw new Error("series_preview indices must be unique and sorted");
  preview.values.forEach((value, index) => finite(value, `series_preview.values[${index}]`));
  if ([...finalUnion].some((index) => !previewIndices.includes(index))) throw new Error("series preview omits candidate");

  if (!Array.isArray(payload.candidates) || payload.candidates.length !== finalUnion.size) throw new Error("candidates must cover final lanes");
  const candidateIndices = new Set<number>();
  for (const [position, rawCandidate] of payload.candidates.entries()) {
    const candidate = object(rawCandidate, `candidates[${position}]`);
    exactFields(candidate, ["candidate_id", "candidate_index", "final_lane", "confidence_grade", "support_count", "support_sources", "quality_flagged", "source_evidence"], `candidates[${position}]`);
    const index = integer(candidate.candidate_index, `candidates[${position}].candidate_index`);
    if (index >= nTest || candidateIndices.has(index)) throw new Error("candidate index invalid or duplicated");
    candidateIndices.add(index);
    if (candidate.candidate_id !== `${runId}:${index}`) throw new Error("candidate_id mismatch");
    const lane = candidate.final_lane as ShadowLane;
    const declaredLane = hard.includes(index) ? "Hard" : shadow.includes(index) ? "Shadow" : priority.includes(index) ? "Priority" : standard.includes(index) ? "Standard" : null;
    if (lane !== declaredLane) throw new Error("candidate lane mismatch");
    if (lane === "Hard") {
      if (candidate.confidence_grade !== confidence[String(index)]) throw new Error("Hard confidence mismatch");
    } else if (candidate.confidence_grade !== null) throw new Error("review candidate confidence must be null");
    if (typeof candidate.quality_flagged !== "boolean") throw new Error("quality_flagged must be boolean");
    const evidence = object(candidate.source_evidence, `candidates[${position}].source_evidence`);
    exactFields(evidence, SHADOW_SOURCE_NAMES, `candidates[${position}].source_evidence`);
    const agreeing: string[] = [];
    for (const source of SHADOW_SOURCE_NAMES) {
      const item = object(evidence[source], `source_evidence.${source}`);
      exactFields(item, ["agrees", "score", "threshold"], `source_evidence.${source}`);
      if (typeof item.agrees !== "boolean") throw new Error("source agrees must be boolean");
      finite(item.threshold, `source_evidence.${source}.threshold`);
      if (item.score !== null) finite(item.score, `source_evidence.${source}.score`);
      if (item.agrees) agreeing.push(source);
      if (item.agrees !== (item.score !== null)) throw new Error("source score/agreement mismatch");
      if (item.agrees && (item.score as number) < (item.threshold as number)) throw new Error("source candidate score is below threshold");
    }
    const supportCount = integer(candidate.support_count, "support_count");
    if (!Array.isArray(candidate.support_sources) || candidate.support_sources.some((item) => typeof item !== "string")) throw new Error("support_sources must be strings");
    if (supportCount !== agreeing.length || !sameStrings(candidate.support_sources as string[], agreeing)) throw new Error("support evidence mismatch");
    if (lane === "Shadow" && (nTrain < 100 || supportCount < 3 || candidate.quality_flagged)) throw new Error("Shadow eligibility failed");
    const expectedLane: ShadowLane = baselineHard.includes(index) ? "Hard"
      : baselinePriority.includes(index) ? "Priority"
      : nTrain >= 100 && supportCount >= 3 && !candidate.quality_flagged ? "Shadow"
      : supportCount >= 2 ? "Priority" : "Standard";
    if (lane !== expectedLane) throw new Error("routing policy mismatch");
    if (lane === "Hard") {
      const expectedConfidence: ConfidenceGrade = supportCount >= 3 ? "A" : supportCount >= 1 ? "B" : "C";
      if (candidate.confidence_grade !== expectedConfidence) throw new Error("Hard confidence policy mismatch");
    }
  }
  if ([...finalUnion].some((index) => !candidateIndices.has(index))) throw new Error("candidates must cover final lanes");

  for (const [field, expected] of [["candidate_count", finalUnion.size], ["hard_count", hard.length], ["shadow_count", shadow.length], ["priority_count", priority.length], ["standard_count", standard.length]] as const) {
    if (integer(payload[field], field) !== expected) throw new Error(`${field} mismatch`);
  }
  const providedHash = sha256(payload.prediction_sha256, "prediction_sha256");
  const unsigned = { ...payload };
  delete unsigned.prediction_sha256;
  if (predictionHash(unsigned) !== providedHash) throw new Error("prediction_sha256 mismatch");
  return payload as unknown as ShadowPrediction;
}

function sameStrings(left: string[], right: string[]) {
  return left.length === right.length && left.every((value) => right.includes(value));
}
