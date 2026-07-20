export const SNAPSHOT_SCHEMA_VERSION = 1;
export const SNAPSHOT_MAX_BYTES = 256 * 1024;
export const SNAPSHOT_STALE_SECONDS = 30;

const ALLOWED_FIELDS = new Set([
  "schema_version",
  "generated_at",
  "source_instance_id",
  "active_experiment",
  "active_label",
  "running",
  "datasets_done",
  "expected_datasets",
  "progress_percent",
  "elapsed_seconds",
  "eta_seconds",
  "current_dataset",
  "queue_items",
  "coverage",
  "throughput",
  "runtime_health",
  "system",
  "process_summary",
  "warning_summary",
  "error_summary",
  "last_completed_experiment",
  "last_completed_label",
  "completed_results",
]);

export type Snapshot = {
  schema_version: number;
  generated_at: string;
  source_instance_id: string;
  active_experiment: string | null;
  active_label: string | null;
  running: boolean;
  datasets_done: number;
  expected_datasets: number;
  progress_percent: number;
  elapsed_seconds: number | null;
  eta_seconds: number | null;
  current_dataset: string | null;
  queue_items: Record<string, unknown>[];
  coverage: Record<string, unknown>;
  throughput: Record<string, unknown>;
  runtime_health: Record<string, unknown>;
  system: Record<string, unknown>;
  process_summary: Record<string, unknown>;
  warning_summary: Record<string, unknown>;
  error_summary: Record<string, unknown>;
  last_completed_experiment: string | null;
  last_completed_label: string | null;
  completed_results: Record<string, unknown>[];
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireFiniteNumber(record: Record<string, unknown>, key: string) {
  if (typeof record[key] !== "number" || !Number.isFinite(record[key])) {
    throw new Error(`${key} must be a finite number`);
  }
}

export function assertValidSnapshot(value: unknown): Snapshot {
  if (!isRecord(value)) throw new Error("snapshot must be an object");
  for (const key of Object.keys(value)) {
    if (!ALLOWED_FIELDS.has(key)) throw new Error(`unknown field: ${key}`);
  }
  if (value.schema_version !== SNAPSHOT_SCHEMA_VERSION) {
    throw new Error(`unsupported schema version: ${String(value.schema_version)}`);
  }
  if (typeof value.generated_at !== "string" || Number.isNaN(Date.parse(value.generated_at))) {
    throw new Error("generated_at must be an ISO timestamp");
  }
  if (typeof value.source_instance_id !== "string" || !value.source_instance_id.trim()) {
    throw new Error("source_instance_id is required");
  }
  if (typeof value.running !== "boolean") throw new Error("running must be boolean");
  for (const key of ["datasets_done", "expected_datasets", "progress_percent"]) {
    requireFiniteNumber(value, key);
  }
  for (const key of ["queue_items", "completed_results"]) {
    if (!Array.isArray(value[key])) throw new Error(`${key} must be an array`);
  }
  for (const key of [
    "coverage",
    "throughput",
    "runtime_health",
    "system",
    "process_summary",
    "warning_summary",
    "error_summary",
  ]) {
    if (!isRecord(value[key])) throw new Error(`${key} must be an object`);
  }
  return value as Snapshot;
}

export function isSnapshotStale(snapshot: Pick<Snapshot, "generated_at">, now = new Date()) {
  return now.getTime() - Date.parse(snapshot.generated_at) > SNAPSHOT_STALE_SECONDS * 1000;
}
