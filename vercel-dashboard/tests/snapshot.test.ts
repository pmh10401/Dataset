import { describe, expect, it } from "vitest";

import {
  SNAPSHOT_SCHEMA_VERSION,
  assertValidSnapshot,
  isSnapshotStale,
} from "../src/lib/snapshot";

const validSnapshot = {
  schema_version: SNAPSHOT_SCHEMA_VERSION,
  generated_at: "2026-07-20T04:00:00.000Z",
  source_instance_id: "macbook-air",
  active_experiment: "experiment_254_sktime_structural_features",
  active_label: "Exp254",
  running: true,
  datasets_done: 321,
  expected_datasets: 1117,
  progress_percent: 28.74,
  elapsed_seconds: 120,
  eta_seconds: 300,
  current_dataset: "FiftyWords_normal_5",
  queue_items: [],
  coverage: { datasets_done: 321, expected_datasets: 1117 },
  throughput: { recent_datasets_per_minute: 10.5 },
  runtime_health: { heartbeat: { age_seconds: 2 } },
  system: { cpu: { percent: 35 }, gpu: { percent: 0 } },
  process_summary: { process_count: 1, cpu: 25, mem: 2 },
  warning_summary: { total_warnings: 0 },
  error_summary: { total_errors: 0 },
  last_completed_experiment: "experiment_253",
  last_completed_label: "Exp253",
  completed_results: [],
};

describe("snapshot contract", () => {
  it("accepts the frozen allowlisted schema", () => {
    expect(assertValidSnapshot(validSnapshot)).toEqual(validSnapshot);
  });

  it("rejects unknown top-level fields such as labels and logs", () => {
    expect(() => assertValidSnapshot({ ...validSnapshot, y_test: [0, 1] })).toThrow(
      /unknown field/i,
    );
    expect(() => assertValidSnapshot({ ...validSnapshot, log_tail: "secret" })).toThrow(
      /unknown field/i,
    );
  });

  it("rejects unsupported schema versions", () => {
    expect(() => assertValidSnapshot({ ...validSnapshot, schema_version: 99 })).toThrow(
      /schema version/i,
    );
  });

  it("marks a snapshot stale after 30 seconds", () => {
    expect(isSnapshotStale(validSnapshot, new Date("2026-07-20T04:00:29.000Z"))).toBe(false);
    expect(isSnapshotStale(validSnapshot, new Date("2026-07-20T04:00:31.000Z"))).toBe(true);
  });
});
