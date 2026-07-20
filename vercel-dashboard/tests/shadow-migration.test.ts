import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const migration = fileURLToPath(
  new URL("../supabase/migrations/202607210001_shadow_validation.sql", import.meta.url),
);

describe("Exp265 shadow ledger migration", () => {
  const sql = readFileSync(migration, "utf8").toLowerCase();

  it("creates immutable prediction and append-only feedback tables", () => {
    expect(sql).toContain("create table if not exists public.shadow_validation_runs");
    expect(sql).toContain("create table if not exists public.shadow_validation_candidates");
    expect(sql).toContain("create table if not exists public.shadow_feedback_events");
    expect(sql).toContain("create table if not exists public.shadow_run_feedback_events");
    expect(sql).not.toMatch(/grant\s+update[^;]*shadow_feedback_events/i);
    expect(sql).not.toMatch(/grant\s+delete[^;]*shadow_feedback_events/i);
  });

  it("enables RLS and denies anonymous access", () => {
    for (const table of [
      "shadow_validation_runs",
      "shadow_validation_candidates",
      "shadow_feedback_events",
      "shadow_run_feedback_events",
    ]) {
      expect(sql).toContain(`alter table public.${table} enable row level security`);
      expect(sql).toContain(`revoke all on public.${table} from anon`);
    }
  });

  it("defines exact verdict and missed-anomaly states", () => {
    expect(sql).toContain("verdict in ('true_anomaly', 'false_alarm', 'deferred')");
    expect(sql).toContain("missed_anomaly_state in ('none', 'exists', 'unknown')");
    expect(sql).toContain("missed_anomaly_count > 0");
  });

  it("provides atomic idempotent ingest and effective feedback views", () => {
    expect(sql).toContain("function public.ingest_shadow_prediction(payload jsonb)");
    expect(sql).toContain("prediction hash conflict");
    expect(sql).toContain("create or replace view public.shadow_effective_candidate_feedback");
    expect(sql).toContain("create or replace view public.shadow_effective_run_feedback");
    expect(sql).toContain("row_number() over");
  });

  it("binds correction events to the same candidate or run", () => {
    expect(sql).toContain("foreign key (corrects_event_id, candidate_id)");
    expect(sql).toContain("foreign key (corrects_event_id, run_id)");
  });

  it("publishes ledger changes for realtime", () => {
    expect(sql).toContain("alter publication supabase_realtime add table public.shadow_validation_runs");
    expect(sql).toContain("alter publication supabase_realtime add table public.shadow_feedback_events");
  });
});
