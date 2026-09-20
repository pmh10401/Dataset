import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const migration = fileURLToPath(
  new URL("../supabase/migrations/202607270004_exp397_shadow_ranking.sql", import.meta.url),
);

describe("Exp397 Shadow ranking migration", () => {
  const sql = readFileSync(migration, "utf8").toLowerCase();

  it("creates an isolated immutable ranking ledger", () => {
    expect(sql).toContain("create table public.shadow_ranking_runs");
    expect(sql).toContain("create table public.shadow_ranking_candidates");
    expect(sql).toContain("exp397-review-ranking-replay-v1");
    expect(sql).toContain("retrospective_replay");
    expect(sql).toContain("prediction hash conflict");
    expect(sql).toContain("enable row level security");
    expect(sql).toContain("ingest_shadow_ranking_prediction");
  });

  it("keeps prediction rows label-free and append-only", () => {
    expect(sql).toContain("contains_test_labels = false");
    expect(sql).toContain("contains_evaluation_metrics = false");
    expect(sql).toContain("automatic_policy_changed = false");
    expect(sql).not.toMatch(/create policy[^;]+for update/s);
    expect(sql).not.toMatch(/create policy[^;]+for delete/s);
  });
});
