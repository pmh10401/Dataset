import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const migration = fileURLToPath(
  new URL("../supabase/migrations/202607210002_shadow_feedback_indexes.sql", import.meta.url),
);

describe("Exp265 feedback index migration", () => {
  const sql = readFileSync(migration, "utf8").toLowerCase();

  it("covers feedback reviewer and correction foreign keys", () => {
    expect(sql).toContain("on public.shadow_feedback_events (reviewer_id)");
    expect(sql).toContain("on public.shadow_feedback_events (corrects_event_id, candidate_id)");
    expect(sql).toContain("on public.shadow_run_feedback_events (reviewer_id)");
    expect(sql).toContain("on public.shadow_run_feedback_events (corrects_event_id, run_id)");
  });
});
