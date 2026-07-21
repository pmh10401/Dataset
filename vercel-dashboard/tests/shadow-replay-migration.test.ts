import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const migration = fileURLToPath(
  new URL("../supabase/migrations/202607210003_shadow_replay.sql", import.meta.url),
);

describe("Exp265 retrospective replay migration", () => {
  const sql = readFileSync(migration, "utf8").toLowerCase();

  it("separates replay rows and stores each candidate waveform", () => {
    expect(sql).toContain("retrospective_replay");
    expect(sql).toContain("exp265-replay-v1");
    expect(sql).toContain("add column if not exists series_preview jsonb");
    expect(sql).toContain("collection_mode, observed_at desc");
  });
});
