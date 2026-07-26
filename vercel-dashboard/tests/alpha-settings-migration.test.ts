import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const migrationPath = join(
  process.cwd(),
  "supabase/migrations/202607270001_operational_alpha_settings.sql",
);

describe("operational alpha settings migration", () => {
  it("creates current state and append-only event tables with RLS", () => {
    const sql = readFileSync(migrationPath, "utf8");
    expect(sql).toMatch(/create table if not exists public\.operational_alpha_settings/i);
    expect(sql).toMatch(/create table if not exists public\.operational_alpha_setting_events/i);
    expect(sql).toMatch(/alter table public\.operational_alpha_settings enable row level security/i);
    expect(sql).toMatch(/alter table public\.operational_alpha_setting_events enable row level security/i);
    expect(sql).toMatch(/desired_alpha in \(0\.005, 0\.010\)/i);
  });

  it("provides an atomic authenticated writer function", () => {
    const sql = readFileSync(migrationPath, "utf8");
    expect(sql).toMatch(/create or replace function public\.request_operational_alpha_change/i);
    expect(sql).toMatch(/security definer/i);
    expect(sql).toMatch(/auth\.uid\(\)/i);
    expect(sql).toMatch(/dashboard_members/i);
    expect(sql).toMatch(/can_write/i);
    expect(sql).toMatch(/insert into public\.operational_alpha_setting_events/i);
    expect(sql).toMatch(/insert into public\.operational_alpha_settings/i);
  });

  it("allows members to read but revokes direct writes", () => {
    const sql = readFileSync(migrationPath, "utf8");
    expect(sql).toMatch(/members can read alpha settings/i);
    expect(sql).toMatch(/members can read alpha setting events/i);
    expect(sql).toMatch(/revoke all on public\.operational_alpha_settings from authenticated/i);
    expect(sql).toMatch(/grant select on public\.operational_alpha_settings to authenticated/i);
    expect(sql).toMatch(/grant execute on function public\.request_operational_alpha_change/i);
  });
});
