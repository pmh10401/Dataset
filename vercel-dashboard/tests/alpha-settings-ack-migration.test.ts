import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const migration = readFileSync(
  join(process.cwd(), "supabase/migrations/202607270002_operational_alpha_acknowledgement.sql"),
  "utf8",
);
const permissionsMigration = readFileSync(
  join(process.cwd(), "supabase/migrations/202607270003_operational_alpha_function_permissions.sql"),
  "utf8",
);

describe("operational alpha acknowledgement migration", () => {
  it("adds a protected, version-aware acknowledgement function", () => {
    expect(migration).toMatch(/create or replace function public\.acknowledge_operational_alpha_change/i);
    expect(migration).toMatch(/security definer/i);
    expect(migration).toMatch(/auth\.uid\(\)/i);
    expect(migration).toMatch(/dashboard_members/i);
    expect(migration).toMatch(/can_write/i);
    expect(migration).toMatch(/version\s*=\s*p_version/i);
    expect(migration).toMatch(/desired_alpha\s*=\s*p_alpha/i);
  });

  it("records applied and failed outcomes without granting direct writes", () => {
    expect(migration).toMatch(/status\s*=\s*case when p_success then 'applied' else 'failed' end/i);
    expect(migration).toMatch(/insert into public\.operational_alpha_setting_events/i);
    expect(migration).toMatch(/revoke all on function public\.acknowledge_operational_alpha_change/i);
    expect(migration).toMatch(/grant execute on function public\.acknowledge_operational_alpha_change/i);
  });

  it("explicitly removes Supabase anon execution defaults", () => {
    expect(permissionsMigration).toMatch(/request_operational_alpha_change[\s\S]*from public, anon/i);
    expect(permissionsMigration).toMatch(/acknowledge_operational_alpha_change[\s\S]*from public, anon/i);
    expect(permissionsMigration).toMatch(/request_operational_alpha_change[\s\S]*to authenticated/i);
    expect(permissionsMigration).toMatch(/acknowledge_operational_alpha_change[\s\S]*to authenticated/i);
  });
});
