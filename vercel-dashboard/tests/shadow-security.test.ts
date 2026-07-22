import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

function files(root: string): string[] {
  return readdirSync(root).flatMap((name) => {
    const path = join(root, name);
    return statSync(path).isDirectory() ? files(path) : [path];
  });
}

describe("shadow browser security", () => {
  it("keeps server and machine secrets out of browser modules", () => {
    const browserFiles = [...files(join(process.cwd(), "app/ui")), join(process.cwd(), "src/lib/supabase/browser.ts")]
      .filter((path) => /\.(ts|tsx)$/.test(path));
    const forbidden = ["SUPABASE_SERVICE_ROLE_KEY", "DASHBOARD_INGEST_TOKEN", "machine_password", "eldkqmffh"];
    for (const path of browserFiles) {
      const source = readFileSync(path, "utf8");
      for (const secret of forbidden) expect(source, `${secret} leaked into ${path}`).not.toContain(secret);
    }
  });

  it("does not call the shadow ingest RPC from browser modules", () => {
    const source = readFileSync(join(process.cwd(), "app/ui/shadow-validation.tsx"), "utf8");
    expect(source).not.toContain("ingest_shadow_prediction");
    expect(source).not.toContain("service_role");
  });
});
