import { describe, expect, it } from "vitest";

import { isPublicRoute } from "../proxy";

describe("dashboard proxy route policy", () => {
  it("allows token-protected ingest endpoints to reach their own authentication", () => {
    expect(isPublicRoute("/api/ingest")).toBe(true);
    expect(isPublicRoute("/api/shadow/ingest")).toBe(true);
  });

  it("keeps dashboard and read APIs behind the user session", () => {
    expect(isPublicRoute("/")).toBe(false);
    expect(isPublicRoute("/shadow")).toBe(false);
    expect(isPublicRoute("/api/shadow/report")).toBe(false);
    expect(isPublicRoute("/api/shadow/runs")).toBe(false);
  });
});
