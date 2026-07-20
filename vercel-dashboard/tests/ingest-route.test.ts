import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { NextRequest } from "next/server";

import { POST } from "../app/api/ingest/route";
import { SNAPSHOT_MAX_BYTES } from "../src/lib/snapshot";

describe("POST /api/ingest boundary", () => {
  beforeEach(() => {
    process.env.DASHBOARD_INGEST_TOKEN = "test-ingest-token";
  });

  afterEach(() => {
    delete process.env.DASHBOARD_INGEST_TOKEN;
  });

  it("returns 401 before reading an unauthenticated body", async () => {
    const request = new NextRequest("http://localhost/api/ingest", { method: "POST", body: "{}" });
    expect((await POST(request)).status).toBe(401);
  });

  it("returns 400 for an authenticated invalid schema", async () => {
    const request = new NextRequest("http://localhost/api/ingest", {
      method: "POST",
      headers: { authorization: "Bearer test-ingest-token", "content-type": "application/json" },
      body: JSON.stringify({ schema_version: 99 }),
    });
    expect((await POST(request)).status).toBe(400);
  });

  it("returns 413 before parsing an oversized body", async () => {
    const request = new NextRequest("http://localhost/api/ingest", {
      method: "POST",
      headers: { authorization: "Bearer test-ingest-token", "content-type": "application/json" },
      body: "x".repeat(SNAPSHOT_MAX_BYTES + 1),
    });
    expect((await POST(request)).status).toBe(413);
  });
});
