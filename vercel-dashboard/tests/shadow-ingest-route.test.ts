import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

import { SHADOW_MAX_BYTES } from "../src/lib/shadow";
import { validShadowPayload } from "./shadow-contract.test";

const getUser = vi.fn();
const rpc = vi.fn();

vi.mock("../src/lib/supabase/admin", () => ({
  createIngestClient: () => ({ auth: { getUser }, rpc }),
}));

import { POST } from "../app/api/shadow/ingest/route";

function request(body: string, headers: Record<string, string> = {}) {
  return new NextRequest("http://localhost/api/shadow/ingest", {
    method: "POST",
    body,
    headers: { "content-type": "application/json", ...headers },
  });
}

describe("POST /api/shadow/ingest", () => {
  beforeEach(() => {
    process.env.DASHBOARD_INGEST_TOKEN = "ingest";
    getUser.mockReset().mockResolvedValue({ data: { user: { id: "user-1" } }, error: null });
    rpc.mockReset().mockResolvedValue({ data: { ok: true, idempotent: false }, error: null });
  });

  afterEach(() => delete process.env.DASHBOARD_INGEST_TOKEN);

  it("rejects bearer token before reading the body", async () => {
    expect((await POST(request("not-json"))).status).toBe(401);
  });

  it("rejects oversized input before parsing", async () => {
    const response = await POST(request("x".repeat(SHADOW_MAX_BYTES + 1), {
      authorization: "Bearer ingest",
      "x-supabase-access-token": "session",
    }));
    expect(response.status).toBe(413);
  });

  it("requires a valid writer session", async () => {
    getUser.mockResolvedValue({ data: { user: null }, error: { message: "expired" } });
    const response = await POST(request(JSON.stringify(validShadowPayload()), {
      authorization: "Bearer ingest",
      "x-supabase-access-token": "session",
    }));
    expect(response.status).toBe(401);
  });

  it("maps prediction hash conflict to 409", async () => {
    rpc.mockResolvedValue({ data: null, error: { code: "23505", message: "prediction hash conflict" } });
    const response = await POST(request(JSON.stringify(validShadowPayload()), {
      authorization: "Bearer ingest",
      "x-supabase-access-token": "session",
    }));
    expect(response.status).toBe(409);
  });

  it("passes a valid payload to the atomic RPC", async () => {
    const payload = validShadowPayload();
    const response = await POST(request(JSON.stringify(payload), {
      authorization: "Bearer ingest",
      "x-supabase-access-token": "session",
    }));
    expect(response.status).toBe(200);
    expect(rpc).toHaveBeenCalledWith("ingest_shadow_prediction", { payload });
  });
});
