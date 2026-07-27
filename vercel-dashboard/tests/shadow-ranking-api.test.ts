import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

import { validShadowRankingPayload } from "./shadow-ranking-contract.test";

const getUser = vi.fn();
const rpc = vi.fn();
const serverGetUser = vi.fn();

vi.mock("../src/lib/supabase/admin", () => ({
  createIngestClient: () => ({ auth: { getUser }, rpc }),
}));
vi.mock("../src/lib/supabase/server", () => ({
  createServerSupabaseClient: async () => ({ auth: { getUser: serverGetUser } }),
}));

import { POST } from "../app/api/shadow/rankings/ingest/route";
import { GET } from "../app/api/shadow/rankings/route";

function request(body: string, headers: Record<string, string> = {}) {
  return new NextRequest("http://localhost/api/shadow/rankings/ingest", {
    method: "POST",
    body,
    headers: { "content-type": "application/json", ...headers },
  });
}

describe("POST /api/shadow/rankings/ingest", () => {
  beforeEach(() => {
    process.env.DASHBOARD_INGEST_TOKEN = "ingest";
    getUser.mockReset().mockResolvedValue({ data: { user: { id: "user-1" } }, error: null });
    serverGetUser.mockReset().mockResolvedValue({ data: { user: null }, error: { message: "unauthorized" } });
    rpc.mockReset().mockResolvedValue({ data: { ok: true, idempotent: false }, error: null });
  });

  afterEach(() => delete process.env.DASHBOARD_INGEST_TOKEN);

  it("rejects unauthenticated machine ingestion", async () => {
    expect((await POST(request("{}"))).status).toBe(401);
  });

  it("rejects unauthenticated ranking reads", async () => {
    expect((await GET()).status).toBe(401);
  });

  it("maps prediction hash conflict to 409", async () => {
    rpc.mockResolvedValue({ data: null, error: { code: "23505", message: "prediction hash conflict" } });
    const response = await POST(request(JSON.stringify(validShadowRankingPayload()), {
      authorization: "Bearer ingest",
      "x-supabase-access-token": "session",
    }));
    expect(response.status).toBe(409);
  });

  it("passes the verified payload to the isolated RPC", async () => {
    const payload = validShadowRankingPayload();
    const response = await POST(request(JSON.stringify(payload), {
      authorization: "Bearer ingest",
      "x-supabase-access-token": "session",
    }));
    expect(response.status).toBe(200);
    expect(rpc).toHaveBeenCalledWith("ingest_shadow_ranking_prediction", { payload });
  });
});
