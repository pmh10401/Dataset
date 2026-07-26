import { beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

const getUser = vi.fn();
const from = vi.fn();
const rpc = vi.fn();

vi.mock("../src/lib/supabase/server", () => ({
  createServerSupabaseClient: async () => ({ auth: { getUser }, from, rpc }),
}));

import { GET, POST } from "../app/api/settings/alpha/route";

const AUTH_USER_ID = "11111111-1111-1111-1111-111111111111";

function request(body: unknown) {
  return new NextRequest("http://localhost/api/settings/alpha", {
    method: "POST",
    body: JSON.stringify(body),
    headers: { "content-type": "application/json" },
  });
}

function query(result: unknown) {
  return {
    select: vi.fn().mockReturnThis(),
    eq: vi.fn().mockReturnThis(),
    order: vi.fn().mockReturnThis(),
    limit: vi.fn().mockResolvedValue(result),
    maybeSingle: vi.fn().mockResolvedValue(result),
  };
}

describe("alpha settings route", () => {
  beforeEach(() => {
    getUser.mockReset().mockResolvedValue({ data: { user: { id: AUTH_USER_ID } }, error: null });
    rpc.mockReset().mockResolvedValue({
      data: {
        source_id: "macbook-air",
        desired_alpha: 0.01,
        effective_alpha: 0.005,
        status: "pending",
      },
      error: null,
    });
    from.mockReset().mockImplementation((table: string) => {
      if (table === "dashboard_members") {
        return query({ data: { can_write: true }, error: null });
      }
      if (table === "operational_alpha_settings") {
        return query({
          data: {
            source_id: "macbook-air",
            desired_alpha: 0.005,
            effective_alpha: 0.005,
            status: "applied",
          },
          error: null,
        });
      }
      if (table === "operational_alpha_setting_events") {
        return query({
          data: [{ id: 1, requested_alpha: 0.005, status: "applied", reason: "초기 운영값" }],
          error: null,
        });
      }
      throw new Error(`unexpected table ${table}`);
    });
  });

  it("requires authentication", async () => {
    getUser.mockResolvedValue({ data: { user: null }, error: null });
    expect((await GET()).status).toBe(401);
    expect((await POST(request({ alpha: 0.005, reason: "운영 변경 요청", acknowledged: true }))).status).toBe(401);
  });

  it("returns setting, history, permission and frozen presets", async () => {
    const response = await GET();
    const body = await response.json();
    expect(response.status).toBe(200);
    expect(body.canWrite).toBe(true);
    expect(body.setting.effective_alpha).toBe(0.005);
    expect(body.history).toHaveLength(1);
    expect(body.presets.map((item: { value: number }) => item.value)).toEqual([0.005, 0.01]);
    expect(body.automaticHardAlertsUnchanged).toBe(true);
  });

  it("rejects an unregistered alpha", async () => {
    const response = await POST(request({ alpha: 0.02, reason: "운영 변경 요청", acknowledged: true }));
    expect(response.status).toBe(400);
    expect(rpc).not.toHaveBeenCalled();
  });

  it("creates a pending request through the protected database function", async () => {
    const response = await POST(request({ alpha: 0.01, reason: "강한 신뢰 표시 범위 확대", acknowledged: true }));
    const body = await response.json();
    expect(response.status).toBe(201);
    expect(rpc).toHaveBeenCalledWith("request_operational_alpha_change", {
      p_source_id: "macbook-air",
      p_alpha: 0.01,
      p_reason: "강한 신뢰 표시 범위 확대",
    });
    expect(body.status).toBe("pending");
  });

  it("maps a database permission failure to forbidden", async () => {
    rpc.mockResolvedValue({ data: null, error: { code: "42501", message: "write permission required" } });
    const response = await POST(request({ alpha: 0.005, reason: "오탐 억제 우선 적용", acknowledged: true }));
    expect(response.status).toBe(403);
  });
});
