import { beforeEach, describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

const getUser = vi.fn();
const from = vi.fn();

vi.mock("../src/lib/supabase/server", () => ({
  createServerSupabaseClient: async () => ({ auth: { getUser }, from }),
}));

import { POST as postCandidate } from "../app/api/shadow/candidates/[candidateId]/feedback/route";
import { POST as postRun } from "../app/api/shadow/runs/[runId]/feedback/route";

const AUTH_USER_ID = "11111111-1111-1111-1111-111111111111";

function request(body: unknown) {
  return new NextRequest("http://localhost/api/shadow/feedback", {
    method: "POST",
    body: JSON.stringify(body),
    headers: { "content-type": "application/json" },
  });
}

function tableMock(table: string) {
  const candidate = { select: vi.fn().mockReturnThis(), eq: vi.fn().mockReturnThis(), maybeSingle: vi.fn().mockResolvedValue({ data: { candidate_id: "run-1:2" }, error: null }) };
  const run = { select: vi.fn().mockReturnThis(), eq: vi.fn().mockReturnThis(), maybeSingle: vi.fn().mockResolvedValue({ data: { run_id: "run-1" }, error: null }) };
  const insert = vi.fn().mockReturnValue({ select: vi.fn().mockReturnValue({ single: vi.fn().mockResolvedValue({ data: { id: 7 }, error: null }) }) });
  if (table === "shadow_validation_candidates") return candidate;
  if (table === "shadow_validation_runs") return run;
  if (table === "shadow_feedback_events" || table === "shadow_run_feedback_events") return { insert };
  throw new Error(`unexpected table ${table}`);
}

describe("shadow feedback routes", () => {
  beforeEach(() => {
    getUser.mockReset().mockResolvedValue({ data: { user: { id: AUTH_USER_ID } }, error: null });
    from.mockReset().mockImplementation(tableMock);
  });

  it("uses authenticated user id as candidate reviewer and ignores a forged id", async () => {
    const response = await postCandidate(request({ verdict: "true_anomaly", reviewer_id: "forged" }), { params: Promise.resolve({ candidateId: "run-1:2" }) });
    expect(response.status).toBe(201);
    const eventTable = from.mock.results.find((entry) => entry.value.insert)?.value;
    expect(eventTable.insert).toHaveBeenCalledWith(expect.objectContaining({ reviewer_id: AUTH_USER_ID }));
  });

  it("rejects invalid verdict and oversized notes", async () => {
    expect((await postCandidate(request({ verdict: "yes" }), { params: Promise.resolve({ candidateId: "run-1:2" }) })).status).toBe(400);
    expect((await postCandidate(request({ verdict: "deferred", note: "x".repeat(2001) }), { params: Promise.resolve({ candidateId: "run-1:2" }) })).status).toBe(400);
  });

  it("requires authentication", async () => {
    getUser.mockResolvedValue({ data: { user: null }, error: null });
    expect((await postCandidate(request({ verdict: "false_alarm" }), { params: Promise.resolve({ candidateId: "run-1:2" }) })).status).toBe(401);
  });

  it("appends valid run feedback with an exact missed count", async () => {
    const response = await postRun(request({ missed_anomaly_state: "exists", missed_anomaly_count: 2 }), { params: Promise.resolve({ runId: "run-1" }) });
    expect(response.status).toBe(201);
    const eventTable = from.mock.results.find((entry) => entry.value.insert)?.value;
    expect(eventTable.insert).toHaveBeenCalledWith(expect.objectContaining({ reviewer_id: AUTH_USER_ID, missed_anomaly_count: 2 }));
  });

  it("rejects inconsistent run feedback", async () => {
    const response = await postRun(request({ missed_anomaly_state: "none", missed_anomaly_count: 3 }), { params: Promise.resolve({ runId: "run-1" }) });
    expect(response.status).toBe(400);
  });
});
