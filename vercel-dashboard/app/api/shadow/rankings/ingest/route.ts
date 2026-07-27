import { NextRequest, NextResponse } from "next/server";

import { hasValidBearerToken } from "@/src/lib/auth";
import {
  assertValidShadowRankingPrediction,
  SHADOW_RANKING_MAX_BYTES,
} from "@/src/lib/shadow-ranking";
import { createIngestClient } from "@/src/lib/supabase/admin";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  if (!hasValidBearerToken(request.headers.get("authorization"), process.env.DASHBOARD_INGEST_TOKEN)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const contentLength = Number(request.headers.get("content-length") ?? 0);
  if (contentLength > SHADOW_RANKING_MAX_BYTES) return NextResponse.json({ error: "prediction too large" }, { status: 413 });
  const text = await request.text();
  if (Buffer.byteLength(text, "utf8") > SHADOW_RANKING_MAX_BYTES) {
    return NextResponse.json({ error: "prediction too large" }, { status: 413 });
  }
  let payload;
  try {
    payload = assertValidShadowRankingPrediction(JSON.parse(text));
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "invalid prediction" }, { status: 400 });
  }
  const accessToken = request.headers.get("x-supabase-access-token");
  if (!accessToken) return NextResponse.json({ error: "missing writer session" }, { status: 401 });
  let supabase;
  try {
    supabase = createIngestClient(accessToken);
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "ingest unavailable" }, { status: 503 });
  }
  const { data: authData, error: authError } = await supabase.auth.getUser(accessToken);
  if (authError || !authData.user) return NextResponse.json({ error: "invalid writer session" }, { status: 401 });
  const { data, error } = await supabase.rpc("ingest_shadow_ranking_prediction", { payload });
  if (error) {
    if (error.code === "23505" || /prediction hash conflict/i.test(error.message ?? "")) {
      return NextResponse.json({ error: "prediction hash conflict" }, { status: 409 });
    }
    return NextResponse.json({ error: error.message }, { status: 503 });
  }
  return NextResponse.json({ ok: true, receipt: data }, { headers: { "Cache-Control": "no-store" } });
}
