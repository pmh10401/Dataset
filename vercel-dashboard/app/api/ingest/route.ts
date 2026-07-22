import { NextRequest, NextResponse } from "next/server";

import { hasValidBearerToken } from "@/src/lib/auth";
import { shouldWriteHistory } from "@/src/lib/history";
import {
  SNAPSHOT_MAX_BYTES,
  assertValidSnapshot,
  type Snapshot,
} from "@/src/lib/snapshot";
import { createIngestClient } from "@/src/lib/supabase/admin";

export const runtime = "nodejs";

function historyStatus(snapshot: Snapshot) {
  const errorCount = Number(snapshot.error_summary.total_errors ?? 0);
  return {
    running: snapshot.running,
    active_experiment: snapshot.active_experiment,
    datasets_done: snapshot.datasets_done,
    error_count: Number.isFinite(errorCount) ? errorCount : 0,
  };
}

export async function POST(request: NextRequest) {
  if (!hasValidBearerToken(request.headers.get("authorization"), process.env.DASHBOARD_INGEST_TOKEN)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const contentLength = Number(request.headers.get("content-length") ?? 0);
  if (contentLength > SNAPSHOT_MAX_BYTES) {
    return NextResponse.json({ error: "snapshot too large" }, { status: 413 });
  }
  const text = await request.text();
  if (Buffer.byteLength(text, "utf8") > SNAPSHOT_MAX_BYTES) {
    return NextResponse.json({ error: "snapshot too large" }, { status: 413 });
  }

  let snapshot: Snapshot;
  try {
    snapshot = assertValidSnapshot(JSON.parse(text));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "invalid snapshot" },
      { status: 400 },
    );
  }

  const accessToken = request.headers.get("x-supabase-access-token");
  if (!accessToken) return NextResponse.json({ error: "missing writer session" }, { status: 401 });
  const supabase = createIngestClient(accessToken);
  const { data: authData, error: authError } = await supabase.auth.getUser(accessToken);
  if (authError || !authData.user) {
    return NextResponse.json({ error: "invalid writer session" }, { status: 401 });
  }
  const { data: latest, error: latestReadError } = await supabase
    .from("experiment_status_latest")
    .select("snapshot,received_at")
    .eq("source_id", snapshot.source_instance_id)
    .maybeSingle();
  if (latestReadError) return NextResponse.json({ error: latestReadError.message }, { status: 503 });

  const receivedAt = new Date().toISOString();
  const { error: upsertError } = await supabase.from("experiment_status_latest").upsert({
    source_id: snapshot.source_instance_id,
    generated_at: snapshot.generated_at,
    received_at: receivedAt,
    snapshot,
  });
  if (upsertError) return NextResponse.json({ error: upsertError.message }, { status: 503 });

  const previousSnapshot = latest?.snapshot ? assertValidSnapshot(latest.snapshot) : null;
  const secondsSincePrevious = latest?.received_at
    ? (Date.parse(receivedAt) - Date.parse(latest.received_at)) / 1000
    : Number.POSITIVE_INFINITY;
  if (shouldWriteHistory(previousSnapshot ? historyStatus(previousSnapshot) : null, historyStatus(snapshot), secondsSincePrevious)) {
    const { error: historyError } = await supabase.from("experiment_status_history").insert({
      source_id: snapshot.source_instance_id,
      generated_at: snapshot.generated_at,
      received_at: receivedAt,
      active_experiment: snapshot.active_experiment,
      running: snapshot.running,
      datasets_done: snapshot.datasets_done,
      error_count: historyStatus(snapshot).error_count,
      snapshot,
    });
    if (historyError) return NextResponse.json({ error: historyError.message }, { status: 503 });
  }
  return NextResponse.json({ ok: true, received_at: receivedAt }, { headers: { "Cache-Control": "no-store" } });
}
