import { NextRequest, NextResponse } from "next/server";

import { createServerSupabaseClient } from "@/src/lib/supabase/server";

const STATES = new Set(["none", "exists", "unknown"]);

export async function POST(request: NextRequest, context: { params: Promise<{ runId: string }> }) {
  const supabase = await createServerSupabaseClient();
  const { data: authData, error: authError } = await supabase.auth.getUser();
  if (authError || !authData.user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }
  const state = String(body.missed_anomaly_state);
  const count = body.missed_anomaly_count;
  if (!STATES.has(state)) return NextResponse.json({ error: "invalid missed anomaly state" }, { status: 400 });
  const consistent = (state === "none" && count === 0)
    || (state === "exists" && Number.isInteger(count) && (count as number) > 0)
    || (state === "unknown" && (count === null || count === undefined));
  if (!consistent) return NextResponse.json({ error: "missed anomaly state/count mismatch" }, { status: 400 });
  if (body.note !== undefined && body.note !== null && (typeof body.note !== "string" || body.note.length > 2000)) {
    return NextResponse.json({ error: "note must be at most 2000 characters" }, { status: 400 });
  }
  const corrects = body.corrects_event_id;
  if (corrects !== undefined && corrects !== null && (!Number.isInteger(corrects) || (corrects as number) < 1)) {
    return NextResponse.json({ error: "invalid correction reference" }, { status: 400 });
  }
  const { runId } = await context.params;
  const { data: run, error: runError } = await supabase.from("shadow_validation_runs").select("run_id").eq("run_id", runId).maybeSingle();
  if (runError) return NextResponse.json({ error: runError.message }, { status: 503 });
  if (!run) return NextResponse.json({ error: "run not found" }, { status: 404 });
  if (corrects) {
    const { data: previous, error } = await supabase.from("shadow_run_feedback_events").select("id,run_id").eq("id", corrects).maybeSingle();
    if (error) return NextResponse.json({ error: error.message }, { status: 503 });
    if (!previous || previous.run_id !== runId) return NextResponse.json({ error: "correction target mismatch" }, { status: 400 });
  }
  const { data, error } = await supabase.from("shadow_run_feedback_events").insert({
    run_id: runId,
    missed_anomaly_state: state,
    missed_anomaly_count: state === "unknown" ? null : count,
    reviewer_id: authData.user.id,
    note: body.note ?? null,
    corrects_event_id: corrects ?? null,
  }).select("id,run_id,missed_anomaly_state,missed_anomaly_count,created_at").single();
  if (error) return NextResponse.json({ error: error.message }, { status: 403 });
  return NextResponse.json({ ok: true, event: data }, { status: 201, headers: { "Cache-Control": "no-store" } });
}
