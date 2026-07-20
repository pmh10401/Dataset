import { NextRequest, NextResponse } from "next/server";

import { createServerSupabaseClient } from "@/src/lib/supabase/server";

const VERDICTS = new Set(["true_anomaly", "false_alarm", "deferred"]);

export async function POST(request: NextRequest, context: { params: Promise<{ candidateId: string }> }) {
  const supabase = await createServerSupabaseClient();
  const { data: authData, error: authError } = await supabase.auth.getUser();
  if (authError || !authData.user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }
  if (!VERDICTS.has(String(body.verdict))) return NextResponse.json({ error: "invalid verdict" }, { status: 400 });
  if (body.note !== undefined && body.note !== null && (typeof body.note !== "string" || body.note.length > 2000)) {
    return NextResponse.json({ error: "note must be at most 2000 characters" }, { status: 400 });
  }
  const corrects = body.corrects_event_id;
  if (corrects !== undefined && corrects !== null && (!Number.isInteger(corrects) || (corrects as number) < 1)) {
    return NextResponse.json({ error: "invalid correction reference" }, { status: 400 });
  }
  const { candidateId } = await context.params;
  const { data: candidate, error: candidateError } = await supabase.from("shadow_validation_candidates").select("candidate_id").eq("candidate_id", candidateId).maybeSingle();
  if (candidateError) return NextResponse.json({ error: candidateError.message }, { status: 503 });
  if (!candidate) return NextResponse.json({ error: "candidate not found" }, { status: 404 });
  if (corrects) {
    const { data: previous, error } = await supabase.from("shadow_feedback_events").select("id,candidate_id").eq("id", corrects).maybeSingle();
    if (error) return NextResponse.json({ error: error.message }, { status: 503 });
    if (!previous || previous.candidate_id !== candidateId) return NextResponse.json({ error: "correction target mismatch" }, { status: 400 });
  }
  const { data, error } = await supabase.from("shadow_feedback_events").insert({
    candidate_id: candidateId,
    verdict: body.verdict,
    reviewer_id: authData.user.id,
    note: body.note ?? null,
    corrects_event_id: corrects ?? null,
  }).select("id,candidate_id,verdict,created_at").single();
  if (error) return NextResponse.json({ error: error.message }, { status: 403 });
  return NextResponse.json({ ok: true, event: data }, { status: 201, headers: { "Cache-Control": "no-store" } });
}
