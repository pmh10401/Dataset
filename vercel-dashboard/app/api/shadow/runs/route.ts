import { NextRequest, NextResponse } from "next/server";

import { createServerSupabaseClient } from "@/src/lib/supabase/server";

const LANES = new Set(["Hard", "Shadow", "Priority", "Standard"]);
const CONFIDENCE = new Set(["A", "B", "C"]);
const VERDICTS = new Set(["true_anomaly", "false_alarm", "deferred", "unreviewed"]);

export async function GET(request: NextRequest) {
  const supabase = await createServerSupabaseClient();
  const { data: authData, error: authError } = await supabase.auth.getUser();
  if (authError || !authData.user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const params = request.nextUrl.searchParams;
  const limit = Math.min(100, Math.max(1, Number.parseInt(params.get("limit") ?? "25", 10) || 25));
  const offset = Math.max(0, Number.parseInt(params.get("offset") ?? "0", 10) || 0);
  const lane = params.get("lane");
  const confidence = params.get("confidence");
  const verdict = params.get("verdict");
  const missing = params.get("missing");
  if (lane && !LANES.has(lane)) return NextResponse.json({ error: "invalid lane filter" }, { status: 400 });
  if (confidence && !CONFIDENCE.has(confidence)) return NextResponse.json({ error: "invalid confidence filter" }, { status: 400 });
  if (verdict && !VERDICTS.has(verdict)) return NextResponse.json({ error: "invalid verdict filter" }, { status: 400 });
  if (missing && !["yes", "no"].includes(missing)) return NextResponse.json({ error: "invalid missing filter" }, { status: 400 });

  let query = supabase.from("shadow_validation_runs").select("*", { count: "exact" }).order("observed_at", { ascending: false });
  if (missing) query = query.eq("has_missing", missing === "yes");
  const { data: runs, error: runError, count } = await query.range(offset, offset + limit - 1);
  if (runError) return NextResponse.json({ error: runError.message }, { status: 503 });
  const runIds = (runs ?? []).map((run) => run.run_id);
  if (!runIds.length) return NextResponse.json({ runs: [], total: count ?? 0, limit, offset }, { headers: { "Cache-Control": "no-store" } });
  let candidateQuery = supabase.from("shadow_validation_candidates").select("*").in("run_id", runIds).order("candidate_index");
  if (lane) candidateQuery = candidateQuery.eq("final_lane", lane);
  if (confidence) candidateQuery = candidateQuery.eq("confidence_grade", confidence);
  const { data: candidates, error: candidateError } = await candidateQuery.limit(10000);
  if (candidateError) return NextResponse.json({ error: candidateError.message }, { status: 503 });
  const candidateIds = (candidates ?? []).map((candidate) => candidate.candidate_id);
  const feedbackResult = candidateIds.length
    ? await supabase.from("shadow_effective_candidate_feedback").select("*").in("candidate_id", candidateIds).limit(10000)
    : { data: [], error: null };
  if (feedbackResult.error) return NextResponse.json({ error: feedbackResult.error.message }, { status: 503 });
  const verdictByCandidate = new Map((feedbackResult.data ?? []).map((row) => [row.candidate_id, row]));
  const joinedCandidates = (candidates ?? []).map((candidate) => ({ ...candidate, feedback: verdictByCandidate.get(candidate.candidate_id) ?? null }))
    .filter((candidate) => !verdict || (verdict === "unreviewed" ? candidate.feedback === null : candidate.feedback?.verdict === verdict));
  const visibleRunIds = lane || confidence || verdict ? new Set(joinedCandidates.map((candidate) => candidate.run_id)) : new Set(runIds);
  const runFeedbackResult = await supabase.from("shadow_effective_run_feedback").select("*").in("run_id", runIds).limit(1000);
  if (runFeedbackResult.error) return NextResponse.json({ error: runFeedbackResult.error.message }, { status: 503 });
  const runFeedback = new Map((runFeedbackResult.data ?? []).map((row) => [row.run_id, row]));
  const result = (runs ?? []).filter((run) => visibleRunIds.has(run.run_id)).map((run) => ({
    ...run,
    candidates: joinedCandidates.filter((candidate) => candidate.run_id === run.run_id),
    run_feedback: runFeedback.get(run.run_id) ?? null,
  }));
  return NextResponse.json({ runs: result, total: count ?? 0, limit, offset }, { headers: { "Cache-Control": "no-store" } });
}
