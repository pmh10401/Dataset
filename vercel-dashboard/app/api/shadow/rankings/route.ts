import { NextResponse } from "next/server";

import { loadShadowRankingRows } from "@/src/lib/shadow-ranking-data";
import { createServerSupabaseClient } from "@/src/lib/supabase/server";

export async function GET() {
  const supabase = await createServerSupabaseClient();
  const { data: authData, error: authError } = await supabase.auth.getUser();
  if (authError || !authData.user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  try {
    const runs = await loadShadowRankingRows(supabase);
    const candidates = runs.reduce((total, run) => total + run.candidates.length, 0);
    return NextResponse.json({
      collection_mode: "retrospective_replay",
      policy_version: "exp397-review-ranking-replay-v1",
      datasets: runs.length,
      candidates,
      automatic_policy_changed: false,
      contains_test_labels: false,
      contains_evaluation_metrics: false,
      retrospective: true,
      prospective: false,
      deployment_validated: false,
      runs,
    }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "ranking unavailable" }, { status: 503 });
  }
}
