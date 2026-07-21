import { NextRequest, NextResponse } from "next/server";

import { loadShadowReportRows } from "@/src/lib/shadow-data";
import { buildShadowReport } from "@/src/lib/shadow-metrics";
import { predictionHash, SHADOW_SCHEMA_VERSION } from "@/src/lib/shadow";
import { modeCopy, parseShadowCollectionMode, policyVersionFor } from "@/src/lib/shadow-mode";
import { createIngestClient } from "@/src/lib/supabase/admin";
import { createServerSupabaseClient } from "@/src/lib/supabase/server";

export async function GET(request: NextRequest) {
  const accessToken = request.headers.get("x-supabase-access-token");
  const supabase = accessToken ? createIngestClient(accessToken) : await createServerSupabaseClient();
  const { data: authData, error: authError } = await supabase.auth.getUser(accessToken ?? undefined);
  if (authError || !authData.user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  try {
    const mode = parseShadowCollectionMode(request.nextUrl.searchParams.get("mode"));
    if (!mode) return NextResponse.json({ error: "invalid collection mode" }, { status: 400 });
    const policyVersion = policyVersionFor(mode);
    const scope = modeCopy(mode);
    const runs = await loadShadowReportRows(supabase, mode);
    if (runs.some((run) => run.collection_mode !== mode || run.policy_version !== policyVersion)) {
      throw new Error("mixed shadow policy versions in report source");
    }
    const observed = runs.map((run) => run.observed_at).filter((value): value is string => Boolean(value)).sort();
    const unsigned: Record<string, unknown> = {
      schema_version: SHADOW_SCHEMA_VERSION,
      collection_mode: mode,
      policy_version: policyVersion,
      generated_at: new Date().toISOString(),
      collected_from: observed[0] ?? null,
      collected_to: observed.at(-1) ?? null,
      config_sha256: [...new Set(runs.map((run) => run.config_sha256).filter(Boolean))].sort(),
      ...buildShadowReport(runs),
      automatic_policy_changed: false,
      deployment_validated: false,
      prospective: scope.prospective,
      retrospective: scope.retrospective,
    };
    return NextResponse.json({ ...unsigned, report_sha256: predictionHash(unsigned) }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "report unavailable" }, { status: 503 });
  }
}
