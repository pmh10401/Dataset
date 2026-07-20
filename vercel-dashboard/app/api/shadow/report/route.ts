import { NextResponse } from "next/server";

import { loadShadowReportRows } from "@/src/lib/shadow-data";
import { buildShadowReport } from "@/src/lib/shadow-metrics";
import { createServerSupabaseClient } from "@/src/lib/supabase/server";

export async function GET() {
  const supabase = await createServerSupabaseClient();
  const { data: authData, error: authError } = await supabase.auth.getUser();
  if (authError || !authData.user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  try {
    const runs = await loadShadowReportRows(supabase);
    return NextResponse.json({ ...buildShadowReport(runs), generated_at: new Date().toISOString() }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "report unavailable" }, { status: 503 });
  }
}
