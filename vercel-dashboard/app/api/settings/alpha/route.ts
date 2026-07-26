import { NextRequest, NextResponse } from "next/server";

import {
  ALPHA_PRESETS,
  DEFAULT_ALPHA_SOURCE_ID,
  parseAlphaRequest,
} from "@/src/lib/alpha-settings";
import { createServerSupabaseClient } from "@/src/lib/supabase/server";

const NO_STORE = { "Cache-Control": "no-store" };

export async function GET() {
  const supabase = await createServerSupabaseClient();
  const { data: authData, error: authError } = await supabase.auth.getUser();
  if (authError || !authData.user) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401, headers: NO_STORE });
  }

  const [memberResult, settingResult, historyResult] = await Promise.all([
    supabase
      .from("dashboard_members")
      .select("can_write")
      .eq("user_id", authData.user.id)
      .maybeSingle(),
    supabase
      .from("operational_alpha_settings")
      .select("source_id,desired_alpha,effective_alpha,status,version,reason,requested_at,applied_at,last_error")
      .eq("source_id", DEFAULT_ALPHA_SOURCE_ID)
      .maybeSingle(),
    supabase
      .from("operational_alpha_setting_events")
      .select("id,source_id,previous_desired_alpha,requested_alpha,status,reason,created_at")
      .eq("source_id", DEFAULT_ALPHA_SOURCE_ID)
      .order("created_at", { ascending: false })
      .limit(10),
  ]);

  const error = memberResult.error || settingResult.error || historyResult.error;
  if (error) {
    return NextResponse.json({ error: error.message }, { status: 503, headers: NO_STORE });
  }

  return NextResponse.json(
    {
      sourceId: DEFAULT_ALPHA_SOURCE_ID,
      setting: settingResult.data,
      history: historyResult.data ?? [],
      canWrite: memberResult.data?.can_write === true,
      presets: ALPHA_PRESETS,
      automaticHardAlertsUnchanged: true,
      retrospectiveOnly: true,
    },
    { headers: NO_STORE },
  );
}

export async function POST(request: NextRequest) {
  const supabase = await createServerSupabaseClient();
  const { data: authData, error: authError } = await supabase.auth.getUser();
  if (authError || !authData.user) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401, headers: NO_STORE });
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400, headers: NO_STORE });
  }

  let parsed;
  try {
    parsed = parseAlphaRequest(body);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "invalid request" },
      { status: 400, headers: NO_STORE },
    );
  }

  const { data, error } = await supabase.rpc("request_operational_alpha_change", {
    p_source_id: DEFAULT_ALPHA_SOURCE_ID,
    p_alpha: parsed.alpha,
    p_reason: parsed.reason,
  });
  if (error) {
    const forbidden = error.code === "42501" || /permission|authentication/i.test(error.message);
    return NextResponse.json(
      { error: error.message },
      { status: forbidden ? 403 : 503, headers: NO_STORE },
    );
  }

  return NextResponse.json({ setting: data, status: "pending" }, { status: 201, headers: NO_STORE });
}
