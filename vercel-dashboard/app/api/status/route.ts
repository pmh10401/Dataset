import { NextResponse } from "next/server";

import { createServerSupabaseClient } from "@/src/lib/supabase/server";

export async function GET() {
  const supabase = await createServerSupabaseClient();
  const { data: userData } = await supabase.auth.getUser();
  if (!userData.user) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { data, error } = await supabase
    .from("experiment_status_latest")
    .select("source_id,generated_at,received_at,snapshot")
    .order("received_at", { ascending: false })
    .limit(1)
    .maybeSingle();
  if (error) return NextResponse.json({ error: error.message }, { status: 503 });
  if (!data) return NextResponse.json({ error: "no snapshot" }, { status: 404 });
  return NextResponse.json(data, { headers: { "Cache-Control": "no-store" } });
}
