"use client";

import { FormEvent, useState } from "react";
import { LogIn } from "lucide-react";

import { createBrowserSupabaseClient } from "@/src/lib/supabase/browser";

export default function LoginPage() {
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    const form = new FormData(event.currentTarget);
    const supabase = createBrowserSupabaseClient();
    const { error: authError } = await supabase.auth.signInWithPassword({
      email: String(form.get("email")),
      password: String(form.get("password")),
    });
    if (authError) {
      setError("로그인 정보를 확인해 주세요.");
      setBusy(false);
      return;
    }
    window.location.assign("/");
  }

  return (
    <main className="login-shell">
      <form className="login-panel" onSubmit={submit}>
        <div className="brand-mark">TS</div>
        <h1>실험 진행 대시보드</h1>
        <p>승인된 사용자만 운영 현황을 확인할 수 있습니다.</p>
        <label>이메일<input name="email" type="email" autoComplete="username" required /></label>
        <label>암호<input name="password" type="password" autoComplete="current-password" required /></label>
        {error ? <div className="form-error">{error}</div> : null}
        <button type="submit" disabled={busy}><LogIn size={17} />{busy ? "확인 중" : "로그인"}</button>
      </form>
    </main>
  );
}
