"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Clock3, Cpu, Database, LogOut, RefreshCw } from "lucide-react";

import { isSnapshotStale, type Snapshot } from "@/src/lib/snapshot";
import { createBrowserSupabaseClient } from "@/src/lib/supabase/browser";

type StatusResponse = { source_id: string; generated_at: string; received_at: string; snapshot: Snapshot };

function duration(seconds: number | null) {
  if (seconds === null || !Number.isFinite(seconds)) return "-";
  const minutes = Math.floor(seconds / 60);
  return minutes < 60 ? `${minutes}분` : `${Math.floor(minutes / 60)}시간 ${minutes % 60}분`;
}

function numberValue(value: unknown, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export default function Dashboard() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [message, setMessage] = useState("연결 중");
  const [now, setNow] = useState(() => new Date());
  const load = useCallback(async () => {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) {
      setMessage(response.status === 404 ? "아직 전송된 상태가 없습니다" : "상태를 불러오지 못했습니다");
      return;
    }
    setStatus(await response.json());
    setMessage("정상 연결");
  }, []);

  useEffect(() => {
    const supabase = createBrowserSupabaseClient();
    load();
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    const fallback = window.setInterval(load, 30000);
    const channel = supabase
      .channel("dashboard-latest")
      .on("postgres_changes", { event: "*", schema: "public", table: "experiment_status_latest" }, load)
      .subscribe();
    return () => {
      window.clearInterval(timer);
      window.clearInterval(fallback);
      supabase.removeChannel(channel);
    };
  }, [load]);

  const snapshot = status?.snapshot;
  const stale = snapshot ? isSnapshotStale(snapshot, now) : true;
  const queue = snapshot?.queue_items ?? [];
  const completed = snapshot?.completed_results ?? [];
  const errorCount = numberValue(snapshot?.error_summary.total_errors);
  const warningCount = numberValue(snapshot?.warning_summary.total_warnings);
  const cpuInfo = snapshot?.system.cpu as Record<string, unknown> | undefined;
  const gpuInfo = snapshot?.system.gpu as Record<string, unknown> | undefined;
  const cpu = numberValue(cpuInfo?.percent, numberValue(cpuInfo?.cpu_percent));
  const gpu = numberValue(gpuInfo?.percent, numberValue(gpuInfo?.gpu_percent));
  const gpuAvailable = gpuInfo?.available !== false;
  const processCount = numberValue(snapshot?.process_summary.process_count);

  async function logout() {
    const supabase = createBrowserSupabaseClient();
    await supabase.auth.signOut();
    window.location.assign("/login");
  }

  return (
    <main className="dashboard-shell">
      <header className="topbar">
        <div><span className="eyebrow">TIME SERIES RESEARCH</span><h1>실험 진행 대시보드</h1></div>
        <div className="top-actions">
          <span className={`connection ${stale ? "stale" : "fresh"}`}>{stale ? "업데이트 지연" : message}</span>
          <button className="icon-button" onClick={load} title="새로고침"><RefreshCw size={18} /></button>
          <button className="icon-button" onClick={logout} title="로그아웃"><LogOut size={18} /></button>
        </div>
      </header>

      {!snapshot ? <section className="empty-state"><Database size={26} /><strong>{message}</strong><span>Mac 상태 전송기가 연결되면 자동으로 표시됩니다.</span></section> : <>
        <section className="active-band">
          <div className="active-copy"><span>현재 실험</span><h2>{snapshot.active_label || snapshot.active_experiment || "대기 중"}</h2><p>{snapshot.current_dataset || "다음 작업을 기다리고 있습니다."}</p></div>
          <div className="progress-copy"><strong>{snapshot.progress_percent.toFixed(1)}%</strong><span>{snapshot.datasets_done.toLocaleString()} / {snapshot.expected_datasets.toLocaleString()}</span></div>
          <div className="progress-track"><span style={{ width: `${Math.min(100, Math.max(0, snapshot.progress_percent))}%` }} /></div>
        </section>

        <section className="metrics-band">
          <article><Clock3 /><span>남은 시간</span><strong>{duration(snapshot.eta_seconds)}</strong></article>
          <article><Database /><span>최근 처리 속도</span><strong>{numberValue(snapshot.throughput.recent_datasets_per_minute).toFixed(1)} /분</strong></article>
          <article><AlertTriangle /><span>오류 · 경고</span><strong>{errorCount} · {warningCount}</strong></article>
          <article><Cpu /><span>CPU · GPU</span><strong>{cpu.toFixed(0)}% · {gpuAvailable ? `${gpu.toFixed(0)}%` : "확인 불가"}</strong></article>
        </section>

        <section className="content-grid">
          <div className="section-block"><div className="section-title"><h3>실험 대기열</h3><span>{queue.length}개</span></div><div className="table-wrap"><table><thead><tr><th>상태</th><th>실험</th><th>진행률</th></tr></thead><tbody>{queue.slice(0, 12).map((item, index) => <tr key={`${String(item.id)}-${index}`}><td><span className={`status-tag ${String(item.status)}`}>{String(item.status ?? "대기")}</span></td><td>{String(item.label ?? item.id ?? "-")}</td><td>{numberValue(item.progress_percent).toFixed(1)}%</td></tr>)}</tbody></table></div></div>
          <div className="section-block"><div className="section-title"><h3>시스템 상태</h3><span>{snapshot.running ? "실행 중" : "대기"}</span></div><dl className="health-list"><div><dt>마지막 동기화</dt><dd>{new Date(snapshot.generated_at).toLocaleString("ko-KR")}</dd></div><div><dt>실행 시간</dt><dd>{duration(snapshot.elapsed_seconds)}</dd></div><div><dt>Coverage</dt><dd>{numberValue(snapshot.coverage.coverage_percent, snapshot.progress_percent).toFixed(1)}%</dd></div><div><dt>최근 완료</dt><dd>{snapshot.last_completed_label || snapshot.last_completed_experiment || "-"}</dd></div></dl></div>
        </section>

        <section className="section-block completed"><div className="section-title"><h3>최근 완료 결과</h3><span>{completed.length}개 기록 · 프로세스 {processCount}개</span></div><div className="result-list">{completed.slice(0, 8).map((item, index) => <div key={`${String(item.id)}-${index}`}><CheckCircle2 size={16} /><span>{String(item.label ?? item.experiment ?? item.id ?? "완료된 실험")}</span><strong>{item.mean_f1 == null ? "완료" : `F1 ${numberValue(item.mean_f1).toFixed(4)}`}</strong></div>)}</div></section>
      </>}
    </main>
  );
}
