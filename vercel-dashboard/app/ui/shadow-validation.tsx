"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertCircle, ArrowLeft, Check, CircleHelp, Clock3, Database, Eye, RefreshCw, ShieldCheck, X } from "lucide-react";

import type { ShadowLane } from "@/src/lib/shadow";
import type { ShadowReport } from "@/src/lib/shadow-metrics";
import { collectionStatus, metricDisplay, progressLabel, shadowKoreanLabel, sortCandidates, type ShadowCandidateView } from "@/src/lib/shadow-view-model";
import { createBrowserSupabaseClient } from "@/src/lib/supabase/browser";

type Feedback = { id: number; verdict: "true_anomaly" | "false_alarm" | "deferred" } | null;
type Candidate = ShadowCandidateView & { source_evidence: Record<string, { agrees: boolean; score: number | null; threshold: number }>; feedback: Feedback };
type Run = {
  run_id: string; observed_at: string; dataset_key: string; equipment_id: string | null; recipe_id: string | null;
  sensor_id: string | null; step_id: string | null; n_train: number; n_test: number; has_missing: boolean;
  missing_quality_status: string; series_preview: { indices: number[]; values: number[] }; candidates: Candidate[];
  run_feedback: { id: number; missed_anomaly_state: "none" | "exists" | "unknown"; missed_anomaly_count: number | null } | null;
};

const TABS: { label: string; lane: ShadowLane | "all" }[] = [
  { label: "전체", lane: "all" }, { label: "Shadow", lane: "Shadow" }, { label: "우선 검토", lane: "Priority" },
  { label: "자동 경보", lane: "Hard" }, { label: "일반 검토", lane: "Standard" },
];

function Waveform({ run, index }: { run: Run; index: number }) {
  const width = 720; const height = 180; const pad = 14;
  const values = run.series_preview.values;
  const indices = run.series_preview.indices;
  const min = Math.min(...values); const max = Math.max(...values); const span = max - min || 1;
  const x = (position: number) => pad + (position / Math.max(1, values.length - 1)) * (width - pad * 2);
  const y = (value: number) => height - pad - ((value - min) / span) * (height - pad * 2);
  const points = values.map((value, position) => `${x(position)},${y(value)}`).join(" ");
  const position = Math.max(0, indices.indexOf(index));
  return <svg className="shadow-wave" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`후보 ${index} 주변 시계열`}>
    <line x1={pad} y1={height / 2} x2={width - pad} y2={height / 2} className="wave-zero" />
    <polyline points={points} className="wave-line" />
    <line x1={x(position)} y1={pad} x2={x(position)} y2={height - pad} className="wave-marker" />
    <circle cx={x(position)} cy={y(values[position] ?? 0)} r="5" className="wave-point" />
  </svg>;
}

export default function ShadowValidation() {
  const [report, setReport] = useState<ShadowReport | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [activeLane, setActiveLane] = useState<ShadowLane | "all">("all");
  const [confidence, setConfidence] = useState("all");
  const [missing, setMissing] = useState("all");
  const [verdict, setVerdict] = useState("all");
  const [selected, setSelected] = useState<string | null>(null);
  const [message, setMessage] = useState("불러오는 중");
  const [missedCount, setMissedCount] = useState(1);

  const load = useCallback(async () => {
    const [reportResponse, runsResponse] = await Promise.all([
      fetch("/api/shadow/report", { cache: "no-store" }),
      fetch("/api/shadow/runs?limit=100", { cache: "no-store" }),
    ]);
    if (!reportResponse.ok || !runsResponse.ok) { setMessage("Shadow 검증 자료를 불러오지 못했습니다"); return; }
    setReport(await reportResponse.json());
    const data = await runsResponse.json();
    setRuns(data.runs ?? []);
    setMessage("최신 상태");
  }, []);

  useEffect(() => {
    const supabase = createBrowserSupabaseClient();
    load();
    const fallback = window.setInterval(load, 30000);
    const channel = supabase.channel("shadow-validation")
      .on("postgres_changes", { event: "*", schema: "public", table: "shadow_validation_runs" }, load)
      .on("postgres_changes", { event: "*", schema: "public", table: "shadow_feedback_events" }, load)
      .on("postgres_changes", { event: "*", schema: "public", table: "shadow_run_feedback_events" }, load)
      .subscribe();
    return () => { window.clearInterval(fallback); supabase.removeChannel(channel); };
  }, [load]);

  const candidates = useMemo(() => sortCandidates(runs.flatMap((run) => run.candidates.map((candidate) => ({
    ...candidate, verdict: candidate.feedback?.verdict ?? null, run,
  }))).filter((candidate) =>
    (activeLane === "all" || candidate.final_lane === activeLane)
    && (confidence === "all" || candidate.confidence_grade === confidence)
    && (missing === "all" || candidate.run.has_missing === (missing === "yes"))
    && (verdict === "all" || (verdict === "unreviewed" ? candidate.verdict === null : candidate.verdict === verdict)),
  )), [runs, activeLane, confidence, missing, verdict]);
  const selectedCandidate = candidates.find((candidate) => candidate.candidate_id === selected) ?? candidates[0] ?? null;
  const collection = collectionStatus(report?.runs_collected ?? 0);

  async function submitCandidate(candidate: typeof selectedCandidate, nextVerdict: "true_anomaly" | "false_alarm" | "deferred") {
    if (!candidate) return;
    if (candidate.verdict && !window.confirm("기존 판정을 새 정정 기록으로 바꾸시겠습니까?")) return;
    const response = await fetch(`/api/shadow/candidates/${encodeURIComponent(candidate.candidate_id)}/feedback`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ verdict: nextVerdict, corrects_event_id: candidate.feedback?.id ?? null }),
    });
    setMessage(response.ok ? "판정을 기록했습니다" : "판정을 저장하지 못했습니다");
    if (response.ok) await load();
  }

  async function submitRun(state: "none" | "exists" | "unknown") {
    if (!selectedCandidate) return;
    const run = selectedCandidate.run;
    if (run.run_feedback && !window.confirm("기존 run 판정을 새 정정 기록으로 바꾸시겠습니까?")) return;
    const response = await fetch(`/api/shadow/runs/${encodeURIComponent(run.run_id)}/feedback`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ missed_anomaly_state: state, missed_anomaly_count: state === "none" ? 0 : state === "exists" ? missedCount : null, corrects_event_id: run.run_feedback?.id ?? null }),
    });
    setMessage(response.ok ? "run 판정을 기록했습니다" : "run 판정을 저장하지 못했습니다");
    if (response.ok) await load();
  }

  return <main className="dashboard-shell shadow-validation">
    <header className="topbar">
      <div><span className="eyebrow">EXP265 PROSPECTIVE SHADOW</span><h1>신규 run 검증</h1></div>
      <div className="top-actions"><span className="connection fresh">{message}</span><button className="icon-button" onClick={load} title="새로고침"><RefreshCw size={18} /></button><Link className="icon-button" href="/" title="실험 대시보드"><ArrowLeft size={18} /></Link></div>
    </header>
    <section className="shadow-summary">
      <div className="shadow-progress"><span>{progressLabel(report?.runs_collected ?? 0)}</span><strong>{report?.progress_percent ?? 0}%</strong><div><i style={{ width: `${report?.progress_percent ?? 0}%` }} /></div></div>
      <div className="shadow-kpis">
        <div><span>자동 경보</span><strong>{report?.lanes.Hard.candidates ?? 0}</strong></div>
        <div><span>Shadow</span><strong>{report?.lanes.Shadow.candidates ?? 0}</strong></div>
        <div><span>검토 완료</span><strong>{report?.reviewed_candidates ?? 0}</strong></div>
        <div><span>후보 Precision</span><strong>{metricDisplay(report?.candidate_precision, "precision")}</strong></div>
      </div>
    </section>
    <div className={`shadow-collection-state ${collection.state}`}><Database size={18} /><div><strong>{collection.title}</strong><span>{collection.description}</span></div></div>
    <div className="shadow-notice"><ShieldCheck size={18} /><strong>Shadow는 자동 경보가 아닙니다.</strong><span>사용자 검토를 위한 관찰 후보이며 Exp253 자동 경보는 변경하지 않습니다.</span></div>
    <nav className="shadow-tabs" aria-label="검토 lane">{TABS.map((tab) => <button key={tab.lane} className={activeLane === tab.lane ? "active" : ""} onClick={() => setActiveLane(tab.lane)}>{tab.label}</button>)}</nav>
    <section className="shadow-filters">
      <label>신뢰도<select value={confidence} onChange={(event) => setConfidence(event.target.value)}><option value="all">전체</option><option>A</option><option>B</option><option>C</option></select></label>
      <label>결측<select value={missing} onChange={(event) => setMissing(event.target.value)}><option value="all">전체</option><option value="yes">있음</option><option value="no">없음</option></select></label>
      <label>판정<select value={verdict} onChange={(event) => setVerdict(event.target.value)}><option value="all">전체</option><option value="unreviewed">미검토</option><option value="deferred">보류</option><option value="true_anomaly">실제 이상</option><option value="false_alarm">오탐</option></select></label>
      <span>{candidates.length}개 후보</span>
    </section>
    <section className="shadow-workspace">
      <div className="shadow-candidate-list" aria-label="후보 목록">{candidates.map((candidate) => <button key={candidate.candidate_id} className={selectedCandidate?.candidate_id === candidate.candidate_id ? "selected" : ""} onClick={() => setSelected(candidate.candidate_id)}>
        <span className={`lane-dot lane-${candidate.final_lane.toLowerCase()}`} />
        <span><strong>{shadowKoreanLabel(candidate.final_lane)} · #{candidate.candidate_index}</strong><small>{candidate.run.dataset_key}</small></span>
        <span><b>{candidate.confidence_grade ?? `${candidate.support_count}/4`}</b><small>{shadowKoreanLabel(candidate.verdict ?? "unreviewed")}</small></span>
      </button>)}</div>
      <div className="shadow-detail">{selectedCandidate ? <>
        <div className="shadow-detail-title"><div><span>{shadowKoreanLabel(selectedCandidate.final_lane)}</span><h2>{selectedCandidate.run.dataset_key} · #{selectedCandidate.candidate_index}</h2></div><span className="missing-badge">{shadowKoreanLabel(selectedCandidate.run.has_missing ? "missing_present" : "missing_absent")}</span></div>
        <Waveform run={selectedCandidate.run} index={selectedCandidate.candidate_index} />
        <div className="evidence-strip">{Object.entries(selectedCandidate.source_evidence).map(([name, evidence]) => <div key={name} className={evidence.agrees ? "agrees" : ""}><span>{name}</span><strong>{evidence.agrees ? "동의" : "비동의"}</strong><small>{evidence.score == null ? "-" : `${evidence.score.toFixed(3)} / ${evidence.threshold.toFixed(3)}`}</small></div>)}</div>
        <div className="verdict-actions"><button onClick={() => submitCandidate(selectedCandidate, "true_anomaly")}><Check size={17} />실제 이상</button><button onClick={() => submitCandidate(selectedCandidate, "false_alarm")}><X size={17} />오탐</button><button onClick={() => submitCandidate(selectedCandidate, "deferred")}><Clock3 size={17} />보류</button></div>
        <div className="run-feedback"><div><strong>이 run에서 놓친 이상</strong><button className="help-icon" title="Recall/F1은 모든 후보 판정과 run의 정확한 누락 개수가 있어야 계산됩니다."><CircleHelp size={15} /></button></div><div className="run-feedback-controls"><button onClick={() => submitRun("none")}>없음</button><span><input type="number" min="1" value={missedCount} onChange={(event) => setMissedCount(Math.max(1, Number(event.target.value) || 1))} /><button onClick={() => submitRun("exists")}>있음</button></span><button onClick={() => submitRun("unknown")}>확인 불가</button></div></div>
      </> : <div className="shadow-empty"><Eye size={24} /><strong>검토할 후보가 없습니다</strong></div>}</div>
    </section>
    <section className="shadow-metric-footer"><AlertCircle size={17} /><span>Recall {metricDisplay(report?.recall, "recall")} · F1 {metricDisplay(report?.f1, "f1")}</span><small>{report?.metric_scope === "fully_annotated_human_assisted" ? "완전 주석된 run의 사람 보조 지표" : "run 누락 판정이 완료되기 전에는 계산하지 않습니다"}</small></section>
  </main>;
}
