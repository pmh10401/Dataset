"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowLeft, ChevronDown, ChevronUp, RefreshCw, ShieldCheck } from "lucide-react";

import {
  SHADOW_RANKING_ALPHAS,
  type ShadowRankingAlpha,
} from "@/src/lib/shadow-ranking";
import {
  buildRankingRows,
  DEFAULT_SHADOW_RANKING_ALPHA,
} from "@/src/lib/shadow-ranking-view";

type RankingRun = Parameters<typeof buildRankingRows>[0][number];
type RankingResponse = {
  datasets: number;
  candidates: number;
  runs: RankingRun[];
  retrospective: true;
  prospective: false;
};

export default function ShadowRankingComparison({ onClose }: { onClose: () => void }) {
  const [payload, setPayload] = useState<RankingResponse | null>(null);
  const [alpha, setAlpha] = useState<ShadowRankingAlpha>(DEFAULT_SHADOW_RANKING_ALPHA);
  const [datasetFilter, setDatasetFilter] = useState("");
  const [tier, setTier] = useState("all");
  const [direction, setDirection] = useState("all");
  const [showControls, setShowControls] = useState(false);
  const [message, setMessage] = useState("불러오는 중");

  const load = useCallback(async () => {
    const response = await fetch("/api/shadow/rankings", { cache: "no-store" });
    if (!response.ok) {
      setMessage("순위 자료를 불러오지 못했습니다");
      return;
    }
    setPayload(await response.json());
    setMessage("최신 상태");
  }, []);

  useEffect(() => {
    load();
    const timer = window.setInterval(load, 30000);
    return () => window.clearInterval(timer);
  }, [load]);

  const rows = useMemo(() => buildRankingRows(payload?.runs ?? [], alpha).filter((row) =>
    (!datasetFilter || row.dataset.toLowerCase().includes(datasetFilter.toLowerCase()))
    && (tier === "all" || row.exp292_tier === Number(tier))
    && (direction === "all" || row.movement.direction === direction),
  ), [payload, alpha, datasetFilter, tier, direction]);

  return <main className="dashboard-shell shadow-validation shadow-ranking">
    <header className="topbar">
      <div><span className="eyebrow">EXP397 SHADOW RANKING</span><h1>검토 순위 비교</h1></div>
      <div className="top-actions"><span className="connection fresh">{message}</span><button className="icon-button" onClick={load} title="새로고침"><RefreshCw size={18} /></button><button className="icon-button" onClick={onClose} title="Shadow 검토 화면"><ArrowLeft size={18} /></button></div>
    </header>

    <div className="shadow-notice ranking-notice"><ShieldCheck size={18} /><strong>과거 데이터 재생 · 실제 운영 검증 아님</strong><span>자동 경보 2,218건과 기존 검토 lane은 바꾸지 않고 검토 순서만 비교합니다.</span></div>

    <section className="ranking-summary">
      <div><span>데이터셋</span><strong>{payload?.datasets?.toLocaleString("ko-KR") ?? "-"}</strong></div>
      <div><span>Standard 후보</span><strong>{payload?.candidates?.toLocaleString("ko-KR") ?? "-"}</strong></div>
      <div><span>현재 운영 순위</span><strong>Exp292</strong></div>
      <div><span>Shadow 비교</span><strong>Tier 유지 문법 근거</strong></div>
    </section>

    <section className="ranking-controls">
      <label>Alpha<select value={alpha} onChange={(event) => setAlpha(event.target.value as ShadowRankingAlpha)}>{SHADOW_RANKING_ALPHAS.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
      <label>데이터셋<input value={datasetFilter} onChange={(event) => setDatasetFilter(event.target.value)} placeholder="이름 검색" /></label>
      <label>기존 등급<select value={tier} onChange={(event) => setTier(event.target.value)}><option value="all">전체</option><option value="1">1등급</option><option value="2">2등급</option><option value="3">3등급</option></select></label>
      <label>순위 변화<select value={direction} onChange={(event) => setDirection(event.target.value)}><option value="all">전체</option><option value="up">상승</option><option value="same">동일</option><option value="down">하락</option></select></label>
      <button className="control-toggle" onClick={() => setShowControls((value) => !value)}>{showControls ? <ChevronUp size={16} /> : <ChevronDown size={16} />}연구 대조군 보기</button>
      <span>{rows.length.toLocaleString("ko-KR")}개 후보</span>
    </section>

    <section className="ranking-table-wrap">
      <table className="ranking-table">
        <thead><tr><th>데이터셋</th><th>후보</th><th>기존 등급</th><th>Exp292 순위</th><th>Shadow 순위</th><th>변화</th><th>문법 근거</th><th>최소 p-value</th>{showControls ? <><th>Additive</th><th>Grammar-only</th></> : null}</tr></thead>
        <tbody>{rows.map((row) => <tr key={row.candidate_id}>
          <td>{row.dataset}</td><td>#{row.candidate_index}</td><td>{row.exp292_tier}</td><td>{row.exp292_rank}</td><td><strong>{row.shadowRank}</strong></td>
          <td className={`movement-${row.movement.direction}`}>{row.movement.delta > 0 ? `+${row.movement.delta}` : row.movement.delta}</td>
          <td>{row.supportCount}/3</td><td>{row.minimumPValue.toFixed(4)}</td>
          {showControls ? <><td>{row.control_rank.additive_grammar_family[alpha]}</td><td>{row.control_rank.grammar_only_control[alpha]}</td></> : null}
        </tr>)}</tbody>
      </table>
      {!rows.length ? <div className="ranking-empty">표시할 순위 후보가 없습니다.</div> : null}
    </section>
    <p className="ranking-footnote">Alpha 네 값은 모두 사전등록된 연구 비교값이며, 이 화면은 특정 값을 추천하거나 선택하지 않습니다.</p>
  </main>;
}
