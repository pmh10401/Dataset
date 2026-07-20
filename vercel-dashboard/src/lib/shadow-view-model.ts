import type { ConfidenceGrade, ShadowLane } from "./shadow";
import type { CandidateVerdict } from "./shadow-metrics";

export interface ShadowCandidateView {
  candidate_id: string;
  candidate_index: number;
  final_lane: ShadowLane;
  confidence_grade: ConfidenceGrade | null;
  verdict: CandidateVerdict;
  quality_flagged: boolean;
  support_count: number;
}

const LANE_ORDER: Record<ShadowLane, number> = { Shadow: 0, Priority: 1, Hard: 2, Standard: 3 };
const CONFIDENCE_ORDER: Record<string, number> = { A: 0, B: 1, C: 2 };
const VERDICT_ORDER: Record<string, number> = { unreviewed: 0, deferred: 1, true_anomaly: 2, false_alarm: 2 };

export function sortCandidates<T extends ShadowCandidateView>(rows: T[]): T[] {
  return [...rows].sort((left, right) =>
    LANE_ORDER[left.final_lane] - LANE_ORDER[right.final_lane]
    || (CONFIDENCE_ORDER[left.confidence_grade ?? ""] ?? 3) - (CONFIDENCE_ORDER[right.confidence_grade ?? ""] ?? 3)
    || (VERDICT_ORDER[left.verdict ?? "unreviewed"] ?? 3) - (VERDICT_ORDER[right.verdict ?? "unreviewed"] ?? 3)
    || left.candidate_index - right.candidate_index,
  );
}

const LABELS: Record<string, string> = {
  Hard: "자동 경보",
  Shadow: "Shadow 검토",
  Priority: "우선 검토",
  Standard: "일반 검토",
  true_anomaly: "실제 이상",
  false_alarm: "오탐",
  deferred: "판정 보류",
  unreviewed: "미검토",
  missing_present: "결측 있음",
  missing_absent: "결측 없음",
};

export function shadowKoreanLabel(value: string) {
  return LABELS[value] ?? value;
}

export function progressLabel(runsCollected: number) {
  return `신규 run 검증 ${runsCollected}/100`;
}

export function collectionStatus(runsCollected: number) {
  if (runsCollected <= 0) {
    return {
      state: "waiting" as const,
      title: "개발 준비 완료 · 신규 데이터 대기",
      description: "개발 과정에는 신규 run이 없으므로 성능 검증은 시작 전입니다. 새 run이 들어오면 자동으로 집계합니다.",
    };
  }
  if (runsCollected < 100) {
    return {
      state: "collecting" as const,
      title: "신규 데이터 검증 진행 중",
      description: "실제 신규 run만 누적하고 있으며 과거 TEST 데이터는 새 검증으로 계산하지 않습니다.",
    };
  }
  return {
    state: "complete" as const,
    title: "1차 신규 데이터 수집 목표 도달",
    description: "수집 결과는 사용자 판정이 완료된 범위에서만 성능 지표로 계산합니다.",
  };
}

export function metricDisplay(value: number | null | undefined, _metric: "precision" | "recall" | "f1") {
  return value === null || value === undefined ? "계산 전" : `${(value * 100).toFixed(1)}%`;
}
