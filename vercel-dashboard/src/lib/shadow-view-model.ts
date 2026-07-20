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

export function metricDisplay(value: number | null | undefined, _metric: "precision" | "recall" | "f1") {
  return value === null || value === undefined ? "계산 전" : `${(value * 100).toFixed(1)}%`;
}
