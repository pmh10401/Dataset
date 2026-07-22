import type { ShadowLane } from "./shadow";

export type CandidateVerdict = "true_anomaly" | "false_alarm" | "deferred" | null;
export type MissedAnomalyState = "none" | "exists" | "unknown";

export interface ShadowReportCandidate {
  candidate_id: string;
  final_lane: ShadowLane;
  verdict: CandidateVerdict;
}

export interface ShadowReportRun {
  run_id: string;
  observed_at?: string;
  collection_mode?: string;
  policy_version?: string;
  config_sha256?: string;
  candidate_count: number;
  hard_count: number;
  shadow_count: number;
  priority_count: number;
  standard_count: number;
  candidates: ShadowReportCandidate[];
  run_feedback: { missed_anomaly_state: MissedAnomalyState; missed_anomaly_count: number | null } | null;
}

export interface LaneMetrics {
  candidates: number;
  reviewed: number;
  deferred: number;
  true_anomaly: number;
  false_alarm: number;
  precision: number | null;
}

export interface ShadowReport {
  runs_collected: number;
  target_runs: number;
  progress_percent: number;
  candidate_free_runs: number;
  candidates: number;
  reviewed_candidates: number;
  deferred_candidates: number;
  true_anomaly: number;
  false_alarm: number;
  candidate_precision: number | null;
  candidate_feedback_coverage: number | null;
  fully_annotated_runs: number;
  fully_annotated_precision: number | null;
  recall: number | null;
  f1: number | null;
  metric_scope: "collection_only" | "candidate_feedback_only" | "fully_annotated_human_assisted";
  autonomous_scope: "Hard_only";
  lanes: Record<ShadowLane, LaneMetrics>;
}

const LANES: ShadowLane[] = ["Hard", "Shadow", "Priority", "Standard"];

function ratio(numerator: number, denominator: number) {
  return denominator > 0 ? numerator / denominator : null;
}

function laneMetrics(candidates: ShadowReportCandidate[], lane: ShadowLane): LaneMetrics {
  const rows = candidates.filter((candidate) => candidate.final_lane === lane);
  const trueAnomaly = rows.filter((candidate) => candidate.verdict === "true_anomaly").length;
  const falseAlarm = rows.filter((candidate) => candidate.verdict === "false_alarm").length;
  return {
    candidates: rows.length,
    reviewed: trueAnomaly + falseAlarm,
    deferred: rows.filter((candidate) => candidate.verdict === "deferred").length,
    true_anomaly: trueAnomaly,
    false_alarm: falseAlarm,
    precision: ratio(trueAnomaly, trueAnomaly + falseAlarm),
  };
}

function isFullyAnnotated(run: ShadowReportRun) {
  if (run.candidates.length !== run.candidate_count) return false;
  if (run.candidates.some((candidate) => candidate.verdict !== "true_anomaly" && candidate.verdict !== "false_alarm")) return false;
  const feedback = run.run_feedback;
  if (!feedback) return false;
  return (feedback.missed_anomaly_state === "none" && feedback.missed_anomaly_count === 0)
    || (feedback.missed_anomaly_state === "exists" && Number.isInteger(feedback.missed_anomaly_count) && (feedback.missed_anomaly_count ?? 0) > 0);
}

export function buildShadowReport(runs: ShadowReportRun[]): ShadowReport {
  const candidates = runs.flatMap((run) => run.candidates);
  const trueAnomaly = candidates.filter((candidate) => candidate.verdict === "true_anomaly").length;
  const falseAlarm = candidates.filter((candidate) => candidate.verdict === "false_alarm").length;
  const deferred = candidates.filter((candidate) => candidate.verdict === "deferred").length;
  const fullyAnnotated = runs.filter(isFullyAnnotated);
  const eligibleCandidates = fullyAnnotated.flatMap((run) => run.candidates);
  const eligibleTp = eligibleCandidates.filter((candidate) => candidate.verdict === "true_anomaly").length;
  const eligibleFp = eligibleCandidates.filter((candidate) => candidate.verdict === "false_alarm").length;
  const eligibleMissed = fullyAnnotated.reduce((total, run) => total + (run.run_feedback?.missed_anomaly_count ?? 0), 0);
  const eligiblePrecision = ratio(eligibleTp, eligibleTp + eligibleFp);
  const recall = fullyAnnotated.length ? ratio(eligibleTp, eligibleTp + eligibleMissed) : null;
  const f1 = eligiblePrecision !== null && recall !== null && eligiblePrecision + recall > 0
    ? (2 * eligiblePrecision * recall) / (eligiblePrecision + recall)
    : eligiblePrecision === 0 && recall === 0 ? 0 : null;
  const reviewed = trueAnomaly + falseAlarm;
  return {
    runs_collected: runs.length,
    target_runs: 100,
    progress_percent: Math.min(100, runs.length),
    candidate_free_runs: runs.filter((run) => run.candidate_count === 0).length,
    candidates: candidates.length,
    reviewed_candidates: reviewed,
    deferred_candidates: deferred,
    true_anomaly: trueAnomaly,
    false_alarm: falseAlarm,
    candidate_precision: ratio(trueAnomaly, reviewed),
    candidate_feedback_coverage: ratio(reviewed, candidates.length),
    fully_annotated_runs: fullyAnnotated.length,
    fully_annotated_precision: eligiblePrecision,
    recall,
    f1,
    metric_scope: fullyAnnotated.length ? "fully_annotated_human_assisted" : reviewed ? "candidate_feedback_only" : "collection_only",
    autonomous_scope: "Hard_only",
    lanes: Object.fromEntries(LANES.map((lane) => [lane, laneMetrics(candidates, lane)])) as Record<ShadowLane, LaneMetrics>,
  };
}
