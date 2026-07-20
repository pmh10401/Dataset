import { describe, expect, it } from "vitest";

import { buildShadowReport, type ShadowReportRun } from "../src/lib/shadow-metrics";

function run(overrides: Partial<ShadowReportRun> = {}): ShadowReportRun {
  return {
    run_id: "run-1",
    candidate_count: 2,
    hard_count: 1,
    shadow_count: 1,
    priority_count: 0,
    standard_count: 0,
    candidates: [
      { candidate_id: "run-1:0", final_lane: "Hard", verdict: "true_anomaly" },
      { candidate_id: "run-1:1", final_lane: "Shadow", verdict: "false_alarm" },
    ],
    run_feedback: null,
    ...overrides,
  };
}

describe("shadow validation metrics", () => {
  it("does not expose recall or f1 for candidate-only feedback", () => {
    const report = buildShadowReport([run()]);
    expect(report.candidate_precision).toBe(0.5);
    expect(report.recall).toBeNull();
    expect(report.f1).toBeNull();
    expect(report.metric_scope).toBe("candidate_feedback_only");
  });

  it("counts candidate-free runs toward the 100-run target", () => {
    const report = buildShadowReport([run({ candidate_count: 0, hard_count: 0, shadow_count: 0, candidates: [] })]);
    expect(report.runs_collected).toBe(1);
    expect(report.progress_percent).toBe(1);
  });

  it("computes recall and f1 only from fully annotated runs", () => {
    const report = buildShadowReport([
      run({ run_feedback: { missed_anomaly_state: "exists", missed_anomaly_count: 1 } }),
    ]);
    expect(report.metric_scope).toBe("fully_annotated_human_assisted");
    expect(report.recall).toBeCloseTo(0.5);
    expect(report.f1).toBeCloseTo(0.5);
    expect(report.fully_annotated_runs).toBe(1);
  });

  it("excludes deferred candidate feedback from full annotation", () => {
    const candidateDeferred = run({
      candidates: [
        { candidate_id: "run-1:0", final_lane: "Hard", verdict: "true_anomaly" },
        { candidate_id: "run-1:1", final_lane: "Shadow", verdict: "deferred" },
      ],
      run_feedback: { missed_anomaly_state: "none", missed_anomaly_count: 0 },
    });
    const report = buildShadowReport([candidateDeferred]);
    expect(report.fully_annotated_runs).toBe(0);
    expect(report.recall).toBeNull();
  });

  it("keeps autonomous and review lane statistics separate", () => {
    const report = buildShadowReport([run()]);
    expect(report.lanes.Hard).toMatchObject({ candidates: 1, true_anomaly: 1, false_alarm: 0 });
    expect(report.lanes.Shadow).toMatchObject({ candidates: 1, true_anomaly: 0, false_alarm: 1 });
    expect(report.autonomous_scope).toBe("Hard_only");
  });
});
