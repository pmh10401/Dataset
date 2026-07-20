import { describe, expect, it } from "vitest";

import { metricDisplay, progressLabel, shadowKoreanLabel, sortCandidates, type ShadowCandidateView } from "../src/lib/shadow-view-model";

function candidate(final_lane: ShadowCandidateView["final_lane"], confidence_grade: ShadowCandidateView["confidence_grade"] = null, verdict: ShadowCandidateView["verdict"] = null): ShadowCandidateView {
  return { candidate_id: `${final_lane}-${confidence_grade}-${verdict}`, candidate_index: 0, final_lane, confidence_grade, verdict, quality_flagged: false, support_count: 0 };
}

describe("shadow dashboard view model", () => {
  it("sorts review urgency, Hard confidence, and feedback state", () => {
    const rows = [candidate("Standard"), candidate("Hard", "B", "true_anomaly"), candidate("Priority"), candidate("Hard", "A"), candidate("Shadow")];
    expect(sortCandidates(rows).map((row) => row.final_lane)).toEqual(["Shadow", "Priority", "Hard", "Hard", "Standard"]);
    expect(sortCandidates(rows).filter((row) => row.final_lane === "Hard").map((row) => row.confidence_grade)).toEqual(["A", "B"]);
  });

  it("provides concise Korean labels and missing badge", () => {
    expect(shadowKoreanLabel("Shadow")).toBe("Shadow 검토");
    expect(shadowKoreanLabel("true_anomaly")).toBe("실제 이상");
    expect(shadowKoreanLabel("missing_present")).toBe("결측 있음");
  });

  it("formats the fixed 100-run target", () => {
    expect(progressLabel(37)).toBe("신규 run 검증 37/100");
  });

  it("hides recall and f1 when the report is not eligible", () => {
    expect(metricDisplay(null, "recall")).toBe("계산 전");
    expect(metricDisplay(0.84339, "precision")).toBe("84.3%");
  });
});
