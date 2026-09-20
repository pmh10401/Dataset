import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  buildRankingRows,
  DEFAULT_SHADOW_RANKING_ALPHA,
} from "../src/lib/shadow-ranking-view";

describe("Exp397 Shadow ranking view", () => {
  it("defaults to the first preregistered alpha without selecting a best policy", () => {
    expect(DEFAULT_SHADOW_RANKING_ALPHA).toBe("0.005");
    const source = readFileSync(join(process.cwd(), "app/ui/shadow-ranking-comparison.tsx"), "utf8");
    expect(source).toContain("과거 데이터 재생 · 실제 운영 검증 아님");
    expect(source).toContain("연구 대조군 보기");
    expect(source.toLowerCase()).not.toContain("best alpha");
    expect(source).not.toContain("최적 alpha");
  });

  it("sorts candidates by the selected Shadow rank and reports movement", () => {
    const rows = buildRankingRows([{
      run_id: "run-a",
      dataset_key: "A",
      candidates: [{
        candidate_id: "run-a:2",
        run_id: "run-a",
        candidate_index: 2,
        exp292_tier: 1,
        exp292_rank: 100,
        tier_preserving_rank: { "0.005": 30, "0.01": 40, "0.02": 50, "0.05": 60 },
        control_rank: {
          additive_grammar_family: { "0.005": 20, "0.01": 30, "0.02": 40, "0.05": 50 },
          grammar_only_control: { "0.005": 10, "0.01": 20, "0.02": 30, "0.05": 40 },
        },
        grammar_support_count: { "0.005": 1, "0.01": 1, "0.02": 2, "0.05": 3 },
        p_values: { transition: 0.002, dwell: 0.4, recovery: 0.3, composite: 0.01 },
      }],
    }], "0.005");
    expect(rows[0].shadowRank).toBe(30);
    expect(rows[0].movement).toEqual({ delta: 70, direction: "up" });
  });
});
