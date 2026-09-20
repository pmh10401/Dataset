import { describe, expect, it } from "vitest";

import { collectRankingPages, rankMovement } from "../src/lib/shadow-ranking-data";

describe("Shadow ranking data helpers", () => {
  it("collects all 1,117 run rows across the Supabase boundary", async () => {
    const source = Array.from({ length: 1117 }, (_, index) => index);
    const requested: Array<[number, number]> = [];
    const rows = await collectRankingPages(async (from, to) => {
      requested.push([from, to]);
      return { data: source.slice(from, to + 1), error: null };
    });
    expect(rows).toHaveLength(1117);
    expect(requested).toEqual([[0, 999], [1000, 1999]]);
  });

  it("uses positive movement for a higher Shadow priority", () => {
    expect(rankMovement(100, 30)).toEqual({ delta: 70, direction: "up" });
    expect(rankMovement(30, 100)).toEqual({ delta: -70, direction: "down" });
    expect(rankMovement(30, 30)).toEqual({ delta: 0, direction: "same" });
  });
});
