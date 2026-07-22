import { describe, expect, it } from "vitest";

import { collectPages } from "../src/lib/shadow-data";

describe("shadow report pagination", () => {
  it("collects rows beyond the Supabase 1000-row boundary", async () => {
    const source = Array.from({ length: 1117 }, (_, index) => index);
    const requested: Array<[number, number]> = [];
    const rows = await collectPages(async (from, to) => {
      requested.push([from, to]);
      return { data: source.slice(from, to + 1), error: null };
    });

    expect(rows).toHaveLength(1117);
    expect(rows.at(-1)).toBe(1116);
    expect(requested).toEqual([[0, 999], [1000, 1999]]);
  });

  it("stops after an exact final page and one empty page", async () => {
    const source = Array.from({ length: 2000 }, (_, index) => index);
    let calls = 0;
    const rows = await collectPages(async (from, to) => {
      calls += 1;
      return { data: source.slice(from, to + 1), error: null };
    });
    expect(rows).toHaveLength(2000);
    expect(calls).toBe(3);
  });
});
