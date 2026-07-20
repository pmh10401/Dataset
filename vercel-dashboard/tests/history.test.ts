import { describe, expect, it } from "vitest";

import { shouldWriteHistory } from "../src/lib/history";

describe("history sampling", () => {
  it("writes history on meaningful status changes", () => {
    expect(
      shouldWriteHistory(
        { running: true, active_experiment: "exp1", datasets_done: 20, error_count: 0 },
        { running: true, active_experiment: "exp1", datasets_done: 21, error_count: 0 },
        5,
      ),
    ).toBe(true);
  });

  it("samples unchanged status only after 60 seconds", () => {
    const status = { running: true, active_experiment: "exp1", datasets_done: 20, error_count: 0 };
    expect(shouldWriteHistory(status, status, 59)).toBe(false);
    expect(shouldWriteHistory(status, status, 60)).toBe(true);
  });
});
