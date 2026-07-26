import { describe, expect, it } from "vitest";

import { ALPHA_PRESETS, parseAlphaRequest } from "../src/lib/alpha-settings";

describe("operational alpha policy", () => {
  it("exposes only the two preregistered confidence presets", () => {
    expect(ALPHA_PRESETS.map((preset) => preset.value)).toEqual([0.005, 0.01]);
    expect(ALPHA_PRESETS.map((preset) => preset.strongAlertCount)).toEqual([228, 361]);
  });

  it("accepts a confirmed preset with a trimmed reason", () => {
    expect(parseAlphaRequest({
      alpha: 0.01,
      reason: "  검토 범위를 확대하기 위해 변경  ",
      acknowledged: true,
    })).toEqual({
      alpha: 0.01,
      reason: "검토 범위를 확대하기 위해 변경",
    });
  });

  it("rejects arbitrary alpha values", () => {
    expect(() => parseAlphaRequest({
      alpha: 0.02,
      reason: "임의 값을 사용하지 않음",
      acknowledged: true,
    })).toThrow(/alpha/i);
  });

  it("requires a meaningful reason and explicit confirmation", () => {
    expect(() => parseAlphaRequest({
      alpha: 0.005,
      reason: "짧음",
      acknowledged: true,
    })).toThrow(/reason/i);
    expect(() => parseAlphaRequest({
      alpha: 0.005,
      reason: "오탐 억제 우선 정책으로 변경",
      acknowledged: false,
    })).toThrow(/confirm/i);
  });

  it("rejects oversized reasons", () => {
    expect(() => parseAlphaRequest({
      alpha: 0.005,
      reason: "가".repeat(501),
      acknowledged: true,
    })).toThrow(/reason/i);
  });
});
