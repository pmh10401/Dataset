import { describe, expect, it } from "vitest";

import { modeCopy, parseShadowCollectionMode } from "../src/lib/shadow-mode";

describe("shadow collection modes", () => {
  it("keeps prospective collection as the safe default", () => {
    expect(parseShadowCollectionMode(null)).toBe("prospective_shadow");
    expect(parseShadowCollectionMode("unknown")).toBeNull();
  });

  it("labels historical experiment data as retrospective replay", () => {
    expect(parseShadowCollectionMode("retrospective_replay")).toBe("retrospective_replay");
    expect(modeCopy("retrospective_replay")).toMatchObject({
      title: "개발용 재생",
      prospective: false,
    });
  });
});
