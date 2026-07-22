import { describe, expect, it } from "vitest";

import { hasValidBearerToken } from "../src/lib/auth";

describe("ingest authentication", () => {
  it("accepts only the exact bearer token", () => {
    expect(hasValidBearerToken("Bearer correct-token", "correct-token")).toBe(true);
    expect(hasValidBearerToken("Bearer wrong-token", "correct-token")).toBe(false);
    expect(hasValidBearerToken(null, "correct-token")).toBe(false);
  });
});
