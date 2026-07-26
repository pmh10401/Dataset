import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const root = process.cwd();
const dashboard = readFileSync(join(root, "app/ui/dashboard.tsx"), "utf8");
const settings = readFileSync(join(root, "app/ui/alpha-settings.tsx"), "utf8");
const alphaSettings = readFileSync(join(root, "src/lib/alpha-settings.ts"), "utf8");

describe("alpha settings UI structure", () => {
  it("links to alpha settings from the dashboard toolbar", () => {
    expect(dashboard).toContain('href="/settings/alpha"');
    expect(dashboard).toContain('title="Alpha 설정"');
  });

  it("shows the frozen presets and operational safety language", () => {
    expect(alphaSettings).toContain("0.005");
    expect(alphaSettings).toContain("0.01");
    expect(settings).toContain("자동 경보는 바뀌지 않습니다");
    expect(settings).toContain("현재 적용값");
    expect(settings).toContain("요청값");
  });

  it("requires a reason and explicit acknowledgement", () => {
    expect(settings).toContain("<textarea");
    expect(settings).toContain('type="checkbox"');
    expect(settings).toContain("변경 내용을 확인했습니다");
  });

  it("refreshes pending application state without a page reload", () => {
    expect(settings).toContain("window.setInterval");
    expect(settings).toContain("last_error");
  });
});
