export const ALPHA_PRESETS = [
  {
    value: 0.005,
    key: "strict",
    label: "엄격",
    description: "오탐 억제를 우선하고 가장 강한 근거만 강조합니다.",
    strongAlertCount: 228,
    strongTruePositiveCount: 211,
    strongFalsePositiveCount: 17,
    retrospectivePrecision: 0.9254385964912281,
  },
  {
    value: 0.01,
    key: "balanced",
    label: "균형",
    description: "높은 정밀도를 유지하면서 강한 신뢰 표시 범위를 넓힙니다.",
    strongAlertCount: 361,
    strongTruePositiveCount: 330,
    strongFalsePositiveCount: 31,
    retrospectivePrecision: 0.9141274238227147,
  },
] as const;

export const DEFAULT_ALPHA_SOURCE_ID = "macbook-air";

export type AlphaValue = (typeof ALPHA_PRESETS)[number]["value"];

export type AlphaRequest = {
  alpha: AlphaValue;
  reason: string;
};

export type AlphaSetting = {
  source_id: string;
  desired_alpha: AlphaValue;
  effective_alpha: AlphaValue | null;
  status: "pending" | "applied" | "failed";
  version: number;
  reason: string;
  requested_at: string;
  applied_at: string | null;
  last_error: string | null;
};

export type AlphaSettingEvent = {
  id: number;
  source_id: string;
  previous_desired_alpha: AlphaValue | null;
  requested_alpha: AlphaValue;
  status: "requested" | "applied" | "failed";
  reason: string;
  created_at: string;
};

function isAllowedAlpha(value: unknown): value is AlphaValue {
  return ALPHA_PRESETS.some((preset) => preset.value === value);
}

export function parseAlphaRequest(value: unknown): AlphaRequest {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("alpha request must be an object");
  }
  const record = value as Record<string, unknown>;
  if (!isAllowedAlpha(record.alpha)) {
    throw new Error("alpha must be one of the allowed presets");
  }
  if (record.acknowledged !== true) {
    throw new Error("confirmation is required");
  }
  if (typeof record.reason !== "string") {
    throw new Error("reason is required");
  }
  const reason = record.reason.trim();
  if (reason.length < 5 || reason.length > 500) {
    throw new Error("reason must contain 5 to 500 characters");
  }
  return { alpha: record.alpha, reason };
}
