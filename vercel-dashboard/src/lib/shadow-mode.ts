export const SHADOW_COLLECTION_MODES = ["prospective_shadow", "retrospective_replay"] as const;
export type ShadowCollectionMode = typeof SHADOW_COLLECTION_MODES[number];

export function parseShadowCollectionMode(value: string | null): ShadowCollectionMode | null {
  if (value === null) return "prospective_shadow";
  return SHADOW_COLLECTION_MODES.includes(value as ShadowCollectionMode)
    ? value as ShadowCollectionMode
    : null;
}

export function policyVersionFor(mode: ShadowCollectionMode) {
  return mode === "prospective_shadow" ? "exp265-shadow-v1" : "exp265-replay-v1";
}

export function modeCopy(mode: ShadowCollectionMode) {
  return mode === "prospective_shadow"
    ? { title: "신규 검증", prospective: true, retrospective: false }
    : { title: "개발용 재생", prospective: false, retrospective: true };
}
