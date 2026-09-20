"use client";

import { useState } from "react";

import ShadowRankingComparison from "./shadow-ranking-comparison";
import ShadowValidation from "./shadow-validation";

export default function ShadowHub() {
  const [ranking, setRanking] = useState(false);
  return ranking
    ? <ShadowRankingComparison onClose={() => setRanking(false)} />
    : <ShadowValidation onOpenRanking={() => setRanking(true)} />;
}
