from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle, ScoreResult
from experiments.exp239_248_diverse_score_sources.foundation import crossfit_plan
from experiments.exp239_248_diverse_score_sources.source_base import abstained_result


SEGMENTS = (16, 32, 64)
NGRAMS = (2, 3)
ALPHABET = 5


def _paa(row: np.ndarray, segments: int) -> np.ndarray:
    return np.asarray([part.mean() for part in np.array_split(row, min(segments, len(row)))])


class _MotifModel:
    def __init__(self, reference: np.ndarray):
        self.breakpoints = {}
        self.counts = {}
        self.totals = {}
        for segments in SEGMENTS:
            paa_rows = [_paa(row, segments) for row in reference]
            pooled = np.concatenate(paa_rows)
            breaks = np.quantile(pooled, np.linspace(0, 1, ALPHABET + 1)[1:-1])
            self.breakpoints[segments] = breaks
            for width in NGRAMS:
                counts: Counter[tuple[int, ...]] = Counter()
                for values in paa_rows:
                    symbols = np.digitize(values, breaks).astype(int)
                    counts.update(tuple(symbols[i : i + width]) for i in range(len(symbols) - width + 1))
                self.counts[(segments, width)] = counts
                self.totals[(segments, width)] = sum(counts.values())

    def score(self, rows: np.ndarray) -> np.ndarray:
        output = []
        vocabulary = ALPHABET ** max(NGRAMS)
        for row in np.asarray(rows, dtype=float):
            negative_logs = []
            unseen = 0
            total_motifs = 0
            for segments in SEGMENTS:
                symbols = np.digitize(_paa(row, segments), self.breakpoints[segments]).astype(int)
                for width in NGRAMS:
                    counts = self.counts[(segments, width)]
                    denominator = self.totals[(segments, width)] + vocabulary
                    for start in range(len(symbols) - width + 1):
                        motif = tuple(symbols[start : start + width])
                        observed = counts.get(motif, 0)
                        negative_logs.append(-np.log((observed + 1.0) / denominator))
                        unseen += int(observed == 0)
                        total_motifs += 1
            output.append(float(np.mean(negative_logs)) + unseen / max(1, total_motifs))
        return np.asarray(output, dtype=float)


class SymbolicMotifRarity:
    name = "symbolic_motif_rarity"
    minimum_train = 5

    def score(self, bundle: DatasetBundle, seed: int, artifact_dir: Path) -> ScoreResult:
        if len(bundle.train_normal) < self.minimum_train:
            return abstained_result(self.name, bundle, self.minimum_train)
        train_oof = np.empty(len(bundle.train_normal), dtype=float)
        for fit, query in crossfit_plan(len(bundle.train_normal), seed=seed):
            train_oof[query] = _MotifModel(bundle.train_normal[fit]).score(bundle.train_normal[query])
        test_scores = _MotifModel(bundle.train_normal).score(bundle.test_values)
        return ScoreResult(
            self.name,
            train_oof,
            test_scores,
            False,
            None,
            {
                "seed": seed,
                "segments": SEGMENTS,
                "ngrams": NGRAMS,
                "alphabet": ALPHABET,
                "cross_fitted": True,
                "self_reference_violations": 0,
                "contains_test_labels": False,
            },
        )

