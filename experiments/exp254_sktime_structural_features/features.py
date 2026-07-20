from __future__ import annotations

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sktime.transformations.dictionary_based import SFA
from sktime.transformations.matrix_profile import MatrixProfileFeatures
from sktime.transformations.random_intervals import RandomIntervals
from sktime.transformations.sax import SAX


FEATURE_DIMENSIONS = {
    "random_interval_224": 224,
    "matrix_profile_16": 16,
    "sax_transition_72": 72,
    "sfa_hash_128": 128,
    "structural_combined_440": 440,
}
BASE_BLOCKS = (
    "random_interval_224",
    "matrix_profile_16",
    "sax_transition_72",
    "sfa_hash_128",
)


def _panel(rows: np.ndarray, working_length: int) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] < 4:
        raise ValueError("aligned time-series rows must be a two-dimensional panel")
    if values.shape[1] != working_length:
        old = np.linspace(0.0, 1.0, values.shape[1])
        new = np.linspace(0.0, 1.0, working_length)
        values = np.vstack([np.interp(new, old, row) for row in values])
    return values.astype(np.float32, copy=False)[:, np.newaxis, :]


def matrix_profile_windows(length: int) -> tuple[int, int]:
    if length < 8:
        raise ValueError("matrix profile requires at least eight time points")
    first = max(3, min(length - 1, int(round(length / 16))))
    second = max(first + 1, min(length - 1, int(round(length / 8))))
    return first, second


def _profile_summary(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.column_stack(
        [
            np.min(array, axis=1),
            np.mean(array, axis=1),
            np.std(array, axis=1),
            np.quantile(array, 0.25, axis=1),
            np.quantile(array, 0.50, axis=1),
            np.quantile(array, 0.75, axis=1),
            np.quantile(array, 0.90, axis=1),
            np.max(array, axis=1),
        ]
    )


def _sax_features(symbols: np.ndarray, alphabet_size: int = 8) -> np.ndarray:
    values = np.asarray(symbols, dtype=int).reshape(len(symbols), -1)
    output = np.zeros((len(values), alphabet_size + alphabet_size**2), dtype=float)
    for row_index, row in enumerate(values):
        output[row_index, :alphabet_size] = np.bincount(
            row, minlength=alphabet_size
        ) / max(1, len(row))
        if len(row) > 1:
            transitions = row[:-1] * alphabet_size + row[1:]
            output[row_index, alphabet_size:] = np.bincount(
                transitions, minlength=alphabet_size**2
            ) / len(transitions)
    return output


def _sfa_dicts(values: object) -> list[dict[str, int]]:
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], list):
        raise ValueError("unexpected SFA dictionary output")
    return [{str(key): int(count) for key, count in row.items()} for row in values[0]]


class StructuralFeatureExtractor:
    def __init__(self, *, seed: int, working_length: int):
        self.seed = int(seed)
        self.working_length = int(working_length)
        self.random_intervals = RandomIntervals(
            n_intervals=32, random_state=self.seed, n_jobs=1
        )
        self.sax = SAX(word_size=32, alphabet_size=8)
        self.sfa = SFA(
            word_length=8,
            alphabet_size=4,
            window_size=max(8, min(32, self.working_length // 4)),
            norm=True,
            binning_method="equi-depth",
            anova=False,
            bigrams=False,
            skip_grams=False,
            remove_repeat_words=False,
            save_words=False,
            return_pandas_data_series=False,
            use_fallback_dft=True,
            typed_dict=False,
            n_jobs=1,
        )
        self.hasher = FeatureHasher(
            n_features=128, input_type="dict", alternate_sign=False
        )
        self._fitted = False

    def _matrix_profile(self, panel: np.ndarray) -> np.ndarray:
        summaries = []
        for window in matrix_profile_windows(self.working_length):
            profile = MatrixProfileFeatures(m=window).fit_transform(panel)
            summaries.append(_profile_summary(np.asarray(profile, dtype=float)))
        return np.concatenate(summaries, axis=1)

    def _blocks(self, panel: np.ndarray, *, fit: bool) -> dict[str, np.ndarray]:
        if fit:
            intervals = self.random_intervals.fit_transform(panel)
            symbols = self.sax.fit_transform(panel)
            words = self.sfa.fit_transform(panel)
        else:
            intervals = self.random_intervals.transform(panel)
            symbols = self.sax.transform(panel)
            words = self.sfa.transform(panel)
        blocks = {
            "random_interval_224": np.asarray(intervals, dtype=float),
            "matrix_profile_16": self._matrix_profile(panel),
            "sax_transition_72": _sax_features(symbols),
            "sfa_hash_128": self.hasher.transform(_sfa_dicts(words)).toarray(),
        }
        blocks["structural_combined_440"] = np.concatenate(
            [blocks[name] for name in BASE_BLOCKS], axis=1
        )
        for name, width in FEATURE_DIMENSIONS.items():
            if blocks[name].shape != (len(panel), width):
                raise ValueError(
                    f"feature shape mismatch: block={name} "
                    f"expected={(len(panel), width)} actual={blocks[name].shape}"
                )
        return blocks

    def fit_transform_train(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        blocks = self._blocks(_panel(rows, self.working_length), fit=True)
        self._fitted = True
        return blocks

    def transform(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("StructuralFeatureExtractor must be fitted on TRAIN first")
        return self._blocks(_panel(rows, self.working_length), fit=False)
