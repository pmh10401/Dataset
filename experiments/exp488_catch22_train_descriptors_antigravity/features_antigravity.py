"""Independent, unscaled Catch22 descriptors of frozen normal TRAIN rows."""
from contextlib import closing
from hashlib import sha256
from pathlib import Path
import sqlite3
import struct

import numpy as np
from aeon.transformations.collection.feature_based import Catch22
from aeon.transformations.collection.feature_based._catch22 import feature_names

FEATURE_KEYS = ("width", "n_train", *("catch22_" + name for name in feature_names))
CATCH22_CONFIG = dict(features="all", catch24=False, outlier_norm=False,
                      replace_nans=False, use_pycatch22=False, n_jobs=1)


def load_training_rows(database_path, dataset):
    """Match the official router's membership; never load TEST or abnormal rows."""
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    rows, digest, previous = [], sha256(), None
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        for index, blob in connection.execute(
            "SELECT i.instance_index,i.values_blob FROM instances AS i "
            "JOIN datasets AS d ON d.id=i.dataset_id "
            "WHERE d.name=? AND i.label='0' AND i.split='TRAIN' "
            "ORDER BY i.instance_index", (dataset,),
        ):
            if (type(index) is not int or (previous is not None and index <= previous)
                    or not isinstance(blob, bytes) or not blob or len(blob) % 4):
                raise ValueError(f"invalid TRAIN record: {dataset}")
            digest.update(struct.pack(">qQ", index, len(blob)))
            digest.update(blob)
            rows.append(np.frombuffer(blob, dtype="<f4").astype(np.float64))
            previous = index
    if not rows:
        raise ValueError(f"no normal TRAIN rows: {dataset}")
    return rows, digest.hexdigest()


def describe_rows(rows, frozen_width_and_count):
    """Median of finite per-series features, preserving undefined coordinates."""
    arrays = [np.asarray(row, dtype=np.float64) for row in rows]
    if any(row.ndim != 1 or not row.size for row in arrays):
        raise ValueError("TRAIN rows must be nonempty one-dimensional series")
    finite = [row for row in arrays if np.isfinite(row).all()]
    measured = (float(np.median([len(row) for row in finite])), float(len(finite))) if finite else (0.0, 0.0)
    if len(finite) < 2 or measured != tuple(frozen_width_and_count):
        raise ValueError("TRAIN membership does not match frozen width/n_train")
    normalized = []
    for row in finite:
        scale = row.std()
        if len(row) >= 3 and scale > 0.0:
            normalized.append(((row - row.mean()) / scale)[None, :])
    raw = (np.asarray(Catch22(**CATCH22_CONFIG).fit_transform(normalized), dtype=np.float64)
           if normalized else np.empty((0, 22), dtype=np.float64))
    if raw.shape != (len(normalized), 22):
        raise ValueError("Catch22 returned an unexpected shape")
    valid = np.isfinite(raw)
    medians = [float(np.median(column[mask])) if mask.any() else None
               for column, mask in zip(raw.T, valid.T, strict=True)]
    return dict(values=[*map(float, frozen_width_and_count), *medians],
                total_rows=len(arrays), finite_rows=len(finite), eligible_rows=len(normalized),
                feature_valid_counts=valid.sum(axis=0).tolist())
