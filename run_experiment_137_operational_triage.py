from __future__ import annotations

import numpy as np


def route_tiers(high, standard, priority, test_size):
    hard = set(high)
    standard_review = set(standard) - hard
    priority_review = set(priority) - hard - standard_review
    all_indices = hard | standard_review | priority_review
    invalid = sorted(index for index in all_indices if index < 0 or index >= test_size)
    if invalid:
        raise ValueError(f'out-of-bounds indices test_size={test_size} indices={invalid[:10]}')
    return {
        'hard': hard,
        'standard_review': standard_review,
        'priority_review': priority_review,
        'no_alert': set(range(test_size)) - all_indices,
    }


def tier_metrics(y_test, indices):
    selected = set(indices)
    truth = set(np.flatnonzero(np.asarray(y_test, dtype=int) == 1).astype(int).tolist())
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    denominator = 2 * tp + fp + fn
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': 2 * tp / denominator if denominator else 0.0,
    }
