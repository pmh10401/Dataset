"""Fixed smooth worst-normal ranking surrogate; no TEST or threshold selection."""
import numpy as np
from scipy.special import expit, logsumexp


def _inputs(normal, anomalous):
    n, a = np.asarray(normal, dtype=np.float64), np.asarray(anomalous, dtype=np.float64)
    if n.ndim != 2 or a.ndim != 2 or not len(n) or not len(a) or not n.shape[1] or n.shape[1] != a.shape[1]:
        raise ValueError('nonempty aligned normal and anomalous feature matrices required')
    if not np.isfinite(n).all() or not np.isfinite(a).all():
        raise ValueError('nonfinite rank features')
    return n, a


def _objective(w, normal, anomalous, regularization=.1, temperature=.1, margin=.25):
    ns, az = normal @ w, anomalous @ w
    z = ns / temperature; lse = logsumexp(z)
    tail = temperature * lse
    delta = margin + tail - az
    probability = expit(delta)
    loss = np.logaddexp(0., delta).mean() + regularization * np.mean((w - 1.) ** 2)
    gradient = probability.mean() * (np.exp(z - lse) @ normal)
    gradient -= probability @ anomalous / len(anomalous)
    gradient += 2. * regularization * (w - 1.) / len(w)
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError('nonfinite rank objective or gradient')
    return float(loss), gradient


def loss_gradient(weights, normal, anomalous, *, regularization=.1, temperature=.1, margin=.25):
    n, a = _inputs(normal, anomalous); w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n.shape[1],) or not np.isfinite(w).all():
        raise ValueError('finite weight vector matching features required')
    if not np.isfinite([regularization, temperature, margin]).all() or regularization <= 0 or temperature <= 0 or margin < 0:
        raise ValueError('positive regularization/temperature and nonnegative margin required')
    return _objective(w, n, a, regularization, temperature, margin)


def fit_tail_weights(normal, anomalous):
    n, a = _inputs(normal, anomalous)
    if np.any(n < 0) or np.any(a < 0):
        raise ValueError('distance contributions must be nonnegative')
    w = np.ones(n.shape[1], dtype=np.float64)
    first = _objective(w, n, a)[0]; m = np.zeros_like(w); v = np.zeros_like(w)
    for step in range(1, 81):
        _, gradient = _objective(w, n, a)
        m = .9 * m + .1 * gradient; v = .999 * v + .001 * gradient ** 2
        w -= .05 * (m / (1. - .9 ** step)) / (np.sqrt(v / (1. - .999 ** step)) + 1e-8)
        w = np.clip(w, .25, 4.)  # Projection of model weights, never clipping input data.
    last = _objective(w, n, a)[0]
    return w, dict(initial_objective=first, final_objective=last, steps=80,
                   minimum_weight=float(w.min()), maximum_weight=float(w.max()),
                   convergence_claimed=False, backend='numpy_float64')
