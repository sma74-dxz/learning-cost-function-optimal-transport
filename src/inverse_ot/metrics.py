import numpy as np


def relative_frobenius_error(estimate, truth):
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    return np.linalg.norm(estimate - truth, ord="fro") / np.linalg.norm(truth, ord="fro")


def best_scale(estimate, truth):
    """Least-squares scalar aligning estimate to truth."""
    e = np.asarray(estimate, dtype=float).ravel()
    t = np.asarray(truth, dtype=float).ravel()
    denom = np.dot(e, e)
    return 1.0 if denom == 0 else float(np.dot(e, t) / denom)
