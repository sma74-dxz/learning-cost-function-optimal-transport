from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


Array = np.ndarray


def _as_column(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    return x.reshape(-1, 1)


def forward_sinkhorn(
    mu: Array,
    nu: Array,
    cost: Array,
    epsilon: float,
    max_iter: int = 1000,
    tiny: float = 1e-300,
) -> Tuple[Array, Array, Array]:
    """Entropy-regularized forward OT via Sinkhorn scaling.

    Returns dual potentials alpha, beta and the transport plan pi.
    """
    mu = _as_column(mu)
    nu = _as_column(nu)
    cost = np.asarray(cost, dtype=float)

    K = np.exp(-cost / epsilon)
    K = np.maximum(K, tiny)

    u = np.ones_like(mu)
    v = np.ones_like(nu)

    for _ in range(max_iter):
        u = mu / np.maximum(K @ v, tiny)
        v = nu / np.maximum(K.T @ u, tiny)

    alpha = epsilon * np.log(np.maximum(u, tiny))
    beta = epsilon * np.log(np.maximum(v, tiny))
    pi = (u @ v.T) * K

    return alpha, beta, pi


def project_symmetric_zero_diag_nonnegative(cost: Array) -> Array:
    """Projection used in the paper's symmetric synthetic examples."""
    cost = 0.5 * (cost + cost.T)
    np.fill_diagonal(cost, 0.0)
    return np.maximum(cost, 0.0)


def inverse_sinkhorn_cost(
    mu: Array,
    nu: Array,
    observed_plan: Array,
    epsilon: float,
    max_iter: int = 1000,
    projection: Optional[Callable[[Array], Array]] = None,
    initial_cost: Optional[Array] = None,
    tiny: float = 1e-300,
):
    """Algorithm 1: matrix scaling for discrete inverse OT.

    Iteration:
        K = exp(-c / epsilon)
        u = mu / (K v)
        v = nu / (K^T u)
        K = pi_hat / (u v^T)
        c = projection(-epsilon * log K)

    Parameters
    ----------
    projection:
        Prox/projection associated with structural prior R(c).  If omitted,
        the raw cost update is used.
    """
    mu = _as_column(mu)
    nu = _as_column(nu)
    pi_hat = np.asarray(observed_plan, dtype=float)

    m, n = pi_hat.shape
    if initial_cost is None:
        cost = np.zeros((m, n), dtype=float)
    else:
        cost = np.asarray(initial_cost, dtype=float).copy()

    u = np.ones_like(mu)
    v = np.ones_like(nu)

    history = []

    for _ in range(max_iter):
        K = np.exp(-cost / epsilon)
        K = np.maximum(K, tiny)

        u = mu / np.maximum(K @ v, tiny)
        v = nu / np.maximum(K.T @ u, tiny)

        outer = np.maximum(u @ v.T, tiny)
        K_from_plan = np.maximum(pi_hat / outer, tiny)

        raw_cost = -epsilon * np.log(K_from_plan)
        cost = projection(raw_cost) if projection is not None else raw_cost

        reconstructed = outer * np.exp(-cost / epsilon)
        marginal_error = (
            np.linalg.norm(reconstructed.sum(axis=1, keepdims=True) - mu)
            + np.linalg.norm(reconstructed.sum(axis=0, keepdims=True).T - nu)
        )
        history.append(float(marginal_error))

    alpha = epsilon * np.log(np.maximum(u, tiny))
    beta = epsilon * np.log(np.maximum(v, tiny))

    return alpha, beta, cost, np.asarray(history)


def recover_affinity_matrix(cost: Array, G: Array, D: Array) -> Array:
    """Recover A in c = G^T A D by Moore-Penrose pseudoinverses.

    Shapes
    ------
    G: [p, m]
    D: [q, n]
    cost: [m, n]
    A: [p, q]
    """
    G = np.asarray(G, dtype=float)
    D = np.asarray(D, dtype=float)
    cost = np.asarray(cost, dtype=float)

    G_plus = np.linalg.pinv(G)
    D_plus = np.linalg.pinv(D)

    return G_plus.T @ cost @ D_plus
