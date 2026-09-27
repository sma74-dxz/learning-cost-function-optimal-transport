from __future__ import annotations

import numpy as np
import torch


def sample_coupling(plan, x_grid, y_grid, n_samples, *, rng=None):
    """Sample paired observations (x,y) from a discrete coupling matrix."""
    rng = np.random.default_rng() if rng is None else rng
    plan = np.asarray(plan, dtype=float)
    prob = plan.ravel()
    prob = prob / prob.sum()

    flat_idx = rng.choice(prob.size, size=n_samples, replace=True, p=prob)
    i, j = np.unravel_index(flat_idx, plan.shape)

    x = np.asarray(x_grid)[i]
    y = np.asarray(y_grid)[j]

    return x, y


def uniform_collocation(batch_size, x_dim, y_dim, *, device=None):
    """Uniform samples on [0,1]^(x_dim+y_dim), matching the paper experiments."""
    x = torch.rand(batch_size, x_dim, device=device)
    y = torch.rand(batch_size, y_dim, device=device)
    return x, y
