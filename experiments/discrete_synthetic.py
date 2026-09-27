import json
from pathlib import Path

import numpy as np

from inverse_ot.discrete import (
    forward_sinkhorn,
    inverse_sinkhorn_cost,
    project_symmetric_zero_diag_nonnegative,
)
from inverse_ot.metrics import relative_frobenius_error


def main():
    config = json.loads(Path("configs/discrete_synthetic.json").read_text())

    n = config["n"]
    p = config["power"]
    epsilon = config["epsilon"]

    rng = np.random.default_rng(config["seed"])

    grid = np.arange(n) / n
    true_cost = np.abs(grid[:, None] - grid[None, :]) ** p
    true_cost /= true_cost.max()

    mu = rng.random(n)
    nu = rng.random(n)
    mu /= mu.sum()
    nu /= nu.sum()

    _, _, observed_plan = forward_sinkhorn(
        mu,
        nu,
        true_cost,
        epsilon,
        max_iter=config["forward_iterations"],
    )

    _, _, recovered_cost, _ = inverse_sinkhorn_cost(
        mu,
        nu,
        observed_plan,
        epsilon,
        max_iter=config["inverse_iterations"],
        projection=project_symmetric_zero_diag_nonnegative,
    )

    error = relative_frobenius_error(recovered_cost, true_cost)
    print(f"relative Frobenius error: {error:.6e}")


if __name__ == "__main__":
    main()
