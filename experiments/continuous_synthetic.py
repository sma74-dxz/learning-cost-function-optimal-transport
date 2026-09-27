import json
from pathlib import Path

import numpy as np
import torch

from inverse_ot.continuous import ContinuousIOT, ContinuousIOTConfig
from inverse_ot.discrete import forward_sinkhorn
from inverse_ot.networks import CostNet, DualPotentialNet
from inverse_ot.sampling import sample_coupling, uniform_collocation


def main():
    cfg = json.loads(Path("configs/continuous_synthetic.json").read_text())

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = cfg["grid_size"]
    p = cfg["power"]
    epsilon = cfg["epsilon"]

    # Build a synthetic observed coupling on [0,1] x [0,1].
    grid = np.linspace(0.0, 1.0, n, endpoint=False).reshape(-1, 1)
    true_cost = np.abs(grid[:, None, 0] - grid[None, :, 0]) ** p
    true_cost /= true_cost.max()

    rng = np.random.default_rng(cfg["seed"])
    mu = rng.random(n)
    nu = rng.random(n)
    mu /= mu.sum()
    nu /= nu.sum()

    _, _, plan = forward_sinkhorn(mu, nu, true_cost, epsilon, max_iter=2000)

    px, py = sample_coupling(
        plan,
        grid,
        grid,
        cfg["pair_samples"],
        rng=rng,
    )
    paired = np.concatenate([px, py], axis=1).astype(np.float32)

    alpha = DualPotentialNet(1, cfg["hidden_dim"], cfg["hidden_layers"]).to(device)
    beta = DualPotentialNet(1, cfg["hidden_dim"], cfg["hidden_layers"]).to(device)
    cost_net = CostNet(
        1,
        1,
        cfg["hidden_dim"],
        cfg["hidden_layers"],
        feature_mode="abs_diff",
    ).to(device)

    trainer = ContinuousIOT(
        alpha,
        beta,
        cost_net,
        ContinuousIOTConfig(
            epsilon=epsilon,
            learning_rate=cfg["learning_rate"],
            collocation_multiplier=cfg["collocation_multiplier"],
        ),
    )

    batch_size = cfg["batch_size"]
    for iteration in range(cfg["iterations"]):
        idx = rng.integers(0, len(paired), size=batch_size)
        batch = torch.as_tensor(paired[idx], device=device)

        x = batch[:, :1]
        y = batch[:, 1:]

        cx, cy = uniform_collocation(
            batch_size * cfg["collocation_multiplier"],
            1,
            1,
            device=device,
        )

        metrics = trainer.step(x, y, cx, cy)

        if iteration % 100 == 0 or iteration == cfg["iterations"] - 1:
            print(iteration, metrics)

    # Evaluate the learned cost on a regular grid.
    with torch.no_grad():
        x = torch.as_tensor(
            np.repeat(grid, n, axis=0),
            dtype=torch.float32,
            device=device,
        )
        y = torch.as_tensor(
            np.tile(grid, (n, 1)),
            dtype=torch.float32,
            device=device,
        )
        learned = cost_net(x, y).cpu().numpy().reshape(n, n)

    # Cost is identifiable only under structural assumptions / normalization.
    # Normalize both surfaces for an interpretable synthetic diagnostic.
    learned = learned / max(learned.max(), 1e-12)
    rel = np.linalg.norm(learned - true_cost) / np.linalg.norm(true_cost)
    print(f"normalized cost relative error: {rel:.6f}")


if __name__ == "__main__":
    main()
