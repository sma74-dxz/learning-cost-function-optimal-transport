from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


def continuous_iot_loss(
    alpha_net: nn.Module,
    beta_net: nn.Module,
    cost_net: nn.Module,
    paired_x: torch.Tensor,
    paired_y: torch.Tensor,
    collocation_x: torch.Tensor,
    collocation_y: torch.Tensor,
    epsilon: float = 1.0,
    regularization: Optional[torch.Tensor] = None,
):
    """Monte-Carlo approximation of equation (21) in the paper.

    The x/y marginals are taken from the observed paired samples.  The final
    integral is approximated by uniform collocation samples; for a unit-volume
    domain the integral estimator is simply the sample mean.
    """
    alpha = alpha_net(paired_x)
    beta = beta_net(paired_y)
    observed_cost = cost_net(paired_x, paired_y)

    alpha_c = alpha_net(collocation_x)
    beta_c = beta_net(collocation_y)
    cost_c = cost_net(collocation_x, collocation_y)

    exp_term = torch.exp(
        torch.clamp(
            (alpha_c + beta_c - cost_c) / epsilon,
            min=-60.0,
            max=60.0,
        )
    )

    loss = (
        -alpha.mean()
        -beta.mean()
        +observed_cost.mean()
        +epsilon * exp_term.mean()
    )

    if regularization is not None:
        loss = loss + regularization

    parts = {
        "loss": loss.detach(),
        "minus_E_alpha": (-alpha.mean()).detach(),
        "minus_E_beta": (-beta.mean()).detach(),
        "E_pi_cost": observed_cost.mean().detach(),
        "integral_term": (epsilon * exp_term.mean()).detach(),
    }
    return loss, parts


@dataclass
class ContinuousIOTConfig:
    epsilon: float = 1.0
    learning_rate: float = 1e-3
    collocation_multiplier: int = 10


class ContinuousIOT:
    """Trainer for the continuous inverse-OT neural objective."""

    def __init__(self, alpha_net, beta_net, cost_net, config: ContinuousIOTConfig):
        self.alpha_net = alpha_net
        self.beta_net = beta_net
        self.cost_net = cost_net
        self.config = config

        params = (
            list(alpha_net.parameters())
            + list(beta_net.parameters())
            + list(cost_net.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    def step(self, paired_x, paired_y, collocation_x, collocation_y):
        self.optimizer.zero_grad(set_to_none=True)

        loss, parts = continuous_iot_loss(
            self.alpha_net,
            self.beta_net,
            self.cost_net,
            paired_x,
            paired_y,
            collocation_x,
            collocation_y,
            epsilon=self.config.epsilon,
        )
        loss.backward()
        self.optimizer.step()

        return {k: float(v.cpu()) for k, v in parts.items()}
