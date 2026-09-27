from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int = 1):
    layers = []
    d = input_dim
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(d, hidden_dim), nn.Tanh()])
        d = hidden_dim
    layers.append(nn.Linear(d, output_dim))
    return nn.Sequential(*layers)


class DualPotentialNet(nn.Module):
    """Neural dual potential alpha(x) or beta(y)."""

    def __init__(self, input_dim: int, hidden_dim: int = 20, hidden_layers: int = 3):
        super().__init__()
        self.net = _mlp(input_dim, hidden_dim, hidden_layers, 1)

    def forward(self, x):
        return self.net(x)


class CostNet(nn.Module):
    """Nonnegative neural transport cost.

    feature_mode:
      - "concat": network receives [x, y]
      - "abs_diff": network receives |x-y|
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_dim: int = 20,
        hidden_layers: int = 3,
        feature_mode: str = "concat",
    ):
        super().__init__()
        if feature_mode not in {"concat", "abs_diff"}:
            raise ValueError("feature_mode must be 'concat' or 'abs_diff'")
        if feature_mode == "abs_diff" and x_dim != y_dim:
            raise ValueError("abs_diff requires x_dim == y_dim")

        self.feature_mode = feature_mode
        in_dim = x_dim + y_dim if feature_mode == "concat" else x_dim
        self.net = _mlp(in_dim, hidden_dim, hidden_layers, 1)
        self.output_activation = nn.ReLU()

    def features(self, x, y):
        if self.feature_mode == "concat":
            return torch.cat([x, y], dim=-1)
        return torch.abs(x - y)

    def forward(self, x, y):
        return self.output_activation(self.net(self.features(x, y)))
