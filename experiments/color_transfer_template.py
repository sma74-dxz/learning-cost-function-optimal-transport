"""Template for the RGB color-transfer application.

The historical experiment uses source/target images and observed paired RGB
samples.  Image files are not redistributed here.
"""

import numpy as np
import torch

from inverse_ot.networks import CostNet


def rgb_cost_network(hidden_dim=32, hidden_layers=3):
    """Paper-style learned cost g(|x-y|) for RGB vectors."""
    return CostNet(
        x_dim=3,
        y_dim=3,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        feature_mode="abs_diff",
    )


if __name__ == "__main__":
    net = rgb_cost_network()
    x = torch.rand(8, 3)
    y = torch.rand(8, 3)
    print("example RGB cost shape:", net(x, y).shape)
