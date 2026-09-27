"""Template for the paper's marriage-data inverse-OT experiment.

The historical notebooks perform dataset-specific preprocessing, clustering,
cross-validation, and affinity recovery.  Those source data are not redistributed.
"""

import numpy as np

from inverse_ot.discrete import inverse_sinkhorn_cost, recover_affinity_matrix


def recover_marriage_affinity(mu, nu, observed_plan, G, D, epsilon, iterations=1000):
    """Recover a cost matrix and then the linear affinity A in c = G^T A D."""
    _, _, cost, _ = inverse_sinkhorn_cost(
        mu,
        nu,
        observed_plan,
        epsilon,
        max_iter=iterations,
        projection=None,
    )
    A = recover_affinity_matrix(cost, G, D)
    return cost, A


if __name__ == "__main__":
    print(
        "Prepare the marriage-market feature matrices G, D and observed matching "
        "matrix before calling recover_marriage_affinity()."
    )
