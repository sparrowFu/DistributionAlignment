"""
GaussianImageDistribution - Common utilities for baseline models.

Shared functions used by the ProLIP baseline model.
"""

from typing import Tuple

import torch


def merge_distributions_moment_matching(
    mus: torch.Tensor,
    logvars: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge K Gaussian distributions via moment matching.

    Args:
        mus: Distribution means (B, K, D)
        logvars: Distribution log variances (B, K, D)

    Returns:
        (combined_mu, combined_logvar) each (B, D)
    """
    combined_mu = mus.mean(dim=1)
    vars = torch.exp(logvars)
    combined_var = (vars + mus ** 2).mean(dim=1) - combined_mu ** 2
    combined_logvar = torch.log(combined_var + 1e-6)
    return combined_mu, combined_logvar
