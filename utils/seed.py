"""
GaussianImageDistribution - Random Seed Management

This module provides functions for setting random seeds to ensure reproducibility.
"""

import random
import numpy as np
import torch

# Global seed variable
_global_seed = 42


def set_seed(seed: int) -> int:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value

    Returns:
        The seed that was set
    """
    global _global_seed
    _global_seed = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set seed for CUDA operations (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return _global_seed


def get_seed() -> int:
    """
    Get the current global seed value.

    Returns:
        Current random seed
    """
    return _global_seed


if __name__ == "__main__":
    # Test seed setting
    seed = set_seed(42)
    print(f"Random seed set to: {seed}")

    # Test reproducibility
    rand1 = random.random()
    rand2 = random.random()

    # Reset and verify same values
    set_seed(42)
    rand3 = random.random()
    rand4 = random.random()

    print(f"First run: {rand1}, {rand2}")
    print(f"Second run: {rand3}, {rand4}")
    print(f"Reproducibility check: {rand1 == rand3 and rand2 == rand4}")
