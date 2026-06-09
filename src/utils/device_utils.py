"""
Device and hardware detection utilities.
"""

import torch


def has_mps_module() -> bool:
    """Check if torch.mps module is available (PyTorch 2.1+)."""
    return hasattr(torch, 'mps')
