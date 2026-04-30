"""Torch device selection for neural experts, joint MoE, and softmax router."""

from __future__ import annotations

import os

import torch


def default_torch_device() -> torch.device:
    """
    Training/inference device. Defaults to CPU for reproducibility; GPUs are opt-in.

    Environment variables (first match wins):

    - ``MOE_BANDIT_USE_CUDA=1`` — use ``cuda`` if ``torch.cuda.is_available()``.
    - ``MOE_BANDIT_USE_MPS=1`` — use ``mps`` if the Metal backend is available (Apple Silicon).

    If both CUDA and MPS are requested and available, CUDA is used.
    """
    if os.environ.get("MOE_BANDIT_USE_CUDA", "0") == "1" and torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("MOE_BANDIT_USE_MPS", "0") == "1" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
