"""Compatibility helpers for Jetson Torch wheels built against NumPy 1.x.

The Orin wheel used by this project contains the required ``sm_87`` kernels,
but its NumPy bridge can be disabled when imported with NumPy 2.x.  NumPy 2
itself remains required by Pinocchio and the current OpenCV wheel, so callers
use these narrow fallbacks instead of changing the environment ABI.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def install_torch_numpy_compat(torch_module: Any = None) -> Any:
    """Install an idempotent ``torch.from_numpy`` fallback and return Torch."""

    if torch_module is None:
        import torch as torch_module

    if getattr(torch_module, "_m20_numpy_compat_installed", False):
        return torch_module

    original = torch_module.from_numpy

    def from_numpy_compat(array: Any) -> Any:
        try:
            return original(array)
        except RuntimeError as exc:
            if "Numpy is not available" not in str(exc):
                raise
            # torch.as_tensor() uses a separate conversion path in the
            # affected Jetson wheel and remains functional with NumPy 2.x.
            return torch_module.as_tensor(array)

    torch_module.from_numpy = from_numpy_compat
    torch_module._m20_numpy_compat_installed = True
    return torch_module


def tensor_to_numpy(value: Any) -> np.ndarray:
    """Convert a Torch/array-like value without requiring Tensor.numpy()."""

    if isinstance(value, np.ndarray):
        return value
    candidate = value.detach() if hasattr(value, "detach") else value
    candidate = candidate.cpu() if hasattr(candidate, "cpu") else candidate
    if hasattr(candidate, "numpy"):
        try:
            return np.asarray(candidate.numpy())
        except RuntimeError as exc:
            if "Numpy is not available" not in str(exc):
                raise
    if hasattr(candidate, "tolist"):
        return np.asarray(candidate.tolist())
    return np.asarray(candidate)


__all__ = ["install_torch_numpy_compat", "tensor_to_numpy"]
