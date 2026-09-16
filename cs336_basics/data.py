from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample next-token prediction examples from a one-dimensional token array."""
    if dataset.ndim != 1:
        raise ValueError(f"dataset must be one-dimensional, got shape {dataset.shape}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    num_starts = len(dataset) - context_length
    if num_starts <= 0:
        raise ValueError("dataset must contain at least context_length + 1 tokens")

    starts = np.random.randint(0, num_starts, size=batch_size)
    offsets = np.arange(context_length)
    x = np.asarray(dataset[starts[:, None] + offsets[None, :]], dtype=np.int64)
    y = np.asarray(dataset[starts[:, None] + offsets[None, :] + 1], dtype=np.int64)

    return (
        torch.from_numpy(x).to(device=device),
        torch.from_numpy(y).to(device=device),
    )


def load_token_array(
    path: str | os.PathLike,
    dtype: str | np.dtype = np.uint16,
) -> npt.NDArray:
    """Memory-map a .npy file or a headerless binary token file."""
    path = os.fspath(path)
    if path.endswith(".npy"):
        array = np.load(path, mmap_mode="r")
    else:
        array = np.memmap(path, mode="r", dtype=np.dtype(dtype))
    if array.ndim != 1:
        raise ValueError(f"token array must be one-dimensional, got shape {array.shape}")
    return array
