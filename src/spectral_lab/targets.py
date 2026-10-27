from __future__ import annotations

import math
from collections.abc import Sequence

import torch


DEFAULT_FREQUENCIES = (1, 3, 7, 13)


def target_function(
    x: torch.Tensor,
    frequencies: Sequence[int] = DEFAULT_FREQUENCIES,
) -> torch.Tensor:
    scale = 1.0 / len(frequencies)
    out = torch.zeros_like(x)
    for frequency in frequencies:
        out = out + scale * torch.sin(2.0 * math.pi * frequency * x)
    return out


def target_derivative(
    x: torch.Tensor,
    frequencies: Sequence[int] = DEFAULT_FREQUENCIES,
) -> torch.Tensor:
    out = torch.zeros_like(x)
    for frequency in frequencies:
        out = out + (2.0 * math.pi * frequency / len(frequencies)) * torch.cos(
            2.0 * math.pi * frequency * x
        )
    return out
