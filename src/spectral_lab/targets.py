from __future__ import annotations

import math
from collections.abc import Sequence

import torch


DEFAULT_FREQUENCIES = (1, 3, 7, 13)


def target_function(
    x: torch.Tensor,
    frequencies: Sequence[int] = DEFAULT_FREQUENCIES,
) -> torch.Tensor:
    return sum(
        torch.sin(2.0 * math.pi * frequency * x) / len(frequencies)
        for frequency in frequencies
    )


def target_derivative(
    x: torch.Tensor,
    frequencies: Sequence[int] = DEFAULT_FREQUENCIES,
) -> torch.Tensor:
    return sum(
        (2.0 * math.pi * frequency / len(frequencies))
        * torch.cos(2.0 * math.pi * frequency * x)
        for frequency in frequencies
    )
