from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, width: int = 64, depth: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.ReLU()))
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FourierFeatureMLP(nn.Module):
    def __init__(
        self, width: int = 64, depth: int = 3, bands: int = 6
    ) -> None:
        super().__init__()
        frequencies = 2.0 ** torch.arange(bands, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)
        input_width = 2 * bands
        layers: list[nn.Module] = [nn.Linear(input_width, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.ReLU()))
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * x * self.frequencies
        features = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        return self.network(features)


class SineLayer(nn.Module):
    def __init__(
        self,
        input_width: int,
        output_width: int,
        *,
        first: bool = False,
        omega: float = 30.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_width, output_width)
        self.omega = omega
        self.reset_parameters(first=first)

    def reset_parameters(self, *, first: bool) -> None:
        bound = 1.0 / self.linear.in_features if first else math.sqrt(
            6.0 / self.linear.in_features
        ) / self.omega
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(x))


class SIREN(nn.Module):
    def __init__(
        self, width: int = 64, depth: int = 3, omega: float = 30.0
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            SineLayer(1, width, first=True, omega=omega)
        ]
        layers.extend(
            SineLayer(width, width, omega=omega) for _ in range(depth - 1)
        )
        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)
        bound = math.sqrt(6.0 / width) / omega
        nn.init.uniform_(self.output.weight, -bound, bound)
        nn.init.uniform_(self.output.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(x))


def build_model(name: str, width: int = 64, depth: int = 3) -> nn.Module:
    models = {
        "mlp": lambda: MLP(width=width, depth=depth),
        "fourier": lambda: FourierFeatureMLP(width=width, depth=depth),
        "siren": lambda: SIREN(width=width, depth=depth),
    }
    try:
        return models[name]()
    except KeyError as error:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(models)}") from error
