from __future__ import annotations

import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from spectral_lab.models import build_model
from spectral_lab.targets import (
    DEFAULT_FREQUENCIES,
    target_derivative,
    target_function,
)

MODEL_NAMES = ("mlp", "fourier", "siren")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def frequency_errors(
    prediction: torch.Tensor,
    target: torch.Tensor,
    x: torch.Tensor,
    frequencies: tuple[int, ...] = DEFAULT_FREQUENCIES,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for frequency in frequencies:
        phase = 2.0 * torch.pi * frequency * x
        cos_basis = torch.cos(phase)
        sin_basis = torch.sin(phase)
        cosine_error = 2.0 * torch.mean((prediction - target) * cos_basis)
        sine_error = 2.0 * torch.mean((prediction - target) * sin_basis)
        errors[str(frequency)] = torch.sqrt(
            cosine_error.square() + sine_error.square()
        ).item()
    return errors


def _evaluate(
    model: nn.Module, x: torch.Tensor, target: torch.Tensor
) -> tuple[float, float, dict[str, float]]:
    prediction = model(x)
    prediction_derivative = torch.autograd.grad(
        prediction.sum(), x, create_graph=False
    )[0]
    return (
        torch.mean((prediction - target).square()).item(),
        torch.mean((prediction_derivative - target_derivative(x)).square()).item(),
        frequency_errors(prediction.detach(), target, x),
    )


def run_experiment(
    *,
    output_dir: Path,
    steps: int = 1000,
    seeds: tuple[int, ...] = (0, 1, 2),
    train_size: int = 256,
    learning_rate: float = 1e-3,
    checkpoint_every: int = 25,
    width: int = 64,
    depth: int = 3,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    if steps < 1 or train_size < 2 or checkpoint_every < 1:
        raise ValueError(
            "steps and checkpoint_every must be positive; "
            "train_size must be at least 2"
        )
    if learning_rate <= 0 or width < 1 or depth < 1:
        raise ValueError("learning_rate, width, and depth must be positive")
    if not seeds:
        raise ValueError("at least one seed is required")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    evaluation_x = torch.linspace(0.0, 1.0, 2048, device=device_obj).reshape(-1, 1)
    evaluation_y = target_function(evaluation_x)
    records: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for seed in seeds:
        data_generator = torch.Generator(device="cpu").manual_seed(10_000 + seed)
        train_x = torch.rand(train_size, 1, generator=data_generator).to(device_obj)
        train_y = target_function(train_x)

        for model_name in MODEL_NAMES:
            set_seed(seed)
            model = build_model(model_name, width=width, depth=depth).to(device_obj)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            started = time.perf_counter()

            for step in range(1, steps + 1):
                prediction = model(train_x)
                loss = nn.functional.mse_loss(prediction, train_y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if step % checkpoint_every == 0 or step == steps:
                    eval_x = evaluation_x.detach().requires_grad_(True)
                    model.eval()
                    with torch.enable_grad():
                        mse, derivative_mse, spectral_error = _evaluate(
                            model, eval_x, evaluation_y
                        )
                    model.train()
                    record = {
                        "model": model_name,
                        "seed": seed,
                        "step": step,
                        "mse": mse,
                        "derivative_mse": derivative_mse,
                        "frequency_errors": spectral_error,
                    }
                    records.append(record)

            if device_obj.type == "cuda":
                torch.cuda.synchronize(device_obj)
            elapsed = time.perf_counter() - started
            final_record = next(
                row
                for row in reversed(records)
                if row["model"] == model_name
                and row["seed"] == seed
                and row["step"] == steps
            )
            summary.append(
                {
                    **final_record,
                    "training_seconds": elapsed,
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                }
            )

    _write_results(
        output_dir,
        records,
        summary,
        steps=steps,
        seeds=seeds,
        train_size=train_size,
        learning_rate=learning_rate,
        checkpoint_every=checkpoint_every,
        width=width,
        depth=depth,
        device=device,
    )
    return records


def _write_results(
    output_dir: Path,
    records: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    *,
    steps: int,
    seeds: tuple[int, ...],
    train_size: int,
    learning_rate: float,
    checkpoint_every: int,
    width: int,
    depth: int,
    device: str,
) -> None:
    with (output_dir / "learning_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "model",
                "seed",
                "step",
                "mse",
                "derivative_mse",
                "frequency_errors",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **record,
                    "frequency_errors": json.dumps(
                        record["frequency_errors"], sort_keys=True
                    ),
                }
            )

    aggregate = []
    for model_name in MODEL_NAMES:
        rows = [row for row in summary if row["model"] == model_name]
        metric_values = {
            metric: [row[metric] for row in rows]
            for metric in ("mse", "derivative_mse", "training_seconds")
        }
        aggregate.append(
            {
                "model": model_name,
                "seed_count": len(rows),
                "parameter_count": rows[0]["parameter_count"],
                "metrics": {
                    metric: {
                        "mean": statistics.fmean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else None,
                    }
                    for metric, values in metric_values.items()
                },
                "frequency_errors": {
                    frequency: {
                        "mean": statistics.fmean(
                            row["frequency_errors"][frequency] for row in rows
                        ),
                        "std": (
                            statistics.stdev(
                                row["frequency_errors"][frequency] for row in rows
                            )
                            if len(rows) > 1
                            else None
                        ),
                    }
                    for frequency in rows[0]["frequency_errors"]
                },
            }
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "steps": steps,
                "seeds": seeds,
                "train_size": train_size,
                "learning_rate": learning_rate,
                "checkpoint_every": checkpoint_every,
                "width": width,
                "depth": depth,
                "device": device,
                "model_names": MODEL_NAMES,
                "target_frequencies": DEFAULT_FREQUENCIES,
                "evaluation_points": 2048,
                "python_version": sys.version.split()[0],
                "pytorch_version": torch.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
