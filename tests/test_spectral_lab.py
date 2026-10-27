import json

import torch

from spectral_lab.experiment import frequency_errors, run_experiment
from spectral_lab.models import build_model
from spectral_lab.targets import target_derivative, target_function


def test_target_derivative_matches_finite_difference() -> None:
    x = torch.linspace(0.1, 0.9, 20).reshape(-1, 1)
    epsilon = 1e-4
    finite_difference = (
        target_function(x + epsilon) - target_function(x - epsilon)
    ) / (2 * epsilon)
    torch.testing.assert_close(
        target_derivative(x), finite_difference, rtol=2e-3, atol=2e-3
    )


def test_frequency_error_is_zero_for_identical_signal() -> None:
    x = torch.linspace(0.0, 1.0, 1024).reshape(-1, 1)
    y = target_function(x)
    errors = frequency_errors(y, y, x)
    assert all(error < 1e-7 for error in errors.values())


def test_models_return_one_value_per_input() -> None:
    x = torch.rand(12, 1)
    for name in ("mlp", "fourier", "siren"):
        assert build_model(name)(x).shape == (12, 1)


def test_small_experiment_writes_reproducible_outputs(tmp_path) -> None:
    records = run_experiment(
        output_dir=tmp_path,
        steps=2,
        seeds=(0,),
        train_size=16,
        checkpoint_every=1,
        width=8,
        depth=1,
    )

    assert len(records) == 6
    assert (tmp_path / "learning_curves.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "aggregate.json").is_file()
    assert (tmp_path / "config.json").is_file()
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["steps"] == 2
    assert config["train_size"] == 16
    aggregate = json.loads(
        (tmp_path / "aggregate.json").read_text(encoding="utf-8")
    )
    assert len(aggregate) == 3
    assert aggregate[0]["seed_count"] == 1
    assert aggregate[0]["metrics"]["mse"]["std"] is None
