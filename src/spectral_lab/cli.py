from __future__ import annotations

import argparse
from pathlib import Path

from spectral_lab.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frequency learning in MLP, Fourier-feature MLP, and SIREN."
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    run_experiment(
        output_dir=args.output,
        steps=args.steps,
        seeds=tuple(args.seeds),
        train_size=args.train_size,
        learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every,
        width=args.width,
        depth=args.depth,
        device=args.device,
    )
    print(f"Experiment results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
