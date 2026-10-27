# Neural Spectral Lab

A reproducible study of how neural-network architecture affects the learning of low- and high-frequency components in one-dimensional signals.

## Research question

Under the same sampled data and training budget, how do a ReLU MLP, a Fourier-feature MLP, and a SIREN differ in the order and rate at which they learn known frequency components? Do lower prediction error and faster frequency learning also produce lower derivative error?

This repository is an experimental study, not a claim that one architecture is universally superior. Results depend on the target, initialization, optimizer, model size, and compute budget.

## First experiment

The target is a fixed sum of sine waves with frequencies 1, 3, 7, and 13 on `[0, 1]`. Models train on the same randomly sampled points for each seed. The experiment records:

- Overall mean squared prediction error on a dense evaluation grid
- Mean squared derivative error against the analytic target derivative
- Per-frequency coefficient error throughout training
- Final parameter count and wall-clock training time

The comparison is intentionally scoped to frequency-learning dynamics. It is distinct from a broader architecture benchmark, which should separately examine parameter-matched and compute-matched accuracy/efficiency.

## Setup

Python 3.10+ and PyTorch 2.0+ are required. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the default CPU experiment:

```powershell
spectral-lab
```

Or use a short smoke experiment:

```powershell
spectral-lab --steps 100 --seeds 0 --output results-smoke
```

Results include `learning_curves.csv`, per-seed `summary.json`, cross-seed `aggregate.json` (mean and sample standard deviation), and `config.json` in the output directory. CSV frequency errors are stored as JSON objects keyed by frequency. With one seed, aggregate standard deviations are `null`; three seeds are still a small exploratory sample, not strong statistical evidence. Training time is hardware-dependent and should not be compared across machines.

## Initial exploratory run

The `results/` directory contains the default run (1,000 steps, three seeds, one synthetic target). Mean ± sample standard deviation across seeds:

| Model | Parameters | Prediction MSE | Derivative MSE | Training seconds |
|---|---:|---:|---:|---:|
| ReLU MLP | 8,513 | 0.0484 ± 0.0033 | 241.36 ± 11.30 | 1.35 ± 0.04 |
| Fourier-feature MLP | 9,217 | 0.0026 ± 0.0018 | 112.14 ± 46.16 | 1.42 ± 0.02 |
| SIREN | 8,513 | 0.0138 ± 0.0143 | 2,297.55 ± 3,128.01 | 1.52 ± 0.05 |

These are descriptive observations from a single configuration, not a fair, fully tuned ranking: the parameter counts differ, only three seeds were run, and the synthetic target is narrow. The SIREN variation in particular motivates testing initialization and optimization sensitivity before drawing conclusions. Use `results/config.json` to reproduce this run; absolute timing is hardware-dependent.

Run the tests:

```powershell
python -m pytest
```

## Experimental cautions

- The initial result is a controlled synthetic benchmark, not evidence about all real-world tasks.
- A single learning rate and architecture size do not establish that hyperparameters were equally well tuned. Treat this as a first experiment and report that limitation.
- Parameter counts are reported, but equal parameter count does not imply equal FLOPs or wall-clock compute.
- Frequency-wise errors are projections onto the target frequencies; they do not capture every possible error component.
- Repeat runs with more seeds and document any changes to the protocol before drawing conclusions. The default three-seed run is only an initial exploratory baseline.

## Planned extensions

1. Add a frequency-sweep target family and vary sample density and noise.
2. Add confidence intervals over seeds and a plotting utility.
3. Profile compute and memory separately from the spectral-learning analysis.
4. Add other architectures only with a clear research question and verified implementation details.
