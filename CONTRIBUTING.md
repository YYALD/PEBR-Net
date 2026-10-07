# Contributing

Thank you for your interest in PEBR-Net.

## Reporting a problem

Open an issue with:

- the command you ran (or the argument file),
- the last 50 lines of the run log (`runs/.../logs/<run>.log`),
- your Python, PyTorch and operating-system versions,
- whether the problem reproduces on the paired dataset (`python scripts/download_data.py --verify` checks the copy).

## Changing the code

The files of `pebrnet/` share one namespace (see [docs/architecture.md](docs/architecture.md)). Please describe a
proposed change in an issue first. A pull request should:

1. keep `python -m pytest` green (the tests run on a CPU in under a minute),
2. add a test for new behaviour,
3. leave the training defaults of [docs/training.md](docs/training.md) unchanged unless the change is the point of
   the pull request.

## Code style

Line length 120 (the ruff settings are in `pyproject.toml`); every figure is exported through `_save_figure` (PDF,
600-dpi PNG and SVG).
