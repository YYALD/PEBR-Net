# Package layout

`pebrnet/__init__.py` executes the topic files, in the order of its `FILES` tuple, into one package namespace.
Module-level state (run tags, caches) and cross-references therefore behave as in a single module, and names defined
in one file are available in every other file without imports.

| file | content |
|---|---|
| `config.py` | imports, program identity, the dataclass configuration tree (`PathConfig`, `DataConfig`, `ModelConfig`, `LossConfig`, `TrainConfig`, `ContractConfig`, `RuntimeConfig`, `Config`), ablation presets |
| `utils.py` | devices and mixed precision, logging and run directories, JSON and hashing, gate-time axes, output tags, final report |
| `noise.py` | `simulate_late_noise` and the composite-noise helpers |
| `data.py` | paired CSV ingestion, gate-time axes, caches, splits and split audits, neighbour geometry |
| `measured.py` | hooks for a measured profile, inactive while `paths.field_csv` is empty |
| `dataset.py` | `BTEMDenoisingDataset`: station patches and the training arms |
| `model.py` | the network (`PEBRNet`) and its layers |
| `losses.py` | `ProjectLoss` and the augmented-Lagrangian `ConstraintController` |
| `training.py` | loaders, optimiser, EMA, `run_epoch`, `evaluate_model`, checkpoints, pre-flight self-test, `train_model` |
| `inference.py` | sliding windows, edge handling, the paired test forward, TorchScript export |
| `evaluation.py` | late-window and generalisation diagnoses, deliverable contract |
| `figures.py` | figure style and export, profile renderers |
| `gate_loss.py` | the gate-loss cohort |
| `cli.py` | the argument parser, the configuration builder and `main()` |
| `public.py` | the command-line front end and the figure package |
| `__main__.py` | `python -m pebrnet` and the `pebrnet` console command |

Each file is also available as a view module, for example `import pebrnet.gate_loss`; it holds the names that file
defines. Because the files share a namespace, a linter reports their cross-file names as undefined. The ruff
configuration in `pyproject.toml` silences that rule for `pebrnet/`. `tests/test_package.py` instead checks, with
Python's `symtable`, that every global any function reads is defined somewhere in the package.

## Runs

A run writes to four roots (`--output_root`, `--model_root`, `--plot_root`, `--log_root`). The run directory under
`output_root` links `checkpoints/` and `figures/` into the model and plot roots. See [outputs.md](outputs.md).
