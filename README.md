# SCD-Det

Official implementation of **SCD-Det: a scale-consistent detector for dense small objects in industrial production lines**.

Zhi Chen, Maoxiang Chu, and Rongfen Gong. *Journal of Real-Time Image Processing*, 23, article 187 (2026). [Paper / DOI](https://doi.org/10.1007/s11554-026-01982-1).

This repository contains the core modules, model configuration, and basic training and evaluation utilities.

## Repository structure

- `scd_det/`: FREG, CEBA, UMA, and BSC loss implementations.
- `models/`: SCD-Det network configuration.
- `patches/`: integration patch for Ultralytics 8.3.175.
- `configs/` and `data/`: portable training and dataset templates.
- `tools/`: training, validation, and release checks.
- `tests/`: basic CPU tests for the modules and loss.

## Installation

The integration patch targets **Ultralytics 8.3.175**.

```bash
git clone https://github.com/cz1073755937/SCD-Det.git
cd SCD-Det

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

git clone --branch v8.3.175 --depth 1 https://github.com/ultralytics/ultralytics.git vendor/ultralytics
git -C vendor/ultralytics apply "$PWD/patches/ultralytics-8.3.175-scd-det.patch"
python -m pip install -e vendor/ultralytics
python -m pip install -e ".[test]"
```

## Training and evaluation

Edit the dataset paths and class names in `data/dataset.example.yaml`, then adjust the training configuration as needed.

```bash
# Single-class training
python tools/train.py --cfg configs/train_single_class.example.yaml --device 0

# Multiclass training with class-wise BSC
python tools/train.py --cfg configs/train_multiclass.example.yaml --device 0

# Test evaluation
python tools/val.py runs/scd_det/single_class_seed_0/weights/best.pt \
  --data data/dataset.example.yaml --split test --device 0

# Basic component tests
python -m pytest -q
```

The templates default to seed 0. The paper uses seeds 0, 1, and 2 with fixed dataset splits.

MCDD and WDD are not bundled with the code. The paper provides access upon reasonable request to the corresponding author; see [dataset notes](docs/private_dataset_reproducibility.md).

## Citation

If you use SCD-Det, please cite the published paper:

```bibtex
@article{chen2026scddet,
  title   = {{SCD-Det}: a scale-consistent detector for dense small objects in industrial production lines},
  author  = {Chen, Zhi and Chu, Maoxiang and Gong, Rongfen},
  journal = {Journal of Real-Time Image Processing},
  volume  = {23},
  pages   = {187},
  year    = {2026},
  doi     = {10.1007/s11554-026-01982-1},
  url     = {https://doi.org/10.1007/s11554-026-01982-1}
}
```

`187` is the article number. Citation metadata is also available in [CITATION.cff](CITATION.cff).

## License

This repository is released under [AGPL-3.0](LICENSE). See [NOTICE](NOTICE) for third-party notices.
