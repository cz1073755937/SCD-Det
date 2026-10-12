# SCD-Det

Public research implementation of **SCD-Det**, a scale-consistent detector for dense small objects in industrial imagery.

This repository is a focused code release. It contains the method implementation and portable examples, but it does **not** contain private datasets, unpublished reviewer correspondence, internal experiment orchestration, checkpoints, or unfinished benchmark tables.

## What is included

- FREG: max-energy regional routing with bottleneck attention and local edge compensation.
- CEBA: local-detail and background-context multiplicative enhancement.
- UMA: learnable multiscale aggregation aligned to a configurable target level.
- BSC: the paper loss, a class-wise extension for multiclass data, and two controlled alternative expressions.
- The selected SCD-Det-n architecture used by the revision experiments.
- An integration patch pinned to Ultralytics 8.3.175.
- Portable training/data templates, train/validation entry points, and CPU smoke tests.

The released selected architecture uses:

| Component | Public default |
|---|---:|
| FREG | `k=12`, `S=10`, `heads=8`, `ratio=0.25` |
| CEBA | context kernel `9x9` |
| UMA | P3 alignment (`target_level=0`) |
| Single-class BSC | `lambda_cons=0.01`, global foreground grouping |
| CODD-style multiclass BSC | `lambda_cons=0.005`, class-wise grouping |

The multiclass weight is validation-selected for CODD. It should be reselected on the validation split when transferring to a different dataset; the test set must not be used for hyperparameter selection.

## Installation

The implementation is pinned to **Ultralytics 8.3.175** because model-parser and loss APIs change between releases.

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

Verify the installation:

```bash
python -m pytest -q
python -c "from ultralytics import YOLO; YOLO('models/scd_det_n.yaml').info()"
```

Expected model summary: approximately **2.18 M parameters and 12.0 GFLOPs** at 640-pixel input under Ultralytics 8.3.175.

## Dataset format

Copy `data/dataset.example.yaml`, replace the paths and class names, and arrange labels in standard Ultralytics YOLO detection format:

```text
datasets/your_dataset/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Each label line is `class_id x_center y_center width height`, with coordinates normalized to `[0, 1]`. Keep validation and test splits separate.

MCDD and WDD are private industrial datasets and are intentionally not distributed. See [private dataset reproducibility](docs/private_dataset_reproducibility.md) for the metadata that can still be released without exposing images.

## Training and evaluation

From the repository root:

```bash
# Single-class data
python tools/train.py --cfg configs/train_single_class.example.yaml --device 0

# Multiclass data with class-wise BSC
python tools/train.py --cfg configs/train_multiclass.example.yaml --device 0

# Evaluate a fixed checkpoint on the test split
python tools/val.py runs/scd_det/single_class_seed_0/weights/best.pt \
  --data data/your_dataset.yaml --split test --device 0
```

The examples reproduce the revision protocol defaults: 300 epochs, batch 32, image size 640, SGD, no pretrained weights, no mosaic, no scale augmentation, and deterministic seeds. Adjust paths and worker count to the local machine.

## BSC implementation details

BSC is computed only on decoded foreground boxes selected by Ultralytics' task-aligned assigner. For $N_+$ selected boxes with widths $w_i$ and heights $h_i$:

$$
\mathcal{L}_{\mathrm{BSC}} =
\frac{\lambda}{N_+}\sum_i
\left[
\frac{(w_i-\bar{w})^2}{\bar{w}^2+\epsilon}+
\frac{(h_i-\bar{h})^2}{\bar{h}^2+\epsilon}
\right].
$$

The boxes are decoded `xyxy` predictions in feature-grid units at the loss interface. A common coordinate scaling cancels in the relative-variance expression. For multiclass datasets, `bsc_classwise: true` computes the term separately for each assigned class with at least two foreground boxes, then averages valid class losses.

The `gt_scale_center` and `log_scale_huber` modes are released for controlled ablations; the paper model uses `bsc_mode: paper`.

## Release scope and results

The exact public/private boundary is documented in [release scope](docs/release_scope.md). Numerical result tables and weights are deliberately not frozen into this initial code package while the revision experiments are still being completed and audited. Add them only after verifying seeds, splits, checkpoint identity, and timing protocol.

## License

This code is released under the [GNU AGPL-3.0](LICENSE), matching the Ultralytics codebase used by the integration patch. Ultralytics and other third-party components retain their own copyright notices.

## Citation

The manuscript citation should be inserted after its bibliographic record is final. Until then, cite the software:

```bibtex
@software{scddet_code,
  title  = {SCD-Det: Public Research Implementation},
  author = {Chen, Zhi and Chu, Maoxiang and Gong, Rongfen},
  year   = {2026},
  note   = {Research software}
}
```
