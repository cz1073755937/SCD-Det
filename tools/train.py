"""Train SCD-Det from a portable Ultralytics configuration file."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, default=Path("configs/train_single_class.example.yaml"))
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0 or 0,1; use cpu for CPU")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="Resume last run or a .pt path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.cfg.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    model = YOLO("models/scd_det_n.yaml")
    model.train(cfg=str(config), device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
