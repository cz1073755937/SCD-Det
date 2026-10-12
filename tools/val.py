"""Evaluate a trained SCD-Det checkpoint on val or test data."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/dataset.example.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    YOLO(str(args.weights)).val(
        data=str(args.data.resolve()),
        split=args.split,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
