#!/usr/bin/env python3
"""Main entrypoint for training an SVM classification model.

Usage:
  python main.py --dataset-dir ./dataset --model-path ./src/svm_model.pkl

The script expects the dataset directory to have the following structure:

  dataset/
    images/       # original images (jpg/png/...)
    labels/       # JSON annotation files (one per image)
    label.txt     # list of class names, one per line

Each JSON annotation file should include at least the following fields:
  {
    "image": "example.jpg",
    "label": "cat"
  }

The trained model is saved in the `src/` directory (default: src/svm_model.pkl).
"""

import argparse
from pathlib import Path

from src.svm_model import SVMClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SVM classifier on an image dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./dataset"),
        help="Path to dataset directory containing images/, labels/, and label.txt",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("./model/svm_model.pkl"),
        help="Where to save the trained model (default: model/svm_model.pkl)",
    )
    parser.add_argument("--kernel", type=str, default="rbf", help="SVM kernel (e.g. rbf, linear)")
    parser.add_argument("--C", type=float, default=1.0, help="SVM regularization parameter")
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="Kernel coefficient for rbf/poly/sigmoid (scale, auto, or float)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[64, 64],
        metavar=("WIDTH", "HEIGHT"),
        help="Size to resize images for feature extraction",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = args.dataset_dir
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    label_file = dataset_dir / "label.txt"

    for path in (dataset_dir, images_dir, labels_dir, label_file):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset path not found: {path}")

    classifier = SVMClassifier(
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        image_size=tuple(args.image_size),
    )

    stats = classifier.fit(
        images_dir=images_dir,
        annotations_dir=labels_dir,
        label_file=label_file,
        test_size=args.test_size,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(args.model_path)

    print("Trained and saved model to:", args.model_path)
    print("Training stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
