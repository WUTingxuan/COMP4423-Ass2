#!/usr/bin/env python3
"""Main entrypoint for training an SVM classification model.

Usage:
  python main.py --dataset-dir ./dataset --model-path ./src/svm_model.pkl

The script expects the dataset directory to have the following structure:

  dataset/
        data.yaml
        train/
            images/     # training images
            labels/     # YOLOv8 label txt files (one per image)

The trained model is saved in the `src/` directory (default: src/svm_model.pkl).
"""

import argparse
from pathlib import Path
from typing import Callable

from src.svm_model import SVMClassifier, load_yolo_class_names


def _print_label_index(labels):
    print("Class index mapping:")
    for idx, label in enumerate(labels):
        print(f"  [{idx:>2}] {label}")


def _print_confusion_matrix(title: str, matrix, formatter: Callable[[float], str]):
    print(title)
    if not matrix:
        print("  <empty>")
        return

    n = len(matrix)
    formatted_rows = [[formatter(value) for value in row] for row in matrix]
    cell_width = max(6, max(len(cell) for row in formatted_rows for cell in row))
    idx_width = max(5, len(str(n - 1)))

    header_cells = [f"{i:>{cell_width}}" for i in range(n)]
    axis_label = "true\\pred"
    print(f"{axis_label:>{idx_width}} " + " ".join(header_cells))

    for i, row in enumerate(formatted_rows):
        print(f"{i:>{idx_width}} " + " ".join([f"{cell:>{cell_width}}" for cell in row]))


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SVM classifier on an image dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("./dataset"),
        help="Path to YOLOv8 dataset directory containing data.yaml and train/{images,labels}",
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
    data_yaml = dataset_dir / "data.yaml"
    images_dir = dataset_dir / "train" / "images"
    labels_dir = dataset_dir / "train" / "labels"

    for path in (dataset_dir, data_yaml, images_dir, labels_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset path not found: {path}")

    class_names = load_yolo_class_names(data_yaml)

    classifier = SVMClassifier(
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        image_size=tuple(args.image_size),
    )

    stats = classifier.fit(
        images_dir=images_dir,
        annotations_dir=labels_dir,
        class_names=class_names,
        test_size=args.test_size,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(args.model_path)

    print("Trained and saved model to:", args.model_path)
    print("Training stats:")
    summary_keys = [
        "train_accuracy",
        "val_accuracy",
        "val_macro_precision",
        "val_macro_recall",
        "val_macro_f1",
        "val_weighted_precision",
        "val_weighted_recall",
        "val_weighted_f1",
        "num_classes",
        "num_examples",
        "num_groups",
        "used_group_split",
        "num_feature_samples_mixed_train",
        "num_feature_samples_mixed_val",
    ]
    for key in summary_keys:
        print(f"  {key}: {stats[key]}")

    print("\nFeature strategy comparison:")
    print(
        "  mixed: "
        f"train_acc={stats['comparison']['mixed']['train_accuracy']:.4f}, "
        f"val_acc={stats['comparison']['mixed']['val_accuracy']:.4f}, "
        f"val_macro_f1={stats['comparison']['mixed']['val_macro_f1']:.4f}"
    )
    print(
        "  masked_only: "
        f"train_acc={stats['comparison']['masked_only']['train_accuracy']:.4f}, "
        f"val_acc={stats['comparison']['masked_only']['val_accuracy']:.4f}, "
        f"val_macro_f1={stats['comparison']['masked_only']['val_macro_f1']:.4f}"
    )

    print("\nPer-class metrics (validation):")
    for class_name, metrics in stats["class_metrics"].items():
        print(
            f"  {class_name}: "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1_score']:.4f}, "
            f"support={metrics['support']}"
        )

    labels = stats["confusion_matrix"]["labels"]
    _print_label_index(labels)

    _print_confusion_matrix(
        "\nConfusion matrix (counts):",
        stats["confusion_matrix"]["matrix"],
        lambda value: str(int(value)),
    )

    _print_confusion_matrix(
        "\nConfusion matrix (row-normalized):",
        stats["confusion_matrix"]["normalized"],
        lambda value: f"{float(value):.4f}",
    )


if __name__ == "__main__":
    main()
