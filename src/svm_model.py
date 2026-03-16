import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib


def _load_labels(label_file: Path) -> List[str]:
    """Load label names from label.txt."""
    with label_file.open("r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def _load_annotations(annotations_dir: Path) -> Dict[str, str]:
    """Load JSON annotations mapping image filenames to labels.

    Expected JSON format per file:
    {
        "image": "image_01.jpg",
        "label": "cat"
    }
    """
    mapping: Dict[str, str] = {}
    for json_path in sorted(annotations_dir.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            image_name = data.get("image") or data.get("filename")
            label = data.get("label")
            if not image_name or not label:
                continue
            mapping[image_name] = label
        except Exception:
            # Skip invalid JSON files
            continue
    return mapping


def _extract_image_feature(image_path: Path, size: Tuple[int, int]) -> np.ndarray:
    """Extract a flat feature vector from an image.

    This uses a simple approach: convert to grayscale, resize, and flatten.
    """
    with Image.open(image_path) as img:
        img = img.convert("L")
        img = img.resize(size, Image.Resampling.BILINEAR)
        array = np.asarray(img, dtype=np.float32) / 255.0
        return array.ravel()


class SVMClassifier:
    """A simple SVM classifier wrapper."""

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        image_size: Tuple[int, int] = (64, 64),
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.image_size = image_size
        self.label_encoder = LabelEncoder()
        self.model: Optional[BaseEstimator] = None

    def fit(
        self,
        images_dir: Path,
        annotations_dir: Path,
        label_file: Path,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Train a model from a dataset stored in `images_dir` and `annotations_dir`."""
        labels = _load_labels(label_file)
        label_mapping = {label: i for i, label in enumerate(labels)}

        annotations = _load_annotations(annotations_dir)
        X = []
        y = []
        for image_name, label in annotations.items():
            image_path = images_dir / image_name
            if not image_path.exists():
                continue
            if label not in label_mapping:
                continue
            X.append(_extract_image_feature(image_path, self.image_size))
            y.append(label)

        if not X or not y:
            raise ValueError("No training data found. Check dataset paths and annotation files.")

        X = np.stack(X)
        y = np.array(y)
        y_enc = self.label_encoder.fit_transform(y)

        # Stratified split can fail if some classes have fewer than 2 examples.
        stratify = y_enc
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_enc, test_size=test_size, random_state=random_state, stratify=stratify
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_enc, test_size=test_size, random_state=random_state, stratify=None
            )

        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, probability=True)
        self.model.fit(X_train, y_train)

        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)

        return {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "num_classes": len(self.label_encoder.classes_),
            "num_examples": len(y),
        }

    def predict(self, image_path: Path) -> str:
        """Predict a label for a single image."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        feature = _extract_image_feature(image_path, self.image_size).reshape(1, -1)
        class_id = self.model.predict(feature)[0]
        return self.label_encoder.inverse_transform([class_id])[0]

    def save(self, output_path: Path) -> None:
        """Save the model (including label encoder) to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        output = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "image_size": self.image_size,
        }
        joblib.dump(output, output_path)

    @classmethod
    def load(cls, model_path: Path) -> "SVMClassifier":
        """Load a saved model."""
        data = joblib.load(model_path)
        obj = cls(
            kernel=data.get("kernel", "rbf"),
            C=data.get("C", 1.0),
            gamma=data.get("gamma", "scale"),
            image_size=tuple(data.get("image_size", (64, 64))),
        )
        obj.model = data["model"]
        obj.label_encoder = data["label_encoder"]
        return obj
