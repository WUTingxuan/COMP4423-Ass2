import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib


def _load_yolo_class_names(data_yaml: Path) -> List[str]:
    """Load class names from YOLOv8 data.yaml (the `names` field)."""
    names_value = None
    with data_yaml.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("names:"):
                names_value = stripped.split("names:", 1)[1].strip()
                break

    if not names_value:
        raise ValueError(f"Could not find 'names' in {data_yaml}")

    try:
        names = ast.literal_eval(names_value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid names format in {data_yaml}") from exc

    if not isinstance(names, list) or not names:
        raise ValueError(f"'names' in {data_yaml} must be a non-empty list")

    return [str(x) for x in names]


def _parse_yolo_region(tokens: List[float]) -> Optional[Tuple[float, float, float, float]]:
    """Parse YOLO bbox/segmentation tokens into normalized (x1, y1, x2, y2)."""
    if len(tokens) == 4:
        # Detection format: x_center, y_center, width, height
        x_c, y_c, w, h = tokens
        x1 = x_c - w / 2.0
        y1 = y_c - h / 2.0
        x2 = x_c + w / 2.0
        y2 = y_c + h / 2.0
    elif len(tokens) >= 6 and len(tokens) % 2 == 0:
        # Segmentation format: x1 y1 x2 y2 ...
        xs = tokens[0::2]
        ys = tokens[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    else:
        return None

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _parse_yolo_polygon(tokens: List[float]) -> Optional[List[Tuple[float, float]]]:
    """Parse YOLO segmentation tokens into normalized polygon points."""
    if len(tokens) < 6 or len(tokens) % 2 != 0:
        return None
    # Detection format has exactly 4 values and is not a polygon.
    if len(tokens) == 4:
        return None

    polygon: List[Tuple[float, float]] = []
    for i in range(0, len(tokens), 2):
        x = max(0.0, min(1.0, tokens[i]))
        y = max(0.0, min(1.0, tokens[i + 1]))
        polygon.append((x, y))

    return polygon if len(polygon) >= 3 else None


def _load_annotations(annotations_dir: Path, class_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load YOLOv8 txt annotations mapping image stem to label and optional region.

    Each label file line is expected as:
    <class_id> <x_center> <y_center> <width> <height>

    For image-level classification, we use the first object's class_id.
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    for txt_path in sorted(annotations_dir.glob("*.txt")):
        try:
            with txt_path.open("r", encoding="utf-8") as f:
                first_line = next((line.strip() for line in f if line.strip()), "")

            if not first_line:
                continue

            parts = first_line.split()
            class_id = int(parts[0])
            if class_id < 0 or class_id >= len(class_names):
                continue

            region_tokens = [float(v) for v in parts[1:]]
            region = _parse_yolo_region(region_tokens)
            polygon = _parse_yolo_polygon(region_tokens)
            mapping[txt_path.stem] = {
                "label": class_names[class_id],
                "region": region,
                "polygon": polygon,
            }
        except (ValueError, IndexError, OSError):
            continue
    return mapping


def _extract_image_feature(
    image_path: Path,
    size: Tuple[int, int],
    region: Optional[Tuple[float, float, float, float]] = None,
    polygon: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Extract a flat feature vector from an image.

    This uses a simple approach: convert to grayscale, resize, and flatten.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")

        if polygon is not None:
            w, h = img.size
            pixel_polygon = [
                (max(0, min(w - 1, int(round(x * w)))), max(0, min(h - 1, int(round(y * h)))))
                for x, y in polygon
            ]

            # Keep object region and neutralize background to reduce background leakage.
            mask = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(pixel_polygon, fill=255)
            neutral_bg = Image.new("RGB", (w, h), (128, 128, 128))
            img = Image.composite(img, neutral_bg, mask)

        if region is not None:
            x1, y1, x2, y2 = region
            w, h = img.size
            left = int(round(x1 * w))
            top = int(round(y1 * h))
            right = int(round(x2 * w))
            bottom = int(round(y2 * h))
            if right - left >= 2 and bottom - top >= 2:
                img = img.crop((left, top, right, bottom))

        img = img.convert("L")
        img = img.resize(size, Image.Resampling.BILINEAR)
        array = np.asarray(img, dtype=np.float32) / 255.0
        return array.ravel()


def _extract_image_feature_from_pil(
    image: Image.Image,
    size: Tuple[int, int],
    region: Optional[Tuple[float, float, float, float]] = None,
) -> np.ndarray:
    """Extract feature from an in-memory PIL image using the same pipeline as training."""
    img = image.convert("RGB")
    if region is not None:
        x1, y1, x2, y2 = region
        w, h = img.size
        left = int(round(x1 * w))
        top = int(round(y1 * h))
        right = int(round(x2 * w))
        bottom = int(round(y2 * h))
        if right - left >= 2 and bottom - top >= 2:
            img = img.crop((left, top, right, bottom))

    img = img.convert("L")
    img = img.resize(size, Image.Resampling.BILINEAR)
    array = np.asarray(img, dtype=np.float32) / 255.0
    return array.ravel()


def _estimate_object_region(image: Image.Image) -> Optional[Tuple[float, float, float, float]]:
    """Estimate an object-like region using gradient energy to suppress background influence."""
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if gray.size == 0:
        return None

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    energy = gx + gy

    threshold = float(np.percentile(energy, 88))
    mask = energy >= threshold
    ys, xs = np.where(mask)
    if len(xs) < 100:
        return None

    h, w = gray.shape
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Expand a little for context around target.
    pad_x = int(0.06 * w)
    pad_y = int(0.06 * h)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1 / w, y1 / h, x2 / w, y2 / h)


def _group_key_from_stem(image_stem: str) -> str:
    """Build a group key to keep neighboring video frames in the same split."""
    marker = "_frame_"
    if marker in image_stem:
        return image_stem.split(marker, 1)[0]
    return image_stem


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
        class_names: List[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Train a model from a dataset stored in `images_dir` and `annotations_dir`."""
        annotations = _load_annotations(annotations_dir, class_names)
        records: List[Dict[str, Any]] = []
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        image_paths = []
        for pattern in image_extensions:
            image_paths.extend(images_dir.glob(pattern))

        for image_idx, image_path in enumerate(sorted(image_paths)):
            ann = annotations.get(image_path.stem)
            if ann is None:
                continue
            label = ann["label"]
            region = ann.get("region")
            polygon = ann.get("polygon")
            group_key = _group_key_from_stem(image_path.stem)

            global_feature = _extract_image_feature(image_path, self.image_size)
            region_feature = None
            if region is not None:
                region_feature = _extract_image_feature(image_path, self.image_size, region=region)

            masked_feature = None
            if polygon is not None:
                masked_feature = _extract_image_feature(
                    image_path,
                    self.image_size,
                    region=region,
                    polygon=polygon,
                )

            records.append(
                {
                    "image_idx": image_idx,
                    "label": label,
                    "group": group_key,
                    "global_feature": global_feature,
                    "region_feature": region_feature,
                    "masked_feature": masked_feature,
                }
            )

        if not records:
            raise ValueError("No training data found. Check dataset paths and annotation files.")

        image_labels = np.array([item["label"] for item in records])
        image_groups = np.array([item["group"] for item in records])
        y_image_enc = self.label_encoder.fit_transform(image_labels)
        class_to_id = {label: i for i, label in enumerate(self.label_encoder.classes_)}

        def _build_mixed_dataset(indices: np.ndarray, global_stride: int = 3):
            X_samples: List[np.ndarray] = []
            y_samples: List[int] = []

            for idx in indices:
                item = records[int(idx)]
                label_id = class_to_id[item["label"]]
                has_object_focus = item["region_feature"] is not None or item["masked_feature"] is not None

                # Reduce global-context weight to avoid learning background shortcuts.
                if (int(item["image_idx"]) % global_stride == 0) or (not has_object_focus):
                    X_samples.append(item["global_feature"])
                    y_samples.append(label_id)

                if item["region_feature"] is not None:
                    X_samples.append(item["region_feature"])
                    y_samples.append(label_id)

                if item["masked_feature"] is not None:
                    X_samples.append(item["masked_feature"])
                    y_samples.append(label_id)

            return np.stack(X_samples), np.array(y_samples)

        def _build_masked_only_dataset(indices: np.ndarray):
            X_samples: List[np.ndarray] = []
            y_samples: List[int] = []
            for idx in indices:
                item = records[int(idx)]
                label_id = class_to_id[item["label"]]
                feature = item["masked_feature"]
                if feature is None:
                    feature = item["region_feature"]
                if feature is None:
                    feature = item["global_feature"]

                X_samples.append(feature)
                y_samples.append(label_id)
            return np.stack(X_samples), np.array(y_samples)

        def _stratified_split():
            stratify = y_image_enc
            try:
                train_idx, val_idx, _, _ = train_test_split(
                    np.arange(len(records)),
                    y_image_enc,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify,
                )
                return np.array(train_idx), np.array(val_idx)
            except ValueError:
                train_idx, val_idx, _, _ = train_test_split(
                    np.arange(len(records)),
                    y_image_enc,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=None,
                )
                return np.array(train_idx), np.array(val_idx)

        unique_groups = np.unique(image_groups)
        used_group_split = False
        if len(unique_groups) >= 3:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, val_idx = next(
                splitter.split(np.arange(len(records)), y_image_enc, groups=image_groups)
            )
            train_idx = np.array(train_idx)
            val_idx = np.array(val_idx)
            y_train_image = y_image_enc[train_idx]
            y_val_image = y_image_enc[val_idx]

            train_classes = set(np.unique(y_train_image).tolist())
            val_classes = set(np.unique(y_val_image).tolist())
            if val_classes.issubset(train_classes):
                used_group_split = True
            else:
                train_idx, val_idx = _stratified_split()
        else:
            train_idx, val_idx = _stratified_split()

        X_train, y_train = _build_mixed_dataset(train_idx, global_stride=3)
        X_val, y_val = _build_mixed_dataset(val_idx, global_stride=3)

        X_train_masked, y_train_masked = _build_masked_only_dataset(train_idx)
        X_val_masked, y_val_masked = _build_masked_only_dataset(val_idx)

        def _make_model():
            return make_pipeline(
                StandardScaler(),
                PCA(n_components=0.95, svd_solver="full", random_state=random_state),
                SVC(
                    kernel=self.kernel,
                    C=self.C,
                    gamma=self.gamma,
                    class_weight="balanced",
                    probability=True,
                ),
            )

        self.model = _make_model()
        self.model.fit(X_train, y_train)

        masked_only_model = _make_model()
        masked_only_model.fit(X_train_masked, y_train_masked)

        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        class_ids = np.arange(len(self.label_encoder.classes_))
        class_labels = list(self.label_encoder.classes_)

        per_class_precision, per_class_recall, per_class_f1, per_class_support = (
            precision_recall_fscore_support(
                y_val,
                y_val_pred,
                labels=class_ids,
                average=None,
                zero_division=0,
            )
        )

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_val,
            y_val_pred,
            average="macro",
            zero_division=0,
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_val,
            y_val_pred,
            average="weighted",
            zero_division=0,
        )

        masked_train_pred = masked_only_model.predict(X_train_masked)
        masked_val_pred = masked_only_model.predict(X_val_masked)
        masked_train_acc = accuracy_score(y_train_masked, masked_train_pred)
        masked_val_acc = accuracy_score(y_val_masked, masked_val_pred)
        masked_macro_f1 = precision_recall_fscore_support(
            y_val_masked,
            masked_val_pred,
            average="macro",
            zero_division=0,
        )[2]

        confusion = confusion_matrix(y_val, y_val_pred, labels=class_ids)
        row_sums = confusion.sum(axis=1, keepdims=True)
        confusion_normalized = np.divide(
            confusion,
            row_sums,
            out=np.zeros_like(confusion, dtype=np.float64),
            where=row_sums != 0,
        )

        per_class_metrics: Dict[str, Dict[str, float]] = {}
        for idx, class_name in enumerate(class_labels):
            per_class_metrics[class_name] = {
                "precision": float(per_class_precision[idx]),
                "recall": float(per_class_recall[idx]),
                "f1_score": float(per_class_f1[idx]),
                "support": int(per_class_support[idx]),
            }

        return {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "val_macro_precision": float(macro_precision),
            "val_macro_recall": float(macro_recall),
            "val_macro_f1": float(macro_f1),
            "val_weighted_precision": float(weighted_precision),
            "val_weighted_recall": float(weighted_recall),
            "val_weighted_f1": float(weighted_f1),
            "num_classes": len(self.label_encoder.classes_),
            "num_examples": len(records),
            "num_groups": int(len(unique_groups)),
            "used_group_split": bool(used_group_split),
            "num_feature_samples_mixed_train": int(len(y_train)),
            "num_feature_samples_mixed_val": int(len(y_val)),
            "comparison": {
                "masked_only": {
                    "train_accuracy": float(masked_train_acc),
                    "val_accuracy": float(masked_val_acc),
                    "val_macro_f1": float(masked_macro_f1),
                },
                "mixed": {
                    "train_accuracy": float(train_acc),
                    "val_accuracy": float(val_acc),
                    "val_macro_f1": float(macro_f1),
                },
            },
            "class_metrics": per_class_metrics,
            "confusion_matrix": {
                "labels": class_labels,
                "matrix": confusion.tolist(),
                "normalized": confusion_normalized.tolist(),
            },
        }

    def predict(self, image_path: Path) -> str:
        """Predict a label for a single image."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        feature = _extract_image_feature(image_path, self.image_size).reshape(1, -1)
        class_id = self.model.predict(feature)[0]
        return self.label_encoder.inverse_transform([class_id])[0]

    def _predict_proba_with_attention(self, image_path: Path) -> np.ndarray:
        """Predict probabilities with target-attention fusion to reduce background bias."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            # Candidate regions: global + center crops + auto object region.
            candidates: List[Tuple[Optional[Tuple[float, float, float, float]], float]] = [
                (None, 0.15),
                ((0.10, 0.10, 0.90, 0.90), 0.20),
                ((0.20, 0.20, 0.80, 0.80), 0.25),
            ]

            estimated = _estimate_object_region(image)
            if estimated is not None:
                candidates.append((estimated, 0.40))

            total_weight = sum(weight for _, weight in candidates)
            probs_sum = None
            for region, weight in candidates:
                feature = _extract_image_feature_from_pil(image, self.image_size, region=region).reshape(1, -1)
                probs = self.model.predict_proba(feature)[0]
                if probs_sum is None:
                    probs_sum = np.zeros_like(probs, dtype=np.float64)
                probs_sum += weight * probs

            if probs_sum is None:
                # Defensive fallback to global feature.
                feature = _extract_image_feature_from_pil(image, self.image_size).reshape(1, -1)
                return self.model.predict_proba(feature)[0]

            return probs_sum / max(total_weight, 1e-9)

    def predict_with_confidence(
        self,
        image_path: Path,
        top_k: int = 3,
        use_attention: bool = True,
    ) -> Dict[str, Any]:
        """Predict label and return confidence details for a single image."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")

        if hasattr(self.model, "predict_proba"):
            if use_attention:
                probs = self._predict_proba_with_attention(image_path)
            else:
                feature = _extract_image_feature(image_path, self.image_size).reshape(1, -1)
                probs = self.model.predict_proba(feature)[0]
            if hasattr(self.model, "classes_"):
                class_ids = np.asarray(self.model.classes_, dtype=int)
            else:
                class_ids = np.arange(len(probs), dtype=int)

            ranked_indices = np.argsort(probs)[::-1]
            pred_idx = ranked_indices[0]
            pred_class_id = int(class_ids[pred_idx])
            pred_label = self.label_encoder.inverse_transform([pred_class_id])[0]
            pred_confidence = float(probs[pred_idx])

            top_items = []
            for idx in ranked_indices[: max(1, top_k)]:
                class_id = int(class_ids[idx])
                label = self.label_encoder.inverse_transform([class_id])[0]
                top_items.append({"label": label, "probability": float(probs[idx])})
            return {
                "label": pred_label,
                "confidence": pred_confidence,
                "top_k": top_items,
            }

        # Fallback when predict_proba is unavailable
        pred_class_id = int(self.model.predict(feature)[0])
        pred_label = self.label_encoder.inverse_transform([pred_class_id])[0]
        return {
            "label": pred_label,
            "confidence": 1.0,
            "top_k": [{"label": pred_label, "probability": 1.0}],
        }

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


def load_yolo_class_names(data_yaml: Path) -> List[str]:
    """Public helper to read class names from a YOLOv8 data.yaml file."""
    return _load_yolo_class_names(data_yaml)
