from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_canvas = None

from src.svm_model import SVMClassifier


DEFAULT_MODEL_PATH = Path("model/svm_model.pkl")
DEFAULT_DATASET_IMAGES_DIR = Path("dataset/train/images")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@st.cache_resource
def load_model(model_path: str) -> SVMClassifier:
    return SVMClassifier.load(Path(model_path))


def list_dataset_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )


def _enhance_field_photo(image: Image.Image) -> Image.Image:
    """Apply light normalization for field photos to reduce lighting/domain shift."""
    image = image.convert("RGB")
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Color(image).enhance(1.08)
    image = ImageEnhance.Sharpness(image).enhance(1.12)
    return image


def _predict_from_pil_image(
    model: SVMClassifier,
    image: Image.Image,
    suffix: str,
    top_k: int,
    use_attention: bool,
) -> dict:
    suffix = suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        image.save(tmp, format=("PNG" if suffix.lower() == ".png" else "JPEG"))
        tmp_path = Path(tmp.name)

    try:
        return model.predict_with_confidence(tmp_path, top_k=top_k, use_attention=use_attention)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _render_prediction(prediction: dict, threshold: float) -> None:
    label = prediction["label"]
    confidence = float(prediction["confidence"])
    st.success(f"Predicted vegetation class: {label}")
    st.metric("Prediction confidence", f"{confidence * 100:.2f}%")
    st.progress(min(max(confidence, 0.0), 1.0))

    if confidence < threshold:
        st.warning(
            "Low confidence prediction. Consider re-shooting with better lighting/closer focus, "
            "or adding similar photos into training data."
        )

    st.write("Top probabilities:")
    for item in prediction.get("top_k", []):
        st.write(f"- {item['label']}: {item['probability'] * 100:.2f}%")


def _apply_manual_roi(image: Image.Image, key_prefix: str) -> Image.Image:
    """Let user specify a manual ROI and return cropped image if enabled."""
    use_manual_roi = st.checkbox(
        "Use manual ROI crop",
        value=False,
        key=f"{key_prefix}_use_manual_roi",
        help="Manually crop the target object before prediction to reduce background impact.",
    )
    if not use_manual_roi:
        return image

    width, height = image.size

    if st_canvas is None:
        st.error(
            "Manual ROI drag is required but streamlit-drawable-canvas is not available. "
            "Please install compatible versions and restart the app."
        )
        st.stop()
    else:
        st.caption("Draw one rectangle around the target (drag on image).")
        canvas_width = min(900, width)
        scale = canvas_width / float(width)
        canvas_height = max(1, int(round(height * scale)))
        try:
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=2,
                stroke_color="#00ff66",
                background_image=image.resize((canvas_width, canvas_height)),
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                point_display_radius=0,
                key=f"{key_prefix}_roi_canvas",
            )

            left = top = right = bottom = None
            objects = None
            if canvas_result and canvas_result.json_data:
                objects = canvas_result.json_data.get("objects", [])

            if objects:
                rect = objects[-1]
                x = float(rect.get("left", 0.0))
                y = float(rect.get("top", 0.0))
                w = float(rect.get("width", 0.0)) * float(rect.get("scaleX", 1.0))
                h = float(rect.get("height", 0.0)) * float(rect.get("scaleY", 1.0))

                left = int(round(x / scale))
                top = int(round(y / scale))
                right = int(round((x + w) / scale))
                bottom = int(round((y + h) / scale))

            if left is None:
                st.warning("No ROI drawn yet. Using full image.")
                return image
        except AttributeError:
            st.error(
                "Manual ROI drag is incompatible with current streamlit version in this environment. "
                "Please install a compatible version pair (streamlit / streamlit-drawable-canvas)."
            )
            st.stop()

    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(0, min(width, right))
    bottom = max(0, min(height, bottom))

    if right - left < 2 or bottom - top < 2:
        st.warning("ROI is too small. Falling back to full image.")
        return image

    roi_image = image.crop((left, top, right, bottom))
    st.image(roi_image, caption="Manual ROI used for prediction", use_column_width=True)
    return roi_image


def main() -> None:
    st.set_page_config(page_title="Vegetation Classifier", page_icon="🌿", layout="centered")
    st.title("Vegetation Category Prediction")
    st.write("Upload an image or select one from the dataset, then get the predicted vegetation class.")

    model_path_str = st.text_input("Model path", str(DEFAULT_MODEL_PATH))

    try:
        model = load_model(model_path_str)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.info("Train a model first with: python main.py --dataset-dir ./dataset --model-path ./model/svm_model.pkl")
        return

    mode = st.radio("Choose input mode", ["Upload image", "Select dataset image"], horizontal=True)
    enable_enhancement = st.checkbox(
        "Apply field-photo enhancement before prediction",
        value=True,
        help="Helps reduce lighting/background differences for newly captured photos.",
    )
    confidence_threshold = st.slider(
        "Low confidence warning threshold",
        min_value=0.40,
        max_value=0.95,
        value=0.65,
        step=0.01,
    )
    top_k = st.slider("Show top-k probabilities", min_value=1, max_value=5, value=3, step=1)
    use_attention = st.checkbox(
        "Use automatic target attention (recommended)",
        value=True,
        help="Automatically emphasizes likely object regions to reduce background impact.",
    )

    if mode == "Upload image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded_file is None:
            return

        image = Image.open(uploaded_file).convert("RGB")
        image_for_inference = _enhance_field_photo(image) if enable_enhancement else image

        left_col, right_col = st.columns(2)
        left_col.image(image, caption="Original image", use_column_width=True)
        if enable_enhancement:
            right_col.image(image_for_inference, caption="Enhanced image", use_column_width=True)
        else:
            right_col.image(image_for_inference, caption="Input used for inference", use_column_width=True)

        image_for_inference = _apply_manual_roi(image_for_inference, key_prefix="upload")

        if st.button("Predict category"):
            prediction = _predict_from_pil_image(
                model,
                image_for_inference,
                Path(uploaded_file.name).suffix.lower() or ".jpg",
                top_k=top_k,
                use_attention=use_attention,
            )
            _render_prediction(prediction, threshold=confidence_threshold)
        return

    images_dir = Path(
        st.text_input("Dataset image directory", str(DEFAULT_DATASET_IMAGES_DIR))
    )
    image_paths = list_dataset_images(images_dir)

    if not image_paths:
        st.warning("No images found in the selected dataset image directory.")
        return

    selected_name = st.selectbox("Select an image", [p.name for p in image_paths])
    selected_path = next(p for p in image_paths if p.name == selected_name)
    selected_image = Image.open(selected_path).convert("RGB")
    image_for_inference = _enhance_field_photo(selected_image) if enable_enhancement else selected_image

    left_col, right_col = st.columns(2)
    left_col.image(selected_image, caption="Original image", use_column_width=True)
    if enable_enhancement:
        right_col.image(image_for_inference, caption="Enhanced image", use_column_width=True)
    else:
        right_col.image(image_for_inference, caption="Input used for inference", use_column_width=True)

    image_for_inference = _apply_manual_roi(image_for_inference, key_prefix="dataset")

    if st.button("Predict category"):
        prediction = _predict_from_pil_image(
            model,
            image_for_inference,
            selected_path.suffix.lower() or ".jpg",
            top_k=top_k,
            use_attention=use_attention,
        )
        _render_prediction(prediction, threshold=confidence_threshold)


if __name__ == "__main__":
    main()
