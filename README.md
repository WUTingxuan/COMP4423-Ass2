# COMP4423-Ass2

## Overview

This repository contains an example **SVM classification pipeline**.

- `src/svm_model.py`: SVM model implementation (train, predict, save/load).
- `main.py`: Entry point to train the model with dataset parameters.
- `dataset/`: Expected YOLOv8 dataset layout (`data.yaml`, `train/images`, `train/labels`).

## Dataset Format

This project now uses **YOLOv8 detection format** as input and converts it to image-level
classification labels by reading the first object class in each label file.

Expected structure:

```text
dataset/
	data.yaml
	train/
		images/
		labels/
```

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train a model (default saves into `model/`):

```bash
python main.py --dataset-dir ./dataset --model-path ./model/svm_model.pkl
```

Training output now includes a feature-strategy comparison:

- `mixed`: reduced global-context + object-focused features
- `masked_only`: polygon-masked/region-focused features only

This helps you compare robustness against background influence directly in terminal logs.

3. Load and use the trained model:

```python
from pathlib import Path
from src.svm_model import SVMClassifier

model = SVMClassifier.load(Path("model/svm_model.pkl"))
print(model.predict(Path("dataset/train/images/example.jpg")))
```

## Local App (Upload or Select Image)

Run the local web app:

```bash
streamlit run app.py
```

Then open the URL shown in terminal (usually http://localhost:8501).

Usage:

1. Set the model path (default: model/svm_model.pkl).
2. (Recommended for new photos) Enable "Apply field-photo enhancement before prediction".
3. Choose one input mode:
	- Upload image: upload your local image file.
	- Select dataset image: pick an image from dataset/train/images.
4. Click "Predict category" to get:
	- predicted vegetation class,
	- confidence score,
	- top-k class probabilities,
	- low-confidence warning when confidence is below threshold.
	- optional manual ROI crop for stronger target focus (drag rectangle on image).
