# COMP4423-Ass2

## Overview

This repository contains an example **SVM classification pipeline**.

- `src/svm_model.py`: SVM model implementation (train, predict, save/load).
- `main.py`: Entry point to train the model with dataset parameters.
- `dataset/`: Expected dataset layout (images, JSON labels, and `label.txt`).

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train a model (default saves into `model/`):

```bash
python main.py --dataset-dir ./dataset --model-path ./model/svm_model.pkl
```

3. Load and use the trained model:

```python
from pathlib import Path
from src.svm_model import SVMClassifier

model = SVMClassifier.load(Path("model/svm_model.pkl"))
print(model.predict(Path("dataset/images/cat_1.png")))
```
