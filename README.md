# 🔬 CancerNet: Breast Cancer Detection Pipeline

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![AUC](https://img.shields.io/badge/AUC-0.90-success.svg)

**CancerNet** is a deep learning pipeline for detecting Invasive Ductal Carcinoma (IDC) in histopathology patch images. Built as a research and portfolio project, it implements a complete training pipeline with patient-aware data splitting, staged backbone fine-tuning, mixed precision training, and explainability via Grad-CAM.

---

## 🧠 Architecture

The model uses an **EfficientNet-B4** backbone pretrained on ImageNet, fine-tuned for binary patch classification (malignant / benign).

- **Feature extraction:** EfficientNet-B4 CNN backbone
- **Staged training:** Classification head trained first; backbone unfrozen after a configurable number of epochs to prevent early overfitting
- **Output:** Binary — malignant or benign patch

---

## 📊 Results

Evaluated on a held-out test set using strict **patient-level splits** to prevent any data leakage between train, validation, and test sets.

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.90 |
| Accuracy | 83% |
| Sensitivity (Recall) | 66.4% |
| Specificity | 90.1% |
| Precision | 72.4% |
| Threshold | 0.553 (Youden optimal) |

> **Note:** AUC is the primary metric given the class imbalance (~2.5:1 benign:malignant). Sensitivity is measured at patch level — patient-level aggregation across all patches is expected to recover recall significantly.

### ROC Curve
![ROC Curve](results/roc_curve.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

---

## 📁 Dataset

Uses the **Breast Histopathology Images** dataset (IDC detection), publicly available on Kaggle.

- 280 patients total
- Patient-level train / val / test split: 196 / 42 / 42
- ~280,000 patches across all splits (50×50 pixels, 40x magnification)
- Class distribution: ~72% benign, ~28% malignant

---

## ⚙️ Training Details

| Setting | Value |
|---------|-------|
| Backbone | EfficientNet-B4 (pretrained) |
| Optimizer | AdamW |
| Precision | Mixed (AMP / FP16) |
| Loss | Binary Cross-Entropy with class weights |
| Augmentations | Albumentations (flip, rotation, color jitter, normalization) |
| Hardware | NVIDIA GPU (CUDA) |

---

## 👁️ Explainability

The evaluation pipeline generates **Grad-CAM** heatmaps to visualize which regions of a patch the model focuses on when predicting malignancy. This helps verify that the model is responding to relevant tissue features rather than artifacts.

---

## 🚀 Running the Project

### Local Setup

```bash
# Clone the repository
git clone https://github.com/amal0307/cancer-detection.git
cd cancer-detection

# Install dependencies
pip install -r requirements.txt

# Preprocess dataset
python scripts/preprocess.py --config configs/config.yaml

# Train
python scripts/train.py --config configs/config.yaml

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --gradcam
```

### Resume Training

```bash
python scripts/train.py --config configs/config.yaml --resume auto
```

---

## 📁 Repository Structure
```text
cancer-detection/
│
├── configs/            # YAML config (hyperparameters, paths, hardware settings)
├── scripts/            # train.py, evaluate.py, preprocess.py
├── src/
│   ├── models/         # CancerNet model definition
│   ├── training/       # Trainer loop, loss, scheduler
│   ├── evaluation/     # Metrics, Grad-CAM
│   └── utils/          # Dataloaders, augmentations, seed utils
│
└── requirements.txt
```

---

## ⚠️ Disclaimer

This is a research and portfolio project. It has not been clinically validated and is not intended for medical use or diagnosis.