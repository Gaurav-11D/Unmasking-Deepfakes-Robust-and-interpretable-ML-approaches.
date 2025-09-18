# 🎭 Unmasking Deepfakes: Robust and Interpretable ML Approaches

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)  

---

## 📌 Overview
This repository contains the code, models, and results for **Unmasking Deepfakes**, an MSc Data Science project exploring robust and interpretable machine learning methods for detecting deepfake images.  

Our pipeline combines:
- **Handcrafted features** (colour histograms, gradients, texture descriptors)  
- **Frozen deep embeddings** (VGG16)  
- **Classical classifiers** (Support Vector Machine, Random Forest)  

We benchmark on clean datasets and test robustness under:
- JPEG compression  
- Gaussian noise  
- Gaussian blur  

---

## 📂 Repository Structure

```
Unmasking-Deepfakes-Robust-and-interpretable-ML-approaches/
│
├── data/              # Train/val/test splits and labels (CSV indices)
├── features/          # Scripts for handcrafted + VGG16 feature extraction
├── models/            # Training scripts & saved SVM / RF models
├── figures/           # ROC curves, confusion matrices, etc.
├── results/           # Output tables & evaluation metrics
├── main.tex           # CVPR-style paper (LaTeX source)
└── README.md          # Project summary (this file)
```

---

## ⚙️ Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/Gaurav-11D/Unmasking-Deepfakes-Robust-and-interpretable-ML-approaches.git
cd Unmasking-Deepfakes-Robust-and-interpretable-ML-approaches

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

**Dependencies** include:
- `numpy`, `pandas`
- `scikit-learn`
- `tensorflow` / `keras`
- `matplotlib`, `seaborn`
- `opencv-python` or `Pillow`

---

## 🚀 Usage

### 1. Extract Features
```bash
python extract_features.py --input_dir path/to/images --output features/combined_features.npy
```

### 2. Train Classifiers
```bash
python train_svm.py --features features/combined_features.npy --labels data/train_labels.csv --output models/svm_model.pkl
python train_rf.py  --features features/combined_features.npy --labels data/train_labels.csv --output models/rf_model.pkl
```

### 3. Evaluate Models
```bash
python evaluate.py --model models/rf_model.pkl --test_split data/test_index.csv --perturbation none
python evaluate.py --model models/rf_model.pkl --test_split data/test_index_jpeg_50.csv --perturbation jpeg50
```

This produces:
- ROC curves  
- Confusion matrices  
- Tables with Accuracy, Precision, Recall, F1, AUC  

---

## 📊 Results

- **Random Forest**: Higher accuracy & F1 under noisy/blurred conditions, robust to perturbations.  
- **SVM**: High recall on clean data, but weaker precision (more false positives).  
- AUCs remain close to 1.0 across models, confirming strong class separability.  

---

## 📑 Paper

The CVPR-style write-up with full results and references is in [`main.tex`](./main.tex).  

If you use this work, please cite:

```
@misc{dalvi2025deepfakes,
  author       = {Gaurav Dalvi},
  title        = {Unmasking Deepfakes: Robust and Interpretable ML Approaches},
  year         = {2025},
  institution  = {Kingston University London},
  note         = {MSc Data Science Dissertation Project}
}
```

---

🔍 Maintained by **Gaurav Dalvi**.
