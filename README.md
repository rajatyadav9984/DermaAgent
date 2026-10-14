# DermaAgent — AI-Assisted Skin Lesion Screening System

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![License: Research Use](https://img.shields.io/badge/License-Research%20Only-orange.svg)](#disclaimer)

DermaAgent is an AI-assisted skin lesion image classification research prototype. It provides deep learning inference, predictive uncertainty quantification via Shannon entropy, visual explainability via Grad-CAM heatmaps, and a modern glassmorphic web dashboard.

---

## 🏗️ Project Architecture & Directory Structure

```
C:\DermaAgent\
├── app.py                                # Root entry point for web application
├── requirements.txt                      # Project dependencies (PyTorch, Flask, Grad-CAM, etc.)
├── README.md                             # Documentation & user guide
├── .env.example                          # Environment variables template
├── .gitignore                            # Git exclusion rules for large datasets & models
│
├── agents/                               # Multi-Agent clinical decision support layer
│   ├── __init__.py
│   ├── orchestrator.py                   # Central Multi-Agent Orchestrator
│   ├── image_quality_agent.py            # Blurriness, illumination & quality validation
│   ├── prediction_agent.py               # Formats disease class & top-k confidence
│   ├── reliability_agent.py              # Evaluates entropy & generates reliability advisory
│   ├── explainability_agent.py           # Handles Grad-CAM visual heatmap linking
│   ├── risk_assessment_agent.py          # Screening triage priority, urgency & guidance
│   ├── question_agent.py                 # Clinical anamnesis & consultation follow-ups
│   └── skin_analysis_agent.py            # Unified high-level analysis wrapper
│
├── backend/                              # Modular Flask REST API
│   ├── __init__.py
│   ├── app.py                            # Flask server and browser launcher
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── predict_route.py              # POST /api/predict (Inference, upload, Grad-CAM)
│   │   └── health_route.py               # GET /api/health (System status & model availability)
│   ├── services/
│   │   ├── __init__.py
│   │   └── prediction_service.py         # Singleton cached model manager & inference engine
│   └── utils/
│       ├── __init__.py
│       └── image_validator.py            # PIL integrity check, size limit, corrupted file rejection
│
├── frontend/                             # Glassmorphism Web User Interface
│   ├── index.html                        # Main single-page application
│   ├── style.css                         # Primary styles
│   ├── script.js                         # Primary frontend logic & API client
│   ├── css/                              # Mirrored CSS bundle
│   ├── js/                               # Mirrored JavaScript bundle
│   └── pages/
│       └── about.html                    # Architecture & medical research info
│
├── ml/                                   # Machine Learning & Computer Vision Core
│   ├── dataset_loader.py                 # PyTorch ISIC 2019 Dataset (streams directly from zip or disk)
│   ├── preprocess.py                     # Training & inference image transforms
│   ├── train_isic2019.py                 # 8-class ResNet18 training pipeline
│   ├── evaluate_real.py                  # Evaluation script for HAM10000 ResNet18
│   ├── predict.py                        # Standalone real inference script
│   ├── gradcam.py                        # PyTorch Grad-CAM heatmap generator
│   └── uncertainty.py                    # Shannon entropy & uncertainty quantification
│
├── configs/                              # Centralized configuration files
│   ├── dataset_config.yaml               # Paths, class taxonomy, split configurations
│   └── model_config.yaml                 # Backbones, hyperparameters, entropy thresholds
│
├── data_raw/                             # Raw datasets
│   ├── HAM10000/                         # HAM10000 raw image parts (10,015 images)
│   └── ISIC2019/                         # ISIC 2019 zip archive & challenge ground truth
│       ├── ISIC_2019_Training_Input.zip
│       ├── ISIC_2019_Training_GroundTruth.csv
│       ├── ISIC_2019_Training_Metadata.csv
│       ├── isic2019_train_split.csv      # Leakage-aware 80% train manifest
│       └── isic2019_val_split.csv        # Leakage-aware 20% val manifest
│
├── dataset_real/                         # Preserved 7-class HAM10000 train/val/test splits
├── dataset_backup/                       # Preserved backup splits
│
├── model/                                # Checkpoints & weights
│   ├── real_skin_disease_model.pth       # Trained 7-class ResNet18 baseline (PRESERVED)
│   ├── isic2019_skin_disease_model.pth   # 8-class ISIC 2019 checkpoint (Saved when trained)
│   └── gradcam/                          # Generated Grad-CAM heatmap overlays
│
├── results/                              # Evaluation outputs & reports
│   ├── isic2019_dataset_validation_report.md
│   ├── confusion_matrix/                 # Confusion matrix visualizations
│   ├── metrics/                          # Quantitative JSON metrics
│   └── plots/
│
├── tests/                                # Automated Pytest Suite
│   ├── test_app.py                       # API endpoints, bad upload, corrupted image tests
│   ├── test_prediction.py                # Prediction service & entropy math tests
│   └── test_dataset.py                   # Ground truth integrity & 0-leakage split tests
│
├── scripts/                              # Automation & utility scripts
│   ├── check_dataset.py                  # Dataset validation & HAM10000 overlap checker
│   ├── prepare_isic2019.py               # Leakage-aware lesion grouping & split generator
│   └── run_training.ps1                  # One-click PowerShell training launcher
│
├── uploads/                              # Temporary uploaded images
└── logs/                                 # Execution logs
```

---

## 🧬 Disease Taxonomy: Existing vs New Model

| Class Code | Full Condition Name | Category | HAM10000 Model (7 Classes) | ISIC 2019 Model (8 Classes) |
|:---|:---|:---|:---:|:---:|
| **MEL** / `mel` | Melanoma | Malignant | ✅ | ✅ |
| **NV** / `nv` | Melanocytic Nevi (Moles) | Benign | ✅ | ✅ |
| **BCC** / `bcc` | Basal Cell Carcinoma | Malignant | ✅ | ✅ |
| **AK** / `akiec` | Actinic Keratosis / Bowen's | Pre-cancerous | ✅ | ✅ |
| **BKL** / `bkl` | Benign Keratosis | Benign | ✅ | ✅ |
| **DF** / `df` | Dermatofibroma | Benign | ✅ | ✅ |
| **VASC** / `vasc`| Vascular Lesion | Benign | ✅ | ✅ |
| **SCC** | Squamous Cell Carcinoma | Malignant | ❌ | ✅ *(New)* |

> [!IMPORTANT]
> The baseline checkpoint `model/real_skin_disease_model.pth` is permanently preserved. The new ISIC 2019 8-class model trains strictly to `model/isic2019_skin_disease_model.pth`.

---

## ⚡ Quick Start: Windows PowerShell

### 1. Virtual Environment Activation
```powershell
cd C:\DermaAgent
.\.venv\Scripts\Activate.ps1
```

### 2. Dependency Verification
All required packages are installed in `.venv`. To verify:
```powershell
python -m pip list
```

### 3. Run Automated Tests
```powershell
pytest -v tests
```

### 4. Launch the Web Application
Launch using either:
```powershell
python app.py
```
or via the root launcher script:
```powershell
.\run_dermaagent.bat
```
The server will start on `http://127.0.0.1:5000/` and automatically open your default browser.

---

## 🔬 Dataset Preparation, Training & Evaluation

### 1. Leakage-Aware 3-Way Split (Train / Val / Test)
Partitions the 25,331 ISIC 2019 images into strict, 0-leakage 3-way splits using `lesion_id` grouping:
- **Train Split**: 20,258 images (`data_raw/ISIC2019/isic2019_train_split.csv`)
- **Validation Split**: 2,500 images (`data_raw/ISIC2019/isic2019_val_split.csv`)
- **Independent Test Split**: 2,573 images (`data_raw/ISIC2019/isic2019_test_split.csv`)
- **Leakage Metrics**: 0 image overlap, 0 lesion group overlap across all pairs.

To re-run the 3-way split pipeline:
```powershell
python scripts\prepare_isic2019_3way.py
```

### 2. ISIC 2019 8-Class Model Training
Trains a ResNet18 classifier on the 8 ISIC 2019 classes utilizing pre-trained dermoscopy backbone weights, class-weighted cross-entropy loss, and checkpoint verification:
```powershell
python ml\train_isic2019.py --epochs 10 --batch-size 32 --lr 0.0001
```
For a fast smoke test:
```powershell
python ml\train_isic2019.py --smoke-test --epochs 2 --batch-size 16
```
- **Trained Model Checkpoint**: `model/isic2019_skin_disease_model.pth`
- **Class Weights Tensor**: `model/isic2019_class_weights.pt`

### 3. Independent Test Set Evaluation
Evaluates the trained checkpoint on the independent test split (2,573 images), generating classification reports, weighted & macro metrics, and confusion matrix visualizations:
```powershell
python ml\evaluate_isic2019.py
```
- **Evaluation Report**: `results/isic2019_test_evaluation_report.md`
- **Confusion Matrix Plot**: `results/confusion_matrix/isic2019_test_confusion_matrix.png`
- **Quantitative Metrics JSON**: `results/metrics/isic2019_test_evaluation.json`

### 4. Running the Complete Automated Test Suite
Runs all 15 unit and integration tests across Backend APIs, Dataset Splits, and ML Inference:
```powershell
pytest -v tests
```
*Current test status: 15 / 15 Passed (100% Passing)*

---

## 🛡️ Medical Disclaimer

> [!CAUTION]
> **Educational and Research Prototype Only**:
> DermaAgent is an artificial intelligence research experiment designed for educational, prototyping, and technical benchmarking purposes. **It does not provide medical diagnoses or clinical evaluations and is NOT a substitute for a qualified dermatologist or licensed medical physician.**
