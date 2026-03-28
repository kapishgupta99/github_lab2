# 🌸 GitHub Actions for ML Model Training & Versioning - Iris Classification

[![Iris Model Training](https://github.com/kapishgupta99/github_lab2/actions/workflows/model_retraining_on_push.yml/badge.svg)](https://github.com/kapishgupta99/github_lab2/actions/workflows/model_retraining_on_push.yml)
[![Model Calibration](https://github.com/kapishgupta99/github_lab2/actions/workflows/model_calibration_on_push.yml/badge.svg)](https://github.com/kapishgupta99/github_lab2/actions/workflows/model_calibration_on_push.yml)

---

## 📋 Project Overview

This repository demonstrates how to use **GitHub Actions** to automate machine learning workflows including:
- 🔄 **Automated Model Training** - Triggers on every push to main branch
- 📊 **Model Evaluation** - Calculates and stores performance metrics
- 🏷️ **Model Versioning** - Each run creates a timestamped model version
- 🎯 **Probability Calibration** - Ensures well-calibrated prediction probabilities

---

## 🎓 Course Information

| Field | Details |
|-------|---------|
| **Course** | IE 7374 - Machine Learning Operations (MLOps) |
| **University** | Northeastern University |
| **Instructor** | Professor Ramin Mohammadi |
| **Assignment** | Lab Assignment 4 - GitHub Actions Lab 2 |
| **Author** | Kapish Gupta |

---

## ⭐ Modification from Original Lab

> **Original Lab**: Used **synthetic/randomly generated data** for demonstration purposes.
>
> **This Implementation**: Uses the **Iris Dataset** - a real-world flower classification dataset.

### Why Iris Dataset?

| Aspect | Details |
|--------|---------|
| **Samples** | 150 flower samples |
| **Features** | 4 (Sepal Length, Sepal Width, Petal Length, Petal Width) |
| **Classes** | 3 (Setosa, Versicolor, Virginica) |
| **Task** | Multi-class Classification |
| **Source** | scikit-learn's built-in datasets |

This modification demonstrates the pipeline working with **real, well-understood data** rather than synthetic examples.

---

## 🏗️ Project Structure

```
github_lab2/
│
├── .github/
│   └── workflows/
│       ├── model_retraining_on_push.yml    # Training & evaluation workflow
│       └── model_calibration_on_push.yml   # Probability calibration workflow
│
├── src/
│   ├── __init__.py
│   ├── train_model.py          # Model training script (MODIFIED: Iris data)
│   ├── evaluate_model.py       # Model evaluation script
│   └── calibrate_model.py      # Probability calibration script
│
├── models/                      # Trained models (auto-generated)
│   ├── iris_rf_model_latest.joblib
│   ├── iris_rf_model_YYYYMMDD_HHMMSS.joblib
│   ├── iris_rf_calibrated_latest.joblib
│   └── scaler_latest.joblib
│
├── metrics/                     # Evaluation metrics (auto-generated)
│   ├── metrics_latest.json
│   ├── metrics_YYYYMMDD_HHMMSS.json
│   └── classification_report.txt
│
├── data/                        # Data directory
│   └── .gitkeep
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔄 How It Works

### Automated Pipeline Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│  Push Code  │ ──► │ Train Model  │ ──► │  Evaluate   │ ──► │   Commit    │
│  to Main    │     │  (Iris RF)   │     │   Metrics   │     │   Results   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬──────┘
                                                                     │
                                                                     ▼
                                                            ┌─────────────────┐
                                                            │   Calibration   │
                                                            │   Workflow      │
                                                            │   (Auto-runs)   │
                                                            └─────────────────┘
```

### Workflow 1: Model Training & Evaluation
**File:** `.github/workflows/model_retraining_on_push.yml`  
**Trigger:** Push to `main` branch

| Step | Action |
|------|--------|
| 1 | Checkout repository code |
| 2 | Set up Python 3.10 environment |
| 3 | Install dependencies from requirements.txt |
| 4 | Train Random Forest on Iris dataset |
| 5 | Evaluate model (Accuracy, F1, Precision, Recall) |
| 6 | Save model with timestamp versioning |
| 7 | Commit metrics and model back to repository |

### Workflow 2: Model Calibration
**File:** `.github/workflows/model_calibration_on_push.yml`  
**Trigger:** After training workflow completes successfully

| Step | Action |
|------|--------|
| 1 | Load trained model |
| 2 | Apply Platt scaling (sigmoid calibration) |
| 3 | Evaluate calibration improvement (Brier scores) |
| 4 | Save calibrated model |
| 5 | Commit calibrated model to repository |

---

## 📊 Model Specifications

| Component | Configuration |
|-----------|---------------|
| **Algorithm** | Random Forest Classifier |
| **n_estimators** | 100 |
| **max_depth** | 5 |
| **Preprocessing** | StandardScaler |
| **Train/Test Split** | 80/20 with stratification |
| **Calibration Method** | Platt Scaling (Sigmoid) |

---

## 📈 Metrics Tracked

The pipeline automatically tracks and stores:

- **Accuracy** - Overall correct predictions
- **F1 Score** (Weighted & Macro) - Harmonic mean of precision and recall
- **Precision** (Weighted) - True positive rate
- **Recall** (Weighted) - Sensitivity / True positive rate
- **Confusion Matrix** - Per-class performance breakdown
- **Brier Score** - Probability calibration quality (lower is better)

### Sample Output (metrics/metrics_latest.json)

```json
{
    "timestamp": "20260328_120000",
    "dataset": "Iris",
    "model": "RandomForestClassifier",
    "accuracy": 1.0,
    "f1_score_weighted": 1.0,
    "f1_score_macro": 1.0,
    "precision_weighted": 1.0,
    "recall_weighted": 1.0,
    "confusion_matrix": [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
}
```

---

## 🛠️ Local Development Setup

### Prerequisites
- Python 3.10+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/kapishgupta99/github_lab2.git
cd github_lab2

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Train the model
python src/train_model.py

# Evaluate the model
python src/evaluate_model.py

# Calibrate the model
python src/calibrate_model.py
```

---

## 🔧 GitHub Actions Workflow Details

### Key Concepts Demonstrated

| Concept | Description | Implementation |
|---------|-------------|----------------|
| **Triggers** | Events that start workflows | `on: push` to main branch |
| **Jobs** | Units of work in a workflow | `train-and-evaluate`, `calibrate` |
| **Steps** | Individual tasks within a job | Checkout, Setup Python, Run scripts |
| **Artifacts** | Files generated by workflows | Models, metrics |
| **Workflow Chaining** | Sequential workflow execution | Calibration runs after training |
| **Auto-commit** | Bot commits results | GitHub Actions bot pushes changes |

### Workflow Permissions

The workflows require `contents: write` permission to commit generated files back to the repository.

---

## 📝 Files Changed from Original Lab

| File | Modification |
|------|--------------|
| `src/train_model.py` | Changed from synthetic data to **Iris dataset** |
| `src/evaluate_model.py` | Updated for Iris multi-class evaluation |
| `src/calibrate_model.py` | Adapted calibration for Iris classes |
| `README.md` | Complete rewrite with documentation |
| Workflow YMLs | Updated names and configurations |

---

## 🚀 Quick Start

1. **Fork/Clone** this repository
2. **Push** any change to the `main` branch
3. **Watch** the Actions tab - workflows will run automatically
4. **Check** `models/` and `metrics/` folders for outputs

---

## 📚 References

- [Original MLOps Lab2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Scikit-learn Iris Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset)
- [Model Calibration Guide](https://scikit-learn.org/stable/modules/calibration.html)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Professor Ramin Mohammadi** - Course Instructor
- **Northeastern University** - IE 7374 MLOps Course
- **GitHub Actions** - CI/CD Platform
- **scikit-learn** - Machine Learning Library

---

*Last Updated: March 2026*
