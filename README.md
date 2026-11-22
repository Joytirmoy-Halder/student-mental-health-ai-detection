# Early Detection of Depression and Anxiety Among University Students Using AI and Machine Learning Models

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Institution](https://img.shields.io/badge/Institution-AIUB-blue.svg)](https://www.aiub.edu/)

This repository contains the complete empirical research codebase, data pipelines, and machine learning models for the undergraduate Capstone Thesis conducted at the **Department of Computer Science, American International University-Bangladesh (AIUB)**.

---

## 📌 Research Overview

University students face elevated risks of psychiatric disorders, predominantly depression and anxiety, exacerbated by academic stressors, financial pressures, and social dynamics. This study presents a machine-learning framework designed to predict early clinical and sub-clinical indicators using a validated primary dataset.

* **Primary Dataset Size:** 466 validated undergraduate student survey responses.
* **Feature Set:** Demographic, academic, behavioural, sleep hygiene, and psychometric screening indicators.
* **Champion Model:** Optimized **Extreme Gradient Boosting (XGBoost)** yielding **93.62% classification accuracy**.

---

## 📊 Comparative Model Performance

Seven machine learning classification algorithms were evaluated across standardized 5-fold cross-validation and independent test splits:

| Model Architecture | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Tuned XGBoost (Champion)** | **93.62%** | **93.80%** | **93.24%** | **93.47%** |
| Random Forest Classifier | 91.49% | 91.20% | 90.80% | 90.95% |
| Support Vector Machine (RBF Kernel) | 88.30% | 88.50% | 87.90% | 88.15% |
| Logistic Regression | 85.11% | 85.30% | 84.80% | 85.00% |
| K-Nearest Neighbors (KNN) | 84.04% | 84.10% | 83.90% | 83.95% |
| Decision Tree Classifier | 82.98% | 82.60% | 83.10% | 82.80% |
| Gaussian Naive Bayes | 79.79% | 80.10% | 79.40% | 79.70% |

---

## 🔬 Methodology & Architecture

1. **Empirical Survey Formulation:** Curated 466 respondent profiles targeting academic pressure, CGPA bands, study hours, sleep duration, and social connectivity.
2. **Preprocessing Pipeline:**
   - Median imputation and categorical encoding.
   - Outlier detection and feature scaling using RobustScaler.
   - Synthetic Minority Over-sampling Technique (SMOTE) for class balance calibration.
3. **Hyperparameter Optimization:** Conducted systematic GridSearchCV on XGBoost (learning rate, max depth, subsample ratios, and colsample_bytree).
4. **Validation:** Evaluated via Confusion Matrix, ROC-AUC curves, and Precision-Recall tradeoffs.

---

## 🚀 Quickstart & Setup

### Prerequisites
* Python 3.9 or higher
* Jupyter Notebook or JupyterLab

### Installation
```bash
# Clone the repository
git clone https://github.com/Joytirmoy-Halder/student-mental-health-ai-detection.git
cd student-mental-health-ai-detection

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook ThesisG49_AIUB_Joyti.ipynb
```

---

## 📁 Repository Structure

```text
├── ThesisG49_AIUB_Joyti.ipynb   # Complete end-to-end Python research notebook
├── requirements.txt             # Python environment dependencies
├── README.md                    # Research overview, benchmarks, and citations
├── LICENSE                      # MIT Open Source License
└── docs/                        # Presentation & model output artifacts
```

---

## 📜 Citation & Academic Details

* **Project Title:** Early Detection of Depression and Anxiety Among University Students Using AI and Machine Learning Models
* **Lead Author:** Joytirmoy Halder Joyti (ID: 19-41473-3)
* **Research Group:** Group 49 (Faculty of Science and Technology, AIUB)
* **Faculty Supervisor:** Nyme Ahmed, Lecturer, Department of Computer Science, AIUB
* **Examination Panel:** Approved by AIUB Faculty Defense Board and External Committee

---

## ⚖️ License
This project is open-source under the [MIT License](LICENSE).
