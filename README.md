# mlops_telco_churn
# Telco Customer Churn Prediction — MLOps Pipeline

> An end-to-end MLOps project for predicting telecom customer churn using XGBoost, MLflow, DVC, FastAPI, and Docker — with automated CI/CD and data drift monitoring.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Running the Pipeline](#running-the-pipeline)
  - [Using DVC](#using-dvc)
  - [Using the Script Directly](#using-the-script-directly)
- [Model Serving (FastAPI)](#model-serving-fastapi)
  - [Run Locally](#run-locally)
  - [Run with Docker](#run-with-docker)
  - [API Endpoints](#api-endpoints)
- [Experiment Tracking (MLflow)](#experiment-tracking-mlflow)
- [Data Drift Monitoring](#data-drift-monitoring)
- [CI/CD Pipeline](#cicd-pipeline)
- [Dataset](#dataset)
- [Results](#results)

---

## Overview

This project implements a production-ready MLOps pipeline for predicting customer churn in a telecommunications company. The goal is to identify customers likely to leave, enabling proactive retention strategies.

The pipeline covers the full ML lifecycle:

- Data ingestion & validation — schema and quality checks via Great Expectations
- Preprocessing & feature engineering — binary encoding, one-hot encoding, class imbalance handling
- Model training & hyperparameter tuning — XGBoost with learning rate and depth grid search
- Experiment tracking — MLflow logs parameters, metrics, and model artifacts
- Model serving — FastAPI REST API with threshold-based predictions
- Containerization — Docker image pushed to Docker Hub via GitHub Actions
- Data drift monitoring — Evidently AI reports with auto-trigger retraining
- Pipeline versioning — DVC tracks data and model artifacts

---

## Architecture

```
Raw Data (CSV)
      |
      v
+---------------------+
|  Data Validation     |  <- Great Expectations
+----------+----------+
           |
           v
+---------------------+
|  Preprocessing       |  <- Missing values, type casting
+----------+----------+
           |
           v
+---------------------+
|  Feature Engineering |  <- Binary + One-hot encoding
+----------+----------+
           |
           v
+---------------------+
|  Hyperparameter      |  <- XGBoost grid (lr x depth)
|  Tuning + Training   |     tracked by MLflow
+----------+----------+
           |
           +---> model.pkl (best by recall)
           +---> model_top1.pkl
           +---> model_top2.pkl
                    |
                    v
           +----------------+
           |  FastAPI Server |  <- /predict endpoint
           +----------------+
                    |
                    v
           +----------------+
           |  Docker Image   |  <- smruthi18/telco-churn:latest
           +----------------+

Monitoring (Evidently)
           |
           +-- Drift detected? --> Auto-trigger retraining
           +-- Report saved to monitoring/report.html
```

---

## Project Structure

```
mlops_telco_churn/
+-- .github/
|   +-- workflows/
|       +-- ci.yml               # GitHub Actions -- build & push Docker image
+-- .dvc/                        # DVC configuration
+-- data/
|   +-- raw/
|   |   +-- dataset.csv          # Raw Telco churn dataset
|   +-- processed/
|   |   +-- telco_churn_processed.csv
|   +-- load_data.py
|   +-- make_dataset.py
|   +-- preprocess.py
+-- monitoring/
|   +-- data_drift.py            # Evidently drift detection + auto-retraining
+-- notebooks/
|   +-- EDA.ipynb                # Exploratory Data Analysis
|   +-- feature_importance.csv
+-- scripts/
|   +-- run_pipeline.py          # End-to-end training pipeline
|   +-- prepare_preprocessed_data.py
+-- serving/
|   +-- app.py                   # FastAPI prediction server
|   +-- models/                  # Saved model artifacts (.pkl)
+-- src/
|   +-- data/
|   |   +-- load_data.py
|   |   +-- make_dataset.py
|   |   +-- preprocess.py
|   +-- features/
|   |   +-- build_features.py    # Binary + one-hot encoding
|   +-- models/
|   |   +-- train.py
|   |   +-- tune.py
|   |   +-- evaluate.py
|   +-- utils/
|   |   +-- utils.py
|   |   +-- validate_data.py     # Schema & data quality validation
|   +-- logger.py
+-- artifacts/                   # feature_columns.json, threshold.txt
+-- dvc.yaml                     # DVC pipeline definition
+-- dvc.lock
+-- Dockerfile
+-- requirements.txt
+-- README.md
```

---

## Tech Stack

| Category            | Tool / Library       |
|---------------------|----------------------|
| ML Model            | XGBoost              |
| Experiment Tracking | MLflow               |
| Data Versioning     | DVC                  |
| Feature Engineering | Pandas, Scikit-learn |
| Data Validation     | Great Expectations   |
| Model Serving       | FastAPI + Uvicorn    |
| Monitoring          | Evidently AI         |
| Containerization    | Docker               |
| CI/CD               | GitHub Actions       |
| Language            | Python 3.9           |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Git + DVC

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/SmruthiChandrashekar/mlops_telco_churn.git
cd mlops_telco_churn
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Pull data with DVC**

```bash
dvc pull
```

---

## Running the Pipeline

### Using DVC

Run the full versioned pipeline (load -> validate -> preprocess -> feature engineering -> train -> evaluate):

```bash
dvc repro
```

### Using the Script Directly

```bash
python scripts/run_pipeline.py \
  --input data/raw/dataset.csv \
  --target Churn \
  --threshold 0.35 \
  --test_size 0.2 \
  --experiment "Telco Churn"
```

**Arguments:**

| Argument       | Default       | Description                                    |
|----------------|---------------|------------------------------------------------|
| `--input`      | *(required)*  | Path to the raw CSV dataset                    |
| `--target`     | `Churn`       | Name of the target column                      |
| `--threshold`  | `0.35`        | Probability threshold for churn classification |
| `--test_size`  | `0.2`         | Fraction of data reserved for testing          |
| `--experiment` | `Telco Churn` | MLflow experiment name                         |

The pipeline will:
1. Load and validate data
2. Preprocess and engineer features
3. Train XGBoost models across a hyperparameter grid (`lr` in {0.01, 0.03, 0.05} x `depth` in {4, 6})
4. Rank models by **recall** (optimized for churn detection)
5. Save `model_top1.pkl`, `model_top2.pkl`, and `model.pkl` (production model) to `serving/models/`

---

## Model Serving (FastAPI)

### Run Locally

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run with Docker

**Pull from Docker Hub:**

```bash
docker pull smruthi18/telco-churn:latest
docker run -p 8000:8000 smruthi18/telco-churn:latest
```

**Build locally:**

```bash
docker build -t telco-churn .
docker run -p 8000:8000 telco-churn
```

### API Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Churn Prediction API is running"
}
```

#### `POST /predict`
Predict customer churn probability.

**Request Body:**
```json
{
  "tenure": 24,
  "MonthlyCharges": 65.5,
  "TotalCharges": 1572.0,
  "gender": 1,
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "PhoneService": 1,
  "PaperlessBilling": 1,
  "Contract_One year": 0,
  "Contract_Two year": 1
}
```

**Response:**
```json
{
  "prediction": 0,
  "churn_probability": 0.18,
  "threshold_used": 0.35
}
```

| Field               | Description                                  |
|---------------------|----------------------------------------------|
| `prediction`        | `1` = will churn, `0` = will not churn       |
| `churn_probability` | Model's estimated probability of churn       |
| `threshold_used`    | Decision threshold applied (default: `0.35`) |

---

## Experiment Tracking (MLflow)

All training runs are tracked with MLflow. To launch the MLflow UI:

```bash
mlflow ui
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

**Logged per run:**

| Type      | Items                                                              |
|-----------|--------------------------------------------------------------------|
| Params    | `model`, `learning_rate`, `max_depth`, `threshold`                 |
| Metrics   | `precision`, `recall`, `f1`, `roc_auc`, `train_time`, `pred_time` |
| Artifacts | Serialized XGBoost model                                           |
| Tags      | `rank` (for top model identification)                              |

> Models are ranked and selected by **recall** to minimize missed churners.

---

## Data Drift Monitoring

The monitoring module uses **Evidently AI** to detect distributional shifts in the input data and automatically trigger retraining when drift is detected.

**Run drift detection:**

```bash
python monitoring/data_drift.py
```

- Splits the dataset into a reference set and a current set (random or date-based split)
- Generates a full data drift HTML report saved to `monitoring/report.html`
- If drift is detected, automatically triggers model retraining

**To use date-based splitting**, configure in `monitoring/data_drift.py`:

```python
DATE_COLUMN = "signup_date"   # your date column
SPLIT_DATE  = "2024-06-01"    # data before this = reference
```

---

## CI/CD Pipeline

GitHub Actions automatically builds and pushes the Docker image to Docker Hub on every push or pull request to `main` or `ci-cd` branches.

**Workflow file:** `.github/workflows/ci.yml`

```
Push to main / ci-cd
        |
        v
  Checkout Code
        |
        v
  Set up Docker Buildx
        |
        v
  Login to Docker Hub  <- uses DOCKERHUB_USERNAME & DOCKERHUB_TOKEN secrets
        |
        v
  Build & Push Image -> smruthi18/telco-churn:latest
```

**Required GitHub Secrets:**

| Secret               | Description              |
|----------------------|--------------------------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN`    | Docker Hub access token  |

---

## Dataset

The dataset used is the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), which contains information about:

- Customer demographics (gender, age, partners, dependents)
- Account information (tenure, contract type, payment method, charges)
- Services subscribed (phone, internet, streaming, etc.)
- **Target**: `Churn` — whether the customer left within the last month

---

## Results

| Metric    | Value (approx.) |
|-----------|-----------------|
| ROC-AUC   | ~0.85           |
| Recall    | ~0.78           |
| Precision | ~0.65           |
| F1 Score  | ~0.71           |

> Recall is prioritized over precision to minimize the cost of missing a churner. The default decision threshold is set to **0.35** (lower than 0.5) to increase sensitivity.

---

## Author

**Smruthi Chandrashekar**
- GitHub: [@SmruthiChandrashekar](https://github.com/SmruthiChandrashekar)
- Docker Hub: [smruthi18](https://hub.docker.com/u/smruthi18)

---

*Built as part of the 6th Semester MLOps course project.*
