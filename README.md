# Telco Customer Churn MLOps Pipeline

This repository contains an end-to-end MLOps workflow for predicting customer churn in a telecom dataset. It includes data loading, validation, preprocessing, feature engineering, model training, experiment tracking, model versioning, API serving, data drift monitoring, containerization, and CI/CD automation.

## Project Overview

The project trains machine learning models to predict whether a telecom customer is likely to churn. The pipeline uses DVC to manage data and pipeline outputs, MLflow to track experiments, XGBoost for model training, and FastAPI to serve trained models through a prediction API.

## Features

- Data loading and preprocessing for telco churn data
- Data validation before model training
- Feature engineering for model-ready datasets
- XGBoost model training with hyperparameter tuning
- MLflow experiment tracking for parameters, metrics, and models
- DVC pipeline for reproducible training
- FastAPI service for model inference
- Data drift monitoring with Evidently
- Logging support for monitoring workflows
- Docker support for deployment
- GitHub Actions workflow for Docker image build and push

## Repository Structure

```text
.
+-- .github/workflows/       # CI/CD workflow files
+-- data/                    # Raw and processed data tracked with DVC
+-- monitoring/              # Data drift monitoring report generation
+-- notebooks/               # Exploratory analysis notebooks
+-- scripts/                 # Pipeline execution scripts
+-- serving/                 # FastAPI app and trained models
+-- src/                     # Source code for data, features, models, and utilities
+-- Dockerfile               # Docker image definition
+-- dvc.yaml                 # DVC pipeline definition
+-- requirements.txt         # Python dependencies
+-- README.md                # Project documentation
```

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- MLflow
- DVC
- FastAPI
- Uvicorn
- Docker
- GitHub Actions
- Evidently

## Setup Instructions

Clone the repository:

```bash
git clone https://github.com/Vandana-Kumari-Meena/mlops_telco_churn.git
cd mlops_telco_churn
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Training Pipeline

Run the pipeline directly:

```bash
python scripts/run_pipeline.py --input data/raw/dataset.csv --target Churn
```

Or reproduce the DVC pipeline:

```bash
dvc repro
```

The pipeline saves processed data in `data/processed/` and trained models in `serving/models/`, including a final production model at `serving/models/model.pkl`.

## MLflow Tracking

The training pipeline logs model parameters, metrics, and artifacts using MLflow.

Start the MLflow UI:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

## Data Drift Monitoring

Generate a data drift report:

```bash
python monitoring/data_drift.py
```

The report is saved as `monitoring/report.html`. If drift is detected, the monitoring script is designed to trigger model retraining.

## Run the FastAPI App

Start the API server:

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000
```

Health check endpoint:

```text
GET /
```

Prediction endpoint:

```text
POST /predict
```

Example request body:

```json
{
  "feature_1": 1,
  "feature_2": 0,
  "feature_3": 45
}
```

The API loads the production model from `serving/models/model.pkl`, aligns incoming data with the saved feature column order, applies the saved threshold, and returns the churn prediction with its probability.

## Docker

Build the Docker image:

```bash
docker build -t telco-churn-api .
```

Run the container:

```bash
docker run -p 8000:8000 telco-churn-api
```

## CI/CD

The GitHub Actions workflow builds the Docker image on pushes and pull requests to `main` and `ci-cd`. On push events, it can also publish the image to Docker Hub when the required repository secrets are configured.

## Author

Vandana Kumari Meena
