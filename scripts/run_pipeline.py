#!/usr/bin/env python3
"""
Runs sequentially: load → validate → preprocess → feature engineering → train → evaluate
"""

import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# === Fix import path for local modules ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local modules
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


def main(args):
    """
    Main training pipeline function.
    """

    # =========================
    # MLflow SETUP
    # =========================
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment)

    # =========================
    # 1. LOAD DATA
    # =========================
    print("🔄 Loading data...")
    df = load_data(args.input)
    print(f"✅ Data loaded: {df.shape}")

    # =========================
    # 2. VALIDATE DATA
    # =========================
    print("🔍 Validating data...")
    is_valid, failed = validate_telco_data(df)

    if not is_valid:
        raise ValueError(f"❌ Data validation failed: {failed}")

    print("✅ Data validation passed")

    # =========================
    # 3. PREPROCESS
    # =========================
    print("🔧 Preprocessing...")
    df = preprocess_data(df)

    processed_path = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)

    # =========================
    # 4. FEATURE ENGINEERING
    # =========================
    print("🛠️ Building features...")
    target = args.target

    df_enc = build_features(df, target_col=target)

    for c in df_enc.select_dtypes(include=["bool"]).columns:
        df_enc[c] = df_enc[c].astype(int)

    # =========================
    # 5. TRAIN TEST SPLIT
    # =========================
    print("📊 Splitting data...")
    X = df_enc.drop(columns=[target])
    y = df_enc[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=42
    )

    # =========================
    # 6. HANDLE IMBALANCE
    # =========================
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # =========================
    # 7. TUNING LOOP
    # =========================
    learning_rates = [0.01, 0.03, 0.05]
    max_depths = [4, 6]

    all_models = []

    for lr in learning_rates:
        for depth in max_depths:

            with mlflow.start_run():

                print(f"🚀 Training XGBoost | lr={lr}, depth={depth}")

                model = XGBClassifier(
                    n_estimators=300,
                    learning_rate=lr,
                    max_depth=depth,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    random_state=42,
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight
                )

                # Log params
                mlflow.log_param("model", "xgboost")
                mlflow.log_param("learning_rate", lr)
                mlflow.log_param("max_depth", depth)
                mlflow.log_param("threshold", args.threshold)

                # Train
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - t0
                mlflow.log_metric("train_time", train_time)

                # Predict
                t1 = time.time()
                proba = model.predict_proba(X_test)[:, 1]
                y_pred = (proba >= args.threshold).astype(int)
                pred_time = time.time() - t1
                mlflow.log_metric("pred_time", pred_time)

                # Metrics
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, proba)

                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("roc_auc", roc_auc)

                # Save model to MLflow
                mlflow.sklearn.log_model(model, "model")

                # Store for ranking
                all_models.append({
                    "model": model,
                    "recall": recall,
                    "params": {
                        "learning_rate": lr,
                        "max_depth": depth
                    }
                })

                print(f"✅ Done lr={lr}, depth={depth}")

    # =========================
    # 8. SAVE TOP MODELS
    # =========================
    models_dir = os.path.join(project_root, "serving", "models")
    os.makedirs(models_dir, exist_ok=True)

    all_models = sorted(all_models, key=lambda x: x["recall"], reverse=True)

    top_k = 2

    for i in range(top_k):
        best = all_models[i]

        model_path = os.path.join(models_dir, f"model_top{i+1}.pkl")
        joblib.dump(best["model"], model_path)

        with mlflow.start_run(run_name=f"top_model_{i+1}"):

            mlflow.log_param("model", "xgboost")
            mlflow.log_params(best["params"])
            mlflow.log_metric("recall", best["recall"])

            mlflow.set_tag("rank", i+1)

            mlflow.sklearn.log_model(best["model"], "model")

        print(f"✅ Saved model_top{i+1}.pkl → {model_path}")
    # =========================
    # SAVE FINAL PRODUCTION MODEL
    # =========================
    final_model = all_models[0]["model"]  # best model (highest recall)

    final_model_path = os.path.join(models_dir, "model.pkl")
    joblib.dump(final_model, final_model_path)

    print(f"🔥 Final model saved → {final_model_path}")

    print("\n🎯 All tuning runs completed!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline")

    p.add_argument("--input", type=str, required=True)
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")

    args = p.parse_args()
    main(args)