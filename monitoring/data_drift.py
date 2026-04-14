import numpy as np
np.float_ = np.float64
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os
from src.logger import get_logger

logger = get_logger(__name__)

DATA_PATH   = "data/raw/dataset.csv"
REPORT_PATH = "monitoring/report.html"
DATE_COLUMN = None  # set to your date column e.g. "signup_date"
SPLIT_DATE  = None  # e.g. "2024-06-01"

def run_drift():
    logger.info("Loading dataset...")

    if not os.path.exists(DATA_PATH):
        logger.error(f"File not found at {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Dataset loaded: {len(df)} rows")
    except Exception as e:
        logger.error(f"Could not read CSV — {e}")
        return

    logger.info("Splitting data...")

    if DATE_COLUMN and SPLIT_DATE:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        ref  = df[df[DATE_COLUMN] <  SPLIT_DATE]
        curr = df[df[DATE_COLUMN] >= SPLIT_DATE]
        logger.info(f"Date split — Ref: {len(ref)} rows | Current: {len(curr)} rows")
    else:
        ref  = df.sample(frac=0.5, random_state=42)
        curr = df.drop(ref.index)
        logger.warning("Using random split — results may not reflect real drift")

    if ref.empty or curr.empty:
        logger.error("One of the splits is empty. Check your date range.")
        return

    try:
        logger.info("Generating drift report...")
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=curr)

        # drift summary
        drifted = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
        logger.info(f"Dataset drift detected: {drifted}")

        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        report.save_html(REPORT_PATH)
        logger.info(f"Report saved to {REPORT_PATH}")

    except Exception as e:
        logger.error(f"Failed to generate report — {e}")

if __name__ == "__main__":
    run_drift()
