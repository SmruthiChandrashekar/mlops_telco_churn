import numpy as np
np.float_ = np.float64

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def run_drift():
    print("FILE STARTED")
    print("Loading dataset...")

    df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    print("Splitting data...")
    ref = df.sample(frac=0.5, random_state=42)
    curr = df.drop(ref.index)

    print("Generating report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr)

    report.save_html("monitoring/report.html")

    print("Report saved successfully!")

if __name__ == "__main__":
    run_drift()