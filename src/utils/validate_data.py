import pandas as pd
from typing import Tuple, List


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Telco churn validation (GE-style checks without API dependency issues)
    """

    print("🔍 Starting data validation...")

    failed = []

    # =========================
    # SCHEMA CHECKS
    # =========================
    required_cols = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges", "Churn"
    ]

    for col in required_cols:
        if col not in df.columns:
            failed.append(f"missing_column_{col}")

    # =========================
    # BUSINESS RULES
    # =========================
    if not df["gender"].isin(["Male", "Female"]).all():
        failed.append("invalid_gender")

    if not df["Partner"].isin(["Yes", "No"]).all():
        failed.append("invalid_partner")

    if not df["Dependents"].isin(["Yes", "No"]).all():
        failed.append("invalid_dependents")

    if not df["PhoneService"].isin(["Yes", "No"]).all():
        failed.append("invalid_phone_service")

    # =========================
    # NUMERIC CHECKS
    # =========================
    if (df["tenure"] < 0).any() or (df["tenure"] > 120).any():
        failed.append("invalid_tenure_range")

    if (df["MonthlyCharges"] < 0).any() or (df["MonthlyCharges"] > 200).any():
        failed.append("invalid_monthly_charges")

    # 🔥 FIXED ONLY THIS PART
    total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if (total_charges < 0).any():
        failed.append("invalid_total_charges")

    # =========================
    # TARGET CHECK
    # =========================
    if not df["Churn"].isin(["Yes", "No", 0, 1]).all():
        failed.append("invalid_churn_values")

    # =========================
    # FINAL RESULT
    # =========================
    success = len(failed) == 0

    if success:
        print("✅ Data validation PASSED")
    else:
        print(f"❌ Data validation FAILED: {failed}")

    return success, failed