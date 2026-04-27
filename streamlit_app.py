import json
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
import streamlit as st

from src.data.preprocess import preprocess_data


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "serving" / "models" / "model.pkl"
FEATURE_COLUMNS_PATH = ROOT / "artifacts" / "feature_columns.json"
THRESHOLD_PATH = ROOT / "artifacts" / "threshold.txt"
DEFAULT_THRESHOLD = 0.35


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def load_threshold() -> float:
    if THRESHOLD_PATH.exists():
        return float(THRESHOLD_PATH.read_text().strip())
    return DEFAULT_THRESHOLD


def get_feature_columns(model) -> Optional[List[str]]:
    if FEATURE_COLUMNS_PATH.exists():
        return json.loads(FEATURE_COLUMNS_PATH.read_text())

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "get_booster"):
        booster_features = model.get_booster().feature_names
        if booster_features:
            return list(booster_features)

    return None


def encode_single_customer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    yes_no_columns = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
    ]
    for column in yes_no_columns:
        if column in df.columns:
            df[column] = df[column].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1}).fillna(0).astype(int)

    category_columns = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    existing_categories = [column for column in category_columns if column in df.columns]
    if existing_categories:
        df = pd.get_dummies(df, columns=existing_categories, drop_first=True)

    for column in df.select_dtypes(include=["bool"]).columns:
        df[column] = df[column].astype(int)

    return df


def make_prediction(model, raw_input: dict, feature_columns: Optional[List[str]], threshold: float):
    raw_df = pd.DataFrame([raw_input])
    processed_df = preprocess_data(raw_df, target_col="Churn")
    feature_df = encode_single_customer(processed_df)

    if feature_columns:
        feature_df = feature_df.reindex(columns=feature_columns, fill_value=0)

    probability = float(model.predict_proba(feature_df)[:, 1][0])
    prediction = int(probability >= threshold)
    return prediction, probability


st.set_page_config(page_title="Telco Churn Prediction", layout="centered")

st.title("Telco Customer Churn Prediction")
st.write("Enter customer details and predict whether the customer is likely to churn.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = load_model()
except ModuleNotFoundError as exc:
    st.error(f"Missing Python package: {exc.name}")
    st.info("Install project dependencies with: pip install -r requirements.txt")
    st.stop()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

feature_columns = get_feature_columns(model)
threshold = load_threshold()

with st.form("churn_form"):
    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    col3, col4 = st.columns(2)
    with col3:
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    with col4:
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        prediction, probability = make_prediction(model, customer, feature_columns, threshold)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{probability:.2%}")
    st.write(f"Threshold used: `{threshold}`")

    if prediction == 1:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")
