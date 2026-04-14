from fastapi import FastAPI
import joblib
import pandas as pd
import json

app = FastAPI()

# =========================
# LOAD ARTIFACTS
# =========================

# Load trained model
model = joblib.load("serving/models/model.pkl")

# Load feature column order
with open("artifacts/feature_columns.json") as f:
    feature_columns = json.load(f)

# Load threshold
with open("artifacts/threshold.txt") as f:
    THRESHOLD = float(f.read())


# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running 🚀"}


@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input JSON to DataFrame
        df = pd.DataFrame([data])

        # Align columns with training data
        df = df.reindex(columns=feature_columns, fill_value=0)

        # Predict probability
        proba = model.predict_proba(df)[:, 1]

        # Apply threshold
        pred = (proba >= THRESHOLD).astype(int)

        return {
            "prediction": int(pred[0]),
            "churn_probability": float(proba[0]),
            "threshold_used": THRESHOLD
        }

    except Exception as e:
        return {"error": str(e)}