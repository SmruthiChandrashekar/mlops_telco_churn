from fastapi import FastAPI
import joblib

app = FastAPI()

# Load both models
model1 = joblib.load("serving/models/model_top1.pkl")
model2 = joblib.load("serving/models/model_top2.pkl")


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to list
        features = list(data.values())

        # Predictions
        pred1 = model1.predict([features])
        pred2 = model2.predict([features])

        return {
            "model_top1_prediction": int(pred1[0]),
            "model_top2_prediction": int(pred2[0])
        }

    except Exception as e:
        return {"error": str(e)}