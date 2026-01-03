from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Breast Cancer Prediction API")

# Load models
logistic_model = joblib.load("logistic_model.joblib")
tree_model = joblib.load("decision_tree_model.joblib")

# Input schema
class InputData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "ML API is running"}

@app.post("/predict/logistic")
def predict_logistic(data: InputData):
    prediction = logistic_model.predict([data.features])
    return {
        "model": "Logistic Regression",
        "prediction": int(prediction[0])
    }

@app.post("/predict/tree")
def predict_tree(data: InputData):
    prediction = tree_model.predict([data.features])
    return {
        "model": "Decision Tree",
        "prediction": int(prediction[0])
    }
