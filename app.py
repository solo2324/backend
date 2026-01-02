from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Breast Cancer Prediction API")

# CORS settings to allow frontend requests
origins = [
    "https://frontend-lk5q.vercel.app",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained models
logistic_model = joblib.load("logistic_model.joblib")
tree_model = joblib.load("decision_tree_model.joblib")

# Input schema
class InputData(BaseModel):
    features: list[float]

# Root endpoint for quick test
@app.get("/")
def root():
    return {"message": "Breast Cancer Prediction API is running"}

# Logistic Regression prediction
@app.post("/predict/logistic")
def predict_logistic(data: InputData):
    prediction = logistic_model.predict([data.features])[0]
    return {"model": "Logistic Regression", "prediction": int(prediction)}

# Decision Tree prediction
@app.post("/predict/tree")
def predict_tree(data: InputData):
    prediction = tree_model.predict([data.features])[0]
    return {"model": "Decision Tree", "prediction": int(prediction)}
