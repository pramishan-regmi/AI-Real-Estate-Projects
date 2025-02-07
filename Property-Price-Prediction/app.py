# api/app.py
from fastapi import FastAPI
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load("models/property_price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the Property Price Prediction API!"}

@app.post("/predict")
def predict_price(features: dict):
    input_data = pd.DataFrame([features])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return {"predicted_price": round(prediction, 2)}

# Run with: uvicorn app:app --reload
