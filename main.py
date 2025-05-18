from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_transform.pkl")
clf = joblib.load("plant_health_model.pkl")

# Input schema
class SensorInput(BaseModel):
    soilMoisture: float
    temperature: float
    humidity: float
    timeOfDay: int

@app.post("/predict")
async def predict(input: SensorInput):
    X = np.array([[input.soilMoisture, input.temperature, input.humidity, input.timeOfDay]])
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    prediction = clf.predict(X_pca)[0]
    return {"status": prediction}
