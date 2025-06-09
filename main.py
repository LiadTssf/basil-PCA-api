from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://plant-talk.vercel.app"]
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
    explained_var = pca.explained_variance_ratio_.tolist()

    return {
        "status": prediction,
        "pca_components": X_pca.tolist()[0],
        "explained_variance": explained_var
    }
    
@app.get("/api/garden")
async def get_garden_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://gardenpi.duckdns.org/")
        response.raise_for_status()  # Throw 4xx/5xx errors
        return response.json()    
        
