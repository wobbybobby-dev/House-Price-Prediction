from fastapi import FastAPI
from input_schemas import HouseFeatures
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI(title="House Price Prediction API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://127.0.0.1:5500"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained model
model = joblib.load("model/model.pkl")

@app.get("/")
def root():
    return {"message": "Welcome to the House Price Prediction API"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    rooms_per_person = features.AveRooms / features.AveOccup if features.AveOccup > 0 else 0

    # Convert input to model-compatible format
    input_data = np.array([[ 
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude,
        rooms_per_person  # <- engineered feature included here
    ]])

    # Predict using the loaded model
    prediction = model.predict(input_data)[0]
    return {
        "Predicted House Price (in 100,000 USD)": round(prediction, 2),
        "RoomsPerPerson (engineered feature)": round(rooms_per_person, 2)
        }
