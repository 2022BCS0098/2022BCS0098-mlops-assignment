from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.get("/")
def health():
    return {
        "Name": "Kushal Kumar",
        "Roll No": "2022BCS0098"
    }

@app.post("/predict")
def predict(features: list):
    data = np.array(features).reshape(1, -1)
    prediction = model.predict(data)[0]

    return {
        "prediction": int(prediction),
        "Name": "Kushal Kumar",
        "Roll No": "2022BCS0098"
    }

