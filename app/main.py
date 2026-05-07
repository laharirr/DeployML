from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time

app = FastAPI(title="DeployML - ML Deployment Platform")

model = joblib.load("model/placement_model.pkl")

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency"
)


class PredictionInput(BaseModel):
    cgpa: float
    skills: int
    projects: int


@app.get("/")
def home():
    return {
        "message": "Welcome to DeployML"
    }


@app.post("/predict")
def predict(data: PredictionInput):
    start_time = time.time()

    REQUEST_COUNT.inc()

    input_data = np.array([
        [data.cgpa, data.skills, data.projects]
    ])

    prediction = model.predict(input_data)[0]

    REQUEST_LATENCY.observe(time.time() - start_time)

    result = "Placed" if prediction == 1 else "Not Placed"

    return {
        "prediction": result
    }


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )