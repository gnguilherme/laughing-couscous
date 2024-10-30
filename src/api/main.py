"""API module."""

from typing import Any

from fastapi import FastAPI
from pandas import DataFrame
from pydantic import BaseModel, Field


class Feature(BaseModel):
    feature_1: float = Field(..., example=1.0)
    feature_2: float = Field(..., example=2.0)
    feature_3: float = Field(..., example=3.0)


app = FastAPI(
    title="Model Prediction API",
    description="Simple API for model prediction.",
)


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/predict")
def predict(features: Feature) -> Any:

    features = DataFrame([[features.feature_1, features.feature_2, features.feature_3]])
    # load model from mlflow
    # prediction = model.predict(features)
    return {"prediction": features.to_dict(orient="records")}
