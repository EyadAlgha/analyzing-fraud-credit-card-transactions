"""FastAPI service exposing the fraud model for real-time scoring.

Train first (`python train.py` -> artifacts/fraud_model.joblib), then serve:

    uvicorn api:app --reload

    curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
      "merchant": "Kirlin and Sons", "category": "grocery", "amt": 240.5,
      "gender": "M", "city_pop": 120000, "job": "Surveyor",
      "dob": "1980-01-15", "trans_date_trans_time": "2020-06-21 02:14:25"}'
"""

import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from config import ARTIFACT_PATH
from inference import load_model, score

app = FastAPI(title='Fraud Detection API')

_model = None


def get_model():
    # Lazy-load and cache the bundle so the app imports without the artifact present.
    global _model
    if _model is None:
        _model = load_model(ARTIFACT_PATH)
    return _model


class Transaction(BaseModel):
    merchant: str
    category: str
    amt: float
    gender: str
    city_pop: int
    job: str
    dob: str                    # date of birth, e.g. "1980-01-15"
    trans_date_trans_time: str  # transaction timestamp, e.g. "2020-06-21 02:14:25"


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': os.path.exists(ARTIFACT_PATH)}


@app.post('/predict')
def predict(txn: Transaction):
    data = txn.model_dump() if hasattr(txn, 'model_dump') else txn.dict()
    result = score(pd.DataFrame([data]), get_model()).iloc[0]
    return {
        'is_fraud': int(result['is_fraud']),
        'fraud_proba': float(result['fraud_proba']),
        'risk_level': result['risk_level'],   # null when legitimate
        'decision': result['decision'],
    }
