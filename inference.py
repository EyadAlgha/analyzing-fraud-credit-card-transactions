"""Online inference: a persisted model bundle and a scoring path for raw transactions.

`FraudModel` bundles every fitted artifact produced during training so a new
transaction can flow through the same two stages used offline: the XGBoost
classifier (stage 1) and, for flagged frauds, the KMeans risk clusterer (stage 2).
A configurable decision engine then maps the outcome to a business action.
"""

import os
from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

from config import DECISION_MAP
from preprocessing import (
    feature_analysis_preprocessing,
    transform_features,
    apply_merchant_risk,
    FEATURE_COLUMNS,
)
from clustering import predict_risk_level


@dataclass
class FraudModel:
    pipe: Any                  # fitted feature_engine one-hot Pipeline
    freq_maps: dict            # {'merchant': {...}, 'job': {...}} train frequencies
    scaler: Any                # fitted MinMaxScaler
    merchant_risk_dict: dict
    global_fraud_rate: float
    feature_columns: list      # exact model-matrix column order from training
    classifier: Any            # fitted XGBClassifier
    threshold: float           # decision threshold for the fraud probability
    kmeans: Any                # fitted risk clusterer
    risk_map: dict             # cluster id -> risk level (low/medium/high)
    risk_features: list


def save_model(model, path):
    """Persist the model bundle, creating the parent directory if needed."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """Load a model bundle saved by `save_model`."""
    return joblib.load(path)


def build_feature_matrix(raw_df, model):
    """Turn raw transactions into the exact model matrix the classifier expects."""
    df = feature_analysis_preprocessing(raw_df.copy())
    feat_artifacts = {'pipe': model.pipe, 'freq_maps': model.freq_maps, 'scaler': model.scaler}

    X = transform_features(df[FEATURE_COLUMNS].copy(), feat_artifacts)
    X = apply_merchant_risk(X, model.merchant_risk_dict, model.global_fraud_rate)
    X = X.drop(['merchant', 'job'], axis = 1)
    # Align to the training columns (adds any one-hot column missing from this batch)
    return X.reindex(columns = model.feature_columns, fill_value = 0)


def _decide(is_fraud, risk_level):
    if not is_fraud:
        return DECISION_MAP['legitimate']
    return DECISION_MAP.get(risk_level, DECISION_MAP['medium-risk'])


def score(raw_df, model):
    """Score raw transactions.
    Returns a DataFrame with `fraud_proba`, `is_fraud`, `risk_level` (None for legitimate), and `decision`.
    """
    X = build_feature_matrix(raw_df, model)

    proba = model.classifier.predict_proba(X)[:, 1]
    out = pd.DataFrame(index = X.index)
    out['fraud_proba'] = proba
    out['is_fraud'] = (proba >= model.threshold).astype(int)
    out['risk_level'] = None

    # Stage 2 only runs on the transactions flagged as fraud
    fraud_mask = out['is_fraud'] == 1
    if fraud_mask.any():
        X_risk = X.loc[fraud_mask, model.risk_features]
        out.loc[fraud_mask, 'risk_level'] = predict_risk_level(X_risk, model.kmeans, model.risk_map)

    out['decision'] = [_decide(f, r) for f, r in zip(out['is_fraud'], out['risk_level'])]
    return out
