import os
import random

import numpy as np

# Global reproducibility seed (was 52 before, now 18).
SEED = 18

# Train/test split.
TEST_SIZE = 0.3

# Laplace smoothing term for the train-only merchant_risk feature.
MERCHANT_RISK_ALPHA = 10

# Decision threshold for converting fraud probabilities into labels.
DECISION_THRESHOLD = 0.16

# Number of K-Means risk clusters.
N_CLUSTERS = 3

# RandomOverSampler keeps its own fixed seed for reproducibility.
OVERSAMPLE_SEED = 42

# Features used to cluster flagged frauds into risk levels.
RISK_FEATURES = ['amt', 'hour', 'age', 'city_pop', 'job_freq']

# Risk levels ordered from least to most dangerous.
RISK_ORDER = ['low-risk', 'medium-risk', 'high-risk']

# XGBoost base hyperparameters shared by every model variant.
XGB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.2,
    objective='binary:logistic',
    eval_metric='logloss',
)

# Search space for hyperparameter tuning (tune.py), biased toward regularization
# (shallower trees, lower lr, subsampling) to curb the base model's overfitting
# and its overconfident, saturated probabilities.
TUNE_PARAM_DIST = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 5, 10],
    'gamma': [0, 1, 5],
    'reg_lambda': [1.0, 5.0, 10.0],
    'reg_alpha': [0.0, 1.0],
}

# Where the trained model bundle (train.py) is saved and loaded from (api.py).
ARTIFACT_PATH = 'artifacts/fraud_model.joblib'

# Decision engine: map an inference outcome to a business action (business-specific).
DECISION_MAP = {
    'legitimate': 'approve',
    'low-risk': 'monitor',
    'medium-risk': 'review',
    'high-risk': 'decline',
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
