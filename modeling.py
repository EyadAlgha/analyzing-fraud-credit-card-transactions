import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, brier_score_loss

from config import XGB_PARAMS, SEED, DECISION_THRESHOLD, TUNE_PARAM_DIST

def class_weight(y_train):
    neg, pos = np.bincount(y_train)  # get number of frauds and non-frauds from training set
    weight = neg / pos
    return neg, pos, weight


def build_xgb(scale_pos_weight=None, seed=SEED, **params):
    config = {**XGB_PARAMS, **params, 'random_state': seed}
    if scale_pos_weight is not None:
        config['scale_pos_weight'] = scale_pos_weight
    return XGBClassifier(**config)


def train_xgb(X_train, y_train, scale_pos_weight=None, **params):
    model = build_xgb(scale_pos_weight=scale_pos_weight, **params)
    model.fit(X_train, y_train)
    return model


def predict_with_threshold(model, X, threshold=DECISION_THRESHOLD):
    y_proba = model.predict_proba(X)[:, 1]  # Obtain all fraud probabilities
    return (y_proba >= threshold).astype(int)


def classification_metrics(y_true, y_pred):
    # Precision / Recall / F1 for the fraud class (label 1)
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }


def probability_metrics(y_true, y_proba):
    # How well-calibrated / confident the probabilities are. Lower Brier is better;
    # 'saturation' is the fraction of predictions pinned near 0 or 1 (overconfidence).
    saturation = float(((y_proba < 0.01) | (y_proba > 0.99)).mean())
    return {'brier': brier_score_loss(y_true, y_proba), 'saturation': saturation}


def tune_xgb(X, y, n_iter=25, cv=3, scoring='average_precision', search_frac=0.25, seed=SEED):
    """Randomized hyperparameter search over TUNE_PARAM_DIST.

    Searches on a stratified `search_frac` of the data for speed (XGBoost 'hist'),
    scores with average precision (threshold-independent, good for imbalance), then
    refits the best configuration on the FULL training set.
    Returns `(best_model, best_params, best_cv_score)`.
    """
    X_search, y_search = X, y
    if search_frac < 1.0:
        X_search, _, y_search, _ = train_test_split(
            X, y, train_size=search_frac, stratify=y, random_state=seed)

    base = XGBClassifier(tree_method='hist', eval_metric='logloss', random_state=seed)
    search = RandomizedSearchCV(
        base, TUNE_PARAM_DIST, n_iter=n_iter, scoring=scoring, cv=cv,
        random_state=seed, verbose=1)
    search.fit(X_search, y_search)

    best_model = XGBClassifier(
        **search.best_params_, tree_method='hist', eval_metric='logloss', random_state=seed)
    best_model.fit(X, y)  # refit best config on the full training data
    return best_model, search.best_params_, search.best_score_
