from sklearn.model_selection import train_test_split

from config import (set_seed, TEST_SIZE, SEED, MERCHANT_RISK_ALPHA,
                    DECISION_THRESHOLD, RISK_FEATURES, ARTIFACT_PATH)
from data import load_raw_data
from preprocessing import (
    feature_analysis_preprocessing,
    fit_features,
    transform_features,
    fit_merchant_risk,
    apply_merchant_risk,
    MODEL_COLUMNS,
    FEATURE_COLUMNS,
)
from modeling import train_xgb, predict_with_threshold, classification_metrics
from clustering import fit_risk_clusterer
from inference import FraudModel, save_model


def train_model(path='data', test_size=TEST_SIZE, seed=SEED, alpha=MERCHANT_RISK_ALPHA, threshold=DECISION_THRESHOLD):
    """Fit all artifacts on the train split and bundle them into a FraudModel.

    Returns `(model, metrics, silhouette)` where `metrics` are held-out test
    Precision/Recall/F1 and `silhouette` is the train-fraud cluster quality.
    """
    set_seed()

    df = feature_analysis_preprocessing(load_raw_data(path))
    data = df[MODEL_COLUMNS]
    X, y = data[FEATURE_COLUMNS], data['is_fraud']

    # Split BEFORE fitting anything
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = test_size, stratify = y, random_state = seed)

    # Fit feature transformers + merchant_risk on TRAIN only
    X_train, feat = fit_features(X_train.copy())
    X_train, merchant_risk_dict, global_fraud_rate = fit_merchant_risk(X_train, y_train, alpha = alpha)
    X_train = X_train.drop(['merchant', 'job'], axis = 1)
    feature_columns = list(X_train.columns)

    # Stage 1: supervised fraud classifier
    classifier = train_xgb(X_train, y_train)

    # Stage 2: fit the risk clusterer on the TRAIN frauds the classifier flags
    y_train_pred = predict_with_threshold(classifier, X_train, threshold)
    train_frauds = X_train[y_train_pred == 1]
    kmeans, risk_map, silhouette = fit_risk_clusterer(train_frauds, RISK_FEATURES)

    model = FraudModel(
        pipe = feat['pipe'],
        freq_maps = feat['freq_maps'],
        scaler = feat['scaler'],
        merchant_risk_dict = merchant_risk_dict,
        global_fraud_rate = global_fraud_rate,
        feature_columns = feature_columns,
        classifier = classifier,
        threshold = threshold,
        kmeans = kmeans,
        risk_map = risk_map,
        risk_features = RISK_FEATURES,
    )

    # Honest held-out evaluation through the very same fitted artifacts
    X_test = transform_features(X_test.copy(), feat)
    X_test = apply_merchant_risk(X_test, merchant_risk_dict, global_fraud_rate).drop(['merchant', 'job'], axis = 1)
    X_test = X_test.reindex(columns = feature_columns, fill_value = 0)
    y_pred = predict_with_threshold(classifier, X_test, threshold)
    metrics = classification_metrics(y_test, y_pred)

    return model, metrics, silhouette


if __name__ == '__main__':
    model, metrics, silhouette = train_model()
    save_model(model, ARTIFACT_PATH)
    print(f"Saved model -> {ARTIFACT_PATH}")
    print(f"Test  Precision/Recall/F1: {metrics['precision']:.3f} / {metrics['recall']:.3f} / {metrics['f1']:.3f}")
    print(f"Train silhouette: {silhouette:.3f}")
