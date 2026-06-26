from config import set_seed, DECISION_THRESHOLD
from data import load_raw_data
from preprocessing import (
    feature_analysis_preprocessing,
    prepare_model_data,
)
from modeling import train_xgb, predict_with_threshold, classification_metrics
from clustering import assign_risk_levels


def run_pipeline(path='data', threshold=DECISION_THRESHOLD):
    set_seed()

    # Stage 0: load and engineer analysis features.
    df = load_raw_data(path)
    df = feature_analysis_preprocessing(df)

    # Stage 1: model-ready matrices + supervised fraud classifier.
    X_train, X_test, y_train, y_test, scaler, merchant_risk_dict = prepare_model_data(df)
    model = train_xgb(X_train, y_train)
    y_pred = predict_with_threshold(model, X_test, threshold)
    metrics = classification_metrics(y_test, y_pred)

    # Stage 2: cluster the flagged frauds into low/medium/high risk levels.
    fraud_data = X_test[y_pred == 1].copy()
    fraud_data, X_cluster, kmeans, silhouette, cluster_stats, risk_map = assign_risk_levels(fraud_data)

    return {
        'model': model,
        'scaler': scaler,
        'merchant_risk_dict': merchant_risk_dict,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'threshold': threshold,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'fraud_data': fraud_data,
        'kmeans': kmeans,
        'silhouette': silhouette,
        'cluster_stats': cluster_stats,
        'risk_map': risk_map,
    }


if __name__ == '__main__':
    results = run_pipeline()
    print(f"Threshold (T): {results['threshold']}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1: {results['f1']:.3f}")
    print(f"Silhouette score: {results['silhouette']:.3f}")
    print(f"Flagged frauds: {len(results['fraud_data'])}")
