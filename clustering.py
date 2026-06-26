import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import RISK_FEATURES, N_CLUSTERS, SEED, RISK_ORDER


def fit_risk_clusterer(fraud_data, risk_features=RISK_FEATURES, n_clusters=N_CLUSTERS, seed=SEED):
    # Fit KMeans over the risk features and rank the clusters from lowest to
    # highest risk, returning the fitted model + the cluster->label mapping so
    X_cluster = fraud_data[risk_features]

    kmeans = KMeans(n_clusters = n_clusters, random_state = seed)
    labels = kmeans.fit_predict(X_cluster)

    silhouette = silhouette_score(X_cluster, labels)  # Log-transforming age increases this score!

    # Rank clusters by their normalized mean risk score (lowest -> highest)
    cluster_stats = X_cluster.assign(risk_cluster = labels).groupby('risk_cluster')[risk_features].mean()
    normalized_stats = (cluster_stats - cluster_stats.mean()) / cluster_stats.std()
    risk_scores = normalized_stats.mean(axis = 1)

    sorted_clusters = risk_scores.sort_values().index.tolist()
    risk_map = {cluster: label for cluster, label in zip(sorted_clusters, RISK_ORDER)}

    return kmeans, risk_map, silhouette


def predict_risk_level(X_risk, kmeans, risk_map):
    # Assign new frauds to a risk level using an already-fitted clusterer.
    clusters = kmeans.predict(X_risk)
    return pd.Series(clusters, index = X_risk.index).map(risk_map)


def assign_risk_levels(fraud_data, risk_features=RISK_FEATURES, n_clusters=N_CLUSTERS, seed=SEED):
    # Analysis-path helper: fit the clusterer on these frauds and attach the
    # cluster + ordered risk_level columns, returning the validation artifacts.
    X_cluster = fraud_data[risk_features]

    kmeans, risk_map, silhouette = fit_risk_clusterer(fraud_data, risk_features, n_clusters, seed)

    clusters = kmeans.predict(X_cluster)
    fraud_data['risk_cluster'] = clusters
    fraud_data['risk_level'] = pd.Series(clusters, index = fraud_data.index).map(risk_map)

    # Compute mean values of features per cluster (for validation/plots)
    cluster_stats = fraud_data.groupby('risk_cluster')[risk_features].mean()
    cluster_stats['risk_level'] = cluster_stats.index.map(risk_map)

    # Make risk_level an ordered category for plotting consistency
    fraud_data['risk_level'] = pd.Categorical(fraud_data['risk_level'], categories = RISK_ORDER, ordered = True)

    return fraud_data, X_cluster, kmeans, silhouette, cluster_stats, risk_map
