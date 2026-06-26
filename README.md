# Analyzing Fraudulent Credit Card Transactions

A machine learning project that detects fraudulent credit card transactions and, for the transactions flagged as fraud, assigns a risk level (low, medium, or high) describing how threatening each one is to the user and the business.

**Authors:** Eyad Alghamdi, Raed Alghamdi, Nawaf Alshehri

## Overview

Financial fraud is a large and growing problem, with global fraud losses reaching an estimated $486 billion in 2023. Most published work treats fraud detection as a binary problem: a transaction is either fraud or not.

This project adds a second stage. After a transaction is classified as fraud, an unsupervised clustering step sorts it into one of three risk categories. The motivation is the gap identified in the literature review: existing studies provide no risk label that tells an institution how dangerous a given fraud is, which is the kind of signal useful for prioritizing manual investigation and downstream decision-making.

The two-stage design is:

1. **Supervised classification** with XGBoost to separate fraud from legitimate transactions.
2. **Unsupervised clustering** with K-Means over risk-related features to assign a low / medium / high risk label to the flagged frauds.

## Dataset

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) by kartik2112 (Kaggle), pulled at runtime via `kagglehub`.

- ~1,296,675 transactions, 33 original features
- Contamination rate below 0.58% (highly imbalanced)
- No duplicate rows, no missing values
- Mixed feature types: categorical, numerical, datetime, geolocation

## Pipeline

### Feature engineering

- Extracted `age` from the cardholder date of birth, then binned it into age groups (minors, young adults, adults, seniors)
- Decomposed `trans_date_trans_time` into year, month, day, hour, and minute
- Stripped the `fraud_` prefix from merchant names
- Built a `merchant_risk` feature: a Laplace-smoothed historical fraud rate per merchant, computed only on the training split to avoid leakage (smoothing term alpha = 10)

### Preprocessing

- `category` -> One-Hot Encoding
- `gender` -> Label Encoding
- `merchant`, `job` -> Frequency Encoding
- `amt`, `city_pop`, `age` -> log transform
- All numerical features then normalized to the [0, 1] range
- Train/test split: 70/30, stratified on the fraud label

### Classifier (XGBoost)

XGBoost was chosen for its interpretability, strong handling of class imbalance, and fast training. Base configuration:

```
n_estimators = 500
learning_rate = 0.2
objective = 'binary:logistic'
eval_metric = 'logloss'
```

Three strategies for the imbalance were compared:

| Approach                  | Precision | Recall | F1   |
|---------------------------|-----------|--------|------|
| Base model                | 0.97      | 0.84   | 0.90 |
| Random oversampling       | 0.87      | 0.91   | 0.89 |
| Tuning scale_pos_weight   | 0.86      | 0.90   | 0.88 |

To balance catching frauds against keeping manual review costs down, the decision threshold was lowered from the default 0.5. A threshold near 0.15-0.16 brings precision and recall together at roughly 0.89 each (F1 ~ 0.89).

This result edges out the Kaggle state-of-the-art replicated on the same dataset (our F1 0.900 vs the reference 0.882).

### Interpretation

- Feature importance by weight and by gain
- A single exported decision tree (`xgb_tree.png`)
- SHAP analysis for per-feature contribution (commented out in the notebook because the full run takes around 40 minutes)

### Clustering and risk labels

The frauds caught by the classifier are clustered with K-Means (k = 3) over a set of risk-related features: `amt`, `hour`, `age`, `city_pop`, `job_freq`. Clusters are then ranked by a normalized mean risk score and mapped to low / medium / high.

- Silhouette score: 0.830

### Cluster validation

To check that the clusters actually describe different risk levels rather than some unrelated grouping:

- **Histograms** of each feature per risk level (peak and spread)
- **One-way ANOVA** per feature. All selected features came out significant (p < 0.05) except `job_freq` (p = 0.0586)
- **Tukey HSD post-hoc tests** to locate which specific pairs of clusters differ
- **3D t-SNE** visualization of the clusters (perplexity = 125)

The expectation, supported by the analysis, is that low-risk and high-risk groups separate cleanly while medium-risk sits as a fuzzier middle band.

## Repository contents

```
fraud_detection_final.ipynb     Full pipeline: EDA, modeling, clustering, validation
ML_Final_Presentation.pptx      Final presentation slides
README.md
```

## Running it

The notebook is built for an environment with internet access (it downloads the dataset on the fly). Install the dependencies:

```bash
pip install kaggle kagglehub folium feature_engine xgboost \
            scikit-learn imbalanced-learn shap scipy statsmodels \
            pandas numpy matplotlib seaborn
```

Then run the notebook top to bottom:

```bash
jupyter notebook fraud_detection_final.ipynb
```

A global seed (18) is set for reproducibility. Note that t-SNE and SHAP are the slowest steps.
