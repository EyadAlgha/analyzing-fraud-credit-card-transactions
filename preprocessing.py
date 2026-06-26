import numpy as np
import pandas as pd
from feature_engine.encoding import OneHotEncoder
from feature_engine.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from config import TEST_SIZE, SEED, MERCHANT_RISK_ALPHA, OVERSAMPLE_SEED

# Columns kept for modeling (dropped 'city'). 'merchant'/'job' are kept through
# encoding so merchant_risk can use them, then dropped at the end.
MODEL_COLUMNS = ['merchant', 'category', 'amt', 'gender', 'city_pop', 'job',
                 'is_fraud', 'age', 'year', 'month', 'day', 'hour', 'minute']

# Raw feature columns fed to the model (everything in MODEL_COLUMNS except the label).
FEATURE_COLUMNS = [c for c in MODEL_COLUMNS if c != 'is_fraud']

# Columns log-transformed then scaled, and the full set of scaled columns.
LOG_COLS = ['amt', 'city_pop', 'age']
SCALE_COLS = LOG_COLS + ['year', 'merchant_freq', 'job_freq']
FREQ_COLS = ['merchant', 'job']


def feature_analysis_preprocessing(df):
  # Convert features to their suitable type, preprocess features, and feature engineering for aid in visualization

  # Remove fraud_ tag from all merchants
  df['merchant'] = df['merchant'].apply(lambda item : item.replace('fraud_', ''))

  # Convert time-related objects to their correct type (datetime object)
  df['dob'] = pd.to_datetime(df['dob'])
  df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

  # Feature engineering
  #current_year = pd.Timestamp.now().year # Obtain current year
  df['age'] = 2021 - df['dob'].dt.year # Extract age of user

  age_bins = [16, 30, 50, 65, 101] # Group ages into bins
  age_label = ['Young Adult', 'Middle Aged', 'Adult', 'Senior'] # Label for each bin
  df['age_group'] = pd.cut(df['age'], bins = age_bins, labels = age_label, include_lowest = True) # Bin ages to labels (e.g, age 0-18 to young adult, 19-30 to middle aged, etc...)

  # Extract year, month, day, hour, minute of transaction
  df['year'] = df['trans_date_trans_time'].dt.year
  df['month'] = df['trans_date_trans_time'].dt.month
  df['day'] = df['trans_date_trans_time'].dt.day
  df['hour'] = df['trans_date_trans_time'].dt.hour
  df['minute'] = df['trans_date_trans_time'].dt.minute

  # Map related categories to one bin
  df['category'] = df['category'].map({'gas_transport':'gas_transport',
                                                'home': 'home',
                                                'shopping_pos':'shopping',
                                                'shopping_net':'shopping',
                                                'kids_pets':'kids_pets',
                                                'entertainment':'entertainment',
                                                'food_dining':'food_dining',
                                                'personal_care':'personal_care',
                                                'health_fitness':'health_fitness',
                                                'misc_pos':'misc',
                                                'misc_net':'misc',
                                                'grocery_net':'grocery',
                                                'grocery_pos':'grocery',
                                                'travel':'travel'})

  return df


def fit_features(X_train):
  # Fit every encoder/scaler on the TRAIN split only and return the transformed
  # train matrix plus the fitted artifacts so the exact same transforms can be
  # re-applied to test data and to new transactions at inference time.

  # One-Hot encode categorical features
  pipe = Pipeline([('onehot_encoder', OneHotEncoder(variables = ['category'])),
                   ('label_encoder', OneHotEncoder(variables = ['gender'], drop_last = True))])
  pipe.fit(X_train)
  X_train = pipe.transform(X_train)

  # Frequency-Encode 'merchant' and 'job' using TRAIN counts only
  freq_maps = {}
  for col in FREQ_COLS:
    freq_maps[col] = X_train[col].value_counts().to_dict()
    X_train[f'{col}_freq'] = X_train[col].map(freq_maps[col])

  # Log-transform skewed features (element-wise, no fitting required)
  for col in LOG_COLS:
    X_train[col] = np.log1p(X_train[col])  # log(1 + x) to avoid zero logs

  # Scale numerical features
  scaler = MinMaxScaler()
  X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])

  feat_artifacts = {'pipe': pipe, 'freq_maps': freq_maps, 'scaler': scaler}
  return X_train, feat_artifacts


def transform_features(X, feat_artifacts):
  # Apply already-fitted feature transformers to new/test data (no fitting).
  X = feat_artifacts['pipe'].transform(X)

  for col in FREQ_COLS:
    # Merchants/jobs unseen in training appeared 0 times there
    X[f'{col}_freq'] = X[col].map(feat_artifacts['freq_maps'][col]).fillna(0)

  for col in LOG_COLS:
    X[col] = np.log1p(X[col])

  X[SCALE_COLS] = feat_artifacts['scaler'].transform(X[SCALE_COLS])
  return X


def fit_transform_features(X_train, X_test):
  # Convenience wrapper used by the offline analysis path: fit on train, apply to both.
  X_train, feat_artifacts = fit_features(X_train)
  X_test = transform_features(X_test, feat_artifacts)
  return X_train, X_test, feat_artifacts['scaler']


def fit_merchant_risk(X_train, y_train, alpha=MERCHANT_RISK_ALPHA):
  # Laplace-smoothed historical fraud rate per merchant, learned on TRAIN only.
  train_data = X_train.copy()
  train_data['is_fraud'] = y_train

  global_fraud_rate = y_train.mean()

  merchant_stats = train_data.groupby('merchant')['is_fraud'].agg(['mean', 'count']) # Obtain mean and frequency of fraudelent merchants

  # risk_smoothed = mean * freq + all_rate*alpha / (freq + alpha)
  merchant_stats['smoothed_risk'] = ((merchant_stats['mean'] * merchant_stats['count'] + global_fraud_rate * alpha) / (merchant_stats['count'] + alpha))

  merchant_risk_dict = merchant_stats['smoothed_risk'].to_dict()

  X_train = apply_merchant_risk(X_train, merchant_risk_dict, global_fraud_rate)
  return X_train, merchant_risk_dict, global_fraud_rate


def apply_merchant_risk(X, merchant_risk_dict, global_fraud_rate):
  # Map the learned per-merchant risk; merchants unseen in training fall back to the global rate.
  X['merchant_risk'] = X['merchant'].map(merchant_risk_dict).fillna(global_fraud_rate)
  return X


def prepare_model_data(df, test_size=TEST_SIZE, seed=SEED, alpha=MERCHANT_RISK_ALPHA):
  data = df[MODEL_COLUMNS]
  X = data.drop('is_fraud', axis = 1)
  y = data['is_fraud']

  # Split BEFORE fitting any encoder/scaler to prevent leakage
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = y, random_state = seed)

  X_train, X_test, scaler = fit_transform_features(X_train, X_test)

  X_train, merchant_risk_dict, global_fraud_rate = fit_merchant_risk(X_train, y_train, alpha = alpha)
  X_test = apply_merchant_risk(X_test, merchant_risk_dict, global_fraud_rate)

  X_train = X_train.drop(['merchant', 'job'], axis = 1)
  X_test = X_test.drop(['merchant', 'job'], axis = 1)

  return X_train, X_test, y_train, y_test, scaler, merchant_risk_dict


def oversample(X_train, y_train, random_state=OVERSAMPLE_SEED):
  ros = RandomOverSampler(random_state = random_state) # Keep random state to 42 for reproducability
  X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
  return X_train_resampled, y_train_resampled
