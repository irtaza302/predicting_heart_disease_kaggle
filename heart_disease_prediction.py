#!/usr/bin/env python3
"""
Heart Disease Prediction - Kaggle Playground Series S6E2
Predict the likelihood of heart disease using ROC-AUC evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train = pd.read_csv('playground-series-s6e2/train.csv')
test = pd.read_csv('playground-series-s6e2/test.csv')
sample_sub = pd.read_csv('playground-series-s6e2/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Explore target variable
print("\n--- Target Distribution ---")
print(train['Heart Disease'].value_counts())

# Encode target
train['target'] = (train['Heart Disease'] == 'Presence').astype(int)

# Feature columns
feature_cols = [col for col in train.columns if col not in ['id', 'Heart Disease', 'target']]
print(f"\nFeatures: {feature_cols}")

# Prepare data
X = train[feature_cols].copy()
y = train['target'].copy()
X_test = test[feature_cols].copy()

# Feature Engineering
print("\n--- Feature Engineering ---")

def add_features(df):
    """Add engineered features"""
    df = df.copy()
    
    # Age groups
    df['Age_group'] = pd.cut(df['Age'], bins=[0, 40, 50, 60, 70, 100], labels=[0, 1, 2, 3, 4])
    df['Age_group'] = df['Age_group'].astype(int)
    
    # BP categories
    df['BP_category'] = pd.cut(df['BP'], bins=[0, 120, 140, 160, 200, 300], labels=[0, 1, 2, 3, 4])
    df['BP_category'] = df['BP_category'].astype(int)
    
    # Cholesterol categories
    df['Cholesterol_category'] = pd.cut(df['Cholesterol'], bins=[0, 200, 240, 280, 320, 700], labels=[0, 1, 2, 3, 4])
    df['Cholesterol_category'] = df['Cholesterol_category'].astype(int)
    
    # Risk factors
    df['risk_score'] = (df['Age'] > 50).astype(int) + \
                       (df['Sex'] == 1).astype(int) + \
                       (df['Cholesterol'] > 240).astype(int) + \
                       (df['BP'] > 140).astype(int) + \
                       (df['Exercise angina'] == 1).astype(int)
    
    # Heart rate reserve (approximation)
    df['HR_reserve'] = 220 - df['Age'] - df['Max HR']
    
    # Interaction features
    df['Age_BP'] = df['Age'] * df['BP'] / 100
    df['Age_Cholesterol'] = df['Age'] * df['Cholesterol'] / 100
    df['BP_Cholesterol'] = df['BP'] * df['Cholesterol'] / 1000
    
    # ST depression ratio
    df['ST_ratio'] = df['ST depression'] / (df['Max HR'] / 100 + 1)
    
    return df

X = add_features(X)
X_test = add_features(X_test)

print(f"Features after engineering: {X.shape[1]}")

# Model training with cross-validation
print("\n--- Training Models ---")

# Define models
models = {
    'XGBoost': xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        random_state=42,
        verbose=0,
        thread_count=-1
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_scores[name] = scores.mean()
    print(f"{name} CV AUC: {scores.mean():.5f} (+/- {scores.std():.5f})")
    
    # Train on full data and predict
    model.fit(X, y)
    predictions[name] = model.predict_proba(X_test)[:, 1]

# Display results
print("\n--- Model Performance Summary ---")
for name, score in sorted(cv_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.5f}")

# Ensemble predictions (weighted average based on CV scores)
print("\n--- Creating Ensemble ---")
total_score = sum(cv_scores.values())
weights = {name: score/total_score for name, score in cv_scores.items()}

ensemble_pred = np.zeros(len(X_test))
for name, pred in predictions.items():
    ensemble_pred += weights[name] * pred
    print(f"{name} weight: {weights[name]:.4f}")

# Also try simple average
simple_avg_pred = np.mean([predictions[name] for name in predictions], axis=0)

# Use the best performing approach
best_model = max(cv_scores, key=cv_scores.get)
print(f"\nBest single model: {best_model} with AUC: {cv_scores[best_model]:.5f}")

# Create submission files
print("\n--- Creating Submissions ---")

# Best single model submission
sub_best = sample_sub.copy()
sub_best['Heart Disease'] = predictions[best_model]
sub_best.to_csv('submission_best_model.csv', index=False)
print(f"Saved: submission_best_model.csv ({best_model})")

# Weighted ensemble submission
sub_weighted = sample_sub.copy()
sub_weighted['Heart Disease'] = ensemble_pred
sub_weighted.to_csv('submission_weighted_ensemble.csv', index=False)
print("Saved: submission_weighted_ensemble.csv")

# Simple average ensemble submission
sub_avg = sample_sub.copy()
sub_avg['Heart Disease'] = simple_avg_pred
sub_avg.to_csv('submission_simple_avg.csv', index=False)
print("Saved: submission_simple_avg.csv")

# Stacking ensemble with meta-learner
print("\n--- Advanced Stacking Ensemble ---")

# Create out-of-fold predictions for stacking
oof_preds = np.zeros((len(X), len(models)))
test_preds_stacking = np.zeros((len(X_test), len(models)))

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train_fold = X.iloc[train_idx]
    y_train_fold = y.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    
    for i, (name, model) in enumerate(models.items()):
        # Clone model for each fold
        if name == 'XGBoost':
            m = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                use_label_encoder=False, eval_metric='logloss', n_jobs=-1
            )
        elif name == 'LightGBM':
            m = lgb.LGBMClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbose=-1, n_jobs=-1
            )
        elif name == 'CatBoost':
            m = CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.05,
                random_state=42, verbose=0, thread_count=-1
            )
        elif name == 'RandomForest':
            m = RandomForestClassifier(
                n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
            )
        else:
            m = GradientBoostingClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42
            )
        
        m.fit(X_train_fold, y_train_fold)
        oof_preds[val_idx, i] = m.predict_proba(X_val_fold)[:, 1]
        test_preds_stacking[:, i] += m.predict_proba(X_test)[:, 1] / cv.n_splits

# Train meta-learner (Logistic Regression)
meta_model = LogisticRegression(random_state=42)
meta_model.fit(oof_preds, y)
stacking_pred = meta_model.predict_proba(test_preds_stacking)[:, 1]

# Stacking submission
sub_stacking = sample_sub.copy()
sub_stacking['Heart Disease'] = stacking_pred
sub_stacking.to_csv('submission_stacking.csv', index=False)
print("Saved: submission_stacking.csv")

# Evaluate stacking on OOF predictions
stacking_auc = roc_auc_score(y, oof_preds.mean(axis=1))
print(f"Stacking OOF AUC (simple avg): {stacking_auc:.5f}")

# Meta model coefficients
print("\nMeta-learner coefficients:")
for i, name in enumerate(models.keys()):
    print(f"  {name}: {meta_model.coef_[0][i]:.4f}")

print("\n--- All submissions created! ---")
print("Recommended submission order:")
print("1. submission_stacking.csv (most robust)")
print("2. submission_weighted_ensemble.csv")
print("3. submission_best_model.csv")