# -*- coding: utf-8 -*-
"""Tsunami Potential Classification using Random Forest and XGBoost
Based on Historical Earthquake Data in Japan (2000-2025)
"""

# Install required geospatial and mapping packages if running in a fresh environment
# !pip install global_land_mask
# !pip install basemap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve,
                             roc_auc_score, precision_recall_curve)
from global_land_mask import globe
from mpl_toolkits.basemap import Basemap

# ==========================================
# 1. DATA LOADING AND PRE-PROCESSING
# ==========================================

# Load datasets (Adjust file paths as needed for your repository structure)
datagempa = pd.read_csv('data/raw/data2000-2025(csv).csv', delimiter=';')
katalogtsunami = pd.read_csv('data/raw/katalog tsunami 2000-2025 japan.csv', delimiter=';')

# Filter earthquake data based on empirical minimum tsunami thresholds (Magnitude >= 5.3, Depth <= 68 km)
datagempa2 = datagempa[(datagempa['MT_Magnitude(Mw)'] >= 5.3) & (datagempa['MT_Depth(km)'] <= 68)]

datagempa3 = datagempa2[['Origin_Time(UT)', 'MT_Magnitude(Mw)', 'MT_Depth(km)', 'Longitude(deg)', 'Latitude(deg)', 'Rake', 'Strike', 'Dip']].copy()

# Format Origin Time in tsunami catalog to match earthquake catalog
katalogtsunami['Origin_Time(UT)'] = (katalogtsunami['Year'].astype(str) + '/' +
                                  katalogtsunami['Mo'].astype(str).str.zfill(2) + '/' +
                                  katalogtsunami['Dy'].astype(str).str.zfill(2) + ',' +
                                  katalogtsunami['Hr'].astype(str).str.zfill(2) + ':' +
                                  katalogtsunami['Mn'].astype(str).str.zfill(2) + ':' +
                                  katalogtsunami['Sec'].astype(str).replace(',', '.', regex=True))

catalogtsunami2 = katalogtsunami[['Origin_Time(UT)', 'Latitude', 'Longitude']].copy()

datagempa3['Origin_Time(UT)'] = pd.to_datetime(datagempa3['Origin_Time(UT)'].str.replace('"', ''), format='%Y/%m/%d,%H:%M:%S.%f')
catalogtsunami2['Origin_Time(UT)'] = pd.to_datetime(catalogtsunami2['Origin_Time(UT)'].str.replace('"', ''), format='%Y/%m/%d,%H:%M:%S.%f')

datagempa3['Time_dt'] = datagempa3['Origin_Time(UT)']
catalogtsunami2['Time_dt'] = catalogtsunami2['Origin_Time(UT)']

# Add marker column to tsunami catalog
catalogtsunami2['match_found'] = 1

# Sort by time for merge_asof
datagempa3 = datagempa3.sort_values('Time_dt')
catalogtsunami2 = catalogtsunami2.sort_values('Time_dt')

# Integrate earthquake data with tsunami catalog using nearest timestamp matching
datagempa4 = pd.merge_asof(
    datagempa3,
    catalogtsunami2[['Time_dt', 'match_found']],
    on='Time_dt',
    direction='nearest',
    tolerance=pd.Timedelta('1.5min')
)

# Create binary target label (1 if matched with tsunami event, 0 otherwise)
datagempa4['Label_Tsunami'] = datagempa4['match_found'].fillna(0).astype(int)
datagempa4 = datagempa4.drop(columns=['match_found', 'Time_dt'])

print("Labeling Process Complete!")
print(datagempa4['Label_Tsunami'].value_counts())

# Filter offshore earthquakes using global land mask (1 = sea, 0 = land)
def is_it_sea(lat, lon):
    return 0 if globe.is_land(lat, lon) else 1

datagempa4['is_sea'] = datagempa4.apply(lambda x: is_it_sea(x['Latitude(deg)'], x['Longitude(deg)']), axis=1)
datagempa5 = datagempa4[datagempa4['is_sea'] == 1]

# Spatial bounding box filtering for Japan region
min_lat, max_lat = 24.0, 46.0
min_lon, max_lon = 122.0, 148.0

datagempa6 = datagempa5[
    (datagempa5['Latitude(deg)'] >= min_lat) & (datagempa5['Latitude(deg)'] <= max_lat) &
    (datagempa5['Longitude(deg)'] >= min_lon) & (datagempa5['Longitude(deg)'] <= max_lon)
]

print(f"Initial data count: {len(datagempa5)}")
print(f"Filtered data count: {len(datagempa6)}")

# Drop specific spatial outlier
datagempa7 = datagempa6.drop(index=[1029], errors='ignore')

# ==========================================
# 2. FAULT MECHANISM CLASSIFICATION
# ==========================================

def classify_fault_type(rake_val):
    if pd.isna(rake_val):
        return np.nan
    if isinstance(rake_val, str):
        try:
            rake = float(rake_val.split(';')[0])
        except ValueError:
            return np.nan
    else:
        rake = float(rake_val)

    # Classification based on Rake angle (Basili & Tiberti, 2025)
    if 45 <= rake <= 135:
        return 1  # Reverse Fault
    elif -135 <= rake <= -45:
        return 0  # Normal Fault
    else:
        return 2  # Strike-Slip Fault

datagempa7['Fault_Type_New'] = datagempa7['Rake'].apply(classify_fault_type)

if 'Fault_Type_Encoded' in datagempa7.columns:
    datagempa7 = datagempa7.drop('Fault_Type_Encoded', axis=1)

print("New Fault Class Distribution:")
print(datagempa7['Fault_Type_New'].value_counts())
print("\nDescription: 1 = Reverse, 0 = Normal, 2 = Strike-Slip")

# One-hot encoding for fault types
datagempa7 = pd.get_dummies(datagempa7, columns=['Fault_Type_New'], prefix='Fault', dtype=int)
print("Columns after One-Hot Encoding:")
print(datagempa7[['Fault_0', 'Fault_1', 'Fault_2']].head())

# ==========================================
# 3. DATASET SETUP & TRAIN-TEST SPLIT
# ==========================================

# Dataset 2: Includes seismic parameters + fault type features
X = datagempa7[['MT_Magnitude(Mw)', 'MT_Depth(km)', 'Longitude(deg)', 'Latitude(deg)', 'Fault_0', 'Fault_1', 'Fault_2']]
y = datagempa7['Label_Tsunami']

# Dataset 1: Baseline seismic parameters only (without fault type)
A = datagempa7[['MT_Magnitude(Mw)', 'MT_Depth(km)', 'Longitude(deg)']]
B = datagempa7['Label_Tsunami']

# Train-test split (80% train, 20% test with stratification)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2, random_state=42, stratify=B)

# ==========================================
# 4. EVALUATION & VISUALIZATION FUNCTION
# ==========================================

def evaluate_model_complete(model, X_train, y_train, X_test, y_test,
                            threshold=0.5, model_name="Model", feature_names=None,
                            save_path=None):

    print(f"\n" + "="*60)
    print(f"EVALUATION: {model_name.upper()} (THRESHOLD: {threshold})")
    print("="*60)

    # Probability predictions
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Custom threshold predictions
    y_pred_train_custom = (y_prob_train >= threshold).astype(int)
    y_pred_test_custom = (y_prob_test >= threshold).astype(int)

    # Train metrics
    acc_train = accuracy_score(y_train, y_pred_train_custom)
    prec_train = precision_score(y_train, y_pred_train_custom, zero_division=0)
    rec_train = recall_score(y_train, y_pred_train_custom, zero_division=0)
    f1_train = f1_score(y_train, y_pred_train_custom, zero_division=0)
    auc_train = roc_auc_score(y_train, y_prob_train)

    # Test metrics
    acc_test = accuracy_score(y_test, y_pred_test_custom)
    prec_test = precision_score(y_test, y_pred_test_custom, zero_division=0)
    rec_test = recall_score(y_test, y_pred_test_custom, zero_division=0)
    f1_test = f1_score(y_test, y_pred_test_custom, zero_division=0)
    auc_test = roc_auc_score(y_test, y_prob_test)

    print(f"{'Metric':<12} | {'Training Data':<20} | {'Testing Data':<20}")
    print("-" * 60)
    print(f"{'Accuracy':<12} | {acc_train:.4f}               | {acc_test:.4f}")
    print(f"{'Precision':<12} | {prec_train:.4f}               | {prec_test:.4f}")
    print(f"{'Recall':<12} | {rec_train:.4f}               | {rec_test:.4f}")
    print(f"{'F1-Score':<12} | {f1_train:.4f}               | {f1_test:.4f}")
    print(f"{'AUC-ROC':<12} | {auc_train:.4f}               | {auc_test:.4f}")
    print("-" * 60)

    # Visualization plots
    plt.figure(figsize=(24, 5))

    # Plot 1: Confusion Matrix
    plt.subplot(1, 4, 1)
    cm = confusion_matrix(y_test, y_pred_test_custom)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Tsunami (0)', 'Tsunami (1)'],
                yticklabels=['Non-Tsunami (0)', 'Tsunami (1)'])
    plt.title(f'Confusion Matrix\n (Threshold = {threshold})')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')

    # Plot 2: ROC Curve
    plt.subplot(1, 4, 2)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
    plt.plot(fpr_train, tpr_train, label=f'Train AUC: {auc_train:.3f}')
    plt.plot(fpr_test, tpr_test, label=f'Test AUC: {auc_test:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # Plot 3: Precision-Recall Curve
    plt.subplot(1, 4, 3)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob_test)
    plt.plot(rec_curve, prec_curve, marker='.', label='PR Curve')
    plt.axhline(y=prec_test, color='r', linestyle='--', label=f'Threshold {threshold}')
    plt.axvline(x=rec_test, color='r', linestyle='--')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    # Plot 4: Feature Importance
    plt.subplot(1, 4, 4)
    if hasattr(model, 'best_estimator_'):
        importances = model.best_estimator_.feature_importances_
    else:
        importances = model.feature_importances_

    if feature_names is not None:
        imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        ax = sns.barplot(x=imp_series.values, y=imp_series.index, hue=imp_series.index, palette='viridis', legend=False)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')

        for p in ax.patches:
            bar_width = p.get_width()
            ax.annotate(f'{bar_width:.4f}',
                        (bar_width, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center',
                        fontsize=10, color='black',
                        xytext=(5, 0),
                        textcoords='offset points')

        plt.xlim(0, max(imp_series.values) * 1.15)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.1, transparent=False)
        print(f"[INFO] HD plot successfully exported as: {save_path}")
    plt.show()

# ==========================================
# 5. MODEL TRAINING & HYPERPARAMETER TUNING
# ==========================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest Hyperparameter Grid
rf_params = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 4, 5, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'class_weight': ['balanced']
}

# --- Random Forest (Dataset 1) ---
print("\n[INFO] Training Random Forest Dataset 1...")
rf_base_1 = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_search_1 = GridSearchCV(rf_base_1, rf_params, scoring='recall', cv=skf, n_jobs=-1)
rf_search_1.fit(A_train, B_train)
print(f"Best Parameters RF Dataset 1: {rf_search_1.best_params_}")

evaluate_model_complete(rf_search_1.best_estimator_, A_train, B_train, A_test, B_test,
                        threshold=0.42,
                        model_name="Random Forest (Dataset 1)",
                        feature_names=A_train.columns)

# --- Random Forest (Dataset 2) ---
print("\n[INFO] Training Random Forest Dataset 2...")
rf_base_2 = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_search_2 = GridSearchCV(rf_base_2, rf_params, scoring='recall', cv=skf, n_jobs=-1)
rf_search_2.fit(X_train, y_train)
print(f"Best Parameters RF Dataset 2: {rf_search_2.best_params_}")

evaluate_model_complete(rf_search_2.best_estimator_, X_train, y_train, X_test, y_test,
                        threshold=0.472,
                        model_name="Random Forest (Dataset 2)",
                        feature_names=X_train.columns)

# Calculate class imbalance ratios for XGBoost
ratio_imbalance_1 = float(len(y_train[y_train==0]) / len(y_train[y_train==1]))
ratio_imbalance_2 = float(len(B_train[B_train==0]) / len(B_train[B_train==1]))

# XGBoost Hyperparameter Grid
xgb_params_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

# --- XGBoost (Dataset 1) ---
print("\n[INFO] Training XGBoost Dataset 1...")
xgb_base_1 = XGBClassifier(scale_pos_weight=ratio_imbalance_2, random_state=42, eval_metric='logloss')
xgb_grid_1 = GridSearchCV(estimator=xgb_base_1, param_grid=xgb_params_grid, scoring='recall', cv=skf, n_jobs=-1, error_score='raise')
xgb_grid_1.fit(A_train, B_train)
print(f"Best Parameters XGB Dataset 1: {xgb_grid_1.best_params_}")

evaluate_model_complete(xgb_grid_1.best_estimator_, A_train, B_train, A_test, B_test,
                        threshold=0.57,
                        model_name="XGBoost (Dataset 1)",
                        feature_names=A_train.columns)

# --- XGBoost (Dataset 2) ---
print("\n[INFO] Training XGBoost Dataset 2...")
xgb_base_2 = XGBClassifier(scale_pos_weight=ratio_imbalance_1, random_state=42, eval_metric='logloss')
xgb_2 = GridSearchCV(estimator=xgb_base_2, param_grid=xgb_params_grid, scoring='recall', cv=skf, n_jobs=-1, error_score='raise')
xgb_2.fit(X_train, y_train)
print(f"Best Parameters XGB Dataset 2: {xgb_2.best_params_}")

evaluate_model_complete(xgb_2.best_estimator_, X_train, y_train, X_test, y_test,
                        threshold=0.5,
                        model_name="XGBoost (Dataset 2)",
                        feature_names=X_train.columns,
                        save_path="reports/figures/xgboost_evaluation_hd.png")
